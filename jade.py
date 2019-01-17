import numpy as np
from scipy import stats
from concurrent import futures
from logging import getLogger
import datetime
from functools import partial

from de_core import DECore

logger = getLogger(__name__)


class JADE(DECore):
    """
    JADE
    """

    def __init__(self,
                 objective_function: callable,
                 ndim: int,
                 lower_limit: np.ndarray,
                 upper_limit: np.ndarray,
                 minimize: bool = True,
                 c: float = 0.1,
                 p: float = 0.05):
        """

        :param objective_function: f(x) callable function
        :param ndim: dimension of x
        :param lower_limit: lower limit of search space 1d-array
        :param upper_limit: upper limit of search space 1d-array
        :param minimize: minimize flag. if the problem is minimization, then set True.
                                        otherwise set False and turning as maximization.
        :param c
        :param p
        """

        super(JADE, self).__init__(objective_function=objective_function,
                                   ndim=ndim,
                                   lower_limit=lower_limit,
                                   upper_limit=upper_limit,
                                   minimize=minimize)

        self._c = c
        self._p = p
        self._archive = []
        self._mu_cr = 0.5
        self._mu_f = 0.5
        self._s_cr = []
        self._s_f = []

    def initialization(self, x_init=None):
        """

        :param x_init: initial value of x (optional)
        :return:
        """
        # initialize x
        if x_init:
            self._x_current = x_init
        else:
            self._x_current = np.random.rand(self._pop, self._nd) * (self._up_lim - self._low_lim) + self._low_lim

        # initialize archive
        self._archive = []

        # initialize mu_cr, mu_f
        self._mu_cr = 0.5
        self._mu_f = 0.5

    def _generate_cr(self):
        """
        generate Cr_i using N(mu_cr, 0.1)

        :return:
        """
        cr_ = stats.norm.rvs(loc=self._mu_cr, scale=0.1)
        # clipping to [0, 1]
        return np.clip(cr_, 0., 1.)

    def _generate_f(self):
        """
        generate F_i using Cauchy(mu_f, 0.1)

        :return:
        """
        f_ = -1
        while f_ <= 0.:
            f_ = stats.cauchy.rvs(loc=self._mu_f, scale=0.1)
            # if F_i >= 1., then set as 1.
            # if F_i <= 0., then regenerate F_i.
            if f_ >= 1.:
                f_ = 1.
        return f_

    def _update_mu_cr(self):
        """
        update mu_cr

        :return:
        """
        self._mu_cr = (1. - self._c) * self._mu_cr + self._c * np.average(self._s_cr)

    def _update_mu_f(self):
        """
        update mu_f

        :return:
        """
        # Lehmer mean
        mean_f = np.sum(np.array(self._s_f) ** 2) / np.sum(self._s_f)

        self._mu_f = (1. - self._c) * self._mu_f + self._c * mean_f

    def _get_x_best_p(self):
        """
        get x_best_p (randomly choice from top of 100p% best vectors)

        :return:
        """

        # top 100p %
        top_n = max(int(float(self._pop) * self._p), 2)

        top_idx = np.argsort(self._f_current)[:top_n] if self._is_minimize else np.argsort(self._f_current)[::-1][:top_n]

        best_p_idx = np.random.choice(top_idx)

        return self._x_current[best_p_idx]

    def _selection(self, p, u, fu):
        """

        :param p: current index
        :param u: trial vectors
        :param fu: evaluation values of trial vectors
        :return:
        """
        # score is better than current
        q1 = fu <= self._f_current[p] if self._is_minimize else fu >= self._f_current[p]
        # over lower limit
        q2 = np.any(u < self._low_lim)
        # over upper limit
        q3 = np.any(u > self._up_lim)
        # q1 ^ ~q2 ^ ~q3
        q = q1 * ~q2 * ~q3

        f_p1 = fu if q else self._f_current[p]
        x_p1 = u if q else self._x_current[p]

        return f_p1, x_p1, q

    def _mutation(self, current, sf):
        """
        current-to-pbest/1

        :param current: current index of population
        :param sf: scaling factor
        :return:
        """

        # x p-best
        x_pbest = self._get_x_best_p()

        # x r1
        r1 = np.random.choice([n for n in range(self._pop) if n != current])
        x_r1 = self._x_current[r1]

        # x~ r2
        # randomly selection from population ^ archive
        r2_ = np.random.choice([n for n in range(self._pop) if n not in [r1, current]]
                               + list(range(self._pop, self._pop + len(self._archive))))

        if len(self._archive):
            x_r2_ = np.concatenate([self._x_current, np.r_[self._archive]])[r2_]
        else:
            x_r2_ = self._x_current[r2_]

        # v
        v = self._x_current[current] + sf * (x_pbest - self._x_current[current]) + sf * (x_r1 - x_r2_)
        return v

    def _crossover(self, v, x, cr):
        """

        :param v: mutant vector
        :param x: current vector
        :param cr: crossover-rate
        :return:
        """
        # crossover
        r = np.random.choice(range(self._nd))
        u = np.zeros(self._nd)

        # binary crossover
        flg = np.equal(r, np.arange(self._nd)) + np.random.rand(self._nd) < cr

        # from mutant vector
        u[flg] = v[flg]
        # from current vector
        u[~flg] = x[~flg]

        return u

    def _process_1_generation(self, current, gen):
        # set random seed
        # seed = current timestamp + current index + current generation
        seed = int(datetime.datetime.now().timestamp()) + current + gen
        np.random.seed(seed)

        # generate F and Cr
        sf = self._generate_f()
        cr = self._generate_cr()

        # mutation
        v_p = self._mutation(current, sf)

        # crossover
        u_p = self._crossover(v_p, self._x_current[current], cr)

        # selection
        f_p1, x_p1, q = self._selection(current, u_p, self._evaluate_with_check(u_p))
        return current, x_p1, f_p1, q, sf, cr

    def _evaluate(self, params):
        current, u = params
        return current, self._evaluate_with_check(u)

    def optimize_mp(self, k_max: int, population: int = 10, proc: [int, None] = None):
        """

        :param k_max: max-iterations
        :param population: number of populations
        :param proc: number of process. if None, then use maximum process
        :return:

        """
        # set population
        self._pop = population

        # initialize
        self.initialization()

        # get fitness of initial x
        with futures.ProcessPoolExecutor(proc) as executor:
            results = executor.map(self._evaluate, zip(range(self._pop), self._x_current))

        self._f_current = np.array([r[1] for r in sorted(list(results))])

        for k in range(k_max):
            # initialize s_cr, s_f
            self._s_cr = []
            self._s_f = []

            # multi-processing
            with futures.ProcessPoolExecutor(proc) as executor:
                results = executor.map(partial(self._process_1_generation, gen=k), range(self._pop))

            # correct results
            _x_current = []
            _f_current = []
            for n, x, fp, q, sf, cr in sorted(results):
                _x_current.append(x)
                _f_current.append(fp)

                if q:
                    self._archive.append(self._x_current[n].copy())
                    self._s_f.append(sf)
                    self._s_cr.append(cr)

            # update current values
            self._x_current = np.r_[_x_current].copy()
            self._f_current = np.array(_f_current).copy()

            best_score = np.amin(self._f_current) if self._is_minimize else np.amax(self._f_current)
            logger.info('k={} best score = {}, mu_cr = {}, mu_f = {}'.format(k, best_score, self._mu_cr, self._mu_f))

            # remove an individual from archive when size of archive is larger than population.
            if len(self._archive) > self._pop:
                r = np.random.choice(range(len(self._archive)), len(self._archive) - self._pop, replace=False)
                arc = [self._archive[a] for a in range(len(self._archive)) if a not in r]
                self._archive = arc

            # update mu_f, mu_cr
            self._update_mu_f()
            self._update_mu_cr()

        # get best point
        best_idx = np.argmin(self._f_current) if self._is_minimize else np.argmax(self._f_current)
        x_best = self._x_current[best_idx]
        logger.info('global best score = {}'.format(self._f_current[best_idx]))
        logger.info('x_best = {}'.format(x_best))
        return x_best

    def optimize(self, k_max: int, population: int = 10):
        """

        :param k_max: max-iterations
        :param population: number of populations
        :return:
        """

        # set population
        self._pop = population

        # initialize
        self.initialization()

        # get fitness of initial x
        self._f_current = np.array([self._evaluate_with_check(x) for x in self._x_current])

        for k in range(k_max):
            # initialize s_cr, s_f
            self._s_cr = []
            self._s_f = []

            for p in range(self._pop):
                # generate F and Cr
                sf = self._generate_f()
                cr = self._generate_cr()

                # mutation
                v_p = self._mutation(p, sf=sf)

                # crossover
                u_p = self._crossover(v_p, self._x_current[p], cr=cr)

                # selection
                f_p1, x_p1, q = self._selection(p, u_p, self._evaluate_with_check(u_p))

                # storing parent-x, cr, f
                if q:
                    self._archive.append(self._x_current[p].copy())
                    self._s_f.append(sf)
                    self._s_cr.append(cr)

                # update current values
                self._f_current[p] = f_p1
                self._x_current[p] = x_p1

            best_score = np.amin(self._f_current) if self._is_minimize else np.amax(self._f_current)
            logger.info('k={} best score = {}, mu_cr = {}, mu_f = {}'.format(k, best_score, self._mu_cr, self._mu_f))

            # remove an individual from archive when size of archive is larger than population.
            if len(self._archive) > self._pop:
                r = np.random.choice(range(len(self._archive)), len(self._archive) - self._pop, replace=False)
                arc = [self._archive[a] for a in range(len(self._archive)) if a not in r]
                self._archive = arc

            # update mu_f, mu_cr
            self._update_mu_f()
            self._update_mu_cr()

        # get best point
        best_idx = np.argmin(self._f_current) if self._is_minimize else np.argmax(self._f_current)
        x_best = self._x_current[best_idx]
        logger.info('global best score = {}'.format(self._f_current[best_idx]))
        logger.info('x_best = {}'.format(x_best))
        return x_best
