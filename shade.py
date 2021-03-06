import numpy as np
from scipy import stats
from concurrent import futures
from logging import getLogger

from de_core import DECore

logger = getLogger(__name__)


class SHADE(DECore):
    """
    SHADE: Success-History based Adaptive Differential Evolution
    """

    def __init__(self,
                 objective_function: callable,
                 ndim: int,
                 lower_limit: np.ndarray,
                 upper_limit: np.ndarray,
                 minimize: bool = True,
                 h: int = 10,
                 p: float = 0.05):
        """

        :param objective_function: f(x) callable function
        :param ndim: dimension of x
        :param lower_limit: lower limit of search space 1d-array
        :param upper_limit: upper limit of search space 1d-array
        :param minimize: minimize flag. if the problem is minimization, then set True.
                                        otherwise set False and turning as maximization.
        :param h: memory size
        :param p: selection rate for `current-pbest`
        """

        super(SHADE, self).__init__(objective_function=objective_function,
                                    ndim=ndim,
                                    lower_limit=lower_limit,
                                    upper_limit=upper_limit,
                                    minimize=minimize)

        self._h = h
        self._p = p
        self._k = None
        self._archive = []
        self._m_cr = self._m_f = None

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
        self._m_cr = [0.5 for _ in range(self._h)]
        self._m_f = [0.5 for _ in range(self._h)]

        # initialize k
        self._k = 0

        # initialize orbit
        self._orbit = []

    def _generate_cr(self, r):
        """
        generate Cr_i using N(mu_cr, 0.1)

        :return:
        """
        cr_ = stats.norm.rvs(loc=self._m_cr[r], scale=0.1)
        # clipping to [0, 1]
        return np.clip(cr_, 0., 1.)

    def _generate_f(self, r):
        """
        generate F_i using Cauchy(mu_f, 0.1)

        :return:
        """
        f_ = -1
        while f_ <= 0.:
            f_ = stats.cauchy.rvs(loc=self._m_f[r], scale=0.1)
            # if F_i >= 1., then set as 1.
            # if F_i <= 0., then regenerate F_i.
            if f_ >= 1.:
                f_ = 1.
        return f_

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

    def _selection(self, p, u):
        """

        :param p: current index
        :param u: trial vectors
        :return:
        """
        # evaluate optimized function
        fu = self._evaluate_with_check(u)

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

        return p, f_p1, x_p1, q

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

    @staticmethod
    def _lehmer_mean(s):
        return np.sum(np.array(s) ** 2) / np.sum(s)

    def _mutation_crossover(self):
        l_up = []
        l_sf = []
        l_cr = []
        # for each individuals
        for p in range(self._pop):
            # generate F and Cr
            ri = np.random.choice(range(self._h))
            sf = self._generate_f(ri)
            cr = self._generate_cr(ri)

            # mutation
            v_p = self._mutation(p, sf=sf)

            # crossover
            u_p = self._crossover(v_p, self._x_current[p], cr=cr)

            # storing trial vectors, scaling-factor, crossover-rate
            l_up.append(u_p)
            l_sf.append(sf)
            l_cr.append(cr)

        return l_up, l_sf, l_cr

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
            s_cr = []
            s_f = []

            # mutation and crossover
            l_up, l_sf, l_cr = self._mutation_crossover()

            # multi-processing
            with futures.ProcessPoolExecutor(proc) as executor:
                results = executor.map(self._selection, range(self._pop), l_up)

            # correct results
            _x_current = []
            _f_current = []
            for n, fp, x, q in sorted(results):
                _x_current.append(x)
                _f_current.append(fp)

                if q:
                    self._archive.append(self._x_current[n].copy())
                    s_f.append(l_sf[n])
                    s_cr.append(l_cr[n])

            # update current values
            self._x_current = np.r_[_x_current].copy()
            self._f_current = np.array(_f_current).copy()

            best_score = np.amin(self._f_current) if self._is_minimize else np.amax(self._f_current)
            logger.info('k={} best score = {}'.format(k, best_score))
            self._orbit.append(best_score)

            # remove an individual from archive when size of archive is larger than population.
            if len(self._archive) > self._pop:
                r = np.random.choice(range(len(self._archive)), len(self._archive) - self._pop, replace=False)
                arc = [self._archive[a] for a in range(len(self._archive)) if a not in r]
                self._archive = arc

            # update m_f, m_cr
            if len(s_f) > 0 and len(s_cr) > 0:
                self._m_f[self._k] = self._lehmer_mean(s_f)
                self._m_cr[self._k] = self._lehmer_mean(s_cr)
                self._k = np.mod(self._k + 1, self._h)

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
            s_cr = []
            s_f = []

            # mutation and crossover
            l_up, l_sf, l_cr = self._mutation_crossover()

            for p, u_p in enumerate(l_up):
                # selection
                _, f_p1, x_p1, q = self._selection(p, u_p)

                # storing parent-x, cr, f
                if q:
                    self._archive.append(self._x_current[p].copy())
                    s_f.append(l_sf[p])
                    s_cr.append(l_cr[p])

                # update current values
                self._f_current[p] = f_p1
                self._x_current[p] = x_p1

            best_score = np.amin(self._f_current) if self._is_minimize else np.amax(self._f_current)
            logger.info('k={} best score = {}'.format(k, best_score))
            self._orbit.append(best_score)

            # remove an individual from archive when size of archive is larger than population.
            if len(self._archive) > self._pop:
                r = np.random.choice(range(len(self._archive)), len(self._archive) - self._pop, replace=False)
                arc = [self._archive[a] for a in range(len(self._archive)) if a not in r]
                self._archive = arc

            # update m_f, m_cr
            if len(s_f) > 0 and len(s_cr) > 0:
                self._m_f[self._k] = self._lehmer_mean(s_f)
                self._m_cr[self._k] = self._lehmer_mean(s_cr)
                self._k = np.mod(self._k + 1, self._h)

        # get best point
        best_idx = np.argmin(self._f_current) if self._is_minimize else np.argmax(self._f_current)
        x_best = self._x_current[best_idx]
        logger.info('global best score = {}'.format(self._f_current[best_idx]))
        logger.info('x_best = {}'.format(x_best))
        return x_best
