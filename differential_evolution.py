import numpy as np
import multiprocessing as mp
from logging import getLogger
import datetime
from functools import partial

logger = getLogger('__name__')


class DE(object):
    """
    Differential Evolution
    """

    def __init__(self, objective_function, ndim, lower_limit, upper_limit, minimize=True):
        """

        :param objective_function: f(x) callable function
        :param ndim: dimension of x
        :param lower_limit: lower limit of search space 1d-array
        :param upper_limit: upper limit of search space 1d-array
        :param minimize: minimize flag. if the problem is minimization, then set True.
                                        otherwise set False and turning as maximization.
        """
        self._of = objective_function
        self._p = None
        self._nd = ndim
        self._x_current = None
        self._low_lim = lower_limit
        self._up_lim = upper_limit
        self._f_current = None
        self._is_minimize = minimize

    def initialization(self, x_init=None):
        """

        :param x_init: initial value of x (optional)
        :return:
        """
        # initialize x
        if x_init:
            self._x_current = x_init
        else:
            self._x_current = np.random.rand(self._p, self._nd) * (self._up_lim - self._low_lim) + self._low_lim

    def _update(self, u, fu):
        """

        :param u: trial vectors
        :param fu: evaluation values of trial vectors
        :return:
        """
        # score is better than current
        q1 = fu < self._f_current if self._is_minimize else fu > self._f_current
        # over lower limit
        q2 = np.any(u < self._low_lim, axis=1)
        # over upper limit
        q3 = np.any(u > self._up_lim, axis=1)
        # q1 ^ ~q2 ^ ~q3
        q = np.where(q1 * ~q2 * ~q3)

        # update current values
        self._f_current[q] = fu[q].copy()
        self._x_current[q] = u[q].copy()

    def _get_mutant_vector(self, current, method, num, sf):
        """

        :param current: current index of population
        :param method: mutation method
        :param num: number of mutant vectors
        :param sf: scaling factor
        :return:
        """
        # mutant vector
        # best
        if method == 'best':
            r_best = np.argmin(self._f_current) if self._is_minimize else np.argmax(self._f_current)
            r = [r_best]
            r += np.random.choice([n for n in range(self._p) if n != r_best], 2 * num, replace=False).tolist()
            v = self._x_current[r[0]] \
                + sf * np.sum([self._x_current[r[m + 1]] - self._x_current[r[m + 2]] for m in range(num)], axis=0)

        # rand
        elif method == 'rand':
            r = np.random.choice(range(self._p), 2 * num + 1, replace=False).tolist()
            v = self._x_current[r[0]] \
                + sf * np.sum([self._x_current[r[m + 1]] - self._x_current[r[m + 2]] for m in range(num)], axis=0)

        # current-to-rand
        elif method == 'current-to-rand':
            r = [current]
            r += np.random.choice([n for n in range(self._p) if n != current], 2 * num + 1, replace=False).tolist()
            v = self._x_current[r[0]] \
                + np.random.rand() * (self._x_current[r[1]] - self._x_current[r[0]]) \
                + sf * np.sum([self._x_current[r[m + 2]] - self._x_current[r[m + 3]] for m in range(num)], axis=0)

        # current-to-best
        elif method == 'current-to-best':
            r_best = np.argmin(self._f_current) if self._is_minimize else np.argmax(self._f_current)
            r = [r_best, current]
            r += np.random.choice([
                n for n in range(self._p) if n not in [r_best, current]], 2 * num, replace=False).tolist()
            v = self._x_current[r[0]] \
                + sf * (self._x_current[r[1]] - self._x_current[r[0]]) \
                + sf * np.sum([self._x_current[r[m + 2]] - self._x_current[r[m + 3]] for m in range(num)], axis=0)

        else:
            raise ValueError('invalid `method`: {}'.format(method))

        return v

    def _crossover(self, v, x, cross, cr):
        """

        :param v: mutant vector
        :param x: current vector
        :param cross: crossover method
        :param cr: crossover-rate
        :return:
        """
        # crossover
        r = np.random.choice(range(self._nd))
        u = np.zeros(self._nd)

        # binary crossover
        if cross == 'bin':
            flg = np.equal(r, np.arange(self._nd)) + np.random.rand(self._nd) < cr

        # exponential crossover
        elif cross == 'exp':
            flg = np.array([False for _ in range(self._nd)])
            for l in range(self._nd):
                flg[r] = True
                r = (r + 1) % self._nd
                if np.random.rand() >= cr:
                    break
        else:
            raise ValueError('invalid `cross`: {}'.format(cross))

        # from mutant vector
        u[flg] = v[flg]
        # from current vector
        u[~flg] = x[~flg]

        return u

    def _evaluate_with_check(self, x):
        if np.any(x < self._low_lim) or np.any(x > self._up_lim):
            return np.inf
        else:
            return self._of(x)

    def _process_1_generation(self, current, gen, method, num, cross, sf, cr):
        # set random seed
        # seed = current timestamp + current index + current generation
        seed = int(datetime.datetime.now().timestamp()) + current + gen
        np.random.seed(seed)

        # mutation
        v_p = self._get_mutant_vector(current, method, num, sf)

        # crossover
        u_p = self._crossover(v_p, self._x_current[current], cross, cr)

        # evaluation
        return current, u_p, self._evaluate_with_check(u_p)

    def _exec_1_generation(self, gen, method, num, cross, sf, cr, proc):
        with mp.Pool(proc) as pool:
            results = pool.map(partial(self._process_1_generation, gen=gen, method=method, num=num, cross=cross,
                                       sf=sf, cr=cr), range(self._p))

        u_current = []
        fu = []
        for _, u, fp in sorted(results):
            u_current.append(u)
            fu.append(fp)
        return u_current, fu

    def _evaluate(self, current, u):
        return current, self._evaluate_with_check(u)

    def optimize_mp(self, k_max, population=10, method='best', num=1, cross='bin', sf=0.7, cr=0.3, proc=None):
        """

        :param k_max: max-iterations
        :param population: number of populations
        :param method: mutation method ['best', 'rand', 'current-to-best', 'current-to-rand']
        :param num: number of mutant vectors
        :param cross: crossover method ['bin', 'exp']
        :param sf: scaling-factor F
        :param cr: crossover-rate CR
        :param proc: number of process. if None, then use maximum process
        :return:

        ex) DE/rand/1/bin --> method='rand', num=1, cross='bin'
            DE/best/2/exp --> method='best', num=2, cross='exp'
        """
        # set population
        self._p = population

        # initialize
        self.initialization()

        # get fitness of initial x
        with mp.Pool(proc) as pool:
            results = pool.starmap(self._evaluate, [(n, u) for n, u in zip(range(self._p), self._x_current)])
        self._f_current = np.array([r[1] for r in sorted(results)])

        for k in range(k_max):
            u_current, fu = self._exec_1_generation(k, method, num, cross, sf, cr, proc)

            # selection
            self._update(np.stack(u_current, axis=0), np.array(fu))

            best_score = np.amin(self._f_current) if self._is_minimize else np.amax(self._f_current)
            logger.info('k={} best score = {}'.format(k, best_score))

        # get best point
        best_idx = np.argmin(self._f_current) if self._is_minimize else np.argmax(self._f_current)
        x_best = self._x_current[best_idx]
        logger.info('global best score = {}'.format(self._f_current[best_idx]))
        logger.info('x_best = {}'.format(x_best))
        return x_best

    def optimize(self, k_max, population=10, method='best', num=1, cross='bin', sf=0.7, cr=0.3):
        """

        :param k_max: max-iterations
        :param population: number of populations
        :param method: mutation method ['best', 'rand', 'current-to-best', 'current-to-rand']
        :param num: number of mutant vectors
        :param cross: crossover method ['bin', 'exp']
        :param sf: scaling-factor F
        :param cr: crossover-rate CR
        :return:

        ex) DE/rand/1/bin --> method='rand', num=1, cross='bin'
            DE/best/2/exp --> method='best', num=2, cross='exp'
        """
        # set population
        self._p = population

        # initialize
        self.initialization()

        # get fitness of initial x
        self._f_current = np.array([self._of(x) for x in self._x_current])

        for k in range(k_max):
            u_current = []
            fu = []
            for p in range(self._p):
                # mutation
                v_p = self._get_mutant_vector(p, method=method, num=num, sf=sf)

                # crossover
                u_p = self._crossover(v_p, self._x_current[p], cross=cross, cr=cr)

                # evaluation
                u_current.append(u_p)
                fu.append(self._evaluate_with_check(u_p))

            # selection
            self._update(np.stack(u_current, axis=0), np.array(fu))

            best_score = np.amin(self._f_current) if self._is_minimize else np.amax(self._f_current)
            logger.info('k={} best score = {}'.format(k, best_score))

        # get best point
        best_idx = np.argmin(self._f_current) if self._is_minimize else np.argmax(self._f_current)
        x_best = self._x_current[best_idx]
        logger.info('global best score = {}'.format(self._f_current[best_idx]))
        logger.info('x_best = {}'.format(x_best))
        return x_best
