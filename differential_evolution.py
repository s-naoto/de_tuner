import numpy as np
from concurrent import futures
from logging import getLogger

from de_core import DECore

logger = getLogger(__name__)


class DE(DECore):
    """
    Differential Evolution
    """

    def __init__(self,
                 objective_function: callable,
                 ndim: int,
                 lower_limit: np.ndarray,
                 upper_limit: np.ndarray,
                 minimize: bool = True):
        """

        :param objective_function: f(x) callable function
        :param ndim: dimension of x
        :param lower_limit: lower limit of search space 1d-array
        :param upper_limit: upper limit of search space 1d-array
        :param minimize: minimize flag. if the problem is minimization, then set True.
                                        otherwise set False and turning as maximization.
        """

        super(DE, self).__init__(objective_function=objective_function,
                                 ndim=ndim,
                                 lower_limit=lower_limit,
                                 upper_limit=upper_limit,
                                 minimize=minimize)

    def _selection(self, p, u):
        """

        :param p: current index
        :param u: trial vectors
        :return:
        """

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
        return p, f_p1, x_p1

    def _mutation(self, current, mutant, num, sf):
        """

        :param current: current index of population
        :param mutant: mutation method
        :param num: number of mutant vectors
        :param sf: scaling factor
        :return:
        """

        assert num > 0, "'num' must be greater than 0."

        # mutant vector
        # best
        if mutant == 'best':
            r_best = np.argmin(self._f_current) if self._is_minimize else np.argmax(self._f_current)
            r = [r_best]
            r += np.random.choice([n for n in range(self._pop) if n != r_best], 2 * num, replace=False).tolist()
            v = self._x_current[r[0]] \
                + sf * np.sum([self._x_current[r[m + 1]] - self._x_current[r[m + 2]] for m in range(num)], axis=0)

        # rand
        elif mutant == 'rand':
            r = np.random.choice(range(self._pop), 2 * num + 1, replace=False).tolist()
            v = self._x_current[r[0]] \
                + sf * np.sum([self._x_current[r[m + 1]] - self._x_current[r[m + 2]] for m in range(num)], axis=0)

        # current-to-rand
        elif mutant == 'current-to-rand':
            r = [current]
            r += np.random.choice([n for n in range(self._pop) if n != current], 2 * num + 1, replace=False).tolist()
            v = self._x_current[r[0]] \
                + sf * (self._x_current[r[1]] - self._x_current[r[0]]) \
                + sf * np.sum([self._x_current[r[m + 2]] - self._x_current[r[m + 3]] for m in range(num)], axis=0)

        # current-to-best
        elif mutant == 'current-to-best':
            r_best = np.argmin(self._f_current) if self._is_minimize else np.argmax(self._f_current)
            r = [r_best, current]
            r += np.random.choice([
                n for n in range(self._pop) if n not in [r_best, current]], 2 * num, replace=False).tolist()
            v = self._x_current[r[0]] \
                + sf * (self._x_current[r[1]] - self._x_current[r[0]]) \
                + sf * np.sum([self._x_current[r[m + 2]] - self._x_current[r[m + 3]] for m in range(num)], axis=0)

        else:
            raise ValueError('invalid `mutant`: {}'.format(mutant))

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

    def _mutation_crossover(self, mutant, num, sf, cross, cr):
        l_up = []
        # for each individuals
        for p in range(self._pop):
            # mutation
            v_p = self._mutation(p, mutant=mutant, num=num, sf=sf)

            # crossover
            u_p = self._crossover(v_p, self._x_current[p], cross=cross, cr=cr)
            l_up.append(u_p)

        return l_up

    def optimize_mp(self,
                    k_max: int,
                    population: int = 10,
                    mutant: str = 'best',
                    num: int = 1,
                    cross: str = 'bin',
                    sf: float = 0.7,
                    cr: float = 0.3,
                    proc: [int, None] = None):
        """

        :param k_max: max-iterations
        :param population: number of populations
        :param mutant: mutation method ['best', 'rand', 'current-to-best', 'current-to-rand']
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
        self._pop = population

        # initialize
        self.initialization()

        # get fitness of initial x
        with futures.ProcessPoolExecutor(proc) as executor:
            results = executor.map(self._evaluate, zip(range(self._pop), self._x_current))

        self._f_current = np.array([r[1] for r in sorted(list(results))])

        for k in range(k_max):
            # mutation and crossover
            l_up = self._mutation_crossover(mutant, num, sf, cross, cr)

            # multi-processing
            with futures.ProcessPoolExecutor(proc) as executor:
                results = executor.map(self._selection, range(self._pop), l_up)

            # correct results
            _x_current = []
            _f_current = []
            for _, fp, x in sorted(results):
                _x_current.append(x)
                _f_current.append(fp)

            # update current values
            self._x_current = np.r_[_x_current].copy()
            self._f_current = np.array(_f_current).copy()

            best_score = np.amin(self._f_current) if self._is_minimize else np.amax(self._f_current)
            logger.info('k={} best score = {}'.format(k, best_score))
            self._orbit.append(best_score)

        # get best point
        best_idx = np.argmin(self._f_current) if self._is_minimize else np.argmax(self._f_current)
        x_best = self._x_current[best_idx]
        logger.info('global best score = {}'.format(self._f_current[best_idx]))
        logger.info('x_best = {}'.format(x_best))
        return x_best

    def optimize(self,
                 k_max: int,
                 population: int = 10,
                 mutant: str = 'best',
                 num: int = 1,
                 cross: str = 'bin',
                 sf: float = 0.7,
                 cr: float = 0.3):
        """

        :param k_max: max-iterations
        :param population: number of populations
        :param mutant: mutation method ['best', 'rand', 'current-to-best', 'current-to-rand']
        :param num: number of mutant vectors
        :param cross: crossover method ['bin', 'exp']
        :param sf: scaling-factor F
        :param cr: crossover-rate CR
        :return:

        ex) DE/rand/1/bin --> method='rand', num=1, cross='bin'
            DE/best/2/exp --> method='best', num=2, cross='exp'
        """
        # set population
        self._pop = population

        # initialize
        self.initialization()

        # get fitness of initial x
        self._f_current = np.array([self._evaluate_with_check(x) for x in self._x_current])

        for k in range(k_max):
            # mutation and crossover
            l_up = self._mutation_crossover(mutant, num, sf, cross, cr)

            for p, u_p in enumerate(l_up):
                # selection
                _, f_p1, x_p1 = self._selection(p, u_p)

                # update current values
                self._f_current[p] = f_p1
                self._x_current[p] = x_p1

            best_score = np.amin(self._f_current) if self._is_minimize else np.amax(self._f_current)
            logger.info('k={} best score = {}'.format(k, best_score))
            self._orbit.append(best_score)

        # get best point
        best_idx = np.argmin(self._f_current) if self._is_minimize else np.argmax(self._f_current)
        x_best = self._x_current[best_idx]
        logger.info('global best score = {}'.format(self._f_current[best_idx]))
        logger.info('x_best = {}'.format(x_best))
        return x_best
