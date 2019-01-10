import numpy as np
from logging import getLogger

logger = getLogger(__name__)


class DECore(object):
    """
    Core Class of Differential Evolution
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

    def _selection(self, p, u, fu):
        """

        :param p: current index
        :param u: trial vectors
        :param fu: evaluation values of trial vectors
        :return:
        """

        pass

    def _mutation(self, **kwargs):

        pass

    def _crossover(self, **kwargs):

        pass

    def _evaluate_with_check(self, x):
        if np.any(x < self._low_lim) or np.any(x > self._up_lim):
            return np.inf if self._is_minimize else -np.inf
        else:
            try:
                f = self._of(x)
            except Exception as ex:
                logger.error(ex)
                f = np.inf if self._is_minimize else -np.inf
            return f

    def optimize_mp(self, **kwargs):

        pass

    def optimize(self, **kwargs):

        pass
