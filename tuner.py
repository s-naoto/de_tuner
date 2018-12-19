import numpy as np
from sklearn.model_selection import KFold
from tempfile import TemporaryDirectory
import joblib
from differential_evolution import DE
from pathlib import Path
from logging import getLogger, basicConfig

logger = getLogger('__name__')


class HyperTuner(object):
    def __init__(self, model, space, k_fold=5, **params):
        """

        :param model: target model
        :param space: search space
        :param k_fold: number of folders for K-fold CV
        :param params: parameters for optimizer

        space = {
            'parameter': {'scale': linear', 'range': [0, 1.5]},
            'parameter': {'scale': 'log', 'range': [-1, 2]},
            'parameter': {'scale': 'category', 'range': ['a', 'b', 'c']},
            'parameter': {'scale': 'integer', 'range': [0, 10]}
        }
        """
        self._model = model
        assert isinstance(space, dict)
        self._space = space
        self._parameters = list(self._space.keys())
        self._tempdir = TemporaryDirectory()
        self._tempfile = Path(self._tempdir.name + 'temp_data.gz')
        self._eval_function = None
        default_opt_param = {'k_max': 100,
                             'population': 10,
                             'mutant': 'best',
                             'num': 1,
                             'cross': 'bin',
                             'sf': 0.7,
                             'cr': 0.4}
        self._optimizer_param = default_opt_param
        self._optimizer_param.update(params)
        self._kf = k_fold

    def __del__(self):
        self._tempdir.cleanup()

    def _get_search_limits(self):
        lowers = []
        uppers = []
        for k in self._parameters:
            if self._space[k]['scale'] in ['linear', 'log']:
                lowers.append(self._space[k]['range'][0])
                uppers.append(self._space[k]['range'][1])
            elif self._space[k]['scale'] == 'integer':
                lowers.append(self._space[k]['range'][0])
                uppers.append(self._space[k]['range'][1] + 1)
            else:
                lowers.append(0)
                uppers.append(len(self._space[k]['range']))

        return np.array(lowers), np.array(uppers)

    def _translate_to_origin(self, x):
        org_x = {}
        for n, k in enumerate(self._parameters):
            if self._space[k]['scale'] == 'log':
                org_x[k] = np.power(10, x[n])
            elif self._space[k]['scale'] == 'category':
                org_x[k] = self._space[k]['range'][int(x[n])]
            elif self._space[k]['scale'] == 'integer':
                org_x[k] = int(x[n])
            else:
                org_x[k] = x[n]
        return org_x

    def _evaluate(self, x):
        # load data from temporary directory
        input_data, targets = joblib.load(self._tempfile)

        try:
            # set model using parameter x
            param = self._translate_to_origin(x)
            model = self._model.set_params(**param)

            # train model using CV (K-fold)
            skf = KFold(n_splits=self._kf, shuffle=True)
            scores = []
            for train, test in skf.split(input_data, targets):
                x_tr, t_tr = input_data[train], targets[train]
                x_te, t_te = input_data[test], targets[test]

                model.fit(x_tr, t_tr)
                scores.append(self._eval_function(y_pred=model.predict(x_te), y_true=t_te))

            # average score
            score = np.average(scores)
        except Exception as ex:
            logger.error(ex)
            # if some errors are occurred, score is +infinity.
            score = np.inf
        return score

    def tuning(self, eval_function: callable, x, t, minimize=True):
        joblib.dump((x, t), self._tempfile)

        # set DE
        lower_limit, upper_limit = self._get_search_limits()

        # set evaluation function
        self._eval_function = eval_function
        optimizer = DE(objective_function=self._evaluate, ndim=len(self._parameters), lower_limit=lower_limit,
                       upper_limit=upper_limit, minimize=minimize)

        x_best = optimizer.optimize_mp(**self._optimizer_param)

        return self._translate_to_origin(x_best)


if __name__ == '__main__':
    basicConfig(level='INFO')

    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    search_space = {'n_estimators': {'scale': 'category', 'range': [10, 50, 100, 200, 250, 300]},
                    'max_depth': {'scale': 'integer', 'range': [1, 8]},
                    'min_samples_split': {'scale': 'log', 'range': [-3, 0]},
                    'min_samples_leaf': {'scale': 'linear', 'range': [0, 0.5]},
                    'min_weight_fraction_leaf': {'scale': 'linear', 'range': [0, 0.5]},
                    'max_features': {'scale': 'category', 'range': ['auto', 'sqrt', 'log2', None]}}

    dataset = load_digits()

    tuner = HyperTuner(model=RandomForestClassifier(), space=search_space)
    best_param = tuner.tuning(eval_function=accuracy_score, x=dataset.data, t=dataset.target, minimize=False)
    logger.info('best parameter = {}'.format(best_param))
