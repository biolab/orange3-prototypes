import numpy as np

from Orange.base import Learner, Model
from Orange.classification import LogisticRegressionLearner
from Orange.classification.base_classification import LearnerClassification
from Orange.data import Domain, ContinuousVariable, Table
from Orange.evaluation import CrossValidation
from Orange.regression import RidgeRegressionLearner
from Orange.regression.base_regression import LearnerRegression


__all__ = ['StackedLearner', 'StackedClassificationLearner',
           'StackedRegressionLearner']


class StackedModel(Model):
    def __init__(self, models, aggregate):
        self.models = models
        self.aggregate = aggregate

    def predict_storage(self, data):
        probs = [m(data, Model.Probs) for m in self.models]
        X = np.hstack(probs)
        Y = np.repeat(np.nan, X.shape[0])
        stacked_data = Table(self.aggregate.domain, X, Y)
        return self.aggregate(stacked_data, Model.ValueProbs)


class StackedLearner(Learner):
    __returns__ = StackedModel

    def __init__(self, learners, aggregate, k=5, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.learners = learners
        self.aggregate = aggregate
        self.k = k
        self.params = vars()

    def fit_storage(self, data):
        res = CrossValidation(data, self.learners, k=self.k)
        X = np.hstack(res.probabilities)
        dom = Domain([ContinuousVariable('f{}'.format(i))
                      for i in range(1, X.shape[1] + 1)],
                     data.domain.class_var)
        stacked_data = Table(dom, X, res.actual)
        models = [l(data) for l in self.learners]
        aggregate_model = self.aggregate(stacked_data)
        return StackedModel(models, aggregate_model)


class StackedClassificationLearner(StackedLearner, LearnerClassification):
    def __init__(self, learners, aggregate=LogisticRegressionLearner(), k=5):
        super().__init__(learners=learners, aggregate=aggregate, k=k)


class StackedRegressionLearner(StackedLearner, LearnerRegression):
    def __init__(self, learners, aggregate=RidgeRegressionLearner(), k=5):
        super().__init__(learners=learners, aggregate=aggregate, k=k)


if __name__ == '__main__':
    import Orange
    iris = Table('iris')
    knn = Orange.modelling.KNNLearner()
    tree = Orange.modelling.TreeLearner()
    lr = Orange.classification.LogisticRegressionLearner()
    sl = StackedClassificationLearner([tree, knn])
    m = sl(iris[::2])
    print(m(iris[1::2], Model.Value))
