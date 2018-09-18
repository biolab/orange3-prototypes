import unittest
from Orange.data import Table
from Orange.modelling import KNNLearner, TreeLearner
from Orange.evaluation import CA, CrossValidation, MSE

from orangecontrib.prototypes.stack import StackedFitter


class TestStackedFitter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.housing = Table('housing')

    def test_classification(self):
        sf = StackedFitter([TreeLearner(), KNNLearner()])
        results = CrossValidation(self.iris, [sf], k=3)
        ca = CA(results)
        self.assertGreater(ca, 0.9)

    def test_regression(self):
        sf = StackedFitter([TreeLearner(), KNNLearner()])
        results = CrossValidation(self.housing[:50],
                                  [sf, TreeLearner(), KNNLearner()], k=3)
        mse = MSE(results)
        self.assertLess(mse[0], mse[1])
        self.assertLess(mse[0], mse[2])
