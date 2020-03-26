import unittest

from Orange.widgets.tests.base import WidgetTest, GuiTest
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.modelling import RandomForestLearner
import numpy as np

from orangecontrib.prototypes.widgets import owexplpredictions


class TestOwExplainPredictions(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(
            owexplpredictions.OWExplainPredictions)
        self.data = Table('iris')
        self.selection = Table.from_numpy(
            self.data.domain, self.data[1].x.reshape(1, -1), self.data[1].y.reshape(1, -1))
        self.model = RandomForestLearner()(self.data)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.send_signal(self.widget.Inputs.model, self.model)

    def test_sample_too_big(self):
        self.send_signal(self.widget.Inputs.sample, self.data[:5])

        self.assertTrue(self.widget.Error.sample_too_big.is_shown())

        self.send_signal(self.widget.Inputs.sample, self.selection)
        self.assertFalse(self.widget.Error.sample_too_big.is_shown())

    def test_parameter_passing(self):
        self.widget.gui_error = 1
        self.widget.gui_p_val = 0.5
        self.send_signal(self.widget.Inputs.sample,
                         Table.from_table_rows(self.data, [0]))
        self.assertIsNotNone(self.widget.data)
        self.assertIsNotNone(self.widget.model)
        self.assertIsNotNone(self.widget.to_explain)
        self.assertIsNotNone(self.widget.e)
        self.assertEqual(self.widget.e.p_val, 0.5)
        self.assertEqual(self.widget.e.error, 1)


class TestExplainPredictions(unittest.TestCase):

    def setUp(self):
        self.domain = Domain(attributes=[ContinuousVariable('x1'),
                                         ContinuousVariable('x2'),
                                         ContinuousVariable('x3')],
                             class_vars=[ContinuousVariable('y')])
        x = np.random.randint(0, 2, size=(1000, 3))
        y = np.logical_or(np.logical_or(x[:, 0], x[:, 1]), x[:, 2])
        self.error = 0.05
        self.batch_size = 250
        self.or_data = Table.from_numpy(self.domain, x, y)
        self.expl = owexplpredictions.ExplainPredictions(self.or_data,
                                                         RandomForestLearner()(self.or_data),
                                                         error=self.error,
                                                         batch_size=self.batch_size)
        self.num_iter = 20

    def test_tile_instance(self):
        tiled = self.expl.tile_instance(self.or_data[0])
        self.assertEqual(tiled.X.shape, (self.batch_size, tiled.X.shape[1]))

    def test_anytime_explain_or(self):
        x = np.array([0, 1, 1]).reshape(1, -1)
        y = np.array([1])
        instance = Table.from_numpy(self.domain, x, y)
        contributions = np.zeros((3,))
        true_contributions = np.array([-1/24, 1/12, 1/12])
        for _ in range(self.num_iter):
            predicted_val, result = self.expl.anytime_explain(instance[0])
            self.assertEqual(predicted_val, 1)
            contributions += result.X[:, 0]
        contributions /= self.num_iter
        np.testing.assert_allclose(
            contributions, true_contributions, atol=self.error)


if __name__ == "__main__":
    unittest.main()
