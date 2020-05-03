# pylint: disable=missing-docstring
import inspect
import itertools
import unittest

from AnyQt.QtWidgets import QGraphicsLinearLayout

import Orange
from Orange.base import Learner
from Orange.classification import RandomForestLearner, OneClassSVMLearner, \
    IsolationForestLearner, EllipticEnvelopeLearner, LocalOutlierFactorLearner
from Orange.data import Table
from Orange.regression import RandomForestRegressionLearner
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.prototypes.widgets.owexplainpred import StripePlot, \
    OWExplainPrediction


class TestOWExplainPrediction(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")
        cls.heart = Table("heart_disease")
        cls.housing = Table("housing")
        cls.rf_cls = RandomForestLearner(random_state=42)(cls.iris)
        cls.rf_reg = RandomForestRegressionLearner(
            random_state=42)(cls.housing)

    def setUp(self):
        self.widget = self.create_widget(OWExplainPrediction)

    def test_inputs(self):
        self.send_signal(self.widget.Inputs.background_data, self.heart)
        self.send_signal(self.widget.Inputs.data, self.heart[:1])
        rf_cls = RandomForestLearner(random_state=42)(self.heart)
        self.send_signal(self.widget.Inputs.model, rf_cls)
        self.wait_until_finished()
        self.assertPlotNotEmpty(self.widget._stripe_plot)

    def test_classification_data_classification_model(self):
        self.send_signal(self.widget.Inputs.background_data, self.iris)
        self.send_signal(self.widget.Inputs.data, self.iris[:1])
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertPlotNotEmpty(self.widget._stripe_plot)

    def test_classification_data_regression_model(self):
        self.send_signal(self.widget.Inputs.background_data, self.iris)
        self.send_signal(self.widget.Inputs.data, self.iris[:1])
        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget._stripe_plot)
        self.assertTrue(self.widget.Error.domain_transform_err.is_shown())

    def test_regression_data_regression_model(self):
        self.send_signal(self.widget.Inputs.background_data, self.housing)
        self.send_signal(self.widget.Inputs.data, self.housing[:1])
        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.wait_until_finished()
        self.assertPlotNotEmpty(self.widget._stripe_plot)

    def test_regression_data_classification_model(self):
        self.send_signal(self.widget.Inputs.background_data, self.housing)
        self.send_signal(self.widget.Inputs.data, self.housing[:1])
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget._stripe_plot)
        self.assertTrue(self.widget.Error.domain_transform_err.is_shown())

    def test_output_scores(self):
        self.send_signal(self.widget.Inputs.background_data, self.iris)
        self.send_signal(self.widget.Inputs.data, self.iris[:1])
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.scores)
        self.assertIsInstance(output, Table)
        self.assertEqual(list(output.metas.flatten()),
                         [a.name for a in self.iris.domain.attributes])
        self.send_signal(self.widget.Inputs.model, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.scores))

    def test_all_models(self):
        def run(data):
            self.send_signal(self.widget.Inputs.background_data, data)
            self.send_signal(self.widget.Inputs.data, data)
            if not issubclass(cls, Learner) or \
                    issubclass(cls, (EllipticEnvelopeLearner,
                                     LocalOutlierFactorLearner,
                                     IsolationForestLearner,
                                     OneClassSVMLearner)):
                return
            try:
                model = cls()(data)
            except:
                return
            self.send_signal(self.widget.Inputs.model, model)
            self.wait_until_finished(timeout=50000)

        for _, cls in itertools.chain(
                inspect.getmembers(Orange.regression, inspect.isclass),
                inspect.getmembers(Orange.modelling, inspect.isclass)):
            run(self.housing[::4])
        for _, cls in itertools.chain(
                inspect.getmembers(Orange.classification, inspect.isclass),
                inspect.getmembers(Orange.modelling, inspect.isclass)):
            run(self.iris[::4])

    def test_target_combo(self):
        text = "Iris-setosa"
        self.assertEqual(self.widget._target_combo.currentText(), "")
        self.assertTrue(self.widget._target_combo.isEnabled())

        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.assertEqual(self.widget._target_combo.currentText(), text)
        self.assertTrue(self.widget._target_combo.isEnabled())

        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.assertEqual(self.widget._target_combo.currentText(), "")
        self.assertFalse(self.widget._target_combo.isEnabled())

        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.assertEqual(self.widget._target_combo.currentText(), text)
        self.assertTrue(self.widget._target_combo.isEnabled())

        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.assertEqual(self.widget._target_combo.currentText(), "")
        self.assertFalse(self.widget._target_combo.isEnabled())

        self.send_signal(self.widget.Inputs.model, None)
        self.assertEqual(self.widget._target_combo.currentText(), "")
        self.assertTrue(self.widget._target_combo.isEnabled())

    def test_plot(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertPlotEmpty(self.widget._stripe_plot)

        self.send_signal(self.widget.Inputs.background_data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertPlotNotEmpty(self.widget._stripe_plot)

        self.send_signal(self.widget.Inputs.model, None)
        self.assertPlotEmpty(self.widget._stripe_plot)

        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget._stripe_plot)

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget._stripe_plot)

        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertPlotNotEmpty(self.widget._stripe_plot)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertPlotEmpty(self.widget._stripe_plot)

    def test_multiple_instances_info(self):
        self.send_signal(self.widget.Inputs.data, self.iris[:1])
        self.send_signal(self.widget.Inputs.background_data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.assertFalse(self.widget.Information.multiple_instances.is_shown())

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertTrue(self.widget.Information.multiple_instances.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Information.multiple_instances.is_shown())

    def test_send_report(self):
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, self.iris[:1])
        self.send_signal(self.widget.Inputs.background_data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, self.housing[:1])
        self.send_signal(self.widget.Inputs.background_data, self.housing)
        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.widget.send_report()

    def assertPlotEmpty(self, plot: StripePlot):
        self.assertIsNone(plot)

    def assertPlotNotEmpty(self, plot: StripePlot):
        layout = plot.layout()  # type: QGraphicsLinearLayout
        self.assertEqual(layout.count(), 2)


if __name__ == "__main__":
    unittest.main()
