# pylint: disable=missing-docstring
import inspect
import itertools
import unittest

from AnyQt.QtWidgets import QGraphicsGridLayout, QGraphicsSimpleTextItem
import pyqtgraph as pg

import Orange
from Orange.base import Learner
from Orange.classification import RandomForestLearner
from Orange.data import Table, Domain
from Orange.regression import RandomForestRegressionLearner
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.prototypes.widgets.owexplainmodel import OWExplainModel, \
    ViolinPlot, ViolinItem


class TestOWExplainModel(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")
        cls.housing = Table("housing")
        cls.rf_cls = RandomForestLearner()(cls.iris)
        cls.rf_reg = RandomForestRegressionLearner()(cls.housing)

    def setUp(self):
        self.widget = self.create_widget(OWExplainModel)

    def test_classification_data_classification_model(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertDomainInPlot(self.widget._violin_plot, self.iris.domain)

    def test_classification_data_regression_model(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget._violin_plot)
        self.assertTrue(self.widget.Error.domain_transform_err.is_shown())

    def test_regression_data_regression_model(self):
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.wait_until_finished()
        self.assertDomainInPlot(self.widget._violin_plot, self.housing.domain)

    def test_regression_data_classification_model(self):
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget._violin_plot)
        self.assertTrue(self.widget.Error.domain_transform_err.is_shown())

    def test_all_models(self):
        def run(data):
            self.send_signal(self.widget.Inputs.data, data)
            if not issubclass(cls, Learner):
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
            run(self.housing)
        for _, cls in itertools.chain(
                inspect.getmembers(Orange.classification, inspect.isclass),
                inspect.getmembers(Orange.modelling, inspect.isclass)):
            run(self.iris)

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
        self.assertPlotEmpty(self.widget._violin_plot)

        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertDomainInPlot(self.widget._violin_plot, self.iris.domain)

        self.send_signal(self.widget.Inputs.model, None)
        self.assertPlotEmpty(self.widget._violin_plot)

        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget._violin_plot)

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget._violin_plot)

        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertDomainInPlot(self.widget._violin_plot, self.iris.domain)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertPlotEmpty(self.widget._violin_plot)

    def assertPlotEmpty(self, plot: ViolinPlot):
        self.assertIsNone(plot)

    def assertDomainInPlot(self, plot: ViolinPlot, domain: Domain):
        layout = plot.layout()  # type: QGraphicsGridLayout
        n_rows = layout.rowCount()
        self.assertEqual(n_rows, len(domain.attributes) + 1)
        self.assertEqual(layout.columnCount(), 3)
        for i in range(layout.rowCount() - 1):
            item0 = layout.itemAt(i, 0).item
            self.assertIsInstance(item0, QGraphicsSimpleTextItem)
            self.assertIsInstance(layout.itemAt(i, 1), ViolinItem)
        self.assertIsNone(layout.itemAt(n_rows - 1, 0))
        self.assertIsInstance(layout.itemAt(n_rows - 1, 1), pg.AxisItem)


if __name__ == "__main__":
    unittest.main()
