# pylint: disable=missing-docstring
import inspect
import itertools
import unittest

import numpy as np

from AnyQt.QtCore import Qt, QPoint
from AnyQt.QtTest import QTest
from AnyQt.QtWidgets import QGraphicsGridLayout, QGraphicsSimpleTextItem

import pyqtgraph as pg

from orangecanvas.gui.test import mouseMove

import Orange
from Orange.base import Learner
from Orange.classification import RandomForestLearner, OneClassSVMLearner, \
    IsolationForestLearner, EllipticEnvelopeLearner, LocalOutlierFactorLearner
from Orange.data import Table, Domain
from Orange.regression import RandomForestRegressionLearner
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.prototypes.widgets.owexplainmodel import OWExplainModel, \
    ViolinPlot, ViolinItem, Results


def dummy_run(data, model, _):
    if not data or model is None:
        return None
    m, n = data.X.shape
    k = len(data.domain.class_var.values) \
        if data.domain.has_discrete_class else 1
    mask = np.ones(m, dtype=bool)
    mask[150:] = False
    return Results(x=[np.ones((m, n)) for _ in range(k)],
                   colors=np.zeros((m, n) + (3,)),
                   names=[str(i) for i in range(n)],
                   mask=mask)


class TestOWExplainModel(WidgetTest):
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

    def test_output_scores(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.scores)
        self.assertIsInstance(output, Table)
        self.assertEqual(list(output.metas.flatten()),
                         [a.name for a in self.iris.domain.attributes])
        self.send_signal(self.widget.Inputs.model, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.scores))

    def test_selection(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        plot = self.widget._violin_plot
        h = plot.layout().itemAt(0, plot.VIOLIN_COLUMN)
        pos = self.widget.view.mapFromScene(h.scenePos())
        QTest.mousePress(self.widget.view.viewport(), Qt.LeftButton,
                         pos=pos + QPoint(1, 1))
        mouseMove(self.widget.view.viewport(), Qt.LeftButton,
                  pos=pos + QPoint(200, 20))
        QTest.mouseRelease(self.widget.view.viewport(), Qt.LeftButton,
                           pos=pos + QPoint(200, 30))
        selection = self.get_output(self.widget.Outputs.selected_data)
        self.assertIsInstance(selection, Table)
        self.assertEqual(len(selection), 100)

        QTest.mouseClick(self.widget.view.viewport(), Qt.LeftButton,
                         pos=pos + QPoint(1, 1))
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

    def test_saved_selection(self):
        self.send_signal(self.widget.Inputs.data, self.heart)
        rf_cls = RandomForestLearner(random_state=42)(self.heart)
        self.send_signal(self.widget.Inputs.model, rf_cls)
        self.wait_until_finished()
        plot = self.widget._violin_plot
        h = plot.layout().itemAt(0, plot.VIOLIN_COLUMN)
        pos = self.widget.view.mapFromScene(h.scenePos())
        QTest.mousePress(self.widget.view.viewport(), Qt.LeftButton,
                         pos=pos + QPoint(250, 10))
        mouseMove(self.widget.view.viewport(), Qt.LeftButton,
                  pos=pos + QPoint(300, 20))
        QTest.mouseRelease(self.widget.view.viewport(), Qt.LeftButton,
                           pos=pos + QPoint(300, 30))
        saved_selection = self.get_output(self.widget.Outputs.selected_data)

        settings = self.widget.settingsHandler.pack_data(self.widget)
        widget = self.create_widget(OWExplainModel, stored_settings=settings)
        self.send_signal(widget.Inputs.data, self.heart, widget=widget)
        rf_cls = RandomForestLearner(random_state=42)(self.heart)
        self.send_signal(widget.Inputs.model, rf_cls, widget=widget)
        self.wait_until_finished(widget=widget)
        selection = self.get_output(widget.Outputs.selected_data, widget=widget)
        np.testing.assert_array_equal(selection.X, saved_selection.X)

    def test_all_models(self):
        def run(data):
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

    def test_show_legend(self):
        self.widget.controls.show_legend.setChecked(False)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.controls.show_legend.setChecked(True)
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.controls.show_legend.setChecked(False)

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

    @unittest.mock.patch("orangecontrib.prototypes.widgets.owexplainmodel.run")
    def test_data_sampled_info(self, mocked_run):
        mocked_run.side_effect = dummy_run
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertFalse(self.widget.Information.data_sampled.is_shown())

        self.send_signal(self.widget.Inputs.data, self.heart)
        rf_cls = RandomForestLearner(random_state=42)(self.heart)
        self.send_signal(self.widget.Inputs.model, rf_cls)
        self.wait_until_finished()
        self.assertTrue(self.widget.Information.data_sampled.is_shown())

        self.send_signal(self.widget.Inputs.data, self.heart)
        self.send_signal(self.widget.Inputs.model, None)
        self.wait_until_finished()
        self.assertFalse(self.widget.Information.data_sampled.is_shown())

    def test_send_report(self):
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.widget.send_report()

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
