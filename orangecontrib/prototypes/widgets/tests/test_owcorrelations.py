# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table
from Orange.widgets.visualize.owscatterplot import OWScatterPlot
from orangecontrib.prototypes.widgets.owcorrelations import OWCorrelations
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.widget import AttributeList


class TestOWCorrelations(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data_cont = Table("iris")
        cls.data_disc = Table("zoo")
        cls.data_mixed = Table("heart_disease")

    def setUp(self):
        self.widget = self.create_widget(OWCorrelations)

    def test_input_data_cont(self):
        """Check correlation table for dataset with continuous attributes"""
        self.send_signal("Data", self.data_cont)
        n_attrs = len(self.data_cont.domain.attributes)
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 3)
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(),
                         n_attrs * (n_attrs - 1) / 2)
        self.send_signal("Data", None)
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 0)
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 0)

    def test_input_data_disc(self):
        """Check correlation table for dataset with discrete attributes"""
        self.send_signal("Data", self.data_disc)
        self.assertTrue(self.widget.Information.not_enough_vars.is_shown())
        self.send_signal("Data", None)
        self.assertFalse(self.widget.Information.not_enough_vars.is_shown())

    def test_input_data_mixed(self):
        """Check correlation table for dataset with continuous and discrete
        attributes"""
        self.send_signal("Data", self.data_mixed)
        domain = self.data_mixed.domain
        n_attrs = len([a for a in domain.attributes if a.is_continuous])
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 3)
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(),
                         n_attrs * (n_attrs - 1) / 2)

    def test_input_data_one_feature(self):
        """Check correlation table for dataset with one attribute"""
        self.send_signal("Data", self.data_cont[:, [0, 4]])
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 0)
        self.assertTrue(self.widget.Information.not_enough_vars.is_shown())
        self.send_signal("Data", None)
        self.assertFalse(self.widget.Information.not_enough_vars.is_shown())

    def test_input_data_one_instance(self):
        """Check correlation table for dataset with one instance"""
        self.send_signal("Data", self.data_cont[:1])
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 0)
        self.assertTrue(self.widget.Information.not_enough_inst.is_shown())
        self.send_signal("Data", None)
        self.assertFalse(self.widget.Information.not_enough_inst.is_shown())

    def test_output_data(self):
        """Check dataset on output"""
        self.send_signal("Data", self.data_cont)
        self.assertEqual(self.data_cont, self.get_output("Data"))

    def test_output_features(self):
        """Check features on output"""
        self.send_signal("Data", self.data_cont)
        features = self.get_output("Features")
        self.assertIsInstance(features, AttributeList)
        self.assertEqual(len(features), 2)

    def test_output_correlations(self):
        """Check correlation table on on output"""
        self.send_signal("Data", self.data_cont)
        correlations = self.get_output("Correlations")
        self.assertIsInstance(correlations, Table)
        self.assertEqual(len(correlations), 6)
        self.assertEqual(len(correlations.domain.attributes), 1)
        self.assertEqual(len(correlations.domain.metas), 2)

    def test_scatterplot_input_features(self):
        """Check if attributes have been set after sent to scatterplot"""
        self.send_signal("Data", self.data_cont)
        scatterplot_widget = self.create_widget(OWScatterPlot)
        features = self.get_output("Features")
        self.send_signal("Data", self.data_cont, widget=scatterplot_widget)
        self.send_signal("Features", features, widget=scatterplot_widget)
        self.assertIs(scatterplot_widget.attr_x, self.data_cont.domain[2])
        self.assertIs(scatterplot_widget.attr_y, self.data_cont.domain[3])
