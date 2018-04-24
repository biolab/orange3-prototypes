# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import random

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin

from orangecontrib.prototypes.widgets.owlineplot import OWLinePlot


class TestOWLinePLot(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data

    def setUp(self):
        self.widget = self.create_widget(OWLinePlot)
        self.titanic = Table("titanic")
        self.housing = Table("housing")

    def _select_data(self):
        random.seed(42)
        indices = random.sample(range(0, len(self.data)), 20)
        self.widget.graph.select(indices)
        return self.widget.selection

    def test_input_data(self):
        no_data_info = "No data on input."
        self.assertEqual(self.widget.infoLabel.text(), no_data_info)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertEqual(self.widget.cb_attr.count(), 1)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.cb_attr.count(), 0)
        self.assertEqual(self.widget.infoLabel.text(), no_data_info)
        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.assertTrue(self.widget.Information.not_enough_attrs.is_shown())
        self.send_signal(self.widget.Inputs.data, self.housing)

    def test_input_change(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        random.seed(42)
        indices = random.sample(range(0, len(self.data)), 2)
        self.widget.graph.select(indices)
        self.send_signal(self.widget.Inputs.data, self.titanic)

    def test_input_subset_data(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        random.seed(42)
        indices = random.sample(range(0, len(self.data)), 20)
        self.send_signal(self.widget.Inputs.data_subset, self.data[indices])
        self.send_signal(self.widget.Inputs.data, None)
