# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table
from orangecontrib.prototypes.widgets.owfreeviz import OWFreeViz
from Orange.widgets.tests.base import WidgetTest


class TestOWFreeViz(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data = Table("heart_disease")

    def setUp(self):
        self.widget = self.create_widget(OWFreeViz)

    def test_points_combo_boxes(self):
        self.send_signal("Data", self.data)
        self.assertEqual(len(self.widget.controls.attr_color.model()), 17)
        self.assertEqual(len(self.widget.controls.attr_shape.model()), 11)
        self.assertEqual(len(self.widget.controls.attr_size.model()), 8)
        self.assertEqual(len(self.widget.controls.attr_label.model()), 17)
