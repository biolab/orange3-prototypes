# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import random
from unittest.mock import patch

from Orange.data import Table
from orangecontrib.prototypes.widgets.owfreeviz import OWFreeViz
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin, \
    datasets


class TestOWFreeViz(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data
        cls.same_input_output_domain = False
        cls.heart_disease = Table("heart_disease")

    def setUp(self):
        self.widget = self.create_widget(OWFreeViz)

    def test_points_combo_boxes(self):
        self.send_signal("Data", self.heart_disease)
        self.assertEqual(len(self.widget.controls.attr_color.model()), 17)
        self.assertEqual(len(self.widget.controls.attr_shape.model()), 11)
        self.assertEqual(len(self.widget.controls.attr_size.model()), 8)
        self.assertEqual(len(self.widget.controls.attr_label.model()), 17)

    def test_ugly_datasets(self):
        self.send_signal("Data", Table(datasets.path("testing_dataset_cls")))
        self.send_signal("Data", Table(datasets.path("testing_dataset_reg")))

    def test_error_msg(self):
        data = self.data[:, list(range(len(self.data.domain.attributes)))]
        self.send_signal("Data", data)
        self.assertTrue(self.widget.Error.no_class_var.is_shown())
        self.send_signal("Data", None)
        self.assertFalse(self.widget.Error.no_class_var.is_shown())

    def _select_data(self):
        random.seed(42)
        points = sorted(random.sample(range(0, len(self.data)), 10))
        self.widget.select(points)
        self.widget.commit()
        return points

    def test_outputs(self):
        original_send_signal = self.send_signal

        def send_signal_with_commit(*args, **kwargs):
            original_send_signal(*args, **kwargs)
            self.widget.commit()

        with patch.object(self, "send_signal", send_signal_with_commit):
            super().test_outputs()
