import unittest

import numpy as np

from Orange.data import DiscreteVariable, ContinuousVariable, StringVariable, \
    Domain, Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from orangecontrib.prototypes.widgets.owquickselect import OWQuickSelect

class TestOWQuickSelect(WidgetTest):
    def setUp(self):
        self.widget: OWQuickSelect = self.create_widget(OWQuickSelect)
        a = DiscreteVariable("A", values=("a", "b", "c"))
        b = DiscreteVariable("B", values=("x", "y"))
        c = DiscreteVariable("C", values=("1", "2", "3"))
        d, e, f = (ContinuousVariable(c) for c in "DEF")
        g = StringVariable("G")
        self.domain = Domain([d, a, b, e], c, [f, g])

        self.data = Table.from_numpy(
            self.domain,
            X=[[1, 0, 0, 2],
               [0, 1, 1, 3],
               [1, 2, 0, 4],
               [2, np.nan, 1, 5],
               [0, 0, 1, 6]],
            Y=[0, 1, 0, 2, 2],
            metas=[[2, "x"], [np.nan, "y"], [np.nan, "x"], [1, "y"], [2, "x"]]

        )

    def test_model_and_settings(self):
        w = self.widget
        var_combo = w.controls.variable
        value_combo = w.controls.value

        self.send_signal(self.data)
        self.assertEqual(len(w.var_model), 3)
        self.assertIs(w.variable, self.domain["A"])
        self.assertEqual(len(w.value_model), 3)
        self.assertEqual(w.value_model[0], "a")

        simulate.combobox_activate_index(var_combo, 1)
        self.assertEqual(len(w.value_model), 2)
        self.assertEqual(w.value, "x")

        simulate.combobox_activate_index(value_combo, 1)
        self.assertEqual(w.value, "y")

        self.send_signal(None)
        self.assertEqual(len(w.var_model), 0)
        self.assertIsNone(w.variable)
        self.assertEqual(len(w.value_model), 0)
        self.assertEqual(w.value, "")

        self.send_signal(w.Inputs.data, self.data)
        self.assertEqual(len(w.var_model), 3)
        self.assertIs(w.variable, self.domain["B"])
        self.assertEqual(len(w.value_model), 2)
        self.assertEqual(w.value, "y")

    def test_no_categorical(self):
        w = self.widget
        no_cat = self.data.transform(
            Domain(
                [self.domain["D"], self.domain["E"]],
                None,
                [self.domain["F"], self.domain["G"]]))

        self.send_signal(no_cat)
        self.assertEqual(len(w.var_model), 0)
        self.assertIsNone(w.variable)
        self.assertTrue(w.Error.no_categorical.is_shown())
        self.assertIsNone(self.get_output(w.Outputs.annotated))
        w.send_report()

        self.send_signal(None)
        self.assertEqual(len(w.var_model), 0)
        self.assertIsNone(w.variable)
        self.assertFalse(w.Error.no_categorical.is_shown())
        self.assertIsNone(self.get_output(w.Outputs.annotated))
        w.send_report()

        self.send_signal(no_cat)
        self.assertTrue(w.Error.no_categorical.is_shown())
        self.assertIsNone(self.get_output(w.Outputs.annotated))
        w.send_report()

        self.send_signal(self.data)
        self.assertFalse(w.Error.no_categorical.is_shown())
        self.assertIsNotNone(self.get_output(w.Outputs.annotated))
        w.send_report()

        self.send_signal(no_cat)
        self.assertIsNone(self.get_output(w.Outputs.annotated))

    def test_missing_context_value(self):
        w = self.widget
        var_combo = w.controls.variable
        value_combo = w.controls.value

        self.send_signal(self.data)
        simulate.combobox_activate_index(var_combo, 1)
        simulate.combobox_activate_index(value_combo, 1)
        self.assertEqual(w.value, "y")

        new_b = DiscreteVariable("B", values=("x", ))
        new_data = self.data.transform(
            Domain(self.domain.attributes[:2] + (new_b, ),
                   self.domain.class_var,
                   self.domain.metas))

        self.send_signal(None)
        self.assertIsNone(w.variable)
        self.assertEqual(w.value, "")

        self.send_signal(new_data)
        self.assertIs(w.variable, new_b)
        self.assertEqual(w.value, "x")

    def test_output(self):
        w = self.widget
        var_combo = w.controls.variable
        value_combo = w.controls.value

        self.send_signal(self.data)
        simulate.combobox_activate_index(var_combo, 1)

        np.testing.assert_equal(
            self.get_output(w.Outputs.annotated).metas[:, -1],
            [1, 0, 1, 0, 0])
        np.testing.assert_equal(
            self.get_output(w.Outputs.matching).X,
            self.data.X[[0, 2]])
        np.testing.assert_equal(
            self.get_output(w.Outputs.unmatched).X,
            self.data.X[[1, 3, 4]])

        simulate.combobox_activate_index(value_combo, 1)
        np.testing.assert_equal(
            self.get_output(w.Outputs.annotated).metas[:, -1],
            [0, 1, 0, 1, 1])
        np.testing.assert_equal(
            self.get_output(w.Outputs.matching).X,
            self.data.X[[1, 3, 4]])
        np.testing.assert_equal(
            self.get_output(w.Outputs.unmatched).X,
            self.data.X[[0, 2]])

        # Test with nans
        simulate.combobox_activate_index(var_combo, 0)
        self.assertIs(w.variable, self.domain["A"])
        self.assertEqual(w.value, "a")
        np.testing.assert_equal(
            self.get_output(w.Outputs.annotated).metas[:, -1],
            [1, 0, 0, 0, 1])

        simulate.combobox_activate_index(value_combo, 1)
        np.testing.assert_equal(
            self.get_output(w.Outputs.annotated).metas[:, -1],
            [0, 1, 0, 0, 0])

        simulate.combobox_activate_index(value_combo, 2)
        np.testing.assert_equal(
            self.get_output(w.Outputs.annotated).metas[:, -1],
            [0, 0, 1, 0, 0])

        self.send_signal(None)
        self.send_signal(self.data)
        np.testing.assert_equal(
            self.get_output(w.Outputs.annotated).metas[:, -1],
            [0, 0, 1, 0, 0])

if __name__ == "__main__":
    unittest.main()
