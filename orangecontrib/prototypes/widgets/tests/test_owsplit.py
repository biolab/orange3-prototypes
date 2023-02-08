# pylint: disable=missing-docstring,unsubscriptable-object
import os
import unittest

import numpy as np

from Orange.data import Table, StringVariable, Domain
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.prototypes.widgets.owsplit import OWSplit


class TestOWSplit(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSplit)
        test_path = os.path.dirname(os.path.abspath(__file__))
        self.data = Table.from_file(
            os.path.join(test_path, "orange-in-education.tab"))
        self._create_simple_corpus()

    def _set_attr(self, attr, widget=None):
        if widget is None:
            widget = self.widget
        attr_combo = widget.controls.attribute
        idx = attr_combo.model().indexOf(attr)
        attr_combo.setCurrentIndex(idx)
        attr_combo.activated.emit(idx)

    def _create_simple_corpus(self) -> None:
        """
        Creat a simple dataset with 4 documents.
        """
        metas = np.array(
            [
                ["foo,"],
                ["bar,baz "],
                ["foo,bar"],
                [""],
            ]
        )
        text_var = StringVariable("foo")
        domain = Domain([], metas=[text_var])
        self.small_table = Table.from_numpy(
            domain,
            X=np.empty((len(metas), 0)),
            metas=metas,
        )

    def test_data(self):
        """Basic functionality"""
        self.send_signal(self.widget.Inputs.data, self.data)
        self._set_attr(self.data.domain.attributes[1])
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output.domain.attributes),
                         len(self.data.domain.attributes) + 3)
        self.assertTrue("in-class, in hands-on workshops" in output.domain
                        and "in-class, in lectures" in output.domain and
                        "outside the classroom" in output.domain)
        np.testing.assert_array_equal(output[:10, "in-class, in hands-on "
                                                  "workshops"],
                                      np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 0]
                                               ).reshape(-1, 1))
        np.testing.assert_array_equal(output[:10, "in-class, in lectures"],
                                      np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 0]
                                               ).reshape(-1, 1))
        np.testing.assert_array_equal(output[:10, "outside the classroom"],
                                      np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 1]
                                               ).reshape(-1, 1))

    def test_empty_data(self):
        """Do not crash on empty data"""
        self.send_signal(self.widget.Inputs.data, None)

    def test_discrete(self):
        """No crash on data attributes of different types"""
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertEqual(self.widget.attribute, self.data.domain.metas[1])
        self._set_attr(self.data.domain.attributes[1])
        self.assertEqual(self.widget.attribute, self.data.domain.attributes[1])

    def test_numeric_only(self):
        """Error raised when only numeric variables given"""
        housing = Table.from_file("housing")
        self.send_signal(self.widget.Inputs.data, housing)
        self.assertTrue(self.widget.Warning.no_disc.is_shown())

    def test_split_nonexisting(self):
        """Test splitting when delimiter doesn't exist"""
        self.widget.delimiter = "|"
        self.send_signal(self.widget.Inputs.data, self.data)
        new_cols = set(self.data.get_column("Country"))
        self.assertFalse(any(self.widget.delimiter in v for v in new_cols))
        self.assertEqual(len(self.get_output(
            self.widget.Outputs.data).domain.attributes),
                         len(self.data.domain.attributes) + len(new_cols))

    def test_empty_split(self):
        """Test a case of nan column. At the same time, test duplicate
        variable name."""
        self.widget.delimiter = ","
        self.send_signal(self.widget.Inputs.data, self.small_table)
        # new columns will be ["?", "bar", "baz ", "foo (1)"]
        self.assertEqual(len(self.get_output(self.widget.Outputs.data).domain),
                         5)


if __name__ == "__main__":
    unittest.main()
