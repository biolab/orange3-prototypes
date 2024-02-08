# pylint: disable=missing-docstring,unsubscriptable-object
import os
import unittest

import numpy as np

from Orange.data import Table, StringVariable, Domain, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.prototypes.widgets.owtexttocolumns import \
    OWTextToColumns, SplitColumn, get_substrings, OneHotStrings, OneHotDiscrete


class TestComputation(unittest.TestCase):
    def setUp(self):
        domain = Domain([DiscreteVariable("x", values=("a c d", "bb d"))], None,
                        [StringVariable("foo"), StringVariable("bar")])
        self.data = Table.from_numpy(
            domain,
            np.array([1, 0, np.nan])[:, None], None,
            [["a,bbb,d", "e;f o"], ["", "f o"], ["bbb,d", "e;a;o"]]
        )

    def test_get_string_values(self):
        np.testing.assert_equal(
            set(get_substrings({"a bc", "d,e", "", "f,a t", "t"}, " ")),
            {"a", "bc", "d,e", "f,a", "t"})
        np.testing.assert_equal(
            set(get_substrings({"a bc", "d,e", "", "f,a t", "t"}, ",")),
            {"a bc", "d", "e", "f", "a t", "t"})

    def test_split_column(self):
        sc = SplitColumn(self.data, self.data.domain.metas[0], ",")
        shared = sc(self.data)
        self.assertEqual(set(sc.new_values), {"a", "bbb", "d"})
        self.assertEqual(set(shared), set(sc.new_values))
        np.testing.assert_equal(shared["a"], [0])
        np.testing.assert_equal(shared["bbb"], [0, 2])
        np.testing.assert_equal(shared["d"], [0, 2])

        sc = SplitColumn(self.data, self.data.domain.metas[1], ";")
        shared = sc(self.data)
        self.assertEqual(set(sc.new_values), {"a", "e", "f o", "o"})
        self.assertEqual(set(shared), set(sc.new_values))
        np.testing.assert_equal(shared["a"], [2])
        np.testing.assert_equal(shared["e"], [0, 2])
        np.testing.assert_equal(shared["f o"], [0, 1])
        np.testing.assert_equal(shared["o"], [2])

    def test_no_known_values(self):
        sc = SplitColumn(self.data, self.data.domain.metas[0], ",")
        data = Table.from_numpy(
            self.data.domain, np.zeros((3, 1)), None,
            np.array([["x"] * 2] * 3))
        shared = sc(data)
        for attr in ("a", "bbb", "d"):
            self.assertEqual(shared[attr].size, 0)
            oh = OneHotStrings(sc, attr)
            np.testing.assert_equal(oh(data), [0, 0, 0])

    def test_one_hot_strings(self):
        attr = self.data.domain.metas[0]
        sc = SplitColumn(self.data, attr, ",")

        oh = OneHotStrings(sc, "a")
        np.testing.assert_equal(oh(self.data), [1, 0, 0])

        oh = OneHotStrings(sc, "bbb")
        np.testing.assert_equal(oh(self.data), [1, 0, 1])

        data = Table.from_numpy(
            Domain([], None, [attr]),
            np.zeros((5, 0)), None,
            np.array(["bbb,x,y", "", "bbb", "bbb,a", "foo"])[:, None])
        np.testing.assert_equal(oh(data), [1, 0, 1, 1, 0])

    def test_one_hot_discrete(self):
        attr = self.data.domain.attributes[0]

        oh = OneHotDiscrete(attr, " ", "a")
        np.testing.assert_equal(oh(self.data), [0, 1, np.nan])

        oh = OneHotDiscrete(attr, " ", "d")
        np.testing.assert_equal(oh(self.data), [1, 1, np.nan])

        data = Table.from_numpy(
            Domain([attr], None),
            np.array([1, 0, 1, 0, np.nan])[:, None])

        oh = OneHotDiscrete(attr, " ", "a")
        np.testing.assert_equal(oh(data), [0, 1, 0, 1, np.nan])

        oh = OneHotDiscrete(attr, " ", "d")
        np.testing.assert_equal(oh(data), [1, 1, 1, 1, np.nan])

    def test_discrete_metas(self):
        attr = DiscreteVariable("x", values=("a c d", "bb d"))
        domain = Domain([], None, [attr])
        data = Table.from_numpy(domain, np.zeros((3, 0)), None,
                                np.array([1, 0, np.nan])[:, None])
        oh = OneHotDiscrete(attr, " ", "a")
        np.testing.assert_equal(oh(data), [0, 1, np.nan])


class TestOWTextToColumns(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWTextToColumns)
        test_path = os.path.dirname(os.path.abspath(__file__))
        self.data = Table.from_file(os.path.join(test_path, "orange-in-education.tab"))
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
        Create a simple dataset with 4 documents.
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

    def test_output_string(self):
        "Test outputs; at the same time, test for duplicate variables"
        self.widget.delimiter = ","
        self.send_signal(self.widget.Inputs.data, self.small_table)
        out = self.get_output(self.widget.Outputs.data)
        self.assertEqual([attr.name for attr in out.domain.attributes],
                         ["bar", "baz", "foo (1)"])
        np.testing.assert_equal(out.X,
                                [[0, 0, 1],
                                 [1, 1, 0],
                                 [1, 0, 1],
                                 [0, 0, 0]])

    def test_output_discrete(self):
        self.widget.delimiter = " "
        attr = DiscreteVariable("x", values=("bar foo", "bar baz", "crux"))
        data = Table.from_numpy(
            Domain([attr], None),
            np.array([1, 1, 0, 1, 2, np.nan])[:, None], None)
        self.send_signal(self.widget.Inputs.data, data)
        out = self.get_output(self.widget.Outputs.data)
        self.assertEqual([attr.name for attr in out.domain.attributes],
                         ["x", "bar", "baz", "crux", "foo"])
        np.testing.assert_equal(out.X,
                                [[1, 1, 1, 0, 0],
                                 [1, 1, 1, 0, 0],
                                 [0, 1, 0, 0, 1],
                                 [1, 1, 1, 0, 0],
                                 [2, 0, 0, 1, 0],
                                 [np.nan, np.nan, np.nan, np.nan, np.nan]])


if __name__ == "__main__":
    unittest.main()
