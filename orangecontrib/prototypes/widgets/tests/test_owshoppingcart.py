# Tests test protected methods
# pylint: disable=protected-access
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy import sparse as sp

from Orange.data import (
    DiscreteVariable, ContinuousVariable, StringVariable, Domain, Table
)
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.prototypes.widgets import owshoppinglist


def data_without_commit(f):
    def wrapped(self):
        with patch("orangecontrib.prototypes.widgets.owshoppinglist."
                   "OWShoppingList.commit"):
            self.send_signal(self.widget.Inputs.data, self.data)
            f(self)
    return wrapped


def names(variables):
    return [var.name for var in variables]


class TestOWShoppingCartBase(WidgetTest):

    # Tests use this table:
    #
    #           attributes                       metas
    # -----------------------------------   -----------------
    # gender   age     pretzels   telezka   name     greeting
    # disc     cont    cont       disc      string   string
    # -------------------------------------------------------
    # 0        25      3                    ana      hi
    # 0        26      0          1         berta    hello
    # 0        27                 0         cilka
    #          28                                    hi
    # 1                2                    evgen    foo
    # -------------------------------------------------------

    def setUp(self):
        self.widget = self.create_widget(owshoppinglist.OWShoppingList)

        self.domain = Domain(
            [DiscreteVariable("gender", values=("f", "m")),
             ContinuousVariable("age"),
             ContinuousVariable("pretzels"),
             DiscreteVariable("telezka", values=("big", "small"))],
            [],
            [StringVariable("name"), StringVariable("greeting")])
        n = np.nan
        self.data = Table.from_numpy(
            self.domain,
            [[0, 25, 3, n],
             [0, 26, 0, 1],
             [0, 27, n, 0],
             [n, 28, n, n],
             [1, n, 2, n]],
            None,
            [["ana", "hi"],
             ["berta", "hello"],
             ["cilka", ""],
             ["", "hi"],
             ["evgen", "foo"]])


class TestOWShoppingListFunctional(TestOWShoppingCartBase):
    @data_without_commit
    def test_idvar_selection(self):
        self.assertEqual(names(self.widget.idvar_model), ["telezka", "name"])


    def test_context_and_no_data(self):
        widget = self.widget

        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertIsNotNone(self.get_output(widget.Outputs.data))
        self.assertIs(widget.idvar, widget.idvar_model[0])

        widget.idvar = widget.idvar_model[1]

        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(widget.Outputs.data))
        self.assertIsNone(widget.idvar)

        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertIsNotNone(self.get_output(widget.Outputs.data))
        self.assertIs(widget.idvar, widget.idvar_model[1])

    @data_without_commit
    def test_no_suitable_features(self):
        widget = self.widget
        heart = Table("heart_disease")

        self.assertFalse(widget.Warning.no_suitable_features.is_shown())
        self.assertIsNotNone(self.get_output(widget.Outputs.data))
        self.assertIsNotNone(widget.idvar)

        # Sending unsuitable data shows the warning, resets output
        self.send_signal(widget.Inputs.data, heart)
        self.assertTrue(widget.Warning.no_suitable_features.is_shown())
        self.assertIsNone(self.get_output(widget.Outputs.data))
        self.assertIsNone(widget.idvar)

        # Suitable data clears it, gives output
        self.send_signal(widget.Inputs.data, self.data)
        self.assertFalse(widget.Warning.no_suitable_features.is_shown())
        self.assertIsNotNone(self.get_output(widget.Outputs.data))
        self.assertIsNotNone(widget.idvar)

        # Sending unsuitable data again shows the warning, resets output
        self.send_signal(widget.Inputs.data, heart)
        self.assertTrue(widget.Warning.no_suitable_features.is_shown())
        self.assertIsNone(self.get_output(widget.Outputs.data))
        self.assertIsNone(widget.idvar)

        # Removing data resets warning, but still no output
        self.send_signal(widget.Inputs.data, None)
        self.assertFalse(widget.Warning.no_suitable_features.is_shown())
        self.assertIsNone(self.get_output(widget.Outputs.data))
        self.assertIsNone(widget.idvar)

    def test_invalidates(self):
        widget = self.widget
        mock_return = self.data[:1]
        widget._reshape_to_long = lambda *_: mock_return
        widget.Outputs.data.send = send = Mock()

        self.send_signal(self.widget.Inputs.data, self.data)
        send.assert_called_with(mock_return)
        send.reset_mock()

        widget.controls.only_numeric.click()
        send.assert_called_with(mock_return)
        send.reset_mock()

        widget.controls.exclude_zeros.click()
        send.assert_called_with(mock_return)
        send.reset_mock()

        widget.controls.idvar.activated.emit(1)
        send.assert_called_with(mock_return)
        send.reset_mock()


class TestOWShoppingListUnits(TestOWShoppingCartBase):
    @data_without_commit
    def test_is_unique(self):
        domain = self.data.domain
        widget = self.widget

        self.assertTrue(widget._is_unique(domain["name"]))
        self.assertTrue(widget._is_unique(domain["telezka"]))
        self.assertFalse(widget._is_unique(domain["gender"]))
        self.assertFalse(widget._is_unique(domain["greeting"]))

    def test_nonnan_mask(self):
        for arr in ([1., 2, np.nan, 0], ["Ana", "Berta", "", "Dani"]):
            np.testing.assert_equal(
                self.widget._notnan_mask(np.array(arr)),
                [True, True, False, True])

    @data_without_commit
    def test_get_useful_vars(self):
        def assert_useful(expected):
            self.assertEqual(
                [var.name for
                 var, useful in zip(domain.attributes, widget._get_useful_vars())
                 if useful],
                expected)

        domain = self.data.domain
        widget = self.widget

        widget.idvar = domain["name"]
        widget.only_numeric = False
        assert_useful(["gender", "age", "pretzels", "telezka"])

        widget.idvar = domain["name"]
        widget.only_numeric = True
        assert_useful(["age", "pretzels"])

        widget.idvar = domain["telezka"]
        widget.only_numeric = False
        assert_useful(["gender", "age", "pretzels"])

        widget.idvar = domain["telezka"]
        widget.only_numeric = True
        assert_useful(["age", "pretzels"])

    @data_without_commit
    def test_get_item_names(self):
        self.assertEqual(
            self.widget._get_item_names(np.array([False, True, False, True])),
            ("age", "telezka")
        )

    @data_without_commit
    def test_prepare_domain_names(self):
        domain = self.data.domain
        widget = self.widget

        widget.only_numeric = True

        widget.idvar = domain["name"]
        widget.item_var_name = "the item"
        widget.value_var_name = "the value"
        outdomain = self.widget._prepare_domain(
            ["age", "pretzels"], ["Ana", "Berta", "Dani"])
        idvar, itemvar = outdomain.attributes
        self.assertEqual(idvar.name, "name")
        self.assertEqual(itemvar.name, "the item")
        self.assertEqual(outdomain.class_var.name, "the value")

        widget.idvar = domain["telezka"]
        widget.item_var_name = ""
        widget.value_var_name = ""
        outdomain = self.widget._prepare_domain(
            ["age", "pretzels"], ["Ana", "Berta", "Dani"])
        idvar, itemvar = outdomain.attributes
        self.assertEqual(idvar.name, "telezka")
        self.assertEqual(itemvar.name, owshoppinglist.DEFAULT_ITEM_NAME)
        self.assertEqual(outdomain.class_var.name, owshoppinglist.DEFAULT_VALUE_NAME)

    @data_without_commit
    def test_prepare_domain_values(self):
        domain = self.data.domain
        widget = self.widget

        widget.only_numeric = True

        widget.idvar = domain["name"]
        outdomain = self.widget._prepare_domain(
            ["age", "pretzels"], ["Ana", "Berta", "Dani"])
        idvar, itemvar = outdomain.attributes
        self.assertEqual(idvar.values, ("Ana", "Berta", "Dani"))
        self.assertEqual(itemvar.values, ("age", "pretzels"))
        self.assertIsInstance(outdomain.class_var, ContinuousVariable)

        widget.idvar = domain["telezka"]
        outdomain = self.widget._prepare_domain(
            ["age", "pretzels"], None)
        idvar, itemvar = outdomain.attributes
        self.assertIs(idvar, widget.idvar)
        self.assertEqual(itemvar.values, ("age", "pretzels"))
        self.assertIsInstance(outdomain.class_var, ContinuousVariable)

    @data_without_commit
    def test_reshape_dense(self):
        domain = self.data.domain
        widget = self.widget

        widget.idvar = domain["name"]
        widget.only_numeric = True
        widget.exclude_zeros = True
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[0, 0, 25], [0, 1, 3],
             [1, 0, 26],
             [2, 0, 27],
             [3, 1, 2]]
        )

        widget.idvar = domain["name"]
        widget.only_numeric = True
        widget.exclude_zeros = False
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[0, 0, 25], [0, 1, 3],
             [1, 0, 26], [1, 1, 0],
             [2, 0, 27],
             [3, 1, 2]]
        )

        widget.idvar = domain["name"]
        widget.only_numeric = False
        widget.exclude_zeros = True
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[0, 0, 0], [0, 1, 25], [0, 2, 3],
             [1, 0, 0], [1, 1, 26], [1, 3, 1],
             [2, 0, 0], [2, 1, 27], [2, 3, 0],
             [3, 0, 1], [3, 2, 2]]
        )

        widget.idvar = domain["name"]
        widget.only_numeric = False
        widget.exclude_zeros = False
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[0, 0, 0], [0, 1, 25], [0, 2, 3],
             [1, 0, 0], [1, 1, 26], [1, 2, 0], [1, 3, 1],
             [2, 0, 0], [2, 1, 27], [2, 3, 0],
             [3, 0, 1], [3, 2, 2]]
        )

        widget.idvar = domain["telezka"]
        widget.only_numeric = True
        widget.exclude_zeros = True
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[1, 0, 26],
             [0, 0, 27]]
        )

        widget.idvar = domain["telezka"]
        widget.only_numeric = True
        widget.exclude_zeros = False
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[1, 0, 26], [1, 1, 0],
             [0, 0, 27]]
        )

        widget.idvar = domain["telezka"]
        widget.only_numeric = False
        widget.exclude_zeros = True
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[1, 0, 0], [1, 1, 26],
             [0, 0, 0], [0, 1, 27]]
        )

        widget.idvar = domain["telezka"]
        widget.only_numeric = False
        widget.exclude_zeros = False
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[1, 0, 0], [1, 1, 26], [1, 2, 0],
             [0, 0, 0], [0, 1, 27]]
        )

    @data_without_commit
    def test_reshape_sparse(self):
        domain = self.data.domain
        widget = self.widget
        data = self.data

        sparse_table = Table.from_numpy(
            data.domain, sp.csr_matrix(data.X), None, data.metas)
        self.send_signal(widget.Inputs.data, sparse_table)

        widget.idvar = domain["name"]
        widget.only_numeric = True
        for widget.exclude_zeros in (True, False):
            out = widget._reshape_to_long()
            np.testing.assert_equal(
                np.hstack((out.X, np.atleast_2d(out.Y).T)),
                [[0, 0, 25], [0, 1, 3],
                 [1, 0, 26],
                 [2, 0, 27],
                 [3, 1, 2]]
            )

        widget.idvar = domain["name"]
        widget.only_numeric = False
        for widget.exclude_zeros in (True, False):
            out = widget._reshape_to_long()
            np.testing.assert_equal(
                np.hstack((out.X, np.atleast_2d(out.Y).T)),
                [[0, 1, 25], [0, 2, 3],
                 [1, 1, 26], [1, 3, 1],
                 [2, 1, 27],
                 [3, 0, 1], [3, 2, 2]]
            )

        widget.idvar = domain["telezka"]
        widget.only_numeric = True
        for widget.exclude_zeros in (True, False):
            out = widget._reshape_to_long()
            np.testing.assert_equal(
                np.hstack((out.X, np.atleast_2d(out.Y).T)),
                [[1, 0, 26],
                 [0, 0, 27]]
            )

        widget.idvar = domain["telezka"]
        widget.only_numeric = False
        for widget.exclude_zeros in (True, False):
            out = widget._reshape_to_long()
            np.testing.assert_equal(
                np.hstack((out.X, np.atleast_2d(out.Y).T)),
                [[1, 1, 26],
                 [0, 1, 27]]
            )


if __name__ == "__main__":
    unittest.main()
