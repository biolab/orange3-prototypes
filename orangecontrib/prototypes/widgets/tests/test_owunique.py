# Tests test protected methods
# pylint: disable=protected-access
import unittest
from unittest.mock import Mock

import numpy as np

from Orange.data import DiscreteVariable, ContinuousVariable, Domain, Table
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.prototypes.widgets import owunique


class TestOWUnique(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(owunique.OWUnique)  #: OWUnique

        self.domain = Domain(
            [DiscreteVariable(name, values=("a", "b", "c")) for name in "abcd"],
            [ContinuousVariable("e")],
            [DiscreteVariable(name, values=("a", "b", "c")) for name in "fg"])
        self.table = Table.from_numpy(
            self.domain,
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 2, 0, 0],
             [1, 2, 0, 0]],
            np.arange(6),
            np.zeros((6, 2)))

    def test_model(self):
        w = self.widget
        w.unconditional_commit = Mock()

        self.assertEqual(tuple(w.model_key), ())
        self.assertEqual(tuple(w.model_avail), ())

        self.send_signal(w.Inputs.data, self.table)
        self.assertEqual(tuple(w.model_key), ())
        self.assertEqual(tuple(w.model_avail),
                         self.domain.variables + self.domain.metas)
        w.unconditional_commit.assert_called()
        w.unconditional_commit.reset_mock()

        self.send_signal(w.Inputs.data, None)
        self.assertEqual(tuple(w.model_key), ())
        self.assertEqual(tuple(w.model_avail), ())
        w.unconditional_commit.assert_called()
        w.unconditional_commit.reset_mock()

    def test_settings(self):
        w = self.widget
        domain = self.domain
        w.unconditional_commit = Mock()

        self.send_signal(w.Inputs.data, self.table)
        w.model_key.append(w.model_avail.pop(2))

        self.send_signal(w.Inputs.data, None)
        self.assertEqual(tuple(w.model_key), ())
        self.assertEqual(tuple(w.model_avail), ())

        domain = Domain(domain.attributes[2:], domain.class_vars, domain.metas)
        table = self.table.transform(domain)
        self.send_signal(w.Inputs.data, table)
        self.assertEqual(tuple(w.model_key), (self.domain[2], ))
        self.assertEqual(tuple(w.model_avail),
                         self.domain.variables[3:] + self.domain.metas)

        self.send_signal(w.Inputs.data, None)
        self.assertEqual(tuple(w.model_key), ())
        self.assertEqual(tuple(w.model_avail), ())
        w.unconditional_commit.assert_called()
        w.unconditional_commit.reset_mock()

    def test_unconditional_commit(self):
        w = self.widget
        w.autocommit = False

        w._compute_unique_data = cud = Mock()
        cud.return_value = Mock()

        self.send_signal(w.Inputs.data, self.table)
        out = self.get_output(w.Outputs.data)
        self.assertIs(out, cud.return_value)

        self.send_signal(w.Inputs.data, None)
        out = self.get_output(w.Outputs.data)
        self.assertIs(out, None)

    def test_compute(self):
        w = self.widget

        self.send_signal(w.Inputs.data, self.table)
        out = self.get_output(w.Outputs.data)
        self.assertIsNone(out, None)

        w.model_key[:] = w.model_avail[:2]
        del w.model_avail[:2]

        w.tiebreaker = "Last instance"
        w.commit()
        out = self.get_output(w.Outputs.data)
        np.testing.assert_equal(out.Y, [2, 3, 4, 5])

        w.tiebreaker = "First instance"
        w.commit()
        out = self.get_output(w.Outputs.data)
        np.testing.assert_equal(out.Y, [0, 3, 4, 5])

        w.tiebreaker = "Middle instance"
        w.commit()
        out = self.get_output(w.Outputs.data)
        np.testing.assert_equal(out.Y, [1, 3, 4, 5])

        w.tiebreaker = "Discard instances with non-unique keys"
        w.commit()
        out = self.get_output(w.Outputs.data)
        np.testing.assert_equal(out.Y, [3, 4, 5])


if __name__ == "__main__":
    unittest.main()
