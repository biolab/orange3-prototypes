import unittest
from unittest.mock import Mock

import numpy as np
import numpy.testing as npt

from AnyQt.QtCore import QItemSelection

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.widgets.widget import AttributeList

from orangecontrib.prototypes.widgets.owinteractions_new import OWInteractions, Heuristic
from orangecontrib.prototypes.interactions import InteractionScorer


class TestOWInteractions(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")  # continuous data
        cls.zoo = Table("zoo")  # discrete data

    def setUp(self):
        self.widget = self.create_widget(OWInteractions)

    def test_input_data(self):
        """Check table on input data"""
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.model.columnCount(), 0)
        self.assertEqual(self.widget.model.rowCount(), 0)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.model.columnCount(), 4)
        self.assertEqual(self.widget.model.rowCount(), 6)

    def test_input_data_one_feature(self):
        """Check table on input data with single attribute"""
        self.send_signal(self.widget.Inputs.data, self.iris[:, [0, 4]])
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.model.columnCount(), 0)
        self.assertTrue(self.widget.Warning.not_enough_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.not_enough_vars.is_shown())

    def test_input_data_no_target(self):
        """Check table on input data without target"""
        self.send_signal(self.widget.Inputs.data, self.iris[:, :-1])
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.model.columnCount(), 0)
        self.assertTrue(self.widget.Warning.no_class_var.is_shown())

    def test_input_data_one_instance(self):
        """Check table on input data with single instance"""
        self.send_signal(self.widget.Inputs.data, self.iris[:1])
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.model.columnCount(), 0)
        self.assertFalse(self.widget.Information.removed_cons_feat.is_shown())
        self.assertTrue(self.widget.Warning.not_enough_inst.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.not_enough_inst.is_shown())

    def test_input_data_constant_features(self):
        """Check table on input data with constant columns"""
        x = np.array([[0, 2, 1],
                      [0, 2, 0],
                      [0, 0, 1],
                      [0, 1, 2]])
        y = np.array([1, 2, 1, 0])
        labels = ["a", "b", "c"]
        domain_disc = Domain([DiscreteVariable(str(i), labels) for i in range(3)],
                             DiscreteVariable("cls", labels))
        domain_cont = Domain([ContinuousVariable(str(i)) for i in range(3)],
                             DiscreteVariable("cls", labels))

        self.send_signal(self.widget.Inputs.data, Table(domain_disc, x, y))
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.model.rowCount(), 3)
        self.assertFalse(self.widget.Information.removed_cons_feat.is_shown())

        self.send_signal(self.widget.Inputs.data, Table(domain_cont, x, y))
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.model.rowCount(), 1)
        self.assertTrue(self.widget.Information.removed_cons_feat.is_shown())

        x = np.ones((4, 3), dtype=float)
        self.send_signal(self.widget.Inputs.data, Table(domain_cont, x, y))
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.model.columnCount(), 0)
        self.assertTrue(self.widget.Warning.not_enough_vars.is_shown())
        self.assertTrue(self.widget.Information.removed_cons_feat.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Information.removed_cons_feat.is_shown())

    def test_output_features(self):
        """Check output"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.process_events()
        features = self.get_output(self.widget.Outputs.features)
        self.assertIsInstance(features, AttributeList)
        self.assertEqual(len(features), 2)

    def test_input_changed(self):
        """Check commit on input"""
        self.widget.commit = Mock()
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.process_events()
        self.widget.commit.assert_called_once()

        x = np.array([[0, 1, 0],
                      [1, 1, 2],
                      [0, 2, 0],
                      [0, 0, 2]])
        y = np.array([1, 2, 2, 0])
        domain = Domain([DiscreteVariable(str(i), ["a", "b", "c"]) for i in range(3)],
                        DiscreteVariable("cls"))

        self.widget.commit.reset_mock()
        self.send_signal(self.widget.Inputs.data, Table(domain, x, y))
        self.wait_until_finished()
        self.process_events()
        self.widget.commit.assert_called_once()

    def test_saved_selection(self):
        """Check row selection"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.process_events()
        selection = QItemSelection()
        selection.select(self.widget.model.index(2, 0),
                         self.widget.model.index(2, 3))
        self.widget.on_selection_changed(selection)
        settings = self.widget.settingsHandler.pack_data(self.widget)

        w = self.create_widget(OWInteractions, stored_settings=settings)
        self.send_signal(self.widget.Inputs.data, self.iris, widget=w)
        self.wait_until_finished(w)
        self.process_events()
        sel_row = w.rank_table.selectionModel().selectedRows()[0].row()
        self.assertEqual(sel_row, 2)

    def test_feature_combo(self):
        """Check feature combobox"""
        feature_combo = self.widget.controls.feature
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(len(feature_combo.model()), 5)

        self.wait_until_stop_blocking()
        self.send_signal(self.widget.Inputs.data, self.zoo)
        self.assertEqual(len(feature_combo.model()), 17)

    def test_select_feature(self):
        """Check feature selection"""
        feature_combo = self.widget.controls.feature
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.model.rowCount(), 6)
        self.assertSetEqual(
            {a.name for a in self.get_output(self.widget.Outputs.features)},
            {"sepal width", "sepal length"}
        )

        simulate.combobox_activate_index(feature_combo, 3)
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.model.rowCount(), 3)
        self.assertSetEqual(
            {a.name for a in self.get_output(self.widget.Outputs.features)},
            {"petal length", "sepal width"}
        )

        simulate.combobox_activate_index(feature_combo, 0)
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.model.rowCount(), 6)
        self.assertSetEqual(
            {a.name for a in self.get_output(self.widget.Outputs.features)},
            {"petal length", "sepal width"}
        )

    def test_send_report(self):
        """Check report"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.report_button.click()
        self.wait_until_stop_blocking()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.report_button.click()

    def test_compute_score(self):
        self.widget.scorer = InteractionScorer(self.zoo)
        npt.assert_almost_equal(self.widget.compute_score((1, 0)),
                                [-0.0771,  0.3003,  0.3307], 4)

    def test_row_for_state(self):
        row = self.widget.row_for_state((-0.2, 0.2, 0.1), (1, 0))
        self.assertListEqual(row, [-0.2, 0.1, 1, 0])

    def test_iterate_states(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertListEqual(list(self.widget._iterate_all(None)),
                             [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
        self.assertListEqual(list(self.widget._iterate_all((1, 0))),
                             [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
        self.assertListEqual(list(self.widget._iterate_all((2, 1))),
                             [(2, 1), (3, 0), (3, 1), (3, 2)])
        self.widget.feature_index = 2
        self.assertListEqual(list(self.widget._iterate_by_feature(None)),
                             [(2, 0), (2, 1), (2, 3)])
        self.assertListEqual(list(self.widget._iterate_by_feature((2, 0))),
                             [(2, 0), (2, 1), (2, 3)])
        self.assertListEqual(list(self.widget._iterate_by_feature((2, 3))),
                             [(2, 3)])

    def test_state_count(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget.state_count(), 6)
        self.widget.feature_index = 2
        self.assertEqual(self.widget.state_count(), 3)


class TestInteractionScorer(unittest.TestCase):
    def test_compute_score(self):
        """Check score calculation"""
        x = np.array([[1, 1], [0, 1], [1, 1], [0, 0]])
        y = np.array([0, 1, 1, 1])
        domain = Domain([DiscreteVariable(str(i)) for i in range(2)], DiscreteVariable("3"))
        data = Table(domain, x, y)
        self.scorer = InteractionScorer(data)
        npt.assert_almost_equal(self.scorer(0, 1), -0.1226, 4)
        npt.assert_almost_equal(self.scorer.class_entropy, 0.8113, 4)
        npt.assert_almost_equal(self.scorer.information_gain[0], 0.3113, 4)
        npt.assert_almost_equal(self.scorer.information_gain[1], 0.1226, 4)

    def test_nans(self):
        """Check score calculation with nans"""
        x = np.array([[1, 1], [0, 1], [1, 1], [0, 0], [1, np.nan], [np.nan, 0], [np.nan, np.nan]])
        y = np.array([0, 1, 1, 1, 0, 0, 1])
        domain = Domain([DiscreteVariable(str(i)) for i in range(2)], DiscreteVariable("3"))
        data = Table(domain, x, y)
        self.scorer = InteractionScorer(data)
        npt.assert_almost_equal(self.scorer(0, 1), 0.0167, 4)
        npt.assert_almost_equal(self.scorer.class_entropy, 0.9852, 4)
        npt.assert_almost_equal(self.scorer.information_gain[0], 0.4343, 4)
        npt.assert_almost_equal(self.scorer.information_gain[1], 0.0343, 4)


class TestHeuristic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zoo = Table("zoo")

    def test_heuristic(self):
        """Check attribute pairs returned by heuristic"""
        scorer = InteractionScorer(self.zoo)
        heuristic = Heuristic(scorer.information_gain,
                              type=Heuristic.INFO_GAIN)
        self.assertListEqual(list(heuristic.get_states(None))[:9],
                             [(14, 6), (14, 10), (14, 15), (6, 10),
                              (14, 5), (6, 15), (14, 11), (6, 5), (10, 15)])

        states = heuristic.get_states(None)
        _ = next(states)
        self.assertListEqual(list(heuristic.get_states(next(states)))[:8],
                             [(14, 10), (14, 15), (6, 10), (14, 5),
                              (6, 15), (14, 11), (6, 5), (10, 15)])


if __name__ == "__main__":
    unittest.main()
