import unittest
from unittest.mock import patch, Mock

import numpy as np
import numpy.testing as npt

from AnyQt.QtCore import Qt

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.widgets.visualize.owscatterplot import OWScatterPlot
from Orange.widgets.widget import AttributeList
from orangecontrib.prototypes.widgets.owinteractions import \
	OWInteractions, Heuristic, HeuristicType, Interaction, InteractionRank


class TestOWInteractions(WidgetTest):
	@classmethod
	def setUpClass(cls):
		super().setUpClass()
		cls.data = Table("iris")
		cls.disc_data = Table("zoo")

	def setUp(self):
		self.widget = self.create_widget(OWInteractions)

	def test_input_data(self):
		"""Check interaction table"""
		self.send_signal(self.widget.Inputs.data, None)
		self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 4)
		self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 0)
		self.send_signal(self.widget.Inputs.data, self.data)
		self.wait_until_finished()
		n_attrs = len(self.data.domain.attributes)
		self.process_events()
		self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 4)
		self.assertEqual(self.widget.vizrank.rank_model.rowCount(), n_attrs*(n_attrs-1)/2)

	def test_input_data_one_feature(self):
		"""Check interaction table for dataset with one attribute"""
		self.send_signal(self.widget.Inputs.data, self.data[:, [0, 4]])
		self.wait_until_finished()
		self.process_events()
		self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 4)
		self.assertTrue(self.widget.Warning.not_enough_vars.is_shown())
		self.send_signal(self.widget.Inputs.data, None)
		self.assertFalse(self.widget.Warning.not_enough_vars.is_shown())

	def test_data_no_class(self):
		"""Check interaction table for dataset without class variable"""
		self.send_signal(self.widget.Inputs.data, self.data[:, :-1])
		self.wait_until_finished()
		self.process_events()
		self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 4)
		self.assertTrue(self.widget.Warning.no_class_var.is_shown())

	def test_input_data_one_instance(self):
		"""Check interaction table for dataset with one instance"""
		self.send_signal(self.widget.Inputs.data, self.data[:1])
		self.wait_until_finished()
		self.process_events()
		self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 4)
		self.assertFalse(self.widget.Information.removed_cons_feat.is_shown())
		self.assertTrue(self.widget.Warning.not_enough_inst.is_shown())
		self.send_signal(self.widget.Inputs.data, None)
		self.assertFalse(self.widget.Warning.not_enough_inst.is_shown())

	def test_input_data_with_constant_features(self):
		"""Check interaction table for dataset with constant columns"""
		np.random.seed(0)
		x = np.random.randint(3, size=(4, 3)).astype(float)
		x[:, 2] = 1
		y = np.random.randint(3, size=4).astype(float)

		domain_disc = Domain([DiscreteVariable(str(i), ["a", "b", "c"]) for i in range(3)], DiscreteVariable("cls"))
		domain_cont = Domain([ContinuousVariable(str(i)) for i in range(3)], DiscreteVariable("cls"))

		self.send_signal(self.widget.Inputs.data, Table(domain_disc, x, y))
		self.wait_until_finished()
		self.process_events()
		self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 3)
		self.assertFalse(self.widget.Information.removed_cons_feat.is_shown())

		self.send_signal(self.widget.Inputs.data, Table(domain_cont, x, y))
		self.wait_until_finished()
		self.process_events()
		self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 1)
		self.assertTrue(self.widget.Information.removed_cons_feat.is_shown())

		x = np.ones((4, 3), dtype=float)
		self.send_signal(self.widget.Inputs.data, Table(domain_cont, x, y))
		self.wait_until_finished()
		self.process_events()
		self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 4)
		self.assertTrue(self.widget.Warning.not_enough_vars.is_shown())
		self.assertTrue(self.widget.Information.removed_cons_feat.is_shown())

		self.send_signal(self.widget.Inputs.data, None)
		self.assertFalse(self.widget.Information.removed_cons_feat.is_shown())

	def test_output_features(self):
		"""Check features on output"""
		self.send_signal(self.widget.Inputs.data, self.data)
		self.wait_until_finished()
		self.process_events()
		features = self.get_output(self.widget.Outputs.features)
		self.assertIsInstance(features, AttributeList)
		self.assertEqual(len(features), 2)

	def test_output_interactions(self):
		"""Check interaction table on output"""
		self.send_signal(self.widget.Inputs.data, self.data)
		self.wait_until_finished()
		n_attrs = len(self.data.domain.attributes)
		self.process_events()
		interactions = self.get_output(self.widget.Outputs.interactions)
		self.assertIsInstance(interactions, Table)
		self.assertEqual(len(interactions), n_attrs*(n_attrs-1)/2)
		self.assertEqual(len(interactions.domain.metas), 2)
		self.assertListEqual(["Interaction"], [m.name for m in interactions.domain.attributes])

	def test_input_changed(self):
		"""Check whether changing input emits commit"""
		self.widget.commit = Mock()
		self.send_signal(self.widget.Inputs.data, self.data)
		self.wait_until_finished()
		self.process_events()
		self.widget.commit.assert_called_once()

		np.random.seed(0)
		x = np.random.randint(3, size=(4, 3)).astype(float)
		y = np.random.randint(3, size=4).astype(float)
		domain = Domain([DiscreteVariable(str(i), ["a", "b", "c"]) for i in range(3)], DiscreteVariable("cls"))

		self.widget.commit.reset_mock()
		self.send_signal(self.widget.Inputs.data, Table(domain, x, y))
		self.wait_until_finished()
		self.process_events()
		self.widget.commit.assert_called_once()

	def test_saved_selection(self):
		"""Select row from settings"""
		self.send_signal(self.widget.Inputs.data, self.data)
		self.wait_until_finished()
		self.process_events()
		attrs = self.widget.disc_data.domain.attributes
		self.widget._vizrank_selection_changed(attrs[1], attrs[3])
		settings = self.widget.settingsHandler.pack_data(self.widget)

		w = self.create_widget(OWInteractions, stored_settings=settings)
		self.send_signal(self.widget.Inputs.data, self.data, widget=w)
		self.wait_until_finished(w)
		self.process_events()
		sel_row = w.vizrank.rank_table.selectionModel().selectedRows()[0].row()
		self.assertEqual(sel_row, 1)

	def test_scatterplot_input_features(self):
		"""Check if attributes have been set after sent to scatterplot"""
		self.send_signal(self.widget.Inputs.data, self.data)
		spw = self.create_widget(OWScatterPlot)
		attrs = self.widget.disc_data.domain.attributes
		self.widget._vizrank_selection_changed(attrs[2], attrs[3])
		features = self.get_output(self.widget.Outputs.features)
		self.send_signal(self.widget.Inputs.data, self.data, widget=spw)
		self.send_signal(spw.Inputs.features, features, widget=spw)
		self.assertIs(spw.attr_x, self.data.domain[2])
		self.assertIs(spw.attr_y, self.data.domain[3])

	@patch("orangecontrib.prototypes.widgets.owinteractions.SIZE_LIMIT", 2000)
	def test_heuristic_type(self):
		h_type = self.widget.controls.heuristic_type
		self.send_signal(self.widget.Inputs.data, self.disc_data)
		self.wait_until_finished()
		self.process_events()
		infogain = list(self.widget.vizrank.heuristic.get_states(None))

		simulate.combobox_activate_item(h_type, "Random Search")
		self.wait_until_finished()
		self.process_events()
		random = list(self.widget.vizrank.heuristic.get_states(None))

		self.assertFalse(infogain == random, msg="Double check results, there is a 1 in 15! chance heuristics are equal.")

	def test_feature_combo(self):
		"""Check content of feature selection combobox"""
		feature_combo = self.widget.controls.feature
		self.send_signal(self.widget.Inputs.data, self.data)
		self.assertEqual(len(feature_combo.model()), 5)

		self.wait_until_stop_blocking()
		self.send_signal(self.widget.Inputs.data, self.disc_data)
		self.assertEqual(len(feature_combo.model()), 17)

	def test_select_feature(self):
		"""Check feature selection"""
		feature_combo = self.widget.controls.feature
		self.send_signal(self.widget.Inputs.data, self.data)
		self.wait_until_finished()
		self.process_events()
		self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 6)
		self.assertListEqual(
			[a.name for a in self.get_output(self.widget.Outputs.features)],
			["sepal length", "sepal width"]
		)

		simulate.combobox_activate_index(feature_combo, 3)
		self.wait_until_finished()
		self.process_events()
		self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 3)
		self.assertListEqual(
			[a.name for a in self.get_output(self.widget.Outputs.features)],
			["petal length", "sepal width"]
		)

		simulate.combobox_activate_index(feature_combo, 0)
		self.wait_until_finished()
		self.process_events()
		self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 6)
		self.assertListEqual(
			[a.name for a in self.get_output(self.widget.Outputs.features)],
			["petal length", "sepal width"]
		)

	@patch("orangecontrib.prototypes.widgets.owinteractions.SIZE_LIMIT", 2000)
	def test_vizrank_use_heuristic(self):
		"""Check heuristic use"""
		self.send_signal(self.widget.Inputs.data, self.data)
		self.wait_until_finished()
		self.process_events()
		self.assertTrue(self.widget.vizrank.use_heuristic)
		self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 6)

	@patch("orangecontrib.prototypes.widgets.owinteractions.SIZE_LIMIT", 2000)
	def test_select_feature_against_heuristic(self):
		"""Check heuristic use when feature selected"""
		feature_combo = self.widget.controls.feature
		self.send_signal(self.widget.Inputs.data, self.data)
		simulate.combobox_activate_index(feature_combo, 2)
		self.wait_until_finished()
		self.process_events()
		self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 3)
		self.assertEqual(self.widget.vizrank.heuristic, None)


class TestInteractionRank(WidgetTest):
	@classmethod
	def setUpClass(cls):
		super().setUpClass()
		x = np.array([[1, 1], [0, 1], [1, 1], [0, 0]])
		y = np.array([0, 1, 1, 1])
		domain = Domain([DiscreteVariable(str(i)) for i in range(2)], DiscreteVariable("3"))
		cls.data = Table(domain, x, y)
		cls.attrs = cls.data.domain.attributes

	def setUp(self):
		self.vizrank = InteractionRank(None)
		self.vizrank.attrs = self.attrs

	def test_row_for_state(self):
		"""Check row calculation"""
		row = self.vizrank.row_for_state((0.1511, 0.3837, 0.1511), (0, 1))
		self.assertEqual(row[0].data(Qt.DisplayRole), "+15.1%")
		self.assertEqual(row[0].data(InteractionRank.IntRole), 0.1511)
		self.assertListEqual(row[0].data(InteractionRank.GainRole), [0.3837, 0.1511])
		self.assertEqual(row[1].data(Qt.DisplayRole), "68.6%")
		self.assertEqual(row[2].data(Qt.DisplayRole), self.attrs[0].name)
		self.assertEqual(row[3].data(Qt.DisplayRole), self.attrs[1].name)


class TestInteractionScorer(unittest.TestCase):
	def test_compute_score(self):
		"""Check score calculation"""
		x = np.array([[1, 1], [0, 1], [1, 1], [0, 0]])
		y = np.array([0, 1, 1, 1])
		domain = Domain([DiscreteVariable(str(i)) for i in range(2)], DiscreteVariable("3"))
		data = Table(domain, x, y)
		self.interaction = Interaction(data)
		npt.assert_almost_equal(self.interaction(0, 1), -0.1226, 4)
		npt.assert_almost_equal(self.interaction.class_h, 0.8113, 4)
		npt.assert_almost_equal(self.interaction.attr_h[0], 1., 4)
		npt.assert_almost_equal(self.interaction.attr_h[1], 0.8113, 4)
		npt.assert_almost_equal(self.interaction.gains[0], 0.3113, 4)
		npt.assert_almost_equal(self.interaction.gains[1], 0.1226, 4)
		npt.assert_almost_equal(self.interaction.removed_h[0, 1], 0.3113, 4)

	def test_nans(self):
		"""Check score calculation with sparse data"""
		x = np.array([[1, 1], [0, 1], [1, 1], [0, 0], [1, np.nan], [np.nan, 0], [np.nan, np.nan]])
		y = np.array([0, 1, 1, 1, 0, 0, 1])
		domain = Domain([DiscreteVariable(str(i)) for i in range(2)], DiscreteVariable("3"))
		data = Table(domain, x, y)
		self.interaction = Interaction(data)
		npt.assert_almost_equal(self.interaction(0, 1), 0.0167, 4)
		npt.assert_almost_equal(self.interaction.class_h, 0.9852, 4)
		npt.assert_almost_equal(self.interaction.attr_h[0], 0.9710, 4)
		npt.assert_almost_equal(self.interaction.attr_h[1], 0.9710, 4)
		npt.assert_almost_equal(self.interaction.gains[0], 0.4343, 4)
		npt.assert_almost_equal(self.interaction.gains[1], 0.0343, 4)
		npt.assert_almost_equal(self.interaction.removed_h[0, 1], 0.4852, 4)


class TestHeuristic(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.zoo = Table("zoo")

	def test_heuristic(self):
		"""Check attribute pairs returned by heuristic"""
		score = Interaction(self.zoo)
		heuristic = Heuristic(score.gains, heuristic_type=HeuristicType.INFOGAIN)
		self.assertListEqual(
			list(heuristic.get_states(None))[:9],
			[(14, 6), (14, 10), (14, 15), (6, 10), (14, 5), (6, 15), (14, 11), (6, 5), (10, 15)]
		)

		states = heuristic.get_states(None)
		_ = next(states)
		self.assertListEqual(
			list(heuristic.get_states(next(states)))[:8],
			[(14, 10), (14, 15), (6, 10), (14, 5), (6, 15), (14, 11), (6, 5), (10, 15)]
		)


if __name__ == "__main__":
	unittest.main()
