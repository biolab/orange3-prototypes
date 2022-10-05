from unittest.mock import Mock

import numpy as np

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.widget import AttributeList

from orangecontrib.prototypes.widgets.owinteractions_new import OWInteractions


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

    def test_feature_combo(self):
        """Check feature combobox"""
        feature_combo = self.widget.controls.feature
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(len(feature_combo.model()), 5)

        self.wait_until_stop_blocking()
        self.send_signal(self.widget.Inputs.data, self.zoo)
        self.assertEqual(len(feature_combo.model()), 17)
