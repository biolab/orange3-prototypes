# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import Orange.widgets
from Orange.data import Table
from orangecontrib.prototypes.widgets.owlookalike import OWLookalike
from Orange.widgets.tests.base import WidgetTest


class TestOWLookalike(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWLookalike)
        self.zoo = Table("zoo-with-images")

    def test_input_neighbors(self):
        """Check widget's neighbors with neighbors on the input"""
        self.assertEqual(self.widget.neighbors, None)
        self.send_signal("Neighbors", self.zoo)
        self.assertEqual(self.widget.neighbors, self.zoo)

    def test_input_neighbors_disconnect(self):
        """Check widget's neighbors after disconnecting neighbors on the input"""
        self.send_signal("Neighbors", self.zoo)
        self.assertEqual(self.widget.neighbors, self.zoo)
        self.send_signal("Neighbors", None)
        self.assertEqual(self.widget.neighbors, None)

    def test_input_reference(self):
        """Check widget's reference with reference on the input"""
        self.assertEqual(self.widget.reference, None)
        self.send_signal("Reference", self.zoo)
        self.assertEqual(self.widget.reference, self.zoo)

    def test_input_reference_disconnect(self):
        """Check reference after disconnecting reference on the input"""
        self.send_signal("Reference", self.zoo)
        self.assertEqual(self.widget.reference, self.zoo)
        self.send_signal("Reference", None)
        self.assertEqual(self.widget.reference, None)

    def test_neighbors_list_view(self):
        """Check content of listbox with neighbors on the input"""
        self.send_signal("Neighbors", self.zoo)
        self.assertEqual(self.widget.neighbors_model.rowCount(), len(self.zoo))
