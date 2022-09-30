from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest

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
        """Check interaction table"""
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.model.columnCount(), 0)
        self.assertEqual(self.widget.model.rowCount(), 0)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.model.columnCount(), 4)
        self.assertEqual(self.widget.model.rowCount(), 6)
