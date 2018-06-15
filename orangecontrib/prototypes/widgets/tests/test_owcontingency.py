from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.prototypes.widgets.owcontingency import OWContingencyTable


class TestOWContingencyTable(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWContingencyTable)
        self.titanic = Table("titanic")

    def test_input(self):
        """
        Test whether correct outputs appear when input is provided.
        """
        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.contingency))

        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.annotated_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.contingency))

    def test_attribute_selection(self):
        """
        Test selection of row an column attributes.
        """
        self.send_signal(self.widget.Inputs.data, self.titanic)

        self.widget.rows = self.titanic.domain["survived"]
        self.widget.columns = self.titanic.domain["survived"]
        self.widget._attribute_changed()
        self.assertEqual((2, 2), self.get_output(self.widget.Outputs.contingency).X.shape)

        self.widget.rows = self.titanic.domain["survived"]
        self.widget.columns = self.titanic.domain["status"]
        self.widget._attribute_changed()
        self.assertEqual((2, 4), self.get_output(self.widget.Outputs.contingency).X.shape)

        self.widget.rows = self.titanic.domain["status"]
        self.widget.columns = self.titanic.domain["survived"]
        self.widget._attribute_changed()
        self.assertEqual((4, 2), self.get_output(self.widget.Outputs.contingency).X.shape)

    def test_filtering(self):
        """
        Test data filtering based on selected cells in the table.
        """
        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.widget.rows = self.titanic.domain["survived"]
        self.widget.columns = self.titanic.domain["status"]

        self.widget.selection = {}
        self.widget.commit()
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

        self.widget.selection = {(0, 0)}
        self.widget.commit()
        self.assertEqual(673, len(self.get_output(self.widget.Outputs.selected_data)))

        self.widget.selection = {(0, 0), (1, 0)}
        self.widget.commit()
        self.assertEqual(673 + 212, len(self.get_output(self.widget.Outputs.selected_data)))

        self.widget.selection = {(0, 0), (0, 1)}
        self.widget.commit()
        self.assertEqual(673 + 122, len(self.get_output(self.widget.Outputs.selected_data)))
