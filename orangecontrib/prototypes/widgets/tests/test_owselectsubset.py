from Orange.data import Table, Domain
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.prototypes.widgets.owselectsubset import OWSelectSubset


class TestOWSelectiSubset(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWSelectSubset)

    def test_subset(self):
        data = Table("iris")
        data_subset = data[20:40].transform(Domain([]))  # destroy domain
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.data_subset, data_subset)
        out = self.get_output(self.widget.Outputs.data)
        self.assertEqual(list(data[20:40]), list(out))

    def test_subset_nosubset(self):
        data = Table("iris")
        data_subset = Table("titanic")
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.data_subset, data_subset)
        out = self.get_output(self.widget.Outputs.data)
        self.assertTrue(self.widget.Warning.instances_not_matching.is_shown())
        self.assertEqual([], list(out))
