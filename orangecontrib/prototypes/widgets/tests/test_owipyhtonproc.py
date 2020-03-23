from orangewidget.tests.base import WidgetTest

from orangecontrib.prototypes.widgets.owipythonproc import OWIPythonConsole


class TestOWIPythonProc(WidgetTest):
    def setUp(self):
        super().setUp()
        self.widget = self.create_widget(
            OWIPythonConsole, stored_settings={
                "content": "out_object = in_object"
            }
        )

    def tearDown(self):
        self.widget.onDeleteWidget()
        self.widget.deleteLater()
        self.widget = None
        super().tearDown()

    def test_simple(self):
        w = self.widget
        self.send_signal(w.Inputs.object_, "a", widget=w)
        out = self.get_output(w.Outputs.object_, widget=w)
        self.assertEqual(out, "a")

        self.send_signal(w.Inputs.object_, 42, widget=w)
        out = self.get_output(w.Outputs.object_, widget=w)
        self.assertEqual(out, 42)

        w.set_source("out_object = in_object + 1")
        self.send_signal(w.Inputs.object_, 42, widget=w)
        out = self.get_output(w.Outputs.object_, widget=w)
        self.assertEqual(out, 43)
