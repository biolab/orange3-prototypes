from unittest import mock
from contextlib import ExitStack

from AnyQt.QtTest import QSignalSpy

from Orange.widgets.tests.base import WidgetTest
from orangecontrib.prototypes.widgets.owdatasets import OWDataSets
from orangecontrib.prototypes.widgets import owdatasets


class TestOWDataSets(WidgetTest):
    def setUp(self):
        super().setUp()
        remote = {
            ("a", "b"): {"title": "K", "size": 10},
            ("a", "c"): {"title": "T", "size": 20},
            ("a", "d"): {"title": "Y", "size": 0},
        }
        self.exit = ExitStack()
        self.exit.__enter__()
        self.exit.enter_context(
            mock.patch.object(owdatasets, "list_remote", lambda: remote)
        )
        self.exit.enter_context(
            mock.patch.object(owdatasets, "list_local", lambda: {})
        )

        self.widget = self.create_widget(
            OWDataSets, stored_settings={
                "selected_id": ("a", "c"),
                "auto_commit": False,
            }
        )  # type: OWDataSets

    def tearDown(self):
        super().tearDown()
        self.exit.__exit__(None, None, None)

    def test_init(self):
        if self.widget.isBlocking():
            spy = QSignalSpy(self.widget.blockingStateChanged)
            assert spy.wait(1000)
        self.assertFalse(self.widget.isBlocking())

        model = self.widget.view.model()
        self.assertEqual(model.rowCount(), 3)

        di = self.widget.selected_dataset()
        self.assertEqual((di.prefix, di.filename), ("a", "c"))
