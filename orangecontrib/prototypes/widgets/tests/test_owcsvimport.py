import os
import unittest

from Orange.widgets.tests.base import WidgetTest, GuiTest

from orangecontrib.prototypes.widgets.utils import textimport
from orangecontrib.prototypes.widgets import owcsvimport


class TestOWCSVFileImport(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(owcsvimport.OWCSVFileImport)

    def tearDown(self):
        self.widgets.remove(self.widget)
        self.widget.onDeleteWidget()
        self.widget = None

    def test_basic(self):
        w = self.widget
        w.activate_recent(0)
        w.cancel()


class TestImportDialog(GuiTest):
    def test_dialog(self):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, "test_owgrep_file.txt")
        d = owcsvimport.CSVImportDialog()
        d.setPath(path)
        ColumnTypes = owcsvimport.Options.ColumnType
        RowSpec = owcsvimport.Options.RowSpec
        opts = owcsvimport.Options(
            encoding="utf-8",
            dialect=owcsvimport.textimport.Dialect(
                " ", "\"", "\\", True, True
            ),
            columntypes=[
                (range(0, 2), ColumnTypes.Numeric),
                (range(2, 3), ColumnTypes.Categorical)
            ],
            rowspec=[
                (range(0, 4), RowSpec.Skipped),
                (range(4, 5), RowSpec.Header),
                (range(8, 13), RowSpec.Skipped),
            ]
        )
        d.setOptions(opts)
        d.restoreDefaults()
        opts1 = d.options()
        d.reset()
        opts1 = d.options()
