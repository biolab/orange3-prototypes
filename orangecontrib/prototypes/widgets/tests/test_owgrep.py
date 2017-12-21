# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
# Tests test protected methods
# pylint: disable=protected-access
import os
import unittest
from unittest.mock import Mock, patch

import numpy as np
from AnyQt.QtWidgets import QApplication

from Orange.widgets.utils.filedialogs import RecentPath
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.prototypes.widgets import owgrep


def patch_file_dlg(filename):
    return patch.object(owgrep.QFileDialog, "getOpenFileName",
                        return_value=(filename, None))


class TestOWGrep(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.curdir = os.path.split(__file__)[0]
        cls.test_base = "test_owgrep_file.txt"
        cls.test_file = os.path.join(cls.curdir, cls.test_base)
        with open(cls.test_file) as f:
            cls.test_content = f.read().rstrip()

    def setUp(self):
        self.widget = self.create_widget(owgrep.OWGrep)  #: OWGrep

    def assertCalledAgain(self, func: Mock):
        self.assertTrue(func.called)
        func.reset_mock()

    ############
    # General tests: select a file and check the output signal

    def test_select_file(self):
        widget = self.widget
        widget.pattern = "ef"
        widget.has_header_row = True
        widget.skip_lines = 3
        widget.block_length = 2
        widget.recent_paths = [
            RecentPath(__file__, *os.path.split(__file__)),
            RecentPath(os.path.join(self.curdir, self.test_base),
                       self.curdir, self.test_base)]

        widget.select_file(1)
        table = self.get_output(widget.Outputs.data)
        self.assertEqual([x.name for x in table.domain.variables],
                         ["a", "b", "c"])

    def test_load_recent_file(self):
        widget = self.create_widget(
            owgrep.OWGrep,
            stored_settings=dict(
                recent_paths=[
                    RecentPath(os.path.join(self.curdir, self.test_base),
                               self.curdir, self.test_base)],
                pattern="ef",
                has_header_row=True, skip_lines=3, block_length=2))
        table = self.get_output(widget.Outputs.data, widget)
        self.assertEqual([x.name for x in table.domain.variables],
                         ["a", "b", "c"])

    @patch_file_dlg("test_owgrep_file.txt")
    def test_browse_file(self, file_mock):
        widget = self.widget

        widget.browse_file()
        table = self.get_output(widget.Outputs.data)

        file_mock.return_value = ("", None)
        widget.browse_file()
        self.assertIs(self.get_output(widget.Outputs.data), table)

    @patch_file_dlg("test_owgrep_file.txt")
    def test_auto_commit(self, _):
        widget = self.widget
        widget.browse_file()
        table = self.get_output(widget.Outputs.data)

        widget.controls.auto_send.setChecked(False)
        widget.pattern = "def"
        widget.controls.pattern.returnPressed.emit()
        self.assertIs(self.get_output(widget.Outputs.data), table)

        widget.controls.auto_send.setChecked(True)
        self.assertIsNot(self.get_output(widget.Outputs.data), table)

    @patch_file_dlg("test_owgrep_file.txt")
    def test_no_lines(self, _):
        widget = self.widget
        widget.browse_file()

        widget.pattern = "foo"
        widget.controls.pattern.returnPressed.emit()
        self.assertIsNone(self.get_output(widget.Outputs.data))
        self.assertTrue(widget.Warning.no_lines.is_shown())

        widget.pattern = "def"
        widget.controls.pattern.returnPressed.emit()
        self.assertIsNotNone(self.get_output(widget.Outputs.data))
        self.assertFalse(widget.Warning.no_lines.is_shown())

    @patch_file_dlg("test_owgrep_file.txt")
    @patch.object(owgrep.Table, "from_file")
    def test_unreadable(self, table_mock, _):
        widget = self.widget
        widget.pattern = "ef"

        table_mock.side_effect = ValueError
        widget.browse_file()
        self.assertIsNone(self.get_output(widget.Outputs.data))
        self.assertTrue(widget.Error.unreadable.is_shown())

        table_mock.side_effect = None
        widget.browse_file()
        self.assertIsNotNone(self.get_output(widget.Outputs.data))
        self.assertFalse(widget.Error.unreadable.is_shown())

    @patch_file_dlg("test_owgrep_file.txt")
    def test_find(self, _):
        widget = self.widget
        widget.browse_file()

        widget.find_text = "Line"  # case insensitive
        widget.controls.find_text.returnPressed.emit()
        pos1 = widget.in_view.textCursor().position()  # First occurrence
        widget.controls.find_text.returnPressed.emit()
        pos2 = widget.in_view.textCursor().position()  # Second occurrence
        widget.controls.find_text.returnPressed.emit()
        pos3 = widget.in_view.textCursor().position()  # Wrap around
        self.assertEqual(pos1, pos3)
        self.assertNotEqual(pos1, pos2)

        widget.find_text = "foo"
        widget.controls.find_text.returnPressed.emit()
        pos4 = widget.in_view.textCursor().position()  # Doesn't move
        self.assertEqual(pos1, pos4)

        widget.find_text = ""
        widget.controls.find_text.returnPressed.emit()
        pos4 = widget.in_view.textCursor().position()  # Doesn't move
        self.assertEqual(pos1, pos4)

    @patch_file_dlg("test_owgrep_file.txt")
    def test_copy(self, _):
        widget = self.widget
        widget.browse_file()

        widget.find_text = "Longer Line"  # case insensitive
        widget.controls.find_text.returnPressed.emit()
        cursor = widget.in_view.textCursor()
        cursor.select(cursor.BlockUnderCursor)
        widget.copy_to_clipboard()
        self.assertEqual(QApplication.clipboard().text(), "longer line")


    ############
    # Tests of individual methods (with more assumptions about implementation)

    def test_open_file(self):
        widget = self.widget
        widget.grep_lines = Mock()

        widget.last_path = Mock(return_value=self.test_file)
        widget.open_file()
        self.assertEqual(widget.current_file, self.test_file)
        self.assertEqual(widget.in_view.toPlainText(), self.test_content)
        self.assertFalse(widget.Error.file_not_found.is_shown())
        self.assertCalledAgain(widget.grep_lines)

        widget.last_path = Mock(return_value="foo")
        widget.open_file()
        self.assertEqual(widget.current_file, None)
        self.assertEqual(widget.in_view.toPlainText(), "")
        self.assertTrue(widget.Error.file_not_found.is_shown())
        self.assertCalledAgain(widget.grep_lines)

        widget.last_path = Mock(return_value=self.test_file)
        widget.open_file()
        self.assertEqual(widget.current_file, self.test_file)
        self.assertFalse(widget.Error.file_not_found.is_shown())
        self.assertCalledAgain(widget.grep_lines)

        widget.last_path = Mock(return_value=None)
        widget.open_file()
        self.assertEqual(widget.current_file, None)
        self.assertEqual(widget.in_view.toPlainText(), "")
        self.assertFalse(widget.Error.file_not_found.is_shown())
        self.assertCalledAgain(widget.grep_lines)

        widget.last_path = Mock(return_value="foo")
        widget.open_file()
        self.assertTrue(widget.Error.file_not_found.is_shown())
        self.assertCalledAgain(widget.grep_lines)

    def _grep_and_check(self, expected):
        widget = self.widget
        widget.set_out_view = Mock()
        widget.commit = Mock()
        widget.grep_lines()
        self.assertEqual(widget.selected_lines, expected)
        self.assertTrue(widget.set_out_view.called)
        self.assertTrue(widget.commit.called)

    def test_grep_lines_no_file(self):
        widget = self.widget
        widget.pattern = "ef"
        widget.block_length = 1
        widget.skip_lines = 0

        widget.current_file = self.test_file
        self._grep_and_check(["def", "def"])

        widget.current_file = None
        self._grep_and_check([])

    def test_grep_lines_no_pattern(self):
        widget = self.widget
        widget.current_file = self.test_file
        widget.block_length = 1
        widget.skip_lines = 0
        widget.commit = Mock()

        widget.pattern = "ef"
        self._grep_and_check(["def", "def"])
        self.assertCalledAgain(widget.commit)

        widget.pattern = ""
        self._grep_and_check([])
        self.assertFalse(widget.Warning.no_lines.is_shown())
        self.assertCalledAgain(widget.commit)

        widget.pattern = "foo"
        self._grep_and_check([])
        self.assertTrue(widget.Warning.no_lines.is_shown())
        self.assertCalledAgain(widget.commit)

        widget.pattern = ""
        self._grep_and_check([])
        self.assertFalse(widget.Warning.no_lines.is_shown())
        self.assertCalledAgain(widget.commit)

    def test_grep_lines_skip_and_length(self):
        widget = self.widget
        widget.current_file = self.test_file

        widget.pattern = "ef"
        widget.block_length = 1
        widget.skip_lines = 3
        self._grep_and_check(["a b c", "d e f"])

        widget.pattern = "ef"
        widget.skip_lines = 3
        widget.block_length = 2
        self._grep_and_check(["a b c", "1 2.123 blue", "d e f", "3.1 1 red"])

        widget.pattern = "long"
        widget.skip_lines = 0
        widget.block_length = 1
        self._grep_and_check(["a longer line", "another long line"])

        widget.pattern = "long"
        widget.skip_lines = 0
        widget.block_length = 2
        self._grep_and_check(
            ["a longer line", "---", "another long line", "---"])

        widget.pattern = "long"
        widget.skip_lines = 1
        widget.block_length = 1
        self._grep_and_check(["---", "---"])

    def test_grep_lines_regular_expression(self):
        widget = self.widget
        widget.current_file = self.test_file
        widget.skip_lines = 0
        widget.block_length = 1
        widget.pattern = "long.*li"

        widget.regular_expression = False
        self._grep_and_check([])

        widget.regular_expression = True
        self._grep_and_check(["a longer line", "another long line"])

    def test_grep_lines_case_sensitive(self):
        widget = self.widget
        widget.current_file = self.test_file
        widget.skip_lines = 0
        widget.block_length = 1
        widget.pattern = "lONg"

        widget.regular_expression = True
        widget.case_sensitive = False
        self._grep_and_check(["a longer line", "another long line"])

        widget.regular_expression = False
        widget.case_sensitive = False
        self._grep_and_check(["a longer line", "another long line"])

        widget.regular_expression = True
        widget.case_sensitive = True
        self._grep_and_check([])

        widget.regular_expression = False
        widget.case_sensitive = True
        self._grep_and_check([])

    def test_grep_lines_eof(self):
        widget = self.widget
        widget.current_file = self.test_file

        widget.pattern = "ef"
        widget.skip_lines = 3
        widget.block_length = 4
        self._grep_and_check(
            ["a b c", "1 2.123 blue", "2.4 1.1 red", "2.5 1.235 red",
             "d e f", "3.1 1 red", "1.3 ? blue"])

    def test_grep_nothing_found(self):
        widget = self.widget
        widget.current_file = self.test_file
        widget.skip_lines = 0
        widget.block_length = 1

        widget.pattern = "def"
        self._grep_and_check(["def", "def"])

        widget.pattern = "foo"
        self._grep_and_check([])
        self.assertTrue(widget.Warning.no_lines.is_shown())

        widget.pattern = "def"
        self._grep_and_check(["def", "def"])
        self.assertFalse(widget.Warning.no_lines.is_shown())

    def test_set_out_view(self):
        widget = self.widget
        widget.selected_lines = list("abcde")
        text = "\n".join("abcde")

        widget.has_header_row = False
        widget.block_length = 1
        widget.set_out_view()
        self.assertEqual(widget.out_view.toPlainText(), text)

        widget.has_header_row = True
        widget.block_length = 1
        widget.set_out_view()
        self.assertEqual(widget.out_view.toPlainText(), text)

        widget.has_header_row = False
        widget.block_length = 2
        widget.set_out_view()
        self.assertEqual(widget.out_view.toPlainText(), text)

        widget.has_header_row = True
        widget.block_length = 1
        widget.set_out_view()
        self.assertEqual(widget.out_view.toPlainText(), text)

    @patch("os.remove", wraps=os.remove)
    def test_construct_table_with_header(self, remove_mock):
        widget = self.widget

        widget.selected_lines = \
            ["a b c", "1 2.123 blue", "2.4 1.1 red",
             "d e f", "3.1 1 red"]
        widget.block_length = 3

        table = widget._construct_table(True)
        np.testing.assert_almost_equal(
            table.X, np.array([[1, 2.123, 0], [2.4, 1.1, 1], [3.1, 1, 1]]))
        self.assertEqual([x.name for x in table.domain.variables],
                         ["a", "b", "c"])
        self.assertCalledAgain(remove_mock)

        widget.selected_lines = ["1 2.123 blue", "2.4 1.1 red", "3.1 1 red"]

    @patch("os.remove", wraps=os.remove)
    def test_construct_table_without_header(self, remove_mock):
        widget = self.widget
        widget.selected_lines = \
            ["1 2.123 blue", "2.4 1.1 red", "3.1 1 red",
             "1.3 ? blue", "2.5 1.235 red"]

        for widget.block_length in range(1, 7):
            table = widget._construct_table(False)
            np.testing.assert_almost_equal(
                table.X, np.array([
                    [1, 2.123, 0], [2.4, 1.1, 1], [3.1, 1, 1],
                    [1.3, np.nan, 0], [2.5, 1.235, 1]]),
                err_msg="at block length={}".format(widget.block_length))
            self.assertEqual([x.name for x in table.domain.variables],
                             ["var001", "var002", "var003"])
            self.assertCalledAgain(remove_mock)

    def test_get_data_block_too_short(self):
        widget = self.widget
        widget.has_header_row = True
        widget.selected_lines = ["1 2.123 blue", "2.4 1.1 red", "3.1 1 red"]

        widget.block_length = 1
        self.assertIsNone(widget._get_data())
        self.assertTrue(widget.Error.block_too_short.is_shown())

        widget.block_length = 2
        self.assertIsNotNone(widget._get_data())
        self.assertFalse(widget.Error.block_too_short.is_shown())

        widget.block_length = 1
        self.assertIsNone(widget._get_data())
        self.assertTrue(widget.Error.block_too_short.is_shown())

        widget.block_length = 1
        widget.selected_lines = []
        self.assertIsNone(widget._get_data())
        self.assertFalse(widget.Error.block_too_short.is_shown())

    def test_commit_header_override(self):
        widget = self.widget
        widget.selected_lines = \
            ["1 2.123 1.1", "2.4 1.1 1", "3.1 1 2", "1.3 1 3", "2.5 1.235 2"]
        expected = np.array(np.mat("; ".join(widget.selected_lines)))

        widget.has_header_row = True
        widget.block_length = 5
        table = widget._get_data()
        np.testing.assert_almost_equal(table.X, expected)
        self.assertTrue(widget.Warning.no_header_row.is_shown())
        self.assertEqual([x.name for x in table.domain.variables],
                         ["var001", "var002", "var003"])

        widget.has_header_row = False
        table = widget._get_data()
        np.testing.assert_almost_equal(table.X, expected)
        self.assertFalse(widget.Warning.no_header_row.is_shown())

        widget.has_header_row = True
        widget.block_length = 5
        with patch.object(owgrep.Table, "from_file", side_effect=ValueError):
            self.assertIsNone(widget._get_data())
            self.assertTrue(widget.Error.unreadable.is_shown())
            self.assertFalse(widget.Warning.no_header_row.is_shown())

    def test_header_changed_callback(self):
        widget = self.widget
        widget.commit = Mock()
        widget.set_out_view = Mock()
        widget.controls.has_header_row.setChecked(True)
        widget.controls.has_header_row.setChecked(False)
        self.assertCalledAgain(widget.commit)
        self.assertCalledAgain(widget.set_out_view)
        widget.controls.has_header_row.setChecked(True)
        self.assertCalledAgain(widget.commit)
        self.assertCalledAgain(widget.set_out_view)


if __name__ == "__main__":
    unittest.main()
