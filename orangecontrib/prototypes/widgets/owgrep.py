import os
import re

from tempfile import NamedTemporaryFile

from AnyQt.QtCore import Qt, QTimer
from AnyQt.QtWidgets import QTextEdit, QStyle, QFileDialog, QGridLayout

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting, ContextSetting, ContextHandler
from Orange.widgets.widget import OWWidget, Msg, Output
from Orange.widgets.utils.filedialogs import RecentPathsWComboMixin


class NameContextHandler(ContextHandler):
    def new_context(self, name):
        context = super().new_context()
        context.name = os.path.split(name)[1]
        return context

    # noinspection PyMethodOverriding
    def match(self, context, name):
        name = os.path.split(name)[1]
        return (name == context.name) + \
               (os.path.splitext(name)[1] == os.path.splitext(context.name)[1])

# The data flows through the widget as follows:
# - open_file set current_file (+ in_view) and calls grep_lines;
# - grep_lines greps into selected_lines and calls set_out_view and commit
# - set_out_view shows the grepped lines, with headers in bold
# - commit constructs a table from selected_lines and outputs it
#
# Callbacks enter this at the following points:
# - combo with recent files and the button open call open_file
# - all pattern-related controls except the checkbox for header call grep_lines
# - the checkbox for header calls set_out_view and commit, without re-grepping

class OWGrep(OWWidget, RecentPathsWComboMixin):
    name = "Grep"
    description = "Greps data from text, e.g. log files"
    icon = "icons/Grep.svg"
    priority = 1102

    class Outputs:
        data = Output("Data", Table, default=True)

    settingsHandler = NameContextHandler()

    last_dir = Setting(os.path.expanduser("~"))

    pattern = ContextSetting("")
    case_sensitive = ContextSetting(True)
    regular_expression = ContextSetting(False)
    skip_lines = ContextSetting(0)
    block_length = ContextSetting(1)
    has_header_row = ContextSetting(False)

    auto_send = Setting(True)

    out_css = """
    <style>
        div {
            /* Spaces in monospace are narrower than other characters! */
            font-family: Consolas, Courier, monospace; 
            font-size: 11pt;
            line-height: 120%;
            white-space: pre
        }
        
        .header, .add-header {
            font-weight: 900;
        }
        
        .add-header {
            color: gray;
        }
    </style>
    """

    class Warning(OWWidget.Warning):
        no_lines = Msg("Pattern not found")
        no_header_row = Msg("Blocks do not appear to have headers.")

    class Error(OWWidget.Error):
        unreadable = Msg("Data is not readable.\n{}")
        file_not_found = Msg("File not found")
        block_too_short = Msg("Block with headers must more than 1 line.")

    def __init__(self):
        super().__init__()
        RecentPathsWComboMixin.__init__(self)

        self.current_file = None
        self.find_text = ""
        self.selected_lines = []

        box = gui.widgetBox(
            self.controlArea, box="File", orientation=QGridLayout())
        box.layout().addWidget(self.file_combo, 0, 0, 1, 2)
        self.file_combo.activated[int].connect(self.select_file)
        box.layout().addWidget(
            gui.button(
                None, self, 'Open', callback=self.browse_file,
                autoDefault=False,
                icon=self.style().standardIcon(QStyle.SP_DirOpenIcon)),
            1, 0)
        box.layout().addWidget(
            gui.button(
                None, self, "Reload", callback=self.open_file,
                autoDefault=False,
                icon=self.style().standardIcon(QStyle.SP_BrowserReload)),
            1, 1)

        box = gui.widgetBox(self.controlArea, box="Pattern")
        lineedit = gui.lineEdit(box, self, "pattern")
        lineedit.returnPressed.connect(self.grep_lines)
        gui.checkBox(
            box, self, "case_sensitive", label="Case sensitive",
            callback=self.grep_lines)
        gui.checkBox(
            box, self, "regular_expression", label="Regular expression",
            callback=self.grep_lines)

        box = gui.widgetBox(self.controlArea, box="Format")
        gui.spin(
            box, self, "skip_lines", 0, 10, label="Skipped lines: ",
            callback=self.grep_lines)
        gui.spin(
            box, self, "block_length", 1, 100000, label="Block length: ",
            callback=self.grep_lines)
        gui.checkBox(
            box, self, "has_header_row", label="Block(s) start with header row",
            tooltip="Only the header at the first block will be used.",
            callback=self.has_header_changed)
        gui.rubber(self.controlArea)

        gui.auto_commit(self.controlArea, self, "auto_send", "Send")

        box = gui.hBox(self.mainArea)
        gui.widgetLabel(box, "Input text")
        gui.rubber(box)
        find_line = gui.lineEdit(
            box, self, "find_text", label="Find: ", orientation=Qt.Horizontal,
            callback=self.find_changed, callbackOnType=True)
        find_line.returnPressed.connect(self.find_changed)
        self.in_view = QTextEdit(readOnly=True)
        self.mainArea.layout().addWidget(self.in_view)

        gui.widgetLabel(self.mainArea, "Used lines")
        self.out_view = QTextEdit(readOnly=True)
        self.mainArea.layout().addWidget(self.out_view)

        self.set_file_list()
        # Must not call select_file from within __init__ to avoid reentering
        # the event loop by a progress bar, when we have it
        if self.recent_paths:
            QTimer.singleShot(0, self.select_file)

    def sizeHint(self):  # pragma: no cover
        size = super().sizeHint()
        size.setWidth(850)
        return size

    def select_file(self, n=0):
        """
        Pull the file with the given index in the list to the top and open it.
        """
        assert n < len(self.recent_paths)
        super().select_file(n)
        if self.recent_paths:
            self.closeContext()
            self.openContext(self.last_path())
            self.open_file()
            self.set_file_list()

    def browse_file(self):
        """
        Open the file dialog and open the chosen file.
        """
        start_path = self.last_path() or os.path.expanduser("~")
        filename, _ = QFileDialog.getOpenFileName(
            None, "Open File", start_path, 'All Files (*.*)')
        if not filename:
            return
        self.closeContext()
        self.add_path(filename)
        self.openContext(filename)
        self.open_file()

    def open_file(self):
        """
        Open the file returned by `last_path`.

        Sets `current_file`, shows its contents in `in_view` and calls
        `grep_lines`. This happens even if `current_file` is empty, in which
        case everything is cleared.
        """
        self.Error.file_not_found.clear()
        self.current_file = self.last_path()
        text = ""
        if self.current_file:
            if not os.path.exists(self.current_file):
                self.Error.file_not_found()
                self.current_file = None
            else:
                with open(self.current_file) as f:
                    text = f.read()
        self.in_view.setHtml(self.out_css + "<div>{}</div>".format(text))
        self.grep_lines()

    def grep_lines(self):
        """
        Grep the lines from `current_file` into `selected_lines`.
        If there is no current file, `selected_lines` is set to `[]`.
        Finally, it calls `set_out_view` and `commit`.

        Depends on `current_file` and all settings except `has_header_row`.
        """
        def prepare_re():
            pattern = self.pattern
            if not self.regular_expression:
                pattern = re.escape(pattern)
            flags = re.IGNORECASE if not self.case_sensitive else 0
            return re.compile(pattern, flags)

        self.Warning.no_lines.clear()
        self.selected_lines = []
        if self.pattern and self.current_file:
            pattern = prepare_re()
            with open(self.current_file) as f:
                file_lines = iter(f)
                try:
                    line = next(file_lines)
                    while True:
                        if pattern.search(line):
                            for _ in range(self.skip_lines):
                                line = next(file_lines)
                            for _ in range(self.block_length):
                                self.selected_lines.append(line.strip())
                                line = next(file_lines)
                        else:
                            line = next(file_lines)
                except StopIteration:
                    pass
            self.Warning.no_lines(shown=not self.selected_lines)
        self.set_out_view()
        self.commit()

    def set_out_view(self):
        """
        Show the grepped lines in the output view.

        Directly uses `selected_lines`, `has_header_row` and `block_length`;
        depends on other settings through `selected_lines`.
        """
        if self.has_header_row:
            text = "\n".join(
                "\n".join(
                    ['<span class="{}">{}</span>'.format(
                        "header" if i == 0 else "add-header",
                        self.selected_lines[i])]
                    + self.selected_lines[i + 1: i + self.block_length])
                for i in range(0, len(self.selected_lines), self.block_length))
        else:
            text = "\n".join(self.selected_lines)
        self.out_view.setHtml(self.out_css + "<div>{}</div>".format(text))

    def has_header_changed(self):
        """
        Callback handler for the header checkbox.

        Calls `set_out_view` to update the view (headers are shown in bold)
        and `commit`."""
        self.set_out_view()
        self.commit()

    def commit(self):
        """Output the data table.

        All functionality is in a separate function for simpler control flow.
        """
        self.Outputs.data.send(self._get_data())

    def _get_data(self):
        """
        Return a `Orange.data.Table` with the data from selected lines.

        If the data has a `header_row` but the reader does not recognize it,
        the function adds a header and rereads it, and shows a warning.

        If `selected_lines` are empty, the function returns `None`.
        """
        self.Warning.no_header_row.clear()
        self.Error.block_too_short.clear()
        self.Error.unreadable.clear()
        if not self.selected_lines:
            return None
        if self.has_header_row:
            if self.block_length == 1:
                self.Error.block_too_short()
                return None
            out_data = self._construct_table(with_header_row=True)
            if out_data is None:
                return None
            if all(var.name == "Feature {}".format(i + 1)
                   for i, var in enumerate(out_data.domain.variables)):
                self.Warning.no_header_row()
                return self._construct_table(with_header_row=False)
            else:
                return out_data
        else:
            return self._construct_table(with_header_row=False)

    # pylint: disable=broad-except
    def _construct_table(self, with_header_row):
        """
        Write the `selected_lines` into a temporary file and read it.

        If `with_header_row` is `True`, the first line in every block is
        skipped, except for the first block.

        If `with_header_row` is `False`, a header row is added.
        """
        assert self.selected_lines
        tempf = NamedTemporaryFile("w", suffix=".csv", delete=False)
        if with_header_row:
            data = self.selected_lines[0] + "\n" + \
                "\n".join(
                    "\n".join(self.selected_lines[i + 1:i + self.block_length])
                    for i in range(0, len(self.selected_lines),
                                   self.block_length))
        else:
            n_cols = len(self.selected_lines[0].split())
            data = (
                " ".join("var{:03}".format(i + 1) for i in range(n_cols))
                + "\n"
                + "\n".join(self.selected_lines))
        tempf.write(re.sub(" +", " ", data))
        tempf.close()
        try:
            return Table.from_file(tempf.name)
        except Exception as err:
            self.Error.unreadable(str(err).replace(" " + tempf.name, ""))
        finally:
            os.remove(tempf.name)

    def find_changed(self):
        """Callback for searchin within the file"""
        if not self.find_text:
            return
        in_view = self.in_view
        cursor = in_view.document().find(
            self.find_text, in_view.textCursor().position() + 1)
        if cursor.position() == -1:
            cursor = in_view.document().find(self.find_text, 0)
        if cursor.position() >= 0:
            in_view.setTextCursor(cursor)
            in_view.ensureCursorVisible()
            in_view.verticalScrollBar().setValue(
                in_view.verticalScrollBar().value()
                + in_view.cursorRect(cursor).y() - 20)

    def copy_to_clipboard(self):
        """Overloaded from OWWidget to copy the text from the file"""
        self.in_view.copy()


def main():  # pragma: no cover
    from AnyQt.QtWidgets import QApplication
    a = QApplication([])
    ow = OWGrep()
    ow.show()
    a.exec_()
    ow.saveSettings()


if __name__ == "__main__":  # pragma: no cover
    main()
