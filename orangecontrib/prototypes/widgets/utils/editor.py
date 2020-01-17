import math
import re
import sys
from functools import partial
from typing import cast, Optional, Tuple

from AnyQt.QtCore import Qt, QObject, QEvent, Signal, QRectF, QMimeData
from AnyQt.QtGui import (
    QTextCursor, QKeyEvent, QFontDatabase, QFont, QKeySequence,
    QPaintEvent, QPainter, QFontMetrics
)
from AnyQt.QtWidgets import QPlainTextEdit, QApplication, QSizePolicy

from orangewidget.utils.overlay import OverlayWidget

from qtconsole.pygments_highlighter import PygmentsHighlighter


def qshortcut(string: str, *, macos_command_map=True) -> QKeySequence:
    """
    Create a QShortcut from a portable string (`QKeySequence.PortableText`).

    If `macos_command_map` is True then Ctrl and Command are swapped as per
    usual with Qt on macOS (depending on `Qt.AA_MacDontSwapCtrlAndMeta`).
    If True then the modifiers are not swapped, i.e. 'ctrl+c' remains 'ctrl+c'
    (^c) regardless.
    """
    sh = QKeySequence(string, QKeySequence.PortableText)
    assert sh.count() == 1
    key = sh[0]
    if sys.platform == "darwin":
        swaped = not QApplication.testAttribute(Qt.AA_MacDontSwapCtrlAndMeta)
    else:
        swaped = False

    if not macos_command_map and swaped:
        key_ = key
        if key_ & Qt.ControlModifier:
            key = key & (~Qt.ControlModifier) | Qt.MetaModifier
        if key_ & Qt.MetaModifier:
            key = key & (~Qt.MetaModifier) | Qt.ControlModifier
    return QKeySequence(key)


class EditHelper(QObject):
    """Source code edit helper"""
    def __init__(self, editor: QPlainTextEdit, **kwargs):
        super().__init__(editor, **kwargs)
        self.__editor = editor
        self.__enabled = True
        editor.installEventFilter(self)

    def setEnabled(self, enable):
        if self.__enabled != enable:
            self.__enabled = enable
            if enable:
                self.__editor.installEventFilter(self)
            else:
                self.__editor.removeEventFilter(self)

    def enabled(self):
        return self.__enabled

    def document(self):
        return self.__editor.document()

    def editor(self):
        return self.__editor

    def keyPressEvent(self, event: QKeyEvent) -> bool:
        return False

    def keyReleaseEvent(self, event: QKeyEvent) -> bool:
        return False

    def eventFilter(self, obj, event):
        etype = event.type()
        if etype == QKeyEvent.KeyPress:
            return self.keyPressEvent(event)
        elif etype == QKeyEvent.KeyRelease:
            return self.keyReleaseEvent(event)
        else:
            return super().eventFilter(obj, event)


def iswspace(string: str) -> bool:
    return re.match(r'^\s*$', string) is not None


class PythonSourceEditHelper(EditHelper):
    """Helper for python source code editing."""
    #: The indent spaces
    INDENT = 4

    def keyPressEvent(self, event: QKeyEvent) -> bool:
        if event.modifiers() & (Qt.ControlModifier | Qt.AltModifier |
                                Qt.MetaModifier):
            return False
        editor = self.editor()
        cursor = editor.textCursor()
        key, modifiers = event.key(), event.modifiers()
        if key == Qt.Key_Return:
            text, pos = self.editLineContent()
            indent = len(text) - len(text.lstrip())
            if text.strip() == "pass" or text.strip().startswith("return "):
                indent = max(0, indent - self.INDENT)
            elif text.strip().endswith(":"):
                indent += self.INDENT
            editor.insertPlainText("\n" + (" " * indent))
            return True
        elif key == Qt.Key_Tab:
            cursor = editor.textCursor()
            if cursor.hasSelection():
                self.indentSelection(
                    cursor, -1 if modifiers & Qt.ShiftModifier else 1
                )
            else:
                editor.insertPlainText(" " * self.INDENT)
            return True
        elif key == Qt.Key_Backtab:
            self.indentSelection(cursor, -1)
            return True
        elif key == Qt.Key_Backspace and modifiers == Qt.NoModifier \
                and not cursor.hasSelection():
            text, pos = self.editLineContent()
            head, trailing = text[:pos], text[pos:]
            if trailing and not iswspace(trailing):
                # have trailing non-whitespace
                return False
            m = re.match(f'(^\\s{{1,{self.INDENT}}})', head[::-1])
            if m is None:
                return False
            count = len(m.group())
            cursor = editor.textCursor()
            cursor.clearSelection()
            cursor.movePosition(QTextCursor.PreviousCharacter, QTextCursor.KeepAnchor, count)
            cursor.removeSelectedText()
            return True
        else:
            return super().keyPressEvent(event)

    def editLineContent(self) -> Tuple[str, int]:
        """Return the content of the current line"""
        editor = self.editor()
        cursor = editor.textCursor()
        pos = cursor.positionInBlock()
        cursor.movePosition(QTextCursor.EndOfLine)
        cursor.movePosition(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
        return cursor.selectedText(), pos

    def indentSelection(self, cursor: QTextCursor, direction=1):
        cursor = QTextCursor(cursor)
        if cursor.hasSelection():
            start = cursor.selectionStart()
            end = cursor.selectionEnd()
            cursor.setPosition(end)
            lastblock = cursor.blockNumber()
            cursor.setPosition(start)
            while cursor.blockNumber() <= lastblock:
                self.indentBlock(cursor, direction=direction)
                if not cursor.movePosition(QTextCursor.NextBlock):
                    break

    def indentBlock(self, cursor, direction=1):
        cursor = QTextCursor(cursor)
        cursor.movePosition(QTextCursor.StartOfLine)
        if direction == 1:
            cursor.insertText(' ' * self.INDENT)
        elif direction == -1:
            if cursor.movePosition(QTextCursor.StartOfWord, QTextCursor.KeepAnchor):
                # no whitespace at start of block
                return
            cursor.movePosition(QTextCursor.NextWord, QTextCursor.KeepAnchor)
            if cursor.selectionStart() + self.INDENT < cursor.selectionEnd():
                cursor.clearSelection()
                cursor = QTextCursor(cursor)
                cursor.movePosition(QTextCursor.StartOfLine)
                cursor.movePosition(QTextCursor.NextCharacter,
                                    QTextCursor.KeepAnchor, self.INDENT)
            cursor.removeSelectedText()
        else:
            raise ValueError


def qkeyevent_matches(event: QKeyEvent, shortcut: QKeySequence) -> bool:
    """Does key event match the shortcut sequence."""
    key = (int(event.key()) | int(event.modifiers()))
    key = key & ~int(Qt.KeypadModifier | Qt.GroupSwitchModifier)  # reasons
    return shortcut == QKeySequence(key)


class TextEditShortcutFilter(QObject):
    """
    A shortcut filter for Qt text entry widgets.
    """
    #: The shortcut was activated on widget.
    activated = Signal('QWidget')

    def __init__(self, shortcut: QKeySequence, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.shortcut = shortcut

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.KeyPress:
            event = cast(QKeyEvent, event)
            if qkeyevent_matches(event, self.shortcut):
                self.activated.emit(obj)
                event.accept()
                return True
        return super().eventFilter(obj, event)


class CodeEditor(QPlainTextEdit):
    def __init__(self, parent=None, **kwargs):
        font: Optional[QFont] = kwargs.pop("font", None)
        super().__init__(parent, **kwargs)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        margins = self.viewportMargins()
        margins.setLeft(20)
        self.setViewportMargins(margins)
        margin = self.document().documentMargin()

        self.__lines = _LineNumbersHeader(self, alignment=Qt.AlignLeft)
        self.__lines.setContentsMargins(3, margin, 3, margin)
        self.__lines.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.__lines.setFixedWidth(20)
        self.__lines.setRange(0, self.blockCount())
        self.__lines.setWidget(self)
        self.blockCountChanged.connect(self.__updateForFont)
        self.updateRequest.connect(self.__updateLines)

        if font is None:
            ffont = QFontDatabase.systemFont(QFontDatabase.FixedFont)
            font = QFont()
            font.setFamily(ffont.family())
            font.setStyle(ffont.style())
            self.setFont(font)
            self.setAttribute(Qt.WA_SetFont, False)
        else:
            self.setFont(font)

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.FontChange:
            self.__updateForFont()
        super().changeEvent(event)

    def __updateForFont(self):
        fm = self.fontMetrics()
        blocks = self.blockCount()
        places = int(math.ceil(math.log10(max(blocks, 1)))) + 1
        w = fm.width("0" * max(places, 4))
        margins = self.viewportMargins()
        margins.setLeft(w)
        self.setViewportMargins(margins)
        self.__lines.setFixedWidth(w)

    def insertFromMimeData(self, source: QMimeData) -> None:
        if source.hasText():
            self.insertPlainText(source.text())
        elif source.hasUrls():
            url = source.urls()[0]
            if url.isLocalFile():
                try:
                    contents = open(url.toLocalFile(), "rt", "utf-8")
                except (OSError, UnicodeDecodeError):
                    return
                cursor = self.textCursor()
                cursor.select(QTextCursor.Document)
                cursor.insertText(contents)

    def __updateLines(self):
        self.__lines.setRange(0, self.blockCount())
        self.__lines.setOffset(self.verticalScrollBar().value())


class _LineNumbersHeader(OverlayWidget):
    """
    Display line numbers left of a QPlainTextEditor viewport.
    """
    __offset = 0
    __range = (0, 0)

    def setOffset(self, offset: int):
        if self.__offset != offset:
            self.__offset = offset
            self.update()

    def offset(self):
        return self.__offset

    def setRange(self, min, max):
        if self.__range != (min, max):
            self.__range = (min, max)
            self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        rect = self.contentsRect()
        fm = self.fontMetrics()
        i = self.__offset
        start, end = self.__range
        bottom = rect.y() + rect.height()
        linerect = QRectF(
            rect.left(), rect.top(), rect.width(), fm.lineSpacing())
        vert_advance = fm.lineSpacing() + 1
        painter = QPainter(self)
        while i < end and linerect.top() < bottom:
            painter.drawText(linerect, str(i + 1))
            linerect.translate(0, vert_advance)
            i += 1
        painter.end()


class PythonCodeEditor(CodeEditor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        doc = self.document()
        PygmentsHighlighter(doc,)
        helper = PythonSourceEditHelper(self)
        helper.installEventFilter(self)


def main():
    import sys
    app = QApplication(sys.argv)
    argv = app.arguments()
    if len(argv) > 1:
        with open(argv[1], 'rt', encoding="utf-8") as f:
            source = f.read()
    else:
        source = 'print("Hello World")'
    w = PythonCodeEditor()
    w.setPlainText(source)
    w.show()
    return app.exec()


if __name__ == "__main__":
    main()