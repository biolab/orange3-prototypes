from itertools import zip_longest

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt


class GridItem(QtGui.QGraphicsWidget):
    def __init__(self, widget, parent=None):
        super().__init__(parent)

        # if hasattr(widget, 'setParentItem'):
        #     widget.setParentItem(self)
        #
        # self.__widget = widget

        self.__rect = QtGui.QGraphicsRectItem(0, 0, 100, 100, self)
        self.__rect.setBrush(QtGui.QBrush(QtGui.QColor(Qt.green)))


class OWGrid(QtGui.QGraphicsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setSizePolicy(QtGui.QSizePolicy.Maximum,
                           QtGui.QSizePolicy.Maximum)
        self.setContentsMargins(10, 10, 10, 10)

        self.__layout = QtGui.QGraphicsGridLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(10)
        self.setLayout(self.__layout)

    def set_items(self, items):
        for i, item in enumerate(items):
            if hasattr(item, 'setParentItem'):
                item.setParentItem(self)
            self.__layout.addItem(item, i, 0)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.reflow(self.size().width())

    def reflow(self, width):
        if not self.layout():
            return

        grid = self.__layout

        left, right, *_ = self.getContentsMargins()
        width -= left + right

        hints = self._hints(Qt.PreferredSize)
        widths = [max(32, h.width()) for h in hints]
        ncol = self._fit_n_cols(widths, grid.horizontalSpacing(), width)

        if ncol == grid.columnCount():
            return

        items = [grid.itemAt(i) for i in range(self.__layout.count())]

        # remove all items from the layout, then re-add them back in updated
        # positions
        for item in items:
            grid.removeItem(item)

        for i, item in enumerate(items):
            grid.addItem(item, i // ncol, i % ncol)

    @staticmethod
    def _fit_n_cols(widths, spacing, constraint):

        def sliced(seq, n_col):
            return [seq[i:i + n_col] for i in range(0, len(seq), n_col)]

        def flow_width(widths, spacing, ncol):
            W = sliced(widths, ncol)
            col_widths = map(max, zip_longest(*W, fillvalue=0))
            return sum(col_widths) + (ncol - 1) * spacing

        ncol_best = 1
        for ncol in range(2, len(widths) + 1):
            w = flow_width(widths, spacing, ncol)
            if w <= constraint:
                ncol_best = ncol
            else:
                break

        return ncol_best

    def _hints(self, which):
        return [item.effectiveSizeHint(which) for item in self.items()]

    def items(self):
        return (self.__layout.itemAt(i) for i in range(self.__layout.count()))
