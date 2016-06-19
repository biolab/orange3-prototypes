from itertools import zip_longest

from PyQt4 import QtGui
from PyQt4.QtCore import Qt


class GridItem(QtGui.QGraphicsWidget):
    def __init__(self, widget, parent=None):
        super().__init__(parent)

        self._widget = widget
        self._widget.setParent(self)


class SelectableGridItem(GridItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setFlags(QtGui.QGraphicsWidget.ItemIsSelectable)

    def paint(self, painter, options, widget=None):
        super().paint(painter, options, widget)


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
            # Place the items in some arbitrary order - they will be rearranged
            # before user sees this ordering
            self.__layout.addItem(item, i, 0)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.reflow(self.size().width())

    def reflow(self, width):
        # When setting the geometry when opened, the layout doesn't yet exist
        if self.layout() is None:
            return

        grid = self.__layout

        left, right, *_ = self.getContentsMargins()
        width -= left + right

        # Get size hints with 32 as the minimum size for each cell
        widths = [max(32, h.width()) for h in self._hints(Qt.PreferredSize)]
        ncol = self._fit_n_cols(widths, grid.horizontalSpacing(), width)

        # The number of columns is already optimal
        if ncol == grid.columnCount():
            return

        # remove all items from the layout, then re-add them back in updated
        # positions
        items = self._items()

        for item in items:
            grid.removeItem(item)

        for i, item in enumerate(items):
            grid.addItem(item, i // ncol, i % ncol)

    @staticmethod
    def _fit_n_cols(widths, spacing, constraint):

        def sliced(seq, n_col):
            """Slice the widths into n lists that contain their respective
            widths. E.g. [5, 5, 5], 2 => [[5, 5], [5]]"""
            return [seq[i:i + n_col] for i in range(0, len(seq), n_col)]

        def flow_width(widths, spacing, ncol):
            w = sliced(widths, ncol)
            col_widths = map(max, zip_longest(*w, fillvalue=0))
            return sum(col_widths) + (ncol - 1) * spacing

        ncol_best = 1
        for ncol in range(2, len(widths) + 1):
            width = flow_width(widths, spacing, ncol)
            if width <= constraint:
                ncol_best = ncol
            else:
                break

        return ncol_best

    def _items(self):
        if not self.__layout:
            return []
        return [self.__layout.itemAt(i) for i in range(self.__layout.count())]

    def _hints(self, which):
        return [item.effectiveSizeHint(which) for item in self._items()]
