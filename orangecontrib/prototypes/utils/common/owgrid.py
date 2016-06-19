from itertools import zip_longest

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt


class GridItem(QtGui.QGraphicsWidget):
    def __init__(self, widget, parent=None):
        super().__init__(parent)
        # For some reason, the super constructor is not setting the parent
        self.setParent(parent)

        self.widget = widget
        if hasattr(self.widget, 'setParent'):
            self.widget.setParentItem(self)
            self.widget.setParent(self)

    def sizeHint(self, size_hint, size_constraint=None, *args, **kwargs):
        return self.widget.sizeHint(
            size_hint, size_constraint, *args, **kwargs)

    def boundingRect(self):
        top_left = self.mapFromItem(
            self.widget, self.widget.childrenBoundingRect().topLeft())
        return QtCore.QRectF(top_left, self.sizeHint(Qt.PreferredSize))


class SelectableGridItem(GridItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setFlags(QtGui.QGraphicsWidget.ItemIsSelectable)

    def paint(self, painter, options, widget=None):
        super().paint(painter, options, widget)
        if self.isSelected():
            rect = self.boundingRect()
            painter.save()
            pen = QtGui.QPen(QtGui.QColor(Qt.black))
            pen.setWidth(4)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            painter.drawRect(rect.adjusted(2, 2, -2, -2))
            painter.restore()


class PaddedGridItem(GridItem):
    def __init__(self, widget, parent=None, padding=20, *args, **kwargs):
        self._padding = padding
        super().__init__(widget, parent, *args, **kwargs)

    def boundingRect(self):
        return super().boundingRect().adjusted(
            -self._padding, -self._padding, self._padding, self._padding)

    def sizeHint(self, size_hint, size_constraint=None, *args, **kwargs):
        return super().sizeHint(size_hint, size_constraint, *args, **kwargs) \
            + QtCore.QSizeF(2 * self._padding, 2 * self._padding)
    

class ZoomableGridItem(GridItem):
    def __init__(self, widget, parent=None, max_size=100, *args, **kwargs):
        self._max_size = QtCore.QSizeF(max_size, max_size)

        super().__init__(widget, parent, *args, **kwargs)

        self._resize_widget()

    def set_max_size(self, max_size):
        self.widget.resetTransform()
        self._max_size = QtCore.QSizeF(max_size, max_size)
        self._resize_widget()

    def _resize_widget(self):
        w = self.widget
        sh = self.sizeHint(Qt.PreferredSize)
        scale_w = sh.width() / w.sizeHint(Qt.PreferredSize).width()
        scale_h = sh.height() / w.sizeHint(Qt.PreferredSize).height()
        w.scale(scale_w, scale_h)

    def sizeHint(self, size_hint, size_constraint=None, *args, **kwargs):
        size = super().sizeHint(Qt.PreferredSize)
        size.scale(self._max_size, Qt.KeepAspectRatio)
        return size


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
        return [item.sizeHint(which) for item in self._items()]
