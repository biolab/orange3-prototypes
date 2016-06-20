from itertools import zip_longest

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt


class GridItem(QtGui.QGraphicsWidget):
    def __init__(self, widget, parent=None, **_):
        super().__init__(parent)
        # For some reason, the super constructor is not setting the parent
        self.setParent(parent)

        self.widget = widget
        if hasattr(self.widget, 'setParent'):
            self.widget.setParentItem(self)
            self.widget.setParent(self)

        # Move the child widget to (0, 0) so that bounding rects match up
        # This is needed because the bounding rect is caluclated with the size
        # hint from (0, 0), regardless of any method override
        rect = self.widget.childrenBoundingRect()
        self.widget.moveBy(-rect.topLeft().x(), -rect.topLeft().y())

        # TODO Remove this
        QtGui.QGraphicsRectItem(self.boundingRect(), self)

    def sizeHint(self, size_hint, size_constraint=None, **kwargs):
        return self.widget.sizeHint(
            size_hint, size_constraint, **kwargs)

    def boundingRect(self):
        return QtCore.QRectF(
            QtCore.QPointF(0, 0), self.widget.childrenBoundingRect().size())


class SelectableGridItem(GridItem):
    def __init__(self, widget, parent=None, **kwargs):
        super().__init__(widget, parent, **kwargs)

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
    def __init__(self, widget, parent=None, padding=20, **kwargs):
        self._padding = padding

        super().__init__(widget, parent, **kwargs)

        # Moving the child widget is important in order to keep the bounding
        # boxes aligned
        self.widget.moveBy(self._padding, self._padding)

    def boundingRect(self):
        return QtCore.QRectF(
            QtCore.QPointF(0, 0),
            super().boundingRect().adjusted(
                -self._padding, -self._padding, self._padding, self._padding
            ).size())

    def sizeHint(self, size_hint, size_constraint=None, *args, **kwargs):
        return super().sizeHint(size_hint, size_constraint, *args, **kwargs) \
            + QtCore.QSizeF(2 * self._padding, 2 * self._padding)


class ZoomableGridItem(GridItem):
    def __init__(self, widget, parent=None, max_size=100, **kwargs):
        self._max_size = QtCore.QSizeF(max_size, max_size)

        super().__init__(widget, parent, **kwargs)

        self._resize_widget()

    def set_max_size(self, max_size):
        self.widget.resetTransform()
        self._max_size = QtCore.QSizeF(max_size, max_size)
        self._resize_widget()

    def _resize_widget(self):
        # First, resize the widget
        w = self.widget
        own_hint = self.sizeHint(Qt.PreferredSize)

        # The effective hint for the actual tree with no padding - if padding
        eff_hint = own_hint
        if hasattr(self, '_padding'):
            eff_hint -= QtCore.QSizeF(2 * self._padding, 2 * self._padding)
            # For proper positioning, move to actual top left corner
            w.moveBy(-self._padding, -self._padding)

        # scale_w = own_hint.width() / full_rect.width()
        scale_w = eff_hint.width() / w.boundingRect().width()
        scale_h = eff_hint.height() / w.boundingRect().height()
        scale = scale_w if scale_w > scale_h else scale_h

        # Move the tranform origin to top left, so it stays in place when
        # scaling
        w.setTransformOriginPoint(w.childrenBoundingRect().topLeft())
        w.setScale(scale)
        # Then, move the scaled widget to the center of the bounding box
        own_rect = self.boundingRect()
        self.widget.moveBy(
            (own_rect.width() - w.boundingRect().width() * scale_w) / 2,
            (own_rect.height() - w.boundingRect().height() * scale_h) / 2)

    def boundingRect(self):
        return QtCore.QRectF(QtCore.QPointF(0, 0),
                             self.sizeHint(Qt.PreferredSize))

    def sizeHint(self, size_hint, size_constraint=None, *args, **kwargs):
        size = super().sizeHint(Qt.PreferredSize)
        size.scale(self._max_size, Qt.KeepAspectRatio)
        return size


class OWGrid(QtGui.QGraphicsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setSizePolicy(QtGui.QSizePolicy.Maximum,
                           QtGui.QSizePolicy.Maximum)
        self.setContentsMargins(0, 0, 0, 0)

        self.__layout = QtGui.QGraphicsGridLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(0)
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
