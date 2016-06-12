from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt


class OWLegend(QtGui.QGraphicsWidget):
    def __init__(self, parent=None, orientation=Qt.Vertical, domain=None,
                 items=None, bg_color=QtGui.QColor('#dddddd'), font=None):
        super().__init__(parent)

        self.orientation = orientation
        self._domain = domain
        self._items = items
        self._bg_color = QtGui.QBrush(bg_color)

        # Set default font if none is given
        if font is None:
            self._font = QtGui.QFont()
            self._font.setPointSize(10)
        else:
            self._font = font

        self.setFlags(QtGui.QGraphicsWidget.ItemIsMovable |
                      QtGui.QGraphicsItem.ItemIgnoresTransformations)

        self._layout = None
        self._setup_layout()

        if self._has_domain():
            self.set_domain(self._domain)
        elif self._has_items():
            self.set_finite_items(self._items)

    def _setup_layout(self):
        self._layout = QtGui.QGraphicsLinearLayout(self.orientation)
        self.setLayout(self._layout)

    def set_domain(self, domain):
        class_var = domain.class_var
        if class_var.is_discrete:
            self.set_finite_items(zip(class_var.values, class_var.colors.tolist()))
        else:
            # TODO implement for continuous
            pass

    def set_finite_items(self, values):
        for class_name, color in values:
            legend_item = LegendItem(
                QtGui.QColor(*color), class_name, self, font=self._font)
            self._layout.addItem(legend_item)

    def _has_domain(self):
        return self._domain is not None

    def _has_items(self):
        return self._items is not None

    def paint(self, painter, options, widget=None):
        painter.save()
        pen = QtGui.QPen(QtGui.QColor(Qt.gray))
        pen.setWidth(0)
        pen.setJoinStyle(Qt.RoundJoin)
        brush = QtGui.QBrush(QtGui.QColor(self._bg_color))

        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawRect(self.boundingRect())
        painter.restore()


class LegendItem(QtGui.QGraphicsLinearLayout):
    def __init__(self, color, title, parent, font=None):
        super().__init__()

        self.__parent = parent
        self.__color_indicator = LegendItemSquare(color, parent)
        self.__title_label = LegendItemTitle(title, parent, font=font)

        self.addItem(self.__color_indicator)
        self.addItem(self.__title_label)

        # Make sure items are aligned properly, since the color box and text
        # won't be the same height.
        self.setAlignment(self.__color_indicator, Qt.AlignCenter)
        self.setAlignment(self.__title_label, Qt.AlignCenter)


class LegendItemSquare(QtGui.QGraphicsWidget):
    SIZE = QtCore.QSizeF(12, 12)

    def __init__(self, color, parent):
        super().__init__(parent)

        height, width = self.SIZE.height(), self.SIZE.width()
        self.__square = QtGui.QGraphicsRectItem(0, 0, height, width)
        self.__square.setBrush(QtGui.QBrush(color))
        self.__square.setParentItem(self)

    def sizeHint(self, size_hint, size_constraint=None, *args, **kwargs):
        return QtCore.QSizeF(self.__square.boundingRect().size())


class LegendItemTitle(QtGui.QGraphicsWidget):
    def __init__(self, text, parent, font):
        super().__init__(parent)

        self.__text = QtGui.QGraphicsTextItem(text)
        self.__text.setParentItem(self)
        self.__text.setFont(font)

    def sizeHint(self, size_hint, size_constraint=None, *args, **kwargs):
        return QtCore.QSizeF(self.__text.boundingRect().size())
