"""
Legend classes to use with `QGraphicsScene` objects.
"""
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt


class ColorIndicator(QtGui.QGraphicsWidget):
    pass


class LegendItemSquare(ColorIndicator):
    """Legend square item.

    The legend square item is a small colored square image that can be plugged
    into the legend in front of the text object.

    This should only really be used in conjunction with ˙LegendItem˙.

    Parameters
    ----------
    color : QtGui.QColor
        The color of the square.
    parent : QtGui.QGraphicsItem

    See Also
    --------
    LegendItemCircle

    """

    SIZE = QtCore.QSizeF(12, 12)

    def __init__(self, color, parent):
        super().__init__(parent)

        height, width = self.SIZE.height(), self.SIZE.width()
        self.__square = QtGui.QGraphicsRectItem(0, 0, height, width)
        self.__square.setBrush(QtGui.QBrush(color))
        self.__square.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 0)))
        self.__square.setParentItem(self)

    def sizeHint(self, size_hint, size_constraint=None, *args, **kwargs):
        return QtCore.QSizeF(self.__square.boundingRect().size())


class LegendItemCircle(ColorIndicator):
    """Legend circle item.

    The legend circle item is a small colored circle image that can be plugged
    into the legend in front of the text object.

    This should only really be used in conjunction with ˙LegendItem˙.

    Parameters
    ----------
    color : QtGui.QColor
        The color of the square.
    parent : QtGui.QGraphicsItem

    See Also
    --------
    LegendItemSquare

    """

    SIZE = QtCore.QSizeF(12, 12)

    def __init__(self, color, parent):
        super().__init__(parent)

        height, width = self.SIZE.height(), self.SIZE.width()
        self.__circle = QtGui.QGraphicsEllipseItem(0, 0, height, width)
        self.__circle.setBrush(QtGui.QBrush(color))
        self.__circle.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 0)))
        self.__circle.setParentItem(self)

    def sizeHint(self, size_hint, size_constraint=None, *args, **kwargs):
        return QtCore.QSizeF(self.__circle.boundingRect().size())


class LegendItemTitle(QtGui.QGraphicsWidget):
    """Legend item title - the text displayed in the legend.

    This should only really be used in conjunction with ˙LegendItem˙.

    Parameters
    ----------
    text : str
    parent : QtGui.QGraphicsItem
    font : QtGui.QFont
        This

    """

    def __init__(self, text, parent, font):
        super().__init__(parent)

        self.__text = QtGui.QGraphicsTextItem(text)
        self.__text.setParentItem(self)
        self.__text.setFont(font)

    def sizeHint(self, size_hint, size_constraint=None, *args, **kwargs):
        return QtCore.QSizeF(self.__text.boundingRect().size())


class LegendItem(QtGui.QGraphicsLinearLayout):
    """Legend item - one entry in the legend.

    This represents one entry in the legend i.e. a color indicator and the text
    beside it.

    Parameters
    ----------
    color : QtGui.QColor
        The color that the entry will represent.
    title : str
        The text that will be displayed for the color.
    parent : QtGui.QGraphicsItem
    color_indicator_cls : ColorIndicator
        The type of `ColorIndicator` that will be used for the color.
    font : QtGui.QFont, optional

    """

    def __init__(self, color, title, parent, color_indicator_cls, font=None):
        super().__init__()

        self.__parent = parent
        self.__color_indicator = color_indicator_cls(color, parent)
        self.__title_label = LegendItemTitle(title, parent, font=font)

        self.addItem(self.__color_indicator)
        self.addItem(self.__title_label)

        # Make sure items are aligned properly, since the color box and text
        # won't be the same height.
        self.setAlignment(self.__color_indicator, Qt.AlignCenter)
        self.setAlignment(self.__title_label, Qt.AlignCenter)
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(5)


class LegendBuilder:
    """Legend builder.

    In most cases, the Orange.data.domain.Domain object will be available to
    you when displaying a dataset with a legend. This class makes the legend
    building process simpler, by parsing the domain and returing the
    appropriate legend type.

    Attributes
    ----------
    want_binned : bool, optional
        This only applies for continuous classes. Tell the builder if you want
        a binned (ranges of values e.g. 1-2, 2-3, 3-4) legend or if you want a
        smooth gradient legend. The default is False.

    Examples
    --------
    A basic example
    >>> legend = LegendBuilder()(domain)

    An example using the `want_binned` method
    >>> legend = LegendBuilder(want_binned=True)(domain)

    An example passing a kwarg to the `Legend` constructor.
    >>> legend = LegendBuilder()(domain, color_indicator_cls=LegendItemCircle)

    Notes
    -----
    .. note:: Note that even if some arguments are passed to the constructor,
        they may be ignored, based on the type of class variable the domain
        contains e.g. it would make no sense for the `want_binned` parameter to
        have any effect on discrete target classes.

    """

    def __init__(self, want_binned=False):
        self.want_binned = want_binned

    def __call__(self, domain, *args, **kwargs):
        """Build the appropriate legend instance.

        Parameters
        ----------
        domain : Orange.data.domain.Domain
        args
            Any excess arguments will be passed down to the `Legend`
            constructor. See the `Legend` as well as it subclass constructors
            for a list of valid arguments.
        kwargs
            Any key word arguments will be passed down to the `Legend`
            constructor. See the `Legend` as well as it subclass constructors
            for a list of valid arguments.

        Returns
        -------
        Legend

        See Also
        --------
        Legend
        OWDiscreteLegend
        OWContinuousLegend
        OWBinnedContinuousLegend

        """
        if domain.class_var.is_discrete:
            return OWDiscreteLegend(*args, domain=domain, **kwargs)
        else:
            if self.want_binned:
                return OWBinnedContinuousLegend(*args, domain=domain, **kwargs)
            else:
                return OWContinuousLegend(*args, domain=domain, **kwargs)


class Legend(QtGui.QGraphicsWidget):
    """Base legend class.

    This class provides common attributes for any legend derivates:
      - Behaviour on `QGraphicsScene`
      - Appearance of legend

    If you have access to the `domain` property, the `LegendBuilder` class
    can be used to automatically build a legend for you.

    Parameters
    ----------
    parent : QtGui.QGraphicsItem, optional
    orientation : Qt.Orientation, optional
        The default orientation is vertical
    domain : Orange.data.domain.Domain, optional
        This field is left optional as in some cases, we may want to simply
        pass in a list that represents the legend.
    items : Iterable[QtGui.QColor, str]
    bg_color : QtGui.QColor, optional
    font : QtGui.QFont, optional
    color_indicator_cls : ColorIndicator
        The color indicator class that will be used to render the indicators.

    See Also
    --------
    OWDiscreteLegend
    OWContinuousLegend
    OWContinuousLegend

    Notes
    -----
    .. Warning:: If the domain parameter is supplied, the items parameter will
        be ignored.

    """

    def __init__(self, parent=None, orientation=Qt.Vertical, domain=None,
                 items=None, bg_color=QtGui.QColor(232, 232, 232, 196),
                 font=None, color_indicator_cls=LegendItemSquare):
        super().__init__(parent)

        self.orientation = orientation
        self.bg_color = QtGui.QBrush(bg_color)
        self.color_indicator_cls = color_indicator_cls

        # Set default font if none is given
        if font is None:
            self.font = QtGui.QFont()
            self.font.setPointSize(10)
        else:
            self.font = font

        self.setFlags(QtGui.QGraphicsWidget.ItemIsMovable |
                      QtGui.QGraphicsItem.ItemIgnoresTransformations)

        self._layout = QtGui.QGraphicsLinearLayout(self.orientation)
        self._layout.setContentsMargins(10, 5, 10, 5)
        # If horizontal, there needs to be horizontal space between the items
        if self.orientation == Qt.Horizontal:
            self._layout.setSpacing(10)
        # If vertical spacing, vertical space is provided by child layouts
        else:
            self._layout.setSpacing(0)
        self.setLayout(self._layout)

        if domain is not None:
            self.set_domain(domain)
        elif items is not None:
            self.set_items(items)

    def set_domain(self, domain):
        """Handle receiving the domain object.

        Parameters
        ----------
        domain : Orange.data.domain.Domain

        Returns
        -------

        Raises
        ------
        AttributeError
            If the domain does not contain the correct type of class variable.

        """
        raise NotImplemented()

    def set_items(self, values):
        """Handle receiving an array of items.

        Parameters
        ----------
        values : iterable[QtGui.QColor, str]

        Returns
        -------

        """
        raise NotImplemented()

    def paint(self, painter, options, widget=None):
        painter.save()
        pen = QtGui.QPen(QtGui.QColor(196, 197, 193, 200), 1)
        brush = QtGui.QBrush(QtGui.QColor(self.bg_color))

        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawRoundedRect(self.contentsRect(), 2, 2)
        painter.restore()


class OWDiscreteLegend(Legend):
    """Discrete legend.

    Legend for discrete class variables.

    Notes
    -----
    .. Warning:: Ignores `want_binned` parameter - it would make no sense on a
        discrete variable.

    See Also
    --------
    Legend

    """

    def set_domain(self, domain):
        class_var = domain.class_var

        if not class_var.is_discrete:
            raise AttributeError('[OWDiscreteLegend] The class var provided '
                                 'was not discrete.')

        self.set_items(zip(class_var.values, class_var.colors.tolist()))

    def set_items(self, values):
        for class_name, color in values:
            legend_item = LegendItem(
                QtGui.QColor(*color),
                class_name,
                self,
                self.color_indicator_cls,
                font=self.font
            )
            self._layout.addItem(legend_item)


class OWContinuousLegend(Legend):
    def set_domain(self, domain):
        pass

    def set_items(self, values):
        pass


class OWBinnedContinuousLegend(Legend):
    def set_domain(self, domain):
        pass

    def set_items(self, values):
        pass
