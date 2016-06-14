"""
Legend classes to use with `QGraphicsScene` objects.
"""
import numpy as np
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

        self.__text = QtGui.QGraphicsTextItem(text.title())
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


class LegendGradient(QtGui.QGraphicsWidget):
    """Gradient widget.

    A gradient square bar that can be used to display continuous values.

    Parameters
    ----------
    palette : iterable[QtGui.QColor]
    parent : QtGui.QGraphicsWidget
    orientation : Qt.Orientation

    Notes
    -----
    .. Note:: While the gradient does support any number of colors, any more
        than 3 is not very readable. This should not be a problem, since Orange
        only implements 2 or 3 colors.

    """

    # Default sizes (assume gradient is vertical by default)
    GRADIENT_WIDTH = 20
    GRADIENT_HEIGHT = 150

    def __init__(self, palette, parent, orientation):
        super().__init__(parent)

        self.__gradient = QtGui.QLinearGradient()
        num_colors = len(palette)
        for idx, stop in enumerate(palette):
            self.__gradient.setColorAt(idx * (1. / (num_colors - 1)), stop)

        # We need to tell the gradient where it's start and stop points are
        self.__gradient.setStart(QtCore.QPointF(0, 0))
        if orientation == Qt.Vertical:
            final_stop = QtCore.QPointF(0, self.GRADIENT_HEIGHT)
        else:
            final_stop = QtCore.QPointF(self.GRADIENT_HEIGHT, 0)
        self.__gradient.setFinalStop(final_stop)

        # Get the appropriate rectangle dimensions based on orientation
        if orientation == Qt.Vertical:
            w, h = self.GRADIENT_WIDTH, self.GRADIENT_HEIGHT
        elif orientation == Qt.Horizontal:
            w, h = self.GRADIENT_HEIGHT, self.GRADIENT_WIDTH

        self.__rect_item = QtGui.QGraphicsRectItem(0, 0, w, h, self)
        self.__rect_item.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 0)))
        self.__rect_item.setBrush(QtGui.QBrush(self.__gradient))

    def sizeHint(self, size_hint, size_constraint=None, *args, **kwargs):
        return QtCore.QSizeF(self.__rect_item.boundingRect().size())


class ContinuousLegendItem(QtGui.QGraphicsLinearLayout):
    """Continuous legend item.

    Contains a gradient bar with the color ranges, as well as two labels - one
    on each side of the gradient bar.

    Parameters
    ----------
    palette : iterable[QtGui.QColor]
    values : iterable[float...]
        The number of values must match the number of colors in passed in the
        color palette.
    parent : QtGui.QGraphicsWidget
    font : QtGui.QFont
    orientation : Qt.Orientation

    """

    def __init__(self, palette, values, parent, font=None,
                 orientation=Qt.Vertical):
        if orientation == Qt.Vertical:
            super().__init__(Qt.Horizontal)
        else:
            super().__init__(Qt.Vertical)

        self.__parent = parent
        self.__palette = palette
        self.__values = values

        self.__gradient = LegendGradient(palette, parent, orientation)
        self.__labels_layout = QtGui.QGraphicsLinearLayout(orientation)

        str_vals = self._format_values(values)

        self.__start_label = LegendItemTitle(str_vals[0], parent, font=font)
        self.__end_label = LegendItemTitle(str_vals[1], parent, font=font)
        self.__labels_layout.addItem(self.__start_label)
        self.__labels_layout.addStretch(1)
        self.__labels_layout.addItem(self.__end_label)

        # Gradient should be to the left, then labels on the right if vertical
        if orientation == Qt.Vertical:
            self.addItem(self.__gradient)
            self.addItem(self.__labels_layout)
        # Gradient should be on the bottom, labels on top if horizontal
        elif orientation == Qt.Horizontal:
            self.addItem(self.__labels_layout)
            self.addItem(self.__gradient)

    @staticmethod
    def _format_values(values):
        """Get the formatted values to output."""
        return ['{:.3f}'.format(v) for v in values]


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
    >>> legend = LegendBuilder()(domain, dataset)

    An example using the `want_binned` method
    >>> legend = LegendBuilder(want_binned=True)(domain, dataset)

    An example passing a kwarg to the `Legend` constructor.
    >>> legend = LegendBuilder()(
    >>>     domain,
    >>>     dataset,
    >>>     color_indicator_cls=LegendItemCircle)

    Notes
    -----
    .. note:: Note that even if some arguments are passed to the constructor,
        they may be ignored, based on the type of class variable the domain
        contains e.g. it would make no sense for the `want_binned` parameter to
        have any effect on discrete target classes.

    """

    def __init__(self, want_binned=False):
        self.want_binned = want_binned

    def __call__(self, domain, dataset, *args, **kwargs):
        """Build the appropriate legend instance.

        Parameters
        ----------
        domain : Orange.data.domain.Domain
        dataset : Orange.data.table.Table
            Note that no reference to the dataset is kept, it is only required
            to calculate some values that are then passed on to the Legend
            objects.
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
                value_range = [np.min(dataset.Y), np.max(dataset.Y)]
                return OWContinuousLegend(
                    *args, domain=domain, range=value_range, **kwargs)


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
                 font=None, color_indicator_cls=LegendItemSquare, *_, **__):
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

        if domain is not None:
            self.set_domain(domain)
        elif items is not None:
            self.set_items(items)

    def _clear_layout(self):
        self._layout = None
        for child in self.children():
            child.setParent(None)

    def _setup_layout(self):
        self._clear_layout()

        self._layout = QtGui.QGraphicsLinearLayout(self.orientation)
        self._layout.setContentsMargins(10, 5, 10, 5)
        # If horizontal, there needs to be horizontal space between the items
        if self.orientation == Qt.Horizontal:
            self._layout.setSpacing(10)
        # If vertical spacing, vertical space is provided by child layouts
        else:
            self._layout.setSpacing(0)
        self.setLayout(self._layout)

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
        values : iterable[object, QtGui.QColor]

        Returns
        -------

        """
        raise NotImplemented()

    @staticmethod
    def _convert_to_color(obj):
        if isinstance(obj, QtGui.QColor):
            return obj
        elif isinstance(obj, tuple) or isinstance(obj, list):
            assert len(obj) in (3, 4)
            return QtGui.QColor(*obj)
        else:
            return QtGui.QColor(obj)

    def paint(self, painter, options, widget=None):
        painter.save()
        pen = QtGui.QPen(QtGui.QColor(196, 197, 193, 200), 1)
        brush = QtGui.QBrush(QtGui.QColor(self.bg_color))

        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawRect(self.contentsRect())
        painter.restore()


class OWDiscreteLegend(Legend):
    """Discrete legend.

    See Also
    --------
    Legend
    OWContinuousLegend

    """

    def set_domain(self, domain):
        class_var = domain.class_var

        if not class_var.is_discrete:
            raise AttributeError('[OWDiscreteLegend] The class var provided '
                                 'was not discrete.')

        self.set_items(zip(class_var.values, class_var.colors.tolist()))

    def set_items(self, values):
        self._setup_layout()
        for class_name, color in values:
            legend_item = LegendItem(
                color=self._convert_to_color(color),
                title=class_name,
                parent=self,
                color_indicator_cls=self.color_indicator_cls,
                font=self.font
            )
            self._layout.addItem(legend_item)


class OWContinuousLegend(Legend):
    """Continuous legend.

    See Also
    --------
    Legend
    OWDiscreteLegend

    """
    
    def __init__(self, *args, **kwargs):
        # Variables used in the `set_` methods must be set before calling super
        self.__range = kwargs.get('range', ())

        super().__init__(*args, **kwargs)

        self._layout.setContentsMargins(10, 10, 10, 10)

    def set_domain(self, domain):
        class_var = domain.class_var

        if not class_var.is_continuous:
            raise AttributeError('[OWContinuousLegend] The class var provided '
                                 'was not continuous.')

        # The first and last values must represent the range, the rest should
        # be dummy variables, as they are not shown anywhere
        values = self.__range

        start, end, pass_through_black = class_var.colors
        # If pass through black, push black in between and add index to vals
        if pass_through_black:
            colors = [self._convert_to_color(c) for c
                      in [start, '#000000', end]]
            values.insert(1, -1)
        else:
            colors = [self._convert_to_color(c) for c in [start, end]]

        self.set_items(list(zip(values, colors)))

    def set_items(self, values):
        vals, colors = list(zip(*values))

        # If the orientation is vertical, it makes more sense for the smaller
        # value to be shown on the bottom
        if self.orientation == Qt.Vertical and vals[0] < vals[len(vals) - 1]:
            colors, vals = list(reversed(colors)), list(reversed(vals))

        self._setup_layout()
        self._layout.addItem(ContinuousLegendItem(
            palette=colors,
            values=vals,
            parent=self,
            font=self.font,
            orientation=self.orientation
        ))


class OWBinnedContinuousLegend(Legend):
    def set_domain(self, domain):
        pass

    def set_items(self, values):
        pass
