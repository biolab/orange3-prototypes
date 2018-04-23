import sys

from types import SimpleNamespace as namespace
from collections import namedtuple

import numpy as np

from AnyQt.QtCore import Qt, QSize, QRectF
from AnyQt.QtGui import QPainter, QPen, QColor
from AnyQt.QtWidgets import QListWidget, QApplication

import pyqtgraph as pg
from pyqtgraph.graphicsItems.ViewBox import ViewBox
from pyqtgraph.Point import Point

from Orange.data import Table
from Orange.widgets import gui, settings
from Orange.widgets.utils import colorpalette
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.utils.plot import OWPlotGUI, SELECT, PANNING, ZOOMING
from Orange.widgets.widget import OWWidget, Input, Output, Msg


def ccw(a, b, c):
    """
    Checks whether three points are listed in a counterclockwise order.
    """
    return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)


def intersects(a, b, c, d):
    """
    Checks whether line segment a (given points a and b) intersects with line
    segment b (given points c and d).
    """
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def in_rect(a, b, c, x, y=None):
    """
    Checks whether line segment (given points x and y) is contained within a
    rectangle (given points a, b, c, d).

    If y is None, only point x is considered.
    """
    x_in = a.x < x.x < b.x and a.y < x.y < c.y
    return x_in if y is None else x_in and a.x < y.x < b.x and a.y < y.y < c.y


def line_segment_rect_intersect(rect, item):
    """
    Checks if line segment (item.data) intersects with
    rectangle (rect) or if both its endpoints are placed inside it.
    """
    point = namedtuple("point", ["x", "y"])
    rect_a = point(rect.x(), rect.y())
    rect_b = point(rect.x() + rect.width(), rect.y())
    rect_c = point(rect.x() + rect.width(), rect.y() + rect.height())
    rect_d = point(rect.x(), rect.y() + rect.height())
    b_data = None
    for i in range(len(item.xData) - 1):
        a_data = point(item.xData[i], item.yData[i])
        b_data = point(item.xData[i + 1], item.yData[i + 1])
        if intersects(rect_a, rect_b, a_data, b_data) or \
                intersects(rect_b, rect_c, a_data, b_data) or \
                intersects(rect_c, rect_d, a_data, b_data) or \
                intersects(rect_a, rect_d, a_data, b_data):
            return True
    a_data = point(item.xData[0], item.yData[0])
    return in_rect(rect_a, rect_b, rect_c, a_data, b_data)


class LinePlotItem(pg.PlotDataItem):
    def __init__(self, index, x, y, pen, color):
        super().__init__(x=x, y=y, pen=pen, pxMode=True, symbol="o",
                         symbolSize=pen.width(), symbolBrush=color,
                         symbolPen=pen, antialias=True)
        self._pen = pen
        self._color = color
        self.index = index
        self.curve.setClickable(True, width=10)

    def select(self):
        color = QColor(self._color)
        color.setAlpha(255)
        pen = QPen(self._pen)
        pen.setWidth(2)
        pen.setColor(color)
        self.__change_pen(pen)

    def deselect(self):
        self.__change_pen(self._pen)

    def __change_pen(self, pen):
        self.setPen(pen)
        self.setSymbolPen(pen)


class LinePlotViewBox(ViewBox):
    def __init__(self, graph, enable_menu=False):
        ViewBox.__init__(self, enableMenu=enable_menu)
        self._hovered_item = None
        self.graph = graph
        self.setMouseMode(self.PanMode)

    def _update_scale_box(self, button_down_pos, current_pos):
        x, y = current_pos
        if button_down_pos[0] == x:
            x += 1
        if button_down_pos[1] == y:
            y += 1
        self.updateScaleBox(button_down_pos, Point(x, y))

    def mouseDragEvent(self, event, axis=None):
        if self.graph.state == SELECT and axis is None:
            event.accept()
            pos = event.pos()
            if event.button() == Qt.LeftButton:
                self._update_scale_box(event.buttonDownPos(), event.pos())
                if event.isFinish():
                    self.rbScaleBox.hide()
                    pix_rect = QRectF(event.buttonDownPos(event.button()), pos)
                    val_rect = self.childGroup.mapRectFromParent(pix_rect)
                    self.graph.select_by_rectangle(val_rect)
                else:
                    self._update_scale_box(event.buttonDownPos(), event.pos())
        elif self.graph.state == ZOOMING or self.graph.state == PANNING:
            event.ignore()
            super().mouseDragEvent(event, axis=axis)
        else:
            event.ignore()

    def mouseClickEvent(self, event):
        if event.button() == Qt.RightButton:
            self.autoRange()
        else:
            event.accept()
            self.graph.deselect_all()


class LinePlotGraph(pg.PlotWidget):
    def __init__(self, parent):
        super().__init__(parent, viewBox=LinePlotViewBox(self),
                         background="w", enableMenu=False)
        self._items = {}
        self.selection = set()
        self.state = SELECT
        self.master = parent

    def select_by_rectangle(self, rect):
        selection = []
        for item in self.getViewBox().childGroup.childItems():
            if isinstance(item, LinePlotItem) and item.isVisible():
                if line_segment_rect_intersect(rect, item):
                    selection.append(item.index)
        self.select(selection)

    def select_by_click(self, item):
        self.select([item.index])

    def deselect_all(self):
        for i in self.selection:
            self._items[i].deselect()
        self.selection.clear()
        self.master.selection_changed()

    def select(self, indices):
        for i in self.selection:
            self._items[i].deselect()
        keys = QApplication.keyboardModifiers()
        if keys & Qt.ControlModifier:
            self.selection.symmetric_difference_update(indices)
        elif keys & Qt.AltModifier:
            self.selection.difference_update(indices)
        elif keys & Qt.ShiftModifier:
            self.selection.update(indices)
        else:
            self.selection = set(indices)
        for i in self.selection:
            self._items[i].select()
        self.master.selection_changed()

    def reset(self):
        self._items = {}
        self.selection = set()
        self.state = SELECT
        self.clear()
        self.getAxis('bottom').setTicks(None)

    def add_line_plot_item(self, item):
        self._items[item.index] = item
        self.addItem(item)


class OWLinePlot(OWWidget):
    name = "Line Plot"
    description = "Visualization of data profiles (e.g., time series)."
    icon = "icons/LinePlot.svg"
    priority = 1030

    class Inputs:
        data = Input("Data", Table, default=True)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    settingsHandler = settings.PerfectDomainContextHandler()

    group_var = settings.Setting("")                #: Group by group_var's values
    selected_classes = settings.Setting([])         #: List of selected class indices
    display_individual = settings.Setting(False)    #: Show individual profiles
    display_average = settings.Setting(True)        #: Show average profile
    display_quartiles = settings.Setting(True)      #: Show data quartiles
    auto_commit = settings.Setting(True)
    selection = settings.ContextSetting([])

    class Information(OWWidget.Information):
        not_enough_attrs = Msg("Need at least one continuous feature.")

    def __init__(self, parent=None):
        super().__init__(parent)

        self.classes = []

        self.data = None
        self.group_variables = []
        self.graph_variables = []
        self.__groups = None

        # Setup GUI
        infobox = gui.widgetBox(self.controlArea, "Info")
        self.infoLabel = gui.widgetLabel(infobox, "No data on input.")
        displaybox = gui.widgetBox(self.controlArea, "Display")
        gui.checkBox(displaybox, self, "display_individual",
                     "Line plots",
                     callback=self.__update_visibility)
        gui.checkBox(displaybox, self, "display_quartiles", "Error bars",
                     callback=self.__update_visibility)

        group_box = gui.widgetBox(self.controlArea, "Group by")
        self.cb_attr = gui.comboBox(
            group_box, self, "group_var", sendSelectedValue=True,
            callback=self.update_group_var)
        self.group_listbox = gui.listBox(
            group_box, self, "selected_classes", "classes",
            selectionMode=QListWidget.MultiSelection,
            callback=self.__on_class_selection_changed)
        self.unselectAllClassedQLB = gui.button(
            group_box, self, "Unselect all",
            callback=self.__select_all_toggle)

        self.gui = OWPlotGUI(self)
        self.box_zoom_select(self.controlArea)

        gui.rubber(self.controlArea)

        gui.auto_commit(self.controlArea, self, "auto_commit",
                        "Send Selection", "Send Automatically")

        self.graph = LinePlotGraph(self)
        self.graph.getPlotItem().buttonsHidden = True
        self.graph.setRenderHint(QPainter.Antialiasing, True)
        self.mainArea.layout().addWidget(self.graph)

    def box_zoom_select(self, parent):
        g = self.gui
        box_zoom_select = gui.vBox(parent, "Zoom/Select")
        zoom_select_toolbar = g.zoom_select_toolbar(
            box_zoom_select, nomargin=True,
            buttons=[g.StateButtonsBegin, g.SimpleSelect, g.Pan, g.Zoom,
                     g.StateButtonsEnd, g.ZoomReset]
        )
        buttons = zoom_select_toolbar.buttons
        buttons[g.SimpleSelect].clicked.connect(self.select_button_clicked)
        buttons[g.Pan].clicked.connect(self.pan_button_clicked)
        buttons[g.Zoom].clicked.connect(self.zoom_button_clicked)
        buttons[g.ZoomReset].clicked.connect(self.reset_button_clicked)
        return box_zoom_select

    def select_button_clicked(self):
        self.graph.state = SELECT
        self.graph.getViewBox().setMouseMode(self.graph.getViewBox().RectMode)

    def pan_button_clicked(self):
        self.graph.state = PANNING
        self.graph.getViewBox().setMouseMode(self.graph.getViewBox().PanMode)

    def zoom_button_clicked(self):
        self.graph.state = ZOOMING
        self.graph.getViewBox().setMouseMode(self.graph.getViewBox().RectMode)

    def reset_button_clicked(self):
        self.graph.getViewBox().autoRange()

    def selection_changed(self):
        self.selection = list(self.graph.selection)
        self.commit()

    def sizeHint(self):
        return QSize(800, 600)

    def clear(self):
        """
        Clear/reset the widget state.
        """
        self.cb_attr.clear()
        self.group_listbox.clear()
        self.data = None
        self.__groups = None
        self.graph.reset()
        self.infoLabel.setText("No data on input.")

    @Inputs.data
    def set_data(self, data):
        """
        Set the input profile dataset.
        """
        self.closeContext()
        self.clear()
        self.clear_messages()

        self.data = data
        if data is not None:
            n_instances = len(data)
            n_attrs = len(data.domain.attributes)
            self.infoLabel.setText("%i instances on input\n%i attributes" % (
                n_instances, n_attrs))

            self.graph_variables = [var for var in data.domain.attributes
                                    if var.is_continuous]
            if len(self.graph_variables) < 1:
                self.Information.not_enough_attrs()
            else:
                groupvars = [var for var in data.domain.variables +
                             data.domain.metas if var.is_discrete]

                if len(groupvars) > 0:
                    self.cb_attr.addItems([str(var) for var in groupvars])
                    self.group_var = str(groupvars[0])
                    self.group_variables = groupvars
                    self.update_group_var()
                else:
                    self._setup_plot()

        self.selection = []
        self.openContext(data)
        self.select_data_instances()
        self.commit()

    def select_data_instances(self):
        if self.data is None or not len(self.data) or not len(self.selection):
            return
        if max(self.selection) >= len(self.data):
            self.selection = []
        self.graph.select(self.selection)

    def _plot_curve(self, X, color, data, indices):
        dark_pen = QPen(color.darker(110), 4)
        dark_pen.setCosmetic(True)

        color.setAlpha(150)
        light_pen = QPen(color, 1)
        light_pen.setCosmetic(True)
        items = []
        for index, instance in zip(indices, data):
            item = LinePlotItem(index, X, instance.x, light_pen, color)
            item.sigClicked.connect(self.graph.select_by_click)
            items.append(item)
            self.graph.add_line_plot_item(item)

        mean = np.nanmean(data.X, axis=0)
        meancurve = pg.PlotDataItem(
            x=X, y=mean, pen=dark_pen, symbol="o", pxMode=True,
            symbolSize=5, antialias=True
        )
        self.graph.addItem(meancurve)

        q1, q2, q3 = np.nanpercentile(data.X, [25, 50, 75], axis=0)
        errorbar = pg.ErrorBarItem(
            x=X, y=mean,
            bottom=np.clip(mean - q1, 0, mean - q1),
            top=np.clip(q3 - mean, 0, q3 - mean),
            beam=0.01
        )
        self.graph.addItem(errorbar)
        return items, mean, meancurve, errorbar

    def _setup_plot(self):
        """Setup the plot with new curve data."""
        assert self.data is not None
        self.graph.reset()

        data, domain = self.data, self.data.domain
        self.graph.getAxis('bottom').setTicks([
            [(i+1, str(a)) for i, a in enumerate(self.graph_variables)]
        ])

        X = np.arange(1, len(self.graph_variables)+1)
        groups = []

        if not self.selected_classes:
            group_data = data[:, self.graph_variables]
            items, mean, meancurve, errorbar = self._plot_curve(
                X, QColor(Qt.darkGray), group_data,
                list(range(len(self.data))))
            groups.append(
                namespace(
                    data=group_data,
                    profiles=items,
                    mean=meancurve,
                    boxplot=errorbar)
            )
        else:
            var = domain[self.group_var]
            class_col_data, _ = data.get_column_view(var)
            group_indices = [np.flatnonzero(class_col_data == i)
                             for i in range(len(self.classes))]

            for i, indices in enumerate(group_indices):
                if len(indices) == 0:
                    groups.append(None)
                else:
                    if self.classes:
                        color = self.class_colors[i]
                    else:
                        color = QColor(Qt.darkGray)

                    group_data = data[indices, self.graph_variables]
                    items, mean, meancurve, errorbar = self._plot_curve(
                        X, color, group_data, indices)

                    groups.append(
                        namespace(
                            data=group_data, indices=indices,
                            profiles=items, mean=meancurve,
                            boxplot=errorbar)
                    )

        self.__groups = groups
        self.__update_visibility()

    def __update_visibility(self):
        if self.__groups is None:
            return
        if self.classes and self.selected_classes:
            selected = lambda i: i in self.selected_classes
        else:
            selected = lambda i: True
        for i, group in enumerate(self.__groups):
            if group is not None:
                isselected = selected(i)
                for item in group.profiles:
                    item.setVisible(isselected and self.display_individual)
                group.mean.setVisible(isselected)
                group.boxplot.setVisible(isselected and self.display_quartiles)

    def __select_all_toggle(self):
        allselected = len(self.selected_classes) == len(self.classes)
        if allselected:
            self.selected_classes = []
        else:
            self.selected_classes = list(range(len(self.classes)))

        self.__on_class_selection_changed()

    def __on_class_selection_changed(self):
        mask = [i in self.selected_classes
                for i in range(self.group_listbox.count())]
        self.unselectAllClassedQLB.setText(
            "Select all" if not all(mask) else "Unselect all")

        self.__update_visibility()
        self.graph.deselect_all()

    def update_group_var(self):
        data_attr, _ = self.data.get_column_view(self.group_var)
        class_vals = self.data.domain[self.group_var].values
        self.classes = list(class_vals)
        self.class_colors = \
            colorpalette.ColorPaletteGenerator(len(class_vals))
        self.selected_classes = list(range(len(class_vals)))
        for i in range(len(class_vals)):
            item = self.group_listbox.item(i)
            item.setIcon(colorpalette.ColorPixmap(self.class_colors[i]))

        self._setup_plot()
        self.__on_class_selection_changed()

    def commit(self):
        selected = self.data[self.selection] \
            if self.data is not None and len(self.selection) > 0 else None
        annotated = create_annotated_table(self.data, self.selection)
        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(annotated)


def test_main(argv=sys.argv):
    from AnyQt.QtWidgets import QApplication
    a = QApplication(argv)
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "brown-selected"
    w = OWLinePlot()
    d = Table(filename)

    w.set_data(d)
    w.show()
    r = a.exec_()
    w.saveSettings()
    return r


if __name__ == "__main__":
    sys.exit(test_main())
