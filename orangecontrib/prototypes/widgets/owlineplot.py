import sys
from enum import IntEnum
from types import SimpleNamespace as namespace
from collections import namedtuple

import numpy as np

from AnyQt.QtCore import Qt, QSize, QRectF
from AnyQt.QtGui import QPainter, QPen, QColor
from AnyQt.QtWidgets import QApplication, QSizePolicy

import pyqtgraph as pg
from pyqtgraph.graphicsItems.ViewBox import ViewBox
from pyqtgraph.Point import Point

from Orange.data import Table, DiscreteVariable
from Orange.widgets import gui, settings
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.utils.colorpalette import ColorPaletteGenerator
from Orange.widgets.utils.itemmodels import DomainModel
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


class LinePlotColors:
    LIGHT_ALPHA = 120
    DEFAULT_COLOR = QColor(Qt.darkGray)

    def __call__(self, n):
        return ColorPaletteGenerator(n)


class LinePlotItem(pg.PlotCurveItem):
    def __init__(self, index, instance_id, x, y, color):
        color.setAlpha(LinePlotColors.LIGHT_ALPHA)
        pen = QPen(color, 1)
        pen.setCosmetic(True)
        super().__init__(x=x, y=y, pen=pen, pxMode=True, antialias=True)
        self._selected = False
        self._in_subset = False
        self._pen = pen
        self.index = index
        self.id = instance_id
        self.setClickable(True, width=10)

    def into_subset(self):
        self._in_subset = True
        self._change_pen()

    def out_of_subset(self):
        self._in_subset = False
        self._change_pen()

    def select(self):
        self._selected = True
        self._change_pen()

    def deselect(self):
        self._selected = False
        self._change_pen()

    def setColor(self, color):
        self._pen.setColor(color)
        self._change_pen()

    def _change_pen(self):
        pen = QPen(self._pen)
        if self._in_subset and self._selected:
            color = QColor(self._pen.color())
            color.setAlpha(255)
            pen.setWidth(4)
            pen.setColor(color)
        elif not self._in_subset and self._selected:
            color = QColor(self._pen.color())
            color.setAlpha(255)
            pen.setColor(color)
        elif self._in_subset and not self._selected:
            color = QColor(self._pen.color())
            color.setAlpha(LinePlotColors.LIGHT_ALPHA)
            pen.setWidth(4)
            pen.setColor(color)
        else:
            color = QColor(self._pen.color())
            color.setAlpha(LinePlotColors.LIGHT_ALPHA)
            pen.setColor(color)
        self.setPen(pen)


class LinePlotViewBox(ViewBox):
    def __init__(self, graph, enable_menu=False):
        ViewBox.__init__(self, enableMenu=enable_menu)
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
        self._items_by_id = {}
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

    def select_subset(self, ids):
        for i in ids:
            self._items_by_id[i].into_subset()

    def deselect_subset(self):
        for item in self._items.values():
            item.out_of_subset()

    def reset(self):
        self._items = {}
        self._items_by_id = {}
        self.selection = set()
        self.state = SELECT
        self.clear()
        self.getAxis('bottom').setTicks(None)

    def add_line_plot_item(self, item):
        self._items[item.index] = item
        self._items_by_id[item.id] = item
        self.addItem(item, ignoreBounds=True)

    def finished_adding(self):
        vb = self.getViewBox()
        vb.addedItems.extend(list(self._items.values()))
        vb.updateAutoRange()


class LinePlotDisplay(IntEnum):
    INSTANCES, MEAN, INSTANCES_WITH_MEAN = 0, 1, 2


class OWLinePlot(OWWidget):
    name = "Line Plot"
    description = "Visualization of data profiles (e.g., time series)."
    icon = "icons/LinePlot.svg"
    priority = 1030

    class Inputs:
        data = Input("Data", Table, default=True)
        data_subset = Input("Data Subset", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    settingsHandler = settings.PerfectDomainContextHandler()

    group_var = settings.ContextSetting(None)
    display_index = settings.Setting(LinePlotDisplay.INSTANCES)
    display_quartiles = settings.Setting(False)
    auto_commit = settings.Setting(True)
    selection = settings.ContextSetting([])

    class Information(OWWidget.Information):
        not_enough_attrs = Msg("Need at least one continuous feature.")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.__groups = None
        self.__profiles = None
        self.data = None
        self.data_subset = None
        self.subset_selection = []
        self.graph_variables = []

        # Setup GUI
        infobox = gui.widgetBox(self.controlArea, "Info")
        self.infoLabel = gui.widgetLabel(infobox, "No data on input.")
        displaybox = gui.widgetBox(self.controlArea, "Display")
        radiobox = gui.radioButtons(displaybox, self, "display_index",
                                    callback=self.__update_visibility)
        gui.appendRadioButton(radiobox, "Line plot")
        gui.appendRadioButton(radiobox, "Mean")
        gui.appendRadioButton(radiobox, "Line plot with mean")

        showbox = gui.widgetBox(self.controlArea, "Show")
        gui.checkBox(showbox, self, "display_quartiles", "Error bars",
                     callback=self.__update_visibility)

        self.group_vars = DomainModel(
            placeholder="None", separators=False,
            valid_types=DiscreteVariable)
        self.group_view = gui.listView(
            self.controlArea, self, "group_var", box="Group by",
            model=self.group_vars, callback=self.__group_var_changed)
        self.group_view.setEnabled(False)
        self.group_view.setMinimumSize(QSize(30, 100))
        self.group_view.setSizePolicy(QSizePolicy.Expanding,
                                      QSizePolicy.Ignored)

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
        self.__groups = None
        self.__profiles = None
        self.graph_variables = []
        self.graph.reset()
        self.infoLabel.setText("No data on input.")
        self.group_vars.set_domain(None)
        self.group_view.setEnabled(False)

    @Inputs.data
    def set_data(self, data):
        """
        Set the input profile dataset.
        """
        self.closeContext()
        self.clear()
        self.clear_messages()

        self.data = data
        self.selection = []
        if data is not None:
            self.group_vars.set_domain(data.domain)
            self.group_view.setEnabled(len(self.group_vars) > 1)
            self.group_var = data.domain.class_var if \
                data.domain.class_var and data.domain.class_var.is_discrete \
                else None

            self.infoLabel.setText("%i instances on input\n%i attributes" % (
                len(data), len(data.domain.attributes)))
            self.graph_variables = [var for var in data.domain.attributes
                                    if var.is_continuous]
            if len(self.graph_variables) < 1:
                self.Information.not_enough_attrs()
                self.commit()
                return

        self.openContext(data)
        self._setup_plot()
        self.commit()

    @Inputs.data_subset
    def set_data_subset(self, subset):
        """
        Set the supplementary input subset dataset.
        """
        self.data_subset = subset
        if len(self.subset_selection):
            self.graph.deselect_subset()

    def handleNewSignals(self):
        self.subset_selection = []
        if self.data is not None and self.data_subset is not None and \
                len(self.graph_variables):
            intersection = set(self.data.ids).intersection(
                set(self.data_subset.ids))
            self.subset_selection = intersection
            if self.__profiles is not None:
                self.graph.select_subset(self.subset_selection)

    def _setup_plot(self):
        """Setup the plot with new curve data."""
        if self.data is None:
            return
        self.graph.reset()
        ticks = [[(i + 1, str(a)) for i, a in enumerate(self.graph_variables)]]
        self.graph.getAxis('bottom').setTicks(ticks)
        if self.display_index in (LinePlotDisplay.INSTANCES,
                                  LinePlotDisplay.INSTANCES_WITH_MEAN):
            self._plot_profiles()
        self._plot_groups()
        self.__update_visibility()

    def _plot_profiles(self):
        X = np.arange(1, len(self.graph_variables) + 1)
        data = self.data[:, self.graph_variables]
        self.__profiles = []
        for index, inst in zip(range(len(self.data)), data):
            color = self.__get_line_color(index)
            profile = LinePlotItem(index, inst.id, X, inst.x, color)
            profile.sigClicked.connect(self.graph.select_by_click)
            self.graph.add_line_plot_item(profile)
            self.__profiles.append(profile)
        self.graph.finished_adding()
        self.__select_data_instances()

    def _plot_groups(self):
        if self.__groups is not None:
            for group in self.__groups:
                if group is not None:
                    self.graph.getViewBox().removeItem(group.mean)
                    self.graph.getViewBox().removeItem(group.error_bar)

        self.__groups = []
        X = np.arange(1, len(self.graph_variables) + 1)
        if self.group_var is None:
            self.__plot_mean_with_error(X, self.data[:, self.graph_variables])
        else:
            class_col_data, _ = self.data.get_column_view(self.group_var)
            group_indices = [np.flatnonzero(class_col_data == i)
                             for i in range(len(self.group_var.values))]
            for index, indices in enumerate(group_indices):
                if len(indices) == 0:
                    self.__groups.append(None)
                else:
                    group_data = self.data[indices, self.graph_variables]
                    self.__plot_mean_with_error(X, group_data, index)

    def __plot_mean_with_error(self, X, data, index=None):
        pen = QPen(self.__get_line_color(None, index), 4)
        pen.setCosmetic(True)
        mean = np.nanmean(data.X, axis=0)
        mean_curve = pg.PlotDataItem(x=X, y=mean, pen=pen, symbol="o",
                                     symbolSize=5, antialias=True)
        self.graph.addItem(mean_curve)

        q1, q2, q3 = np.nanpercentile(data.X, [25, 50, 75], axis=0)
        bottom = np.clip(mean - q1, 0, mean - q1)
        top = np.clip(q3 - mean, 0, q3 - mean)
        error_bar = pg.ErrorBarItem(x=X, y=mean, bottom=bottom,
                                    top=top, beam=0.01)
        self.graph.addItem(error_bar)
        self.__groups.append(namespace(mean=mean_curve, error_bar=error_bar))

    def __update_visibility(self):
        self.__update_visibility_profiles()
        self.__update_visibility_groups()

    def __update_visibility_groups(self):
        show_mean = self.display_index in (LinePlotDisplay.MEAN,
                                           LinePlotDisplay.INSTANCES_WITH_MEAN)
        if self.__groups is not None:
            for group in self.__groups:
                if group is not None:
                    group.mean.setVisible(show_mean)
                    group.error_bar.setVisible(self.display_quartiles)

    def __update_visibility_profiles(self):
        show_inst = self.display_index in (LinePlotDisplay.INSTANCES,
                                           LinePlotDisplay.INSTANCES_WITH_MEAN)
        if self.__profiles is None and show_inst:
            self._plot_profiles()
            self.graph.select_subset(self.subset_selection)
        if self.__profiles is not None:
            for profile in self.__profiles:
                profile.setVisible(show_inst)

    def __group_var_changed(self):
        if self.data is None or not len(self.graph_variables):
            return
        self.__color_profiles()
        self._plot_groups()
        self.__update_visibility()

    def __color_profiles(self):
        if self.__profiles is not None:
            for profile in self.__profiles:
                profile.setColor(self.__get_line_color(profile.index))

    def __select_data_instances(self):
        if self.data is None or not len(self.data) or not len(self.selection):
            return
        if max(self.selection) >= len(self.data):
            self.selection = []
        self.graph.select(self.selection)

    def __get_line_color(self, data_index=None, mean_index=None):
        color = QColor(LinePlotColors.DEFAULT_COLOR)
        if self.group_var is not None:
            if data_index is not None:
                value = self.data[data_index][self.group_var]
                if np.isnan(value):
                    return color
            index = int(value) if data_index is not None else mean_index
            color = LinePlotColors()(len(self.group_var.values))[index]
        return color.darker(110) if data_index is None else color

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
