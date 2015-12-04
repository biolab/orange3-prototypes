import sys

from types import SimpleNamespace as namespace

import numpy as np
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QStyle, QGraphicsItem, QPen, QColor
from PyQt4.QtCore import Qt, QPointF

import pyqtgraph as pg

import Orange.data

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import colorpalette


def disconnected_curve_data(data, x=None):
    C, P = data.shape
    if x is not None:
        x = np.asarray(x)
        if x.shape != (P,):
            raise ValueError("x must have shape ({},)".format(P))
    else:
        x = np.arange(P)

    validmask = np.isfinite(data)
    validdata = data[validmask]
    row_count = np.sum(validmask, axis=1)
    connect = np.ones(np.sum(row_count), dtype=bool)
    connect[np.cumsum(row_count)[:-1] - 1] = False
    X = np.tile(x, C)[validmask.ravel()]
    return X, validdata, connect


# TODO:
#  * Box plot item

class OWLinePlot(widget.OWWidget):
    name = "Line Plot"
    description = "Visualization of data profiles (e.g., time series)."
    icon = "icons/LinePlot.svg"
    priority = 1030

    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = []
    settingsHandler = settings.DomainContextHandler()

    group_var = settings.Setting("")                #: Group by group_var's values
    selected_classes = settings.Setting([])         #: List of selected class indices
    display_individual = settings.Setting(False)    #: Show individual profiles
    display_average = settings.Setting(True)        #: Show average profile
    display_quartiles = settings.Setting(True)      #: Show data quartiles

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
        gui.checkBox(displaybox, self, "display_quartiles", "Box plot",
                     callback=self.__update_visibility)

        group_box = gui.widgetBox(self.controlArea, "Group by")
        self.cb_attr = gui.comboBox(
            group_box, self, "group_var", sendSelectedValue=True,
            callback=self.update_group_var)
        self.group_listbox = gui.listBox(
            group_box, self, "selected_classes", "classes",
            selectionMode=QtGui.QListWidget.MultiSelection,
            callback=self.__on_class_selection_changed)
        self.unselectAllClassedQLB = gui.button(
            group_box, self, "Unselect all",
            callback=self.__select_all_toggle)

        gui.rubber(self.controlArea)

        self.graph = pg.PlotWidget(background="w", enableMenu=False)
        self.graph.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.mainArea.layout().addWidget(self.graph)

    def sizeHint(self):
        return QtCore.QSize(800, 600)

    def clear(self):
        """
        Clear/reset the widget state.
        """
        self.cb_attr.clear()
        self.group_listbox.clear()
        self.data = None
        self.__groups = None
        self.graph.clear()

    def set_data(self, data):
        """
        Set the input profile dataset.
        """
        self.closeContext()
        self.clear()

        self.data = data
        if data is not None:
            n_instances = len(data)
            n_attrs = len(data.domain.attributes)
            self.infoLabel.setText("%i instances on input\n%i attributes"%(n_instances, n_attrs))

            self.graph_variables = [var for var in data.domain.attributes
                                    if var.is_continuous]

            groupvars = [var for var in data.domain.variables + data.domain.metas
                        if var.is_discrete]
            if len(groupvars) > 0:
                self.cb_attr.addItems([str(var) for var in groupvars])
                self.group_var = str(groupvars[0])
                self.group_variables = groupvars
                self.update_group_var()

            self.openContext(data)

    def _setup_plot(self):
        """Setup the plot with new curve data."""
        assert self.data is not None
        self.graph.clear()

        data, domain = self.data, self.data.domain
        var = domain[self.group_var]
        class_col_data, _ = data.get_column_view(var)
        group_indices = [np.flatnonzero(class_col_data == i)
                         for i in range(len(self.classes))]

        self.graph.getAxis('bottom').setTicks([
            [(i+1, str(a)) for i, a in enumerate(self.graph_variables)]
        ])

        X = np.arange(1, len(self.graph_variables)+1)
        groups = []

        for i, indices in enumerate(group_indices):
            if len(indices) == 0:
                groups.append(None)
            else:
                if self.classes:
                    color = self.class_colors[i]
                else:
                    color = QColor(Qt.darkGray)
                group_data = data[indices, self.graph_variables]
                plot_x, plot_y, connect = disconnected_curve_data(group_data.X, x=X)

                color.setAlpha(200)
                lightcolor = QColor(color.lighter(factor=150))
                lightcolor.setAlpha(150)
                pen = QPen(color, 2)
                pen.setCosmetic(True)

                lightpen = QPen(lightcolor, 1)
                lightpen.setCosmetic(True)

                curve = pg.PlotCurveItem(
                    x=plot_x, y=plot_y, connect=connect,
                    pen=lightpen, symbolSize=2, antialias=True,
                )
                self.graph.addItem(curve)

                mean = np.nanmean(group_data.X, axis=0)

                meancurve = pg.PlotDataItem(
                    x=X, y=mean, pen=pen, size=5, symbol="o", pxMode=True,
                    symbolSize=5, antialias=True
                )
                self.graph.addItem(meancurve)

                q1, q2, q3 = np.nanpercentile(group_data.X, [25, 50, 75], axis=0)
                # TODO: implement and use a box plot item
                errorbar = pg.ErrorBarItem(
                    x=X, y=mean,
                    bottom=np.clip(mean - q1, 0, mean - q1),
                    top=np.clip(q3 - mean, 0, q3 - mean),
                    beam=0.5
                )
                self.graph.addItem(errorbar)
                groups.append(
                    namespace(
                        data=group_data, indices=indices,
                        profiles=curve, mean=meancurve,
                        boxplot=errorbar)
                )

        self.__groups = groups
        self.__update_visibility()

    def __update_visibility(self):
        if self.__groups is None:
            return

        if self.classes:
            selected = lambda i: i in self.selected_classes
        else:
            selected = lambda i: True
        for i, group in enumerate(self.__groups):
            if group is not None:
                isselected = selected(i)
                group.profiles.setVisible(isselected and self.display_individual)
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


def test_main(argv=sys.argv):
    a = QtGui.QApplication(argv)
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "brown-selected"
    w = OWLinePlot()
    d = Orange.data.Table(filename)

    w.set_data(d)
    w.show()
    r = a.exec_()
    w.saveSettings()
    return r

if __name__ == "__main__":
    sys.exit(test_main())
