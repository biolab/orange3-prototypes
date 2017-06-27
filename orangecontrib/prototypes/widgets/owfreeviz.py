import sys
import enum
from xml.sax.saxutils import escape
from types import SimpleNamespace as namespace

import pkg_resources

import numpy

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt, QObject, QEvent, QLineF, QRectF, QCoreApplication
from PyQt4.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

import pyqtgraph as pg

import Orange.data
import Orange.projection
from Orange.canvas import report

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import classdensity
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.utils.plot import OWPlotGUI
from Orange.widgets.visualize import owlinearprojection as linproj
from Orange.widgets.unsupervised.owmds import Mdsplotutils as plotutils

from orangecontrib.prototypes.projection.freeviz import freeviz
from orangecontrib.prototypes.widgets.utils.axisitem import AxisItem


class AsyncUpdateLoop(QObject):
    """
    Run/drive an coroutine from the event loop.

    This is a utility class which can be used for implementing
    asynchronous update loops. I.e. coroutines which periodically yield
    control back to the Qt event loop.

    """
    Next = QEvent.registerEventType()

    #: State flags
    Idle, Running, Cancelled, Finished = 0, 1, 2, 3
    #: The coroutine has yielded control to the caller (with `object`)
    yielded = Signal(object)
    #: The coroutine has finished/exited (either with an exception
    #: or with a return statement)
    finished = Signal()

    #: The coroutine has returned (normal return statement / StopIteration)
    returned = Signal(object)
    #: The coroutine has exited with with an exception.
    raised = Signal(object)
    #: The coroutine was cancelled/closed.
    cancelled = Signal()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__coroutine = None
        self.__next_pending = False  # Flag for compressing scheduled events
        self.__in_next = False
        self.__state = AsyncUpdateLoop.Idle

    @Slot(object)
    def setCoroutine(self, loop):
        """
        Set the coroutine.

        The coroutine will be resumed (repeatedly) from the event queue.
        If there is an existing coroutine set it is first closed/cancelled.

        Raises an RuntimeError if the current coroutine is running.
        """
        if self.__coroutine is not None:
            self.__coroutine.close()
            self.__coroutine = None
            self.__state = AsyncUpdateLoop.Cancelled

            self.cancelled.emit()
            self.finished.emit()

        if loop is not None:
            self.__coroutine = loop
            self.__state = AsyncUpdateLoop.Running
            self.__schedule_next()

    @Slot()
    def cancel(self):
        """
        Cancel/close the current coroutine.

        Raises an RuntimeError if the current coroutine is running.
        """
        self.setCoroutine(None)

    def state(self):
        """
        Return the current state.
        """
        return self.__state

    def isRunning(self):
        return self.__state == AsyncUpdateLoop.Running

    def __schedule_next(self):
        if not self.__next_pending:
            self.__next_pending = True
            QtCore.QTimer.singleShot(10, self.__on_timeout)

    def __next(self):
        if self.__coroutine is not None:
            try:
                rval = next(self.__coroutine)
            except StopIteration as stop:
                self.__state = AsyncUpdateLoop.Finished
                self.returned.emit(stop.value)
                self.finished.emit()
                self.__coroutine = None
            except BaseException as er:
                self.__state = AsyncUpdateLoop.Finished
                self.raised.emit(er)
                self.finished.emit()
                self.__coroutine = None
            else:
                self.yielded.emit(rval)
                self.__schedule_next()

    @Slot()
    def __on_timeout(self):
        assert self.__next_pending
        self.__next_pending = False
        if not self.__in_next:
            self.__in_next = True
            try:
                self.__next()
            finally:
                self.__in_next = False
        else:
            # warn
            self.__schedule_next()

    def customEvent(self, event):
        if event.type() == AsyncUpdateLoop.Next:
            self.__on_timeout()
        else:
            super().customEvent(event)


class PlotToolBox(QtCore.QObject):
    actionTriggered = Signal(QtGui.QAction)
    toolActivated = Signal(linproj.PlotTool)

    class StandardActions(enum.IntEnum):
        NoAction = 0
        #: Reset zoom (zoom to fit) action with CTRL + Key_0 shortcut
        ZoomReset = 1
        #: Zoom in action with QKeySequence.ZoomIn shortcut
        ZoomIn = 2
        #: Zoom out action with QKeySequence.ZoomOut shortcut
        ZoomOut = 4
        # A Select tool action (exclusive with other *Tool)
        SelectTool = 8
        # A Zoom tool action (exclusive with other *Tool)
        ZoomTool = 16
        # A Pan tool  (exclusive with other *Tool)
        PanTool = 32

    NoAction, ZoomReset, ZoomIn, ZoomOut, SelectTool, ZoomTool, PanTool = \
        list(StandardActions)

    DefaultActions = (ZoomReset | ZoomIn | ZoomOut |
                      SelectTool | ZoomTool | PanTool)

    ExclusiveMask = SelectTool | ZoomTool | PanTool

    ActionData = {
        ZoomReset: ("Zoom to fit", "zoom_reset",
                    Qt.ControlModifier + Qt.Key_0),
        ZoomIn: ("Zoom in", "", QtGui.QKeySequence.ZoomIn),
        ZoomOut: ("Zoom out", "", QtGui.QKeySequence.ZoomOut),
        SelectTool: ("Select", "arrow", Qt.ControlModifier + Qt.Key_1),
        ZoomTool: ("Zoom", "zoom", Qt.ControlModifier + Qt.Key_2),
        PanTool: ("Pan", "pan_hand", Qt.ControlModifier + Qt.Key_3),
    }

    def __init__(self, parent=None, standardActions=DefaultActions, **kwargs):
        super().__init__(parent, **kwargs)
        self.__standardActions = standardActions
        self.__actions = {}
        self.__tools = {}
        self.__viewBox = None
        self.__currentTool = None
        self.__toolgroup = QtGui.QActionGroup(self, exclusive=True)

        def on_toolaction(action):
            tool = action.property("tool")
            if self.__currentTool is not None:
                self.__currentTool.setViewBox(None)
            self.__currentTool = tool
            if tool is not None:
                tool.setViewBox(self.__viewBox)
                self.__viewBox.setCursor(tool.cursor)

        self.__toolgroup.triggered[QtGui.QAction].connect(on_toolaction)

        def icon(name):
            path = "icons/Dlg_{}.png".format(name)
            path = pkg_resources.resource_filename(widget.__name__, path)
            return QtGui.QIcon(path)

        isfirsttool = True
        for flag in PlotToolBox.StandardActions:
            if standardActions & flag:
                _text, _iconname, _keyseq = self.ActionData[flag]

                action = QtGui.QAction(
                    _text, self, icon=icon(_iconname),
                    shortcut=QtGui.QKeySequence(_keyseq)
                )

                self.__actions[flag] = action

                if flag & PlotToolBox.ExclusiveMask:
                    action.setCheckable(True)
                    self.__toolgroup.addAction(action)

                    if flag == self.SelectTool:
                        tool = linproj.PlotSelectionTool(self)
                        tool.cursor = Qt.ArrowCursor
                    elif flag == self.ZoomTool:
                        tool = linproj.PlotZoomTool(self)
                        tool.cursor = Qt.ArrowCursor
                    elif flag == self.PanTool:
                        tool = linproj.PlotPanTool(self)
                        tool.cursor = Qt.OpenHandCursor

                    self.__tools[flag] = tool
                    action.setProperty("tool", tool)

                    if isfirsttool:
                        action.setChecked(True)
                        self.__currentTool = tool
                        isfirsttool = False

    def setViewBox(self, box):
        if self.__viewBox is not box and self.__currentTool is not None:
            self.__currentTool.setViewBox(None)
            # TODO: Unset/restore default view box cursor
            self.__viewBox = None

        self.__viewBox = box
        if self.__currentTool is not None:
            self.__currentTool.setViewBox(box)
            if box is not None:
                box.setCursor(self.__currentTool.cursor)

    def viewBox(self):
        return self.__viewBox

    def standardAction(self, action):
        return self.__actions[action]

    def actions(self):
        return list(self.__actions.values())

    def button(self, action, parent=None):
        action = self.standardAction(action)
        b = QtGui.QToolButton(parent)
        b.setToolButtonStyle(Qt.ToolButtonIconOnly)
        b.setDefaultAction(action)
        return b

    def toolGroup(self):
        """Return the exclusive tool action button group"""
        return self.__toolgroup

    def plotTool(self, action):
        return self.__tools[action]

def make_pen(color, width=1.0, style=Qt.SolidLine, cap=Qt.SquareCap,
             join=Qt.BevelJoin, cosmetic=True):
    pen = QtGui.QPen(color, width, style=style, cap=cap, join=join)
    pen.setCosmetic(cosmetic)
    return pen


class OWFreeViz(widget.OWWidget):
    name = "FreeViz"
    description = "FreeViz Visualization"
    icon = "icons/LinearProjection.svg"
    inputs = [("Data", Orange.data.Table, "set_data", widget.Default),
              ("Data Subset", Orange.data.Table, "set_data_subset")]
    outputs = [("Selected Data", Orange.data.Table, widget.Default),
               (ANNOTATED_DATA_SIGNAL_NAME, Orange.data.Table),
               ("Components", Orange.data.Table)]

    settingsHandler = settings.DomainContextHandler()
    #: Initialization type
    Circular, Random = 0, 1
    #: Force law
    ForceLaw = [
        ("Linear", 1),
        ("Square", 2)
    ]

    ReplotIntervals = [
        ("Every iteration", 1),
        ("Every 3 steps", 3),
        ("Every 5 steps", 5,),
        ("Every 10 steps", 10),
        ("Every 20 steps", 20),
        ("Every 50 steps", 50),
        ("Every 100 steps", 100),
        ("None", -1),
    ]
    JitterAmount = [
        ("None", 0),
        ("0.1%", 0.1),
        ("0.5%", 0.5),
        ("1%", 1.0),
        ("2%", 2.0)
    ]

    #: Output coordinate embedding domain role
    NoCoords, Attribute, Meta = 0, 1, 2

    force_law = settings.Setting(0)
    maxiter = settings.Setting(300)
    replot_interval = settings.Setting(3)
    initialization = settings.Setting(Circular)
    min_anchor_radius = settings.Setting(0)
    embedding_domain_role = settings.Setting(Meta)
    autocommit = settings.Setting(True)

    attr_color = settings.ContextSetting(None, exclude_metas=False)
    attr_label = settings.ContextSetting(None, exclude_metas=False)
    attr_shape = settings.ContextSetting(None, exclude_metas=False)
    attr_size = settings.ContextSetting(None, exclude_metas=False)

    point_width = settings.Setting(10)
    alpha_value = settings.Setting(128)
    jitter = settings.Setting(0)
    class_density = settings.Setting(False)

    graph_name = "plot.plotItem"

    class Error(widget.OWWidget.Error):
        no_class_var = widget.Msg("Need a class variable")
        not_enough_class_vas = widget.Msg("Needs discrete class variable " \
                                          "with at lest 2 values")

    def __init__(self):
        super().__init__()

        self.data = None
        self.data_subset = None
        self.plotdata = None

        self.plot = pg.PlotWidget(enableMouse=False, enableMenu=False)
        self.plot.setFrameStyle(QtGui.QFrame.StyledPanel)
        self.plot.plotItem.hideAxis("bottom")
        self.plot.plotItem.hideAxis("left")
        self.plot.plotItem.hideButtons()
        self.plot.setAspectLocked(True)
        self.plot.scene().installEventFilter(self)
        self.replot = self.plot.replot

        box = gui.widgetBox(self.controlArea, "Optimization", spacing=10)
        form = QtGui.QFormLayout(
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QtGui.QFormLayout.AllNonFixedFieldsGrow,
            verticalSpacing=10
        )
        form.addRow(
            "Force law",
            gui.comboBox(box, self, "force_law",
                         items=[text for text, _ in OWFreeViz.ForceLaw],
                         callback=self.__reset_update_interval)
        )
        form.addRow(
            "Max iterations",
            gui.spin(box, self, "maxiter", 10, 10 ** 4)
        )
        form.addRow(
            "Initialization",
            gui.comboBox(box, self, "initialization",
                         items=["Circular", "Random"],
                         callback=self.__reset_initialization)
        )
        form.addRow(
            "Replot",
            gui.comboBox(box, self, "replot_interval",
                         items=[text for text, _ in OWFreeViz.ReplotIntervals],
                         callback=self.__reset_update_interval)
        )
        box.layout().addLayout(form)

        self.start_button = gui.button(
            box, self, "Optimize", self._toogle_start)

        g = OWPlotGUI(self)
        g.point_properties_box(self.controlArea)
        self.models = g.points_models

        box = gui.widgetBox(self.controlArea, "Plot")
        form = QtGui.QFormLayout(
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QtGui.QFormLayout.AllNonFixedFieldsGrow,
            spacing=8,
        )
        box.layout().addLayout(form)

        form.addRow(
            "Jittering",
            gui.comboBox(box, self, "jitter",
                         items=[text for text, _ in self.JitterAmount],
                         callback=self._update_xy)
        )
        self.class_density_cb = gui.checkBox(
            box, self, "class_density", "", callback=self._update_density)
        form.addRow("Class density", self.class_density_cb)

        box = gui.widgetBox(self.controlArea, "Hide anchors")
        rslider = gui.hSlider(
            box, self, "min_anchor_radius", minValue=0, maxValue=100,
            step=5, label="Radius", createLabel=False, ticks=True,
            callback=self._update_anchor_visibility)
        rslider.setTickInterval(0)
        rslider.setPageStep(10)

        box = gui.widgetBox(self.controlArea, "Zoom/Select")
        hlayout = QtGui.QHBoxLayout()
        box.layout().addLayout(hlayout)

        toolbox = PlotToolBox(self)
        hlayout.addWidget(toolbox.button(PlotToolBox.SelectTool))
        hlayout.addWidget(toolbox.button(PlotToolBox.ZoomTool))
        hlayout.addWidget(toolbox.button(PlotToolBox.PanTool))
        hlayout.addSpacing(4)
        hlayout.addWidget(toolbox.button(PlotToolBox.ZoomReset))
        hlayout.addStretch()
        toolbox.standardAction(PlotToolBox.ZoomReset).triggered.connect(
            lambda: self.plot.setRange(QtCore.QRectF(-1.05, -1.05, 2.1, 2.1))
        )
        toolbox.standardAction(PlotToolBox.ZoomIn).triggered.connect(
            lambda: self.plot.getViewBox().scaleBy((1.25, 1.25))
        )
        toolbox.standardAction(PlotToolBox.ZoomIn).triggered.connect(
            lambda: self.plot.getViewBox().scaleBy((1 / 1.25, 1 / 1.25))
        )
        selecttool = toolbox.plotTool(PlotToolBox.SelectTool)
        selecttool.selectionFinished.connect(self.__select_area)
        self.addActions(toolbox.actions())

        self.controlArea.layout().addStretch(1)

        box = gui.widgetBox(self.controlArea, "Output")
        gui.comboBox(box, self, "embedding_domain_role",
                     items=["Original features only",
                            "Coordinates as features",
                            "Coordinates as meta attributes"])
        gui.auto_commit(box, self, "autocommit", "Commit", box=False,
                        callback=self.commit)

        self.legend = linproj.LegendItem()
        self.legend.setParentItem(self.plot.getViewBox())
        self.legend.anchor((1, 0), (1, 0))

        self.plot.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.mainArea.layout().addWidget(self.plot)
        viewbox = self.plot.getViewBox()
        viewbox.grabGesture(Qt.PinchGesture)
        pinchtool = linproj.PlotPinchZoomTool(parent=self)
        pinchtool.setViewBox(viewbox)

        toolbox.setViewBox(viewbox)

        self._loop = AsyncUpdateLoop(parent=self)
        self._loop.yielded.connect(self.__set_projection)
        self._loop.finished.connect(self.__freeviz_finished)
        self._loop.raised.connect(self.__on_error)

    def clear(self):
        """
        Clear/reset the widget state
        """
        self.data = None
        self._clear_plot()
        self._loop.cancel()

    def init_attr_values(self):
        domain = self.data and self.data.domain
        for model in self.models:
            model.set_domain(domain)
        self.attr_color = domain and self.data.domain.class_var or None
        self.attr_shape = None
        self.attr_size = None
        self.attr_label = None

    def set_data(self, data):
        """
        Set the input dataset.
        """
        self.closeContext()
        self.clear()
        self.Error.clear()
        if data is not None:
            if data.domain.class_var is None:
                self.Error.no_class_var()
                data = None
            elif data.domain.class_var.is_discrete and \
                    len(data.domain.class_var.values) < 2:
                self.Error.not_enough_class_vas()
                data = None

        self.data = data
        self.init_attr_values()
        if data is not None:
            self.class_density_cb.setEnabled(data.domain.has_discrete_class)
            self.openContext(data)

    def set_data_subset(self, data):
        """Set the input subset data set."""
        self.data_subset = data
        if self.plotdata is not None:
            self.plotdata.subsetmask = None

    def handleNewSignals(self):
        """Reimplemented."""
        didupdate = False
        if self.data is not None and self.plotdata is None:
            self._setup()
            self._start()
            didupdate = True

        if self.data_subset is not None and self.plotdata is not None and \
                self.plotdata.subsetmask is None:
            # received a new subset data set; need to update the subset mask
            # and brush fill
            self.plotdata.subsetmask = numpy.in1d(
                self.data.ids, self.data_subset.ids)
            self._update_color()
            didupdate = True

        if self.plotdata is not None and not didupdate:
            # the subset dataset was removed; need to update the brush fill
            self._update_color()

    def _toogle_start(self):
        if self._loop.isRunning():
            self._loop.cancel()
            self.start_button.setText("Optimize")
            self.progressBarFinished(processEvents=False)
        else:
            self._start()

    def _clear_plot(self):
        self.plot.clear()
        self.plotdata = None
        self.legend.hide()
        self.legend.clear()

    def _setup(self):
        """
        Setup the plot.
        """
        X = self.data.X
        Y = self.data.Y
        mask = numpy.bitwise_or.reduce(numpy.isnan(X), axis=1)
        mask |= numpy.isnan(Y)
        valid = ~mask
        X = X[valid, :]
        Y = Y[valid]
        if not len(X):
            return

        if self.data.domain.class_var.is_discrete:
            Y = Y.astype(int)
        X = (X - numpy.mean(X, axis=0))
        span = numpy.ptp(X, axis=0)
        X[:, span > 0] /= span[span > 0].reshape(1, -1)

        if self.initialization == OWFreeViz.Circular:
            anchors = linproj.linproj.defaultaxes(X.shape[1]).T
        else:
            anchors = numpy.random.random((X.shape[1], 2)) * 2 - 1

        EX = numpy.dot(X, anchors)
        radius = numpy.max(numpy.linalg.norm(EX, axis=1))

        jittervec = numpy.random.RandomState(4).rand(*EX.shape) * 2 - 1
        jittervec *= 0.01
        _, jitterfactor = self.JitterAmount[self.jitter]

        if self.attr_color is not None:
            colors = plotutils.color_data(self.data, self.attr_color)[valid]
        else:
            colors = numpy.array([[192, 192, 192]])
            colors = numpy.tile(colors, (X.shape[0], 1))

        pendata = plotutils.pen_data(colors * 0.8)
        colors = numpy.hstack(
            [colors, numpy.full((colors.shape[0], 1), float(self.alpha_value))])
        brushdata = plotutils.brush_data(colors)

        shapedata = plotutils.shape_data(self.data, self.attr_shape)[valid]
        sizedata = size_data(
            self.data, self.attr_size, pointsize=self.point_width)[valid]
        if self.attr_label is not None:
            labeldata = plotutils.column_data(self.data, self.attr_label, valid)
            labeldata = [self.attr_label.str_val(val) for val in labeldata]
        else:
            labeldata = None

        coords = (EX / radius) + jittervec * jitterfactor
        item = linproj.ScatterPlotItem(
            x=coords[:, 0],
            y=coords[:, 1],
            brush=brushdata,
            pen=pendata,
            symbols=shapedata,
            size=sizedata,
            data=numpy.flatnonzero(valid),
            antialias=True,
        )

        self.plot.addItem(item)
        self.plot.setRange(QtCore.QRectF(-1.05, -1.05, 2.1, 2.1))

        # minimum visible anchor radius
        minradius = self.min_anchor_radius / 100 + 1e-5
        axisitems = []
        for anchor, var in zip(anchors, self.data.domain.attributes):
            axitem = AxisItem(
                line=QtCore.QLineF(0, 0, *anchor), text=var.name,)
            axitem.setVisible(numpy.linalg.norm(anchor) > minradius)
            axitem.setPen(pg.mkPen((100, 100, 100)))
            axitem.setArrowVisible(False)
            self.plot.addItem(axitem)
            axisitems.append(axitem)

        hidecircle = QtGui.QGraphicsEllipseItem()
        hidecircle.setRect(
            QtCore.QRectF(-minradius, -minradius,
                          2 * minradius, 2 * minradius))

        _pen = QtGui.QPen(Qt.lightGray, 1)
        _pen.setCosmetic(True)
        hidecircle.setPen(_pen)

        self.plot.addItem(hidecircle)

        self.plotdata = namespace(
            validmask=valid,
            embedding_coords=EX,
            jittervec=jittervec,
            anchors=anchors,
            mainitem=item,
            axisitems=axisitems,
            hidecircle=hidecircle,
            basecolors=colors,
            brushdata=brushdata,
            pendata=pendata,
            shapedata=shapedata,
            sizedata=sizedata,
            labeldata=labeldata,
            labelitems=[],
            densityimage=None,
            X=X,
            Y=Y,
            selectionmask=numpy.zeros_like(valid, dtype=bool),
            subsetmask=None
        )
        self._update_legend()
        self._update_labels()
        self._update_density()

    def _update_color(self):
        if self.plotdata is None:
            return

        validmask = self.plotdata.validmask
        selectionmask = self.plotdata.selectionmask
        if self.attr_color is not None:
            colors = plotutils.color_data(self.data, self.attr_color)[validmask]
        else:
            colors = numpy.array([[192, 192, 192]])
            colors = numpy.tile(colors, (self.plotdata.X.shape[0], 1))

        selectedmask = selectionmask[validmask]
        pointstyle = numpy.where(
            selectedmask, plotutils.Selected, plotutils.NoFlags)

        pendata = plotutils.pen_data(colors * 0.8, pointstyle)
        colors = numpy.hstack(
            [colors, numpy.full((colors.shape[0], 1), float(self.alpha_value))])

        brushdata = plotutils.brush_data(colors, )
        if self.plotdata.subsetmask is not None:
            subsetmask = self.plotdata.subsetmask[validmask]
            brushdata[~subsetmask] = QtGui.QBrush(Qt.NoBrush)

        self.plotdata.pendata = pendata
        self.plotdata.brushdata = brushdata
        self.plotdata.mainitem.setPen(pendata)
        self.plotdata.mainitem.setBrush(brushdata)

        self._update_legend()

    def _update_shape(self):
        if self.plotdata is None:
            return
        validmask = self.plotdata.validmask
        shapedata = plotutils.shape_data(self.data, self.attr_shape)
        shapedata = shapedata[validmask]
        self.plotdata.shapedata = shapedata
        self.plotdata.mainitem.setSymbol(shapedata)
        self._update_legend()

    def _update_size(self):
        if self.plotdata is None:
            return
        validmask = self.plotdata.validmask

        sizedata = size_data(
            self.data, self.attr_size, pointsize=self.point_width)[validmask]
        self.plotdata.sizedata = sizedata
        self.plotdata.mainitem.setSize(sizedata)

    def _update_labels(self):
        if self.plotdata is None:
            return

        if self.attr_label is not None:
            labeldata = plotutils.column_data(
                self.data, self.attr_label, self.plotdata.validmask)
            labeldata = [self.attr_label.str_val(val) for val in labeldata]
        else:
            labeldata = None

        if self.plotdata.labelitems:
            for item in self.plotdata.labelitems:
                item.setParentItem(None)
                self.plot.removeItem(item)
            self.plotdata.labelitems = []

        if labeldata is not None:
            coords = self.plotdata.embedding_coords
            coords = coords / numpy.max(numpy.linalg.norm(coords, axis=1))
            for (x, y), text in zip(coords, labeldata):
                item = pg.TextItem(text, anchor=(0.5, 0), color=0.0)
                item.setPos(x, y)
                self.plot.addItem(item)
                self.plotdata.labelitems.append(item)

    update_point_size = update_sizes = _update_size
    update_alpha_value = update_colors = _update_color
    update_shapes = _update_shape
    update_labels = _update_labels

    def _update_legend(self):
        self.legend.clear()
        if self.plotdata is None:
            return

        legend_data = plotutils.legend_data(
            self.attr_color, self.attr_shape)
        self.legend.clear()
        self.legend.setVisible(bool(legend_data))

        for color, symbol, name in legend_data:
            self.legend.addItem(
                linproj.ScatterPlotItem(
                    pen=color, brush=color, symbol=symbol, size=10),
                name)

    def _update_density(self):
        if self.plotdata is None:
            return

        if self.plotdata.densityimage is not None:
            self.plot.removeItem(self.plotdata.densityimage)
            self.plotdata.densityimage = None

        if self.data.domain.has_discrete_class and self.class_density:
            coords = self.plotdata.embedding_coords
            radius = numpy.linalg.norm(coords, axis=1).max()
            coords = coords / radius
            xmin = ymin = -1.05
            xmax = ymax = 1.05
            xdata, ydata = coords.T
            colors = plotutils.color_data(
                self.data, self.data.domain.class_var)[self.plotdata.validmask]
            imgitem = classdensity.class_density_image(
                xmin, xmax, ymin, ymax, 256, xdata, ydata, colors)
            self.plot.addItem(imgitem)
            self.plotdata.densityimage = imgitem

    def _start(self):
        """
        Start the projection optimization.
        """
        if self.plotdata is None:
            return

        X, Y = self.plotdata.X, self.plotdata.Y
        anchors = self.plotdata.anchors
        _, p = OWFreeViz.ForceLaw[self.force_law]

        def update_freeviz(maxiter, itersteps, initial):
            done = False
            anchors = initial
            while not done:
                res = freeviz(X, Y, scale=False, center=False,
                              initial=anchors, p=p,
                              maxiter=min(itersteps, maxiter))
                EX, anchors_new = res[:2]
                yield res[:2]

                if numpy.all(numpy.isclose(anchors, anchors_new,
                                           rtol=1e-5, atol=1e-4)):
                    return

                maxiter = maxiter - itersteps
                if maxiter <= 0:
                    return
                anchors = anchors_new

        _, interval = self.ReplotIntervals[self.replot_interval]
        if interval == -1:
            interval = self.maxiter

        self._loop.setCoroutine(
            update_freeviz(self.maxiter, interval, anchors))
        self.start_button.setText("Stop")
        self.progressBarInit(processEvents=False)
        self.setBlocking(True)
        self.setStatusMessage("Optimizing")

    def __reset_initialization(self):
        """
        Reset the current 'anchor' initialization, and restart the
        optimization if necessary.
        """
        running = self._loop.isRunning()

        if running:
            self._loop.cancel()

        if self.data is not None:
            self._clear_plot()
            self._setup()

        if running:
            self._start()

    def __reset_update_interval(self):
        running = self._loop.isRunning()
        if running:
            self._loop.cancel()
            if self.data is not None:
                self._start()

    def _update_xy(self):
        # Update the plotted embedding coordinates
        if self.plotdata is None:
            return

        item = self.plotdata.mainitem
        coords = self.plotdata.embedding_coords
        radius = numpy.max(numpy.linalg.norm(coords, axis=1))
        coords = coords / radius
        if self.jitter > 0:
            _, factor = self.JitterAmount[self.jitter]
            coords = coords + self.plotdata.jittervec * factor

        item.setData(x=coords[:, 0], y=coords[:, 1],
                     brush=self.plotdata.brushdata,
                     pen=self.plotdata.pendata,
                     size=self.plotdata.sizedata,
                     symbol=self.plotdata.shapedata,
                     data=numpy.flatnonzero(self.plotdata.validmask)
                     )

        for anchor, item in zip(self.plotdata.anchors,
                                self.plotdata.axisitems):
            item.setLine(QtCore.QLineF(0, 0, *anchor))

        for (x, y), item in zip(coords, self.plotdata.labelitems):
            item.setPos(x, y)

    def _update_anchor_visibility(self):
        # Update the anchor/axes visibility
        if self.plotdata is None:
            return

        minradius = self.min_anchor_radius / 100 + 1e-5
        for anchor, item in zip(self.plotdata.anchors,
                                self.plotdata.axisitems):
            item.setVisible(numpy.linalg.norm(anchor) > minradius)
        self.plotdata.hidecircle.setRect(
            QtCore.QRectF(-minradius, -minradius,
                          2 * minradius, 2 * minradius))

    def __set_projection(self, res):
        # Set/update the projection matrix and coordinate embeddings
        assert self.plotdata is not None, "__set_projection call unexpected"
        _, increment = self.ReplotIntervals[self.replot_interval]
        increment = self.maxiter if increment == -1 else increment
        self.progressBarAdvance(
            increment * 100. / self.maxiter, processEvents=False)
        embedding_coords, projection = res
        self.plotdata.embedding_coords = embedding_coords
        self.plotdata.anchors = projection
        self._update_xy()
        self._update_anchor_visibility()
        self._update_density()

    def __freeviz_finished(self):
        # Projection optimization has finished
        self.start_button.setText("Optimize")
        self.setStatusMessage("")
        self.setBlocking(False)
        self.progressBarFinished(processEvents=False)
        self.commit()

    def __on_error(self, err):
        sys.excepthook(type(err), err, getattr(err, "__traceback__"))

    def __select_area(self, selectarea):
        """Select instances in the specified plot area."""
        if self.plotdata is None:
            return

        item = self.plotdata.mainitem

        if item is None:
            return

        indices = [spot.data()
                   for spot in item.points()
                   if selectarea.contains(spot.pos())]
        indices = numpy.array(indices, dtype=int)

        self.select(indices, QtGui.QApplication.keyboardModifiers())

    def select(self, indices, modifiers=Qt.NoModifier):
        """
        Select the instances specified by `indices`

        Parameters
        ----------
        indices : (N,) int ndarray
            Indices of instances to select.
        modifiers : Qt.KeyboardModifier
            Keyboard modifiers.
        """
        if self.plotdata is None:
            return

        current = self.plotdata.selectionmask

        if not modifiers & (Qt.ControlModifier | Qt.ShiftModifier |
                            Qt.AltModifier):
            # no modifiers -> clear current selection
            current = numpy.zeros_like(self.plotdata.validmask, dtype=bool)

        if modifiers & Qt.AltModifier:
            current[indices] = False
        elif modifiers & Qt.ControlModifier:
            current[indices] = ~current[indices]
        else:
            current[indices] = True
        self.plotdata.selectionmask = current
        self._update_color()
        self.commit()

    def commit(self):
        """
        Commit/send the widget output signals.
        """
        data = subset = components = None
        selectedindices = []
        if self.data is not None:
            coords = self.plotdata.embedding_coords
            valid = self.plotdata.validmask
            selection = self.plotdata.selectionmask
            selectedindices = numpy.flatnonzero(valid & selection)

            C1Var = Orange.data.ContinuousVariable(
                "Component1",
            )
            C2Var = Orange.data.ContinuousVariable(
                "Component2"
            )

            attributes = self.data.domain.attributes
            classes = self.data.domain.class_vars
            metas = self.data.domain.metas
            if self.embedding_domain_role == OWFreeViz.Attribute:
                attributes = attributes + (C1Var, C2Var)
            elif self.embedding_domain_role == OWFreeViz.Meta:
                metas = metas + (C1Var, C2Var)

            domain = Orange.data.Domain(attributes, classes, metas)
            data = self.data.from_table(domain, self.data)

            if self.embedding_domain_role == OWFreeViz.Attribute:
                data.X[valid, -2:] = coords
            elif self.embedding_domain_role == OWFreeViz.Meta:
                data.metas[valid, -2:] = coords

            if selectedindices.size:
                subset = data[selectedindices]

            compdomain = Orange.data.Domain(
                self.data.domain.attributes,
                metas=[Orange.data.StringVariable(name='component')])

            metas = numpy.array([["FreeViz 1"], ["FreeViz 2"]])
            components = Orange.data.Table(
                compdomain, self.plotdata.anchors.T,
                metas=metas)
            components.name = 'components'

        self.send("Selected Data", subset)
        self.send(ANNOTATED_DATA_SIGNAL_NAME,
                  create_annotated_table(data, selectedindices))
        self.send("Components", components)

    def sizeHint(self):
        # reimplemented
        return QtCore.QSize(900, 700)

    def eventFilter(self, recv, event):
        # reimplemented
        if event.type() == QtCore.QEvent.GraphicsSceneHelp and \
                recv is self.plot.scene():
            return self._tooltip(event)
        else:
            return super().eventFilter(recv, event)

    def _tooltip(self, event):
        # Handle a help event for the plot's scene
        if self.plotdata is None:
            return False

        item = self.plotdata.mainitem
        pos = item.mapFromScene(event.scenePos())
        points = item.pointsAt(pos)
        indices = [spot.data() for spot in points]
        if not indices:
            return False

        tooltip = format_tooltip(self.data, columns=..., rows=indices)
        QtGui.QToolTip.showText(event.screenPos(), tooltip, widget=self.plot)
        return True

    def send_report(self):
        self.report_plot()
        caption = report.render_items_vert((
            ("Colors", self.attr_color),
            ("Shape", self.attr_shape),
            ("Size", self.attr_size),
            ("Label", self.attr_label),
            ("Jittering", self.jitter > 0 and
             self.controls.jitter.currentText()),
        ))
        self.report_caption(caption)


def format_tooltip(table, columns, rows, maxattrs=5, maxrows=5):
    domain = table.domain

    if columns is ...:
        columns = domain.variables + domain.metas
    else:
        columns = [domain[col] for col in columns]

    def role(domain, var):
        if var in domain.attributes:
            return 0
        elif var in domain.class_vars:
            return 1
        elif var in domain.metas:
            return 2
        else:
            raise ValueError

    attrs, class_vars, metas = [], [], []
    for var in columns:
        [attrs, class_vars, metas][role(domain, var)].append(var)

    tooltip_lines = []
    for row_idx in rows[:maxrows]:
        row = table[row_idx]
        lines = ["Attributes:"]
        lines.extend('   {} = {}'.format(attr.name, row[attr])
                     for attr in attrs[:maxattrs])

        if len(attrs) > maxattrs:
            lines.append("   ... and {} others".format(len(attrs) - maxattrs))

        if class_vars:
            lines.append("Class:")
            lines.extend("   {} = {}".format(var.name, row[var])
                         for var in class_vars)

        if metas:
            lines.append("Metas:")
            lines.extend("   {} = {}".format(var.name, row[var])
                         for var in metas)

        tooltip_lines.append("\n".join(lines))

    if len(rows) > maxrows:
        tooltip_lines.append("... {} more".format(len(rows) - maxrows))
    text = "\n------------------\n".join(tooltip_lines)
    text = ('<span style="white-space:pre">{}</span>'
            .format(escape(text)))
    return text


def size_data(table, var, pointsize=3):
    if var is None:
        return numpy.full(len(table), pointsize, dtype=float)
    else:
        size_data, _ = table.get_column_view(var)
        cmin, cmax = numpy.nanmin(size_data), numpy.nanmax(size_data)
        if cmax - cmin > 0:
            size_data = (size_data - cmin) / (cmax - cmin)
        else:
            size_data = numpy.zeros(len(table))

        size_data = size_data * pointsize + 3
        size_data[numpy.isnan(size_data)] = 1
        return size_data


def main(argv=sys.argv):
    app = QtGui.QApplication(list(argv))
    argv = app.argv()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "zoo"
    data = Orange.data.Table(filename)
    subset = data[numpy.random.choice(len(data), 4)]
    w = OWFreeViz()
    w.show()
    w.raise_()
    w.set_data(data)
    w.set_data_subset(subset)
    w.handleNewSignals()
    app.exec_()
    w.set_data_subset(None)
    w.set_data(None)
    w.handleNewSignals()
    w.saveSettings()
    return 0

if __name__ == "__main__":
    sys.exit(main())
