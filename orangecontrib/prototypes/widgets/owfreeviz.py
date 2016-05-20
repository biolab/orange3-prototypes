import sys
import enum
from xml.sax.saxutils import escape
from types import SimpleNamespace as namespace

import pkg_resources

import numpy

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt, QObject, QEvent, QCoreApplication
from PyQt4.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

import pyqtgraph as pg

import Orange.data
import Orange.projection

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import colorpalette, itemmodels, classdensity
from Orange.widgets.visualize import owlinearprojection as linproj
from Orange.widgets.unsupervised.owmds import mdsplotutils as plotutils

from ..projection.freeviz import freeviz


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


class AxisItem(pg.GraphicsObject):
    def __init__(self, parent=None, line=None, label=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setFlag(pg.GraphicsObject.ItemHasNoContents)

        if line is None:
            line = QtCore.QLineF(0, 0, 1, 0)

        self._spine = QtGui.QGraphicsLineItem(line, self)
        angle = line.angle()

        self._arrow = pg.ArrowItem(parent=self, angle=0)
        self._arrow.setPos(self._spine.line().p2())
        self._arrow.setRotation(angle)

        self._label = pg.TextItem(text=label, color=(10, 10, 10))
        self._label.setParentItem(self)
        self._label.setPos(self._spine.line().p2())

    def setLabel(self, label):
        if label != self._label.textItem.toPlainText():
            self._label.setText(label)

    def setLine(self, *line):
        line = QtCore.QLineF(*line)
        if line != self._spine.line():
            self._spine.setLine(line)
            self.__updateLayout()

    def setPen(self, pen):
        self._spine.setPen(pen)

    def setArrowVisible(self, visible):
        self._arrow.setVisible(visible)

    def paint(self, painter, option, widget):
        pass

    def boundingRect(self):
        return QtCore.QRectF()

    def viewTransformChanged(self):
        self.__updateLayout()

    def __updateLayout(self):
        T = self.sceneTransform()
        if T is None:
            T = QtGui.QTransform()

        # map the axis spine to scene coord. system (it should suffice to
        # map up to PlotItem?)
        viewbox_line = T.map(self._spine.line())
        angle = viewbox_line.angle()
        assert not numpy.isnan(angle)
        # note in Qt the y axis is inverted (90 degree angle 'points' down)
        left_quad = 270 < angle <= 360 or -0.0 <= angle < 90

        # position the text label along the viewbox_line
        label_pos = self._spine.line().pointAt(0.90)

        if left_quad:
            # Anchor the text under the axis spine
            anchor = (0.5, -0.1)
        else:
            # Anchor the text over the axis spine
            anchor = (0.5, 1.1)

        self._label.setPos(label_pos)
        self._label.anchor = pg.Point(*anchor)
        self._label.updateText()
        self._label.setRotation(-angle if left_quad else 180 - angle)

        self._arrow.setPos(self._spine.line().p2())
        self._arrow.setRotation(180 - angle)


def make_pen(color, width=1.0, style=Qt.SolidLine, cap=Qt.SquareCap,
             join=Qt.BevelJoin, cosmetic=True):
    pen = QtGui.QPen(color, width, style=style, cap=cap, join=join)
    pen.setCosmetic(cosmetic)
    return pen


class OWFreeViz(widget.OWWidget):
    name = "FreeViz"
    description = "FreeViz Visualization"
    icon = "icons/LinearProjection.svg"
    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Data Subset", Orange.data.Table, "set_data_subset")]
    outputs = [("Data", Orange.data.Table, widget.Default),
               ("Selected Data", Orange.data.Table),
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

    color_var = settings.ContextSetting("", exclude_metas=False)
    shape_var = settings.ContextSetting("", exclude_metas=False)
    size_var = settings.ContextSetting("", exclude_metas=False)
    label_var = settings.ContextSetting("", exclude_metas=False)

    opacity = settings.Setting(255)
    point_size = settings.Setting(5)
    jitter = settings.Setting(0)
    class_density = settings.Setting(False)

    def __init__(self):
        super().__init__()

        self.data = None
        self.data_subset = None
        self.plotdata = None

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

        self.color_varmodel = itemmodels.VariableListModel(parent=self)
        self.shape_varmodel = itemmodels.VariableListModel(parent=self)
        self.size_varmodel = itemmodels.VariableListModel(parent=self)
        self.label_varmodel = itemmodels.VariableListModel(parent=self)

        box = gui.widgetBox(self.controlArea, "Plot")
        form = QtGui.QFormLayout(
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QtGui.QFormLayout.AllNonFixedFieldsGrow,
            spacing=8,
        )
        box.layout().addLayout(form)
        color_cb = gui.comboBox(
            box, self, "color_var", sendSelectedValue=True,
            emptyString="(Same color)", contentsLength=10,
            callback=self._update_color)

        color_cb.setModel(self.color_varmodel)
        form.addRow("Color", color_cb)
        opacity_slider = gui.hSlider(
            box, self, "opacity", minValue=50, maxValue=255, ticks=True,
            createLabel=False, callback=self._update_color)
        opacity_slider.setTickInterval(0)
        opacity_slider.setPageStep(10)
        form.addRow("Opacity", opacity_slider)

        shape_cb = gui.comboBox(
            box, self, "shape_var", contentsLength=10, sendSelectedValue=True,
            emptyString="(Same shape)", callback=self._update_shape)
        shape_cb.setModel(self.shape_varmodel)
        form.addRow("Shape", shape_cb)

        size_cb = gui.comboBox(
            box, self, "size_var", contentsLength=10, sendSelectedValue=True,
            emptyString="(Same size)", callback=self._update_size)
        size_cb.setModel(self.size_varmodel)
        form.addRow("Size", size_cb)
        size_slider = gui.hSlider(
            box, self, "point_size", minValue=3, maxValue=20, ticks=True,
            createLabel=False, callback=self._update_size)
        form.addRow(None, size_slider)

        label_cb = gui.comboBox(
            box, self, "label_var", contentsLength=10, sendSelectedValue=True,
            emptyString="(No labels)", callback=self._update_labels)
        label_cb.setModel(self.label_varmodel)
        form.addRow("Label", label_cb)

        form.addRow(
            "Jitter",
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
            step=5, label="Hide radius", createLabel=False, ticks=True,
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
        hlayout.addSpacing(2)
        hlayout.addWidget(toolbox.button(PlotToolBox.ZoomReset))
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

        self.plot = pg.PlotWidget(enableMouse=False, enableMenu=False)
        self.plot.setFrameStyle(QtGui.QFrame.StyledPanel)
        self.plot.plotItem.hideAxis("bottom")
        self.plot.plotItem.hideAxis("left")
        self.plot.plotItem.hideButtons()
        self.plot.setAspectLocked(True)
        self.plot.scene().installEventFilter(self)

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

        self.color_varmodel[:] = ["(Same color)"]
        self.shape_varmodel[:] = ["(Same shape)"]
        self.size_varmodel[:] = ["(Same size)"]
        self.label_varmodel[:] = ["(No labels)"]
        self.color_var = self.shape_var = self.size_var = self.label_var = ""

    def set_data(self, data):
        """
        Set the input dataset.
        """
        self.closeContext()
        self.clear()
        error_msg = ""
        if data is not None:
            if data.domain.class_var is None:
                error_msg = "Need a class variable"
                data = None
            elif data.domain.class_var.is_discrete and \
                    len(data.domain.class_var.values) < 2:
                error_msg = "Needs discrete class variable with at" \
                            " lest 2 values"
                data = None

        self.data = data
        self.error(0, error_msg)
        if data is not None:
            separator = itemmodels.VariableListModel.Separator
            domain = data.domain
            colorvars = ["(Same color)"] + list(domain)
            colorvars_meta = [var for var in domain.metas
                              if var.is_primitive()]
            if colorvars_meta:
                colorvars += [separator] + colorvars_meta
            self.color_varmodel[:] = colorvars
            self.color_var = domain.class_var.name

            def is_discrete(var): return var.is_discrete
            def is_continuous(var): return var.is_continuous
            def is_string(var): return var.is_string
            def filter_(func, iterable): return list(filter(func, iterable))
            maxsymbols = len(linproj.ScatterPlotItem.Symbols) - 1
            def can_be_shape(var):
                return is_discrete(var) and len(var.values) < maxsymbols

            shapevars = ["(Same shape)"] + filter_(can_be_shape, domain)
            shapevars_meta = filter_(can_be_shape, domain.metas)
            if shapevars_meta:
                shapevars += [separator] + shapevars_meta
            self.shape_varmodel[:] = shapevars

            sizevars = ["(Same size)"] + filter_(is_continuous, domain)
            sizevars_meta = filter_(is_continuous, domain.metas)
            if sizevars_meta:
                sizevars += [separator] + sizevars_meta
            self.size_varmodel[:] = sizevars

            labelvars = ["(No labels)"]
            labelvars_meta = filter_(is_string, domain.metas)
            if labelvars_meta:
                labelvars += [separator] + labelvars_meta

            self.label_varmodel[:] = labelvars

            self.class_density_cb.setEnabled(domain.has_discrete_class)
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

        colorvar = self._color_var()
        shapevar = self._shape_var()
        sizevar = self._size_var()
        labelvar = self._label_var()

        if colorvar is not None:
            colors = plotutils.color_data(self.data, colorvar)[valid]
        else:
            colors = numpy.array([[192, 192, 192]])
            colors = numpy.tile(colors, (X.shape[0], 1))

        pendata = plotutils.pen_data(colors * 0.8)
        colors = numpy.hstack(
            [colors, numpy.full((colors.shape[0], 1), float(self.opacity))])
        brushdata = plotutils.brush_data(colors)

        shapedata = plotutils.shape_data(self.data, shapevar)[valid]
        sizedata = size_data(
            self.data, sizevar, pointsize=self.point_size)[valid]
        if labelvar is not None:
            labeldata = plotutils.column_data(self.data, labelvar, valid)
            labeldata = [labelvar.str_val(val) for val in labeldata]
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
                line=QtCore.QLineF(0, 0, *anchor), label=var.name,)
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

    def _color_var(self):
        if self.color_var != "":
            return self.data.domain[self.color_var]
        else:
            return None

    def _update_color(self):
        if self.plotdata is None:
            return

        colorvar = self._color_var()
        validmask = self.plotdata.validmask
        selectionmask = self.plotdata.selectionmask
        if colorvar is not None:
            colors = plotutils.color_data(self.data, colorvar)[validmask]
        else:
            colors = numpy.array([[192, 192, 192]])
            colors = numpy.tile(colors, (self.plotdata.X.shape[0], 1))

        selectedmask = selectionmask[validmask]
        pointstyle = numpy.where(
            selectedmask, plotutils.Selected, plotutils.NoFlags)

        pendata = plotutils.pen_data(colors * 0.8, pointstyle)
        colors = numpy.hstack(
            [colors, numpy.full((colors.shape[0], 1), float(self.opacity))])

        brushdata = plotutils.brush_data(colors, )
        if self.plotdata.subsetmask is not None:
            subsetmask = self.plotdata.subsetmask[validmask]
            brushdata[~subsetmask] = QtGui.QBrush(Qt.NoBrush)

        self.plotdata.pendata = pendata
        self.plotdata.brushdata = brushdata
        self.plotdata.mainitem.setPen(pendata)
        self.plotdata.mainitem.setBrush(brushdata)

        self._update_legend()

    def _shape_var(self):
        if self.shape_var != "":
            return self.data.domain[self.shape_var]
        else:
            return None

    def _update_shape(self):
        if self.plotdata is None:
            return
        shapevar = self._shape_var()
        validmask = self.plotdata.validmask
        shapedata = plotutils.shape_data(self.data, shapevar)
        shapedata = shapedata[validmask]
        self.plotdata.shapedata = shapedata
        self.plotdata.mainitem.setSymbol(shapedata)
        self._update_legend()

    def _size_var(self):
        if self.size_var != "":
            return self.data.domain[self.size_var]
        else:
            return None

    def _update_size(self):
        if self.plotdata is None:
            return
        sizevar = self._size_var()
        validmask = self.plotdata.validmask

        sizedata = size_data(
            self.data, sizevar, pointsize=self.point_size)[validmask]
        self.plotdata.sizedata = sizedata
        self.plotdata.mainitem.setSize(sizedata)

    def _label_var(self):
        if self.label_var != "":
            return self.data.domain[self.label_var]
        else:
            return None

    def _update_labels(self):
        if self.plotdata is None:
            return
        labelvar = self._label_var()

        if labelvar is not None:
            labeldata = plotutils.column_data(
                self.data, labelvar, self.plotdata.validmask)
            labeldata = [labelvar.str_val(val) for val in labeldata]
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

    def _update_legend(self):
        self.legend.clear()
        if self.plotdata is None:
            return

        legend_data = plotutils.legend_data(
            self._color_var(), self._shape_var())
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

        self.send("Data", data)
        self.send("Selected Data", subset)
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
