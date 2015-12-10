import sys
from types import SimpleNamespace as namespace

import pkg_resources

import numpy

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt, QObject, QEvent, QCoreApplication
from PyQt4.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

import pyqtgraph as pg

import Orange.data

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import colorpalette
from Orange.widgets.visualize import owlinearprojection as linproj


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
            QCoreApplication.postEvent(
                self, QEvent(AsyncUpdateLoop.Next), - (2 ** 31 - 1))

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

    def customEvent(self, event):
        if event.type() == AsyncUpdateLoop.Next:
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

import enum


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


class OWFreeViz(widget.OWWidget):
    name = "FreeViz"
    description = "FreeViz Visualization"
    icon = "icons/LinearProjection.svg"
    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Data", Orange.data.Table, widget.Default),
               ("Selected Data", Orange.data.Table),
               ("Components", Orange.data.Table)]

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
    #: Output coordinate embedding domain role
    NoCoords, Attribute, Meta = 0, 1, 2

    force_law = settings.Setting(0)
    maxiter = settings.Setting(300)
    replot_interval = settings.Setting(3)
    initialization = settings.Setting(Circular)
    min_anchor_radius = settings.Setting(0)
    embedding_domain_role = settings.Setting(Meta)
    autocommit = settings.Setting(True)

    def __init__(self):
        super().__init__()

        self.data = None
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

        box = gui.widgetBox(self.controlArea, "Hide anchors")
        gui.hSlider(box, self, "min_anchor_radius", minValue=0, maxValue=100,
                    step=5, ticks=10,
                    label="Hide radius", createLabel=False,
                    callback=self.__update_anchor_visibility)

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
                     items=["Origninal features only",
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
        self.projection = None
        self._clear_plot()
        self._loop.cancel()

    def set_data(self, data):
        """
        Set the input dataset.
        """
        self.clear()
        error_msg = ""
        if data is not None:
            if data.domain.class_var is None or \
                    not data.domain.class_var.is_discrete or \
                    len(data.domain.class_var.values) < 2:
                error_msg = "Needs discrete class variable"
                data = None

        self.data = data
        self.error(0, error_msg)

    def handleNewSignals(self):
        """Reimplemented."""
        if self.data is not None:
            self._setup()
            self._start()

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

        X = (X - numpy.mean(X, axis=0))
        span = numpy.ptp(X, axis=0)
        X[:, span > 0] /= span[span > 0].reshape(1, -1)

        if self.initialization == OWFreeViz.Circular:
            anchors = linproj.linproj.defaultaxes(X.shape[1]).T
        else:
            anchors = numpy.random.random((X.shape[1], 2)) * 2 - 1

        EX = numpy.dot(X, anchors)

        colgen = colorpalette.ColorPaletteGenerator(
            len(self.data.domain.class_var.values))
        colors = numpy.array(colgen[Y], dtype=object)

        rad = numpy.max(numpy.linalg.norm(EX, axis=1))

        item = linproj.ScatterPlotItem(
            x=EX[:, 0] / rad,
            y=EX[:, 1] / rad,
            brush=colors,
            pen=QtGui.QPen(Qt.NoPen),
            data=numpy.flatnonzero(valid),
            antialias=True,
        )

        self.plot.addItem(item)
        self.plot.setRange(QtCore.QRectF(-1.05, -1.05, 2.1, 2.1))

        for i, name in enumerate(self.data.domain.class_var.values):
            color = colgen[i]
            self.legend.addItem(
                linproj.ScatterPlotItem(pen=color, brush=color, size=10),
                name,
            )
        self.legend.show()

        # minimum visible anchor radius
        minradius = self.min_anchor_radius / 100
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
            anchors=anchors,
            mainitem=item,
            axisitems=axisitems,
            hidecircle=hidecircle,
            brushdata=colors,
            X=X,
            Y=Y,
            selectionmask=numpy.zeros_like(valid, dtype=bool)
        )

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
                res = freeviz(X, Y, initial=anchors, p=p,
                              maxiter=min(itersteps, maxiter))
                EX, anchors_new = res
                yield res

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

    def __update_xy(self):
        # Update the plotted embedding coordinates
        if self.plotdata is None:
            return

        item = self.plotdata.mainitem
        coords = self.plotdata.embedding_coords
        rad = numpy.max(numpy.linalg.norm(coords, axis=1))

        selection = self.plotdata.selectionmask[self.plotdata.validmask]
        selection = numpy.flatnonzero(selection)
        brushdata = self.plotdata.brushdata
        if selection.size:
            pendata = numpy.full(len(brushdata), QtGui.QPen(Qt.NoPen),
                                 dtype=object)
            pendata[selection] = pg.mkPen((150, 150, 150), width=3)

        else:
            pendata = None
        item.setData(x=coords[:, 0] / rad, y=coords[:, 1] / rad,
                     brush=brushdata,
                     pen=pendata,
                     data=numpy.flatnonzero(self.plotdata.validmask)
                     )
        for anchor, item in zip(self.plotdata.anchors,
                                self.plotdata.axisitems):
            item.setLine(QtCore.QLineF(0, 0, *anchor))

        self.__update_anchor_visibility()

    def __update_anchor_visibility(self):
        # Update the anchor/axes visibility
        if self.plotdata is None:
            return

        minradius = self.min_anchor_radius / 100
        for anchor, item in zip(self.plotdata.anchors, self.plotdata.axisitems):
            item.setVisible(numpy.linalg.norm(anchor) > minradius)
        self.plotdata.hidecircle.setRect(
            QtCore.QRectF(-minradius, -minradius,
                          2 * minradius, 2 * minradius))

    def __set_projection(self, res):
        # Set/update the projection matrix and coordinate embeddings
        assert self.plotdata is not None, "__set_projection called out of turn"
        self.progressBarAdvance(100. / self.maxiter, processEvents=False)
        embedding_coords, projection = res
        self.plotdata.embedding_coords = embedding_coords
        self.plotdata.anchors = projection
        self.__update_xy()

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
        self.__update_xy()

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

            metas = numpy.array([["Component_1"], ["Component_2"]])
            components = Orange.data.Table(
                compdomain, self.plotdata.anchors.T,
                metas=metas)
            components.name = 'components'

        self.send("Data", data)
        self.send("Selected Data", subset)
        self.send("Components", components)

    def sizeHint(self):
        return QtCore.QSize(900, 700)


import numpy
import scipy.spatial


def freeviz_gradient(X, y, embeddings, p=1, weights=None):
    """
    Return the gradient for the FreeViz [1]_ projection.

    Parameters
    ----------
    X : (N, P) ndarray
        The data instance coordinates
    y : (N, 1) ndarray
        The instance target/class values
    embeddings : (N, dim) ndarray
        The current FreeViz point embeddings.
    p : number
        The 'power' of the force (p == 1 is [inverse] linear,  p == 2 is
        [inverse] squared, ...)
    weights : (N, ) ndarray, optional
        Optional vector of sample weights.

    Returns
    -------
    G : (P, dim) ndarray
        The projection gradient.

    .. [1] Janez Demsar, Gregor Leban, Blaz Zupan
           FreeViz - An Intelligent Visualization Approach for Class-Labeled
           Multidimensional Data Sets, Proceedings of IDAMAP 2005, Edinburgh.
    """
    X = numpy.asarray(X)
    y = numpy.asarray(y)
    embeddings = numpy.asarray(embeddings)

    if weights is not None:
        weights = numpy.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("weights.ndim != 1 ({})".format(weights.ndim))

    N, P = X.shape
    _, dim = embeddings.shape

    if not N == embeddings.shape[0]:
        raise ValueError("X and embeddings must have the same length ({}!={})"
                         .format(X.shape[0] != embeddings.shape[0]))

    if weights is not None and X.shape[0] != weights.shape[0]:
        raise ValueError("X.shape[0] != weights.shape[0] ({}!={})"
                         .format(X.shape[0], weights.shape[0]))

    # All pairwise distances between point embeddings
    D = scipy.spatial.distance.pdist(embeddings, "euclidean")
    D = scipy.spatial.distance.squareform(D, checks=False)
    # All pairwise vector differences between embeddings
    DeltaP = embeddings[:, numpy.newaxis, :] - embeddings[numpy.newaxis, :, :]
    assert DeltaP.shape == (N, N, dim)
    assert numpy.all(numpy.isclose(DeltaP[0, 1], embeddings[0] - embeddings[1]))
    assert numpy.all(numpy.isclose(DeltaP[1, 0], -DeltaP[0, 1]))
    # D = numpy.linalg.norm(DeltaP, axis=2)

    nonzero_mask = D >= numpy.finfo(float).eps * 100
    # Normalize to unit direction vectors
    DeltaP /= numpy.where(nonzero_mask, D, 1)[:, :, numpy.newaxis]

    # Handle attractive force
    if p == 1:
        F = -D
    else:
        F = -(D ** p)

    mask = numpy.c_[y] != numpy.r_[y]
    mask &= nonzero_mask

    # Handle repulsive force
    if p == 1:
        F[mask] = 1. / D[mask]
    else:
        F[mask] = 1. / (D[mask] ** p)

    if weights is not None:
        # Multiply in the pairwise weights
        F *= weights.reshape((1, N))
        F *= weights.reshape((N, 1))

    # multiply unit direction vectors with the force magnitude
    F = DeltaP * F[:, :, numpy.newaxis]
    assert F.shape == (N, N, dim)
    # sum all the forces acting on a particle
    F = numpy.sum(F, axis=0)
    assert F.shape == (N, dim)
    # Transfer forces to the 'anchors'
    # (P, dim) array of gradients
    G = X.T.dot(F)
    assert G.shape == (P, dim)
    return G


def _rotate(A, ):
    # Rotate (2D) projection A so the first anchor is aligned with
    # vector (1, 0)
    phi = numpy.arctan2(A[0, 1], A[0, 0])
    R = [[numpy.cos(-phi), numpy.sin(-phi)],
         [-numpy.sin(-phi), numpy.cos(-phi)]]
    return numpy.dot(A, R)


def freeviz(X, y, weights=None, dim=2, p=1, initial=None, maxiter=500,
            lambda_=0.1, atol=1e-5):
    """
    FreeViz

    Compute a linear lower dimensional projection to optimize separation
    between classes ([1]_).

    Parameters
    ----------
    X : (N, P) ndarray
        The input data instances
    y : (N, ) ndarray
        The instance class labels
    weights : (N, ) ndarray, optional
        Instance weights
    dim : int
        The dimension of the projected points
    p : number
        The force 'power', e.g. if p=1 the attractive/repulsive forces
        follow linear/inverse linear law, for p=2 the forces follow
        square/inverse square law, ...
    initial : (P, dim) ndarray, optional
        Initial projection matrix
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    embeddings : (N, dim) ndarray
        The point projections (`= X.dot(P)`)
    P : (P, dim)
        The projection matrix.

    .. [1] Janez Demsar, Gregor Leban, Blaz Zupan
           FreeViz - An Intelligent Visualization Approach for Class-Labeled
           Multidimensional Data Sets, Proceedings of IDAMAP 2005, Edinburgh.
    """
    X = numpy.asarray(X)
    y = numpy.asarray(y)
    N, P = X.shape
    _N, = y.shape
    if N != _N:
        raise ValueError("X and y must have the same length")

    if weights is not None:
        weights = numpy.asarray(weights)

    if initial is not None:
        initial = numpy.asarray(initial)
        if initial.ndim != 2 or initial.shape != (P, dim):
            raise ValueError
    else:
        initial = numpy.random.random((P, dim)) * 2 - 1

    A = initial
    embeddings = numpy.dot(X, A)

    step_i = 0
    while step_i < maxiter:
        G = freeviz_gradient(X, y, embeddings, p=p, weights=weights)

        # Scale the changes (the largest anchor move is lambda * radius)
        maxg = numpy.max(numpy.linalg.norm(G, axis=1))
        maxr = numpy.max(numpy.linalg.norm(A, axis=1))

        step = lambda_ * maxr / maxg
        Anew = A - step * G

        # Center anchors (?? This does not seem right; it changes the
        # projection axes direction somewhat arbitrarily)
        Anew = Anew - numpy.mean(Anew, axis=0)

        # Scale (so that the largest radius is 1)
        maxr = numpy.max(numpy.linalg.norm(Anew, axis=1))
        if maxr >= 0.001:
            Anew /= maxr

        if dim == 2:
            Anew = _rotate(Anew)

        change = numpy.linalg.norm(Anew - A, axis=1)
        if numpy.all(numpy.isclose(change, 0, atol=atol)):
            break

        A = Anew
        embeddings = numpy.dot(X, A)
        step_i = step_i + 1

    return embeddings, A


def main(argv=sys.argv):
    app = QtGui.QApplication(list(argv))
    argv = app.argv()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "zoo"
    data = Orange.data.Table(filename)
    w = OWFreeViz()
    w.show()
    w.raise_()
    w.set_data(data)
    w.handleNewSignals()
    app.exec_()
    w.set_data(None)
    w.handleNewSignals()
    return 0

if __name__ == "__main__":
    sys.exit(main())
