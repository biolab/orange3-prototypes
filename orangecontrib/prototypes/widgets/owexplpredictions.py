import sys
import copy
import logging
import concurrent.futures
from functools import partial
import time
from enum import IntEnum

from AnyQt.QtWidgets import (
    QApplication, QFormLayout, QTableView,  QSplitter, QHeaderView,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsSimpleTextItem,
    QSizePolicy)
from AnyQt.QtCore import (
    Qt, QThread, pyqtSlot, QMetaObject, Q_ARG, QAbstractProxyModel,
    QRectF, QSize)
from AnyQt.QtGui import QPen, QColor, QBrush, QPainter, QFont
import numpy as np
from numpy.random import RandomState
import scipy.stats as st

import Orange
import Orange. evaluation
from Orange.widgets.widget import OWWidget, Output, Input, Msg
from Orange.widgets import gui, widget, settings
from Orange.widgets.utils.itemmodels import TableModel
from Orange.widgets.utils.sql import check_sql_input
from Orange.base import Model
from Orange.data import (
    DiscreteVariable, ContinuousVariable, StringVariable, Domain,
    Table)
from Orange.widgets.utils.concurrent import (
    ThreadExecutor, FutureWatcher, methodinvoke
)


class SortBy(IntEnum):
    NO_SORTING, BY_NAME, ABSOLUTE, POSITIVE, NEGATIVE = 0, 1, 2, 3, 4

    @staticmethod
    def items():
        return["No sorting", "By name", "Absolute contribution",
               "Positive contribution", "Negative contribution"]


class Task:

    future = ...
    watcher = ...
    canceled = False

    def cancel(self):
        self.canceled = True
        self.future.cancel()
        concurrent.futures.wait([self.future])


class ExplainPredictions:
    """
    Class used to explain individual predictions by determining the importance of attribute values.
    All interactions between atributes are accounted for by calculating Shapely value.

    Parameters
    ----------
    data : Orange.data.Table
        table with dataset
    model: Orange.base.Model
        model to be used for prediction
    error: float
        desired error 
    p_val : float
        p value of error
    batch_size : int
        size of batch to be used in making predictions, bigger batch size speeds up the calculations and improves estimations of variance
    max_iter : int
        maximum number of iterations per attribute
    min_iter : int
        minimum number of iterations per attiribute
    seed : int
        seed for the numpy.random generator, default is 42

    Returns:
    -------
    class_value: float
        either index of predicted class or predicted value
    table: Orange.data.Table
        table containing atributes and corresponding contributions

    """

    def __init__(self, data, model, p_val=0.05, error=0.05, batch_size=500, max_iter=10000000, min_iter=1000, seed=42):
        self.model = model
        self.data = data
        self.p_val = p_val
        self.error = error
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.atr_names = [var.name for var in data.domain.attributes]
        self.seed = seed
        """variables, saved for possible restart"""
        self.saved = False
        self.steps = None
        self.mu = None
        self.M2 = None
        self.expl = None
        self.var = None
        self.iterations_reached = None

    def tile_instance(self, instance):
        tiled_x = np.tile(instance._x, (self.batch_size, 1))
        tiled_metas = np.tile(instance._metas, (self.batch_size, 1))
        tiled_y = np.tile(instance._y, (self.batch_size, 1))
        return Table.from_numpy(instance.domain, tiled_x, tiled_y, tiled_metas)

    def init_arrays(self, no_atr):
        if not self.saved:
            self.saved = True
            self.steps = np.zeros((1, no_atr), dtype=float)
            self.mu = np.zeros((1, no_atr), dtype=float)
            self.M2 = np.zeros((1, no_atr), dtype=float)
            self.expl = np.zeros((1, no_atr), dtype=float)
            self.var = np.ones((1, no_atr), dtype=float)
            self.iterations_reached = np.zeros((1, no_atr))
        else:
            self.iterations_reached = np.copy(self.steps)

    def get_atr_column(self, instance):
        """somewhat ugly fix for printing values in column"""
        attr_values = []
        var = instance.domain.attributes
        for idx in range(len(var)):
            if var[idx].is_discrete:
                attr_values.append(str(var[idx].str_val(instance._x[idx])))
            else:
                attr_values.append(str(instance._x[idx]))
        return np.asarray(attr_values)

    def anytime_explain(self, instance, callback=None, update_func=None, update_prediction=None):
        data_rows, no_atr = self.data.X.shape
        class_value = self.model(instance)[0]
        prng = RandomState(self.seed)

        self.init_arrays(no_atr)
        attr_values = self.get_atr_column(instance)

        batch_mx_size = self.batch_size * no_atr
        z_sq = abs(st.norm.ppf(self.p_val/2))**2

        tiled_inst = self.tile_instance(instance)
        inst1 = copy.deepcopy(tiled_inst)
        inst2 = copy.deepcopy(tiled_inst)

        worst_case = self.max_iter*no_atr
        time_point = time.time()
        update_table = False

        domain = Domain([ContinuousVariable("Score"),
                         ContinuousVariable("Error")],
                        metas=[StringVariable(name="Feature"), StringVariable(name="Value")])

        if update_prediction is not None:
            update_prediction(class_value)

        def create_res_table():
            nonzero = self.steps != 0
            expl_scaled = (self.expl[nonzero] /
                           self.steps[nonzero]).reshape(1, -1)
            """ creating return array"""
            ips = np.hstack((expl_scaled.T, np.sqrt(
                z_sq * self.var[nonzero] / self.steps[nonzero]).reshape(-1, 1)))
            table = Table.from_numpy(domain, ips,
                                     metas=np.hstack((np.asarray(self.atr_names)[nonzero[0]].reshape(-1, 1),
                                                      attr_values[nonzero[0]].reshape(-1, 1))))
            return table

        while not(all(self.iterations_reached[0, :] > self.max_iter)):
            prog = 1 - np.sum(self.max_iter -
                              self.iterations_reached)/worst_case
            if (callback(int(prog*100))):
                break
            if not(any(self.iterations_reached[0, :] > self.max_iter)):
                a = np.argmax(prng.multinomial(
                    1, pvals=(self.var[0, :]/(np.sum(self.var[0, :])))))
            else:
                a = np.argmin(self.iterations_reached[0, :])

            perm = (prng.random_sample(batch_mx_size).reshape(
                self.batch_size, no_atr)) > 0.5
            rand_data = self.data.X[prng.randint(0,
                                                 data_rows, size=self.batch_size), :]
            inst1.X = np.copy(tiled_inst.X)
            inst1.X[perm] = rand_data[perm]
            inst2.X = np.copy(inst1.X)

            inst1.X[:, a] = tiled_inst.X[:, a]
            inst2.X[:, a] = rand_data[:, a]
            f1 = self._get_predictions(inst1, class_value)
            f2 = self._get_predictions(inst2, class_value)

            diff = np.sum(f1 - f2)
            self.expl[0, a] += diff

            """update variance"""
            self.steps[0, a] += self.batch_size
            self.iterations_reached[0, a] += self.batch_size
            d = diff - self.mu[0, a]
            self.mu[0, a] += d / self.steps[0, a]
            self.M2[0, a] += d * (diff - self.mu[0, a])
            self.var[0, a] = self.M2[0, a] / (self.steps[0, a] - 1)

            if time.time() - time_point > 1:
                update_table = True
                time_point = time.time()

            if update_table:
                update_table = False
                update_func(create_res_table())

            # exclude from sampling if necessary
            needed_iter = z_sq * self.var[0, a] / (self.error**2)
            if (needed_iter <= self.steps[0, a]) and (self.steps[0, a] >= self.min_iter) or (self.steps[0, a] > self.max_iter):
                self.iterations_reached[0, a] = self.max_iter + 1

        return class_value, create_res_table()

    def _get_predictions(self, inst, class_value):
        if isinstance(self.data.domain.class_vars[0], ContinuousVariable):
            # regression
            return self.model(inst)
        else:
            # classification
            predictions = (self.model(inst) == class_value) * 1
            return predictions


class OWExplainPredictions(OWWidget):

    name = "Explain Predictions"
    description = "Computes attribute contributions to the final prediction with an approximation algorithm for shapely value"
    icon = "icons/ExplainPredictions.svg"
    priority = 200
    gui_error = settings.Setting(0.05)
    gui_p_val = settings.Setting(0.05)
    gui_num_atr = settings.Setting(20)
    sort_index = settings.Setting(SortBy.ABSOLUTE)

    class Inputs:
        data = Input("Data", Table, default=True)
        model = Input("Model", Model, multiple=False)
        sample = Input("Sample", Table)

    class Outputs:
        explanations = Output("Explanations", Table)

    class Error(OWWidget.Error):
        sample_too_big = widget.Msg("Can only explain one sample at the time.")

    class Warning(OWWidget.Warning):
        unknowns_increased = widget.Msg(
            "Number of unknown values increased, Data and Sample domains mismatch.")

    def __init__(self):
        super().__init__()
        self.data = None
        self.model = None
        self.to_explain = None
        self.explanations = None
        self.stop = True
        self.e = None

        self._task = None
        self._executor = ThreadExecutor()

        info_box = gui.vBox(self.controlArea, "Info")
        self.data_info = gui.widgetLabel(info_box, "Data: N/A")
        self.model_info = gui.widgetLabel(info_box, "Model: N/A")
        self.sample_info = gui.widgetLabel(info_box, "Sample: N/A")

        criteria_box = gui.vBox(self.controlArea, "Stopping criteria")
        self.error_spin = gui.spin(criteria_box,
                                   self,
                                   "gui_error",
                                   0.01,
                                   1,
                                   step=0.01,
                                   label="Error < ",
                                   spinType=float,
                                   callback=self._update_error_spin,
                                   controlWidth=80,
                                   keyboardTracking=False)

        self.p_val_spin = gui.spin(criteria_box,
                                   self,
                                   "gui_p_val",
                                   0.01,
                                   1,
                                   step=0.01,
                                   label="Error p-value < ",
                                   spinType=float,
                                   callback=self._update_p_val_spin,
                                   controlWidth=80, keyboardTracking=False)

        plot_properties_box = gui.vBox(self.controlArea, "Display features")
        self.num_atr_spin = gui.spin(plot_properties_box,
                                     self,
                                     "gui_num_atr",
                                     1,
                                     100,
                                     step=1,
                                     label="Show attributes",
                                     callback=self._update_num_atr_spin,
                                     controlWidth=80,
                                     keyboardTracking=False)

        self.sort_combo = gui.comboBox(plot_properties_box,
                                       self,
                                       "sort_index",
                                       label="Rank by",
                                       items=SortBy.items(),
                                       orientation=Qt.Horizontal,
                                       callback=self._update_combo)

        gui.rubber(self.controlArea)

        self.cancel_button = gui.button(self.controlArea,
                                        self,
                                        "Stop Computation",
                                        callback=self.toggle_button,
                                        autoDefault=True,
                                        tooltip="Stops and restarts computation")
        self.cancel_button.setDisabled(True)

        predictions_box = gui.vBox(self.mainArea, "Model prediction")
        self.predict_info = gui.widgetLabel(predictions_box, "")

        self.mainArea.setMinimumWidth(700)
        self.resize(700, 400)

        class _GraphicsView(QGraphicsView):
            def __init__(self, scene, parent, **kwargs):
                for k, v in dict(verticalScrollBarPolicy=Qt.ScrollBarAlwaysOff,
                                 horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOff,
                                 viewportUpdateMode=QGraphicsView.BoundingRectViewportUpdate,
                                 renderHints=(QPainter.Antialiasing |
                                              QPainter.TextAntialiasing |
                                              QPainter.SmoothPixmapTransform),
                                 alignment=(Qt.AlignTop |
                                            Qt.AlignLeft),
                                 sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding,
                                                        QSizePolicy.MinimumExpanding)).items():
                    kwargs.setdefault(k, v)
                super().__init__(scene, parent, **kwargs)

        class GraphicsView(_GraphicsView):
            def __init__(self, scene, parent):
                super().__init__(scene, parent,
                                 verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
                                 styleSheet='QGraphicsView {background: white}')
                self.viewport().setMinimumWidth(500)
                self._is_resizing = False

            w = self

            def resizeEvent(self, resizeEvent):
                self._is_resizing = True
                self.w.draw()
                self._is_resizing = False
                return super().resizeEvent(resizeEvent)

            def is_resizing(self):
                return self._is_resizing

            def sizeHint(self):
                return QSize(600, 300)

        class FixedSizeGraphicsView(_GraphicsView):
            def __init__(self, scene, parent):
                super().__init__(scene, parent,
                                 sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding,
                                                        QSizePolicy.Minimum))

            def sizeHint(self):
                return QSize(600, 30)

        """all will share the same scene, but will show different parts of it"""
        self.box_scene = QGraphicsScene(self)

        self.box_view = GraphicsView(self.box_scene, self)
        self.header_view = FixedSizeGraphicsView(self.box_scene, self)
        self.footer_view = FixedSizeGraphicsView(self.box_scene, self)

        self.mainArea.layout().addWidget(self.header_view)
        self.mainArea.layout().addWidget(self.box_view)
        self.mainArea.layout().addWidget(self.footer_view)

        self.painter = None

    def draw(self):
        """Uses GraphAttributes class to draw the explanaitons """
        self.box_scene.clear()
        wp = self.box_view.viewport().rect()
        header_height = 30
        if self.explanations is not None:
            self.painter = GraphAttributes(self.box_scene, min(
                self.gui_num_atr, self.explanations.Y.shape[0]))
            self.painter.paint(wp, self.explanations, header_h=header_height)

        """set appropriate boxes for different views"""
        rect = QRectF(self.box_scene.itemsBoundingRect().x(),
                      self.box_scene.itemsBoundingRect().y(),
                      self.box_scene.itemsBoundingRect().width(),
                      self.box_scene.itemsBoundingRect().height())

        self.box_scene.setSceneRect(rect)
        self.box_view.setSceneRect(
            rect.x(), rect.y()+header_height+2, rect.width(), rect.height() - 80)
        self.header_view.setSceneRect(
            rect.x(), rect.y(), rect.width(), 10)
        self.header_view.setFixedHeight(header_height)
        self.footer_view.setSceneRect(
            rect.x(), rect.y() + rect.height() - 50, rect.width(), 35)

    def sort_explanations(self):
        """sorts explanations according to users choice from combo box"""
        if self.sort_index == SortBy.POSITIVE:
            self.explanations = self.explanations[np.argsort(
                self.explanations.X[:, 0])][::-1]
        elif self.sort_index == SortBy.NEGATIVE:
            self.explanations = self.explanations[np.argsort(
                self.explanations.X[:, 0])]
        elif self.sort_index == SortBy.ABSOLUTE:
            self.explanations = self.explanations[np.argsort(
                np.abs(self.explanations.X[:, 0]))][::-1]
        elif self.sort_index == SortBy.BY_NAME:
            l = np.array(
                list(map(np.chararray.lower, self.explanations.metas[:, 0])))
            self.explanations = self.explanations[np.argsort(l)]
        else:
            return

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        """Set input 'Data"""
        self.data = data
        self.explanations = None
        self.data_info.setText("Data: N/A")
        self.e = None
        if data is not None:
            model = TableModel(data, parent=None)
            if data.X.shape[0] == 1:
                inst = "1 instance and "
            else:
                inst = str(data.X.shape[0]) + " instances and "
            if data.X.shape[1] == 1:
                feat = "1 feature "
            else:
                feat = str(data.X.shape[1]) + " features"
            self.data_info.setText("Data: " + inst + feat)

    @Inputs.model
    def set_predictor(self, model):
        """Set input 'Model"""
        self.model = model
        self.model_info.setText("Model: N/A")
        self.explanations = None
        self.e = None
        if model is not None:
            self.model_info.setText("Model: " + str(model.name))

    @Inputs.sample
    @check_sql_input
    def set_sample(self, sample):
        """Set input 'Sample', checks if size is appropriate"""
        self.to_explain = sample
        self.explanations = None
        self.Error.sample_too_big.clear()
        self.sample_info.setText("Sample: N/A")
        if sample is not None:
            if len(sample.X) != 1:
                self.to_explain = None
                self.Error.sample_too_big()
            else:
                if sample.X.shape[1] == 1:
                    feat = "1 feature"
                else:
                    feat = str(sample.X.shape[1]) + " features"
                self.sample_info.setText("Sample: " + feat)
                if self.e is not None:
                    self.e.saved = False

    def handleNewSignals(self):
        if self._task is not None:
            self.cancel()
        assert self._task is None

        self.predict_info.setText("")
        self.Warning.unknowns_increased.clear()
        self.stop = True
        self.cancel_button.setText("Stop Computation")
        self.commit_calc_or_output()

    def commit_calc_or_output(self):
        if self.data is not None and self.to_explain is not None:
            self.commit_calc()
        else:
            self.commit_output()

    def commit_calc(self):
        num_nan = np.count_nonzero(np.isnan(self.to_explain.X[0]))

        self.to_explain = self.to_explain.transform(self.data.domain)
        if num_nan != np.count_nonzero(np.isnan(self.to_explain.X[0])):
            self.Warning.unknowns_increased()
        if self.model is not None:
            # calculate contributions
            if self.e is None:
                self.e = ExplainPredictions(self.data,
                                            self.model,
                                            batch_size=min(
                                                len(self.data.X), 500),
                                            p_val=self.gui_p_val,
                                            error=self.gui_error)
            self._task = task = Task()

            def callback(progress):
                nonlocal task
                # update progress bar
                QMetaObject.invokeMethod(
                    self, "set_progress_value", Qt.QueuedConnection, Q_ARG(int, progress))
                if task.canceled:
                    return True
                return False

            def callback_update(table):
                QMetaObject.invokeMethod(
                    self, "update_view", Qt.QueuedConnection, Q_ARG(Orange.data.Table, table))

            def callback_prediction(class_value):
                QMetaObject.invokeMethod(
                    self, "update_model_prediction", Qt.QueuedConnection, Q_ARG(float, class_value))

            self.was_canceled = False
            explain_func = partial(
                self.e.anytime_explain, self.to_explain[0], callback=callback, update_func=callback_update, update_prediction=callback_prediction)

            self.progressBarInit(processEvents=None)
            task.future = self._executor.submit(explain_func)
            task.watcher = FutureWatcher(task.future)
            task.watcher.done.connect(self._task_finished)
            self.cancel_button.setDisabled(False)

    @pyqtSlot(Orange.data.Table)
    def update_view(self, table):
        self.explanations = table
        self.sort_explanations()
        self.draw()
        self.commit_output()

    @pyqtSlot(float)
    def update_model_prediction(self, value):
        self._print_prediction(value)

    @pyqtSlot(int)
    def set_progress_value(self, value):
        self.progressBarSet(value, processEvents=False)

    @pyqtSlot(concurrent.futures.Future)
    def _task_finished(self, f):
        """
        Parameters:
        ----------
        f: conncurent.futures.Future
            future instance holding the result of learner evaluation
        """
        assert self.thread() is QThread.currentThread()
        assert self._task is not None
        assert self._task.future is f
        assert f.done()

        self._task = None

        if not self.was_canceled:
            self.cancel_button.setDisabled(True)

        try:
            results = f.result()
        except Exception as ex:
            log = logging.getLogger()
            log.exception(__name__, exc_info=True)
            self.error("Exception occured during evaluation: {!r}".format(ex))

            for key in self.results.keys():
                self.results[key] = None
        else:
            self.update_view(results[1])

        self.progressBarFinished(processEvents=False)

    def commit_output(self):
        """
        Sends best-so-far results forward
        """
        self.Outputs.explanations.send(self.explanations)

    def toggle_button(self):
        if self.stop:
            self.stop = False
            self.cancel_button.setText("Restart Computation")
            self.cancel()
        else:
            self.stop = True
            self.cancel_button.setText("Stop Computation")
            self.commit_calc_or_output()

    def cancel(self):
        """
        Cancel the current task (if any).
        """
        if self._task is not None:
            self._task.cancel()
            assert self._task.future.done()
            # disconnect the `_task_finished` slot
            self._task.watcher.done.disconnect(self._task_finished)
            self.was_canceled = True
            self._task_finished(self._task.future)

    def _print_prediction(self, class_value):
        """
        Parameters
        ----------
        class_value: float 
            Number representing either index of predicted class value, looked up in domain, or predicted value (regression)
        """
        name = self.data.domain.class_vars[0].name
        if isinstance(self.data.domain.class_vars[0], ContinuousVariable):
            self.predict_info.setText(name + ":      " + str(class_value))
        else:
            self.predict_info.setText(
                name + ":      " + self.data.domain.class_vars[0].values[int(class_value)])

    def _update_error_spin(self):
        self.cancel()
        if self.e is not None:
            self.e.error = self.gui_error
        self.handleNewSignals()

    def _update_p_val_spin(self):
        self.cancel()
        if self.e is not None:
            self.e.p_val = self.gui_p_val
        self.handleNewSignals()

    def _update_num_atr_spin(self):
        self.cancel()
        self.handleNewSignals()

    def _update_combo(self):
        if self.explanations != None:
            self.sort_explanations()
            self.draw()
            self.commit_output()

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()


class GraphAttributes:
    """
    Creates entire graph of explanations, paint function is the main one, it delegates painting of attributes to draw_attribute, 
    header and scale are dealt with in draw_header_footer. Header is fixed in size.

    Parameters
    ----------
    scene: QGraphicsScene
        scene to add elements to
    num_of_atr : int
        number of attributes to plot
    space: int
        space between columns with atr names, values
    offset_y : int
        distance between the line border of attribute box plot and the box itself
    rect_height : int
        height of a rectangle, representing score of the attribute
    """

    def __init__(self, scene, num_of_atr=3, space=35, offset_y=10, rect_height=40):
        self.scene = scene
        self.num_of_atr = num_of_atr
        self.space = space
        self.graph_space = 80
        self.offset_y = offset_y
        self.black_pen = QPen(Qt.black, 2)
        self.gray_pen = QPen(Qt.gray, 1)
        self.light_gray_pen = QPen(QColor("#DFDFDF"), 1)
        self.light_gray_pen.setStyle(Qt.DashLine)
        self.brush = QBrush(QColor(0x33, 0x88, 0xff, 0xc0))
        self.blue_pen = QPen(QBrush(QColor(0x33, 0x00, 0xff)), 2)
        """placeholders"""
        self.rect_height = rect_height
        self.max_contrib = None
        self.atr_area_h = None
        self.atr_area_w = None
        self.scale = None

    def get_needed_offset(self, explanations):
        max_n = 0
        word = ""
        max_v = 0
        val = ""
        for e in explanations:
            if max_n < len(str(e._metas[0])):
                word = str(e._metas[0])
                max_n = len(str(e._metas[0]))
            if max_v < len(str(e._metas[1])):
                val = str(e._metas[1])
                max_v = len(str(e._metas[1]))
        w = QGraphicsSimpleTextItem(word, None)
        v = QGraphicsSimpleTextItem(val, None)
        return w.boundingRect().width(), v.boundingRect().width()

    def paint(self, wp, explanations=None, header_h=100):
        """
        Coordinates drawing
        Parameters
        ----------
        wp : QWidget
            current viewport
        explanations : Orange.data.table
            data table with name, value, score and error of attributes to plot
        header_h : int
            space to be left on the top and the bottom of graph for header and scale
        """
        self.name_w, self.val_w = self.get_needed_offset(explanations)
        self.offset_left = self.space + self.name_w + \
            self.space + self.val_w + self.graph_space
        self.offset_right = self.graph_space + 50

        self.atr_area_h = wp.height()/2 - header_h
        self.atr_area_w = (wp.width() - self.offset_left -
                           self.offset_right) / 2

        coords = self.split_boxes_area(
            self.atr_area_h, self.num_of_atr, header_h)
        self.max_contrib = np.max(
            abs(explanations.X[:, 0]) + explanations.X[:, 1])
        self.unit = self.get_scale()
        unit_pixels = np.floor(self.atr_area_w/(self.max_contrib/self.unit))
        self.scale = unit_pixels / self.unit

        self.draw_header_footer(
            wp, header_h, unit_pixels, coords[self.num_of_atr - 1], coords[0])

        for y, e in zip(coords, explanations[:self.num_of_atr]):
            self.draw_attribute(y, atr_name=str(e._metas[0]), atr_val=str(
                e._metas[1]), atr_contrib=e._x[0], error=e._x[1])

    def draw_header_footer(self, wp, header_h, unit_pixels, last_y, first_y, marking_len=15):
        """header"""
        max_x = self.max_contrib * self.scale

        atr_label = QGraphicsSimpleTextItem("Name", None)
        val_label = QGraphicsSimpleTextItem("Value", None)
        score_label = QGraphicsSimpleTextItem("Score", None)

        font = score_label.font()
        font.setBold(True)
        font.setPointSize(13)
        atr_label.setFont(font)
        val_label.setFont(font)
        score_label.setFont(font)

        white_pen = QPen(Qt.white, 3)

        fix = self.offset_left + self.atr_area_w

        self.place_left(val_label, -self.atr_area_h - header_h*0.85)
        self.place_left_edge(atr_label, -self.atr_area_h - header_h*0.85)
        self.place_right(score_label, -self.atr_area_h - header_h*0.85)

        self.scene.addLine(-max_x + fix, -self.atr_area_h - header_h,
                           max_x + fix, -self.atr_area_h - header_h, white_pen)

        """footer"""
        line_y = max(first_y + wp.height() + header_h/2 - 10,
                     last_y + header_h/2 + self.rect_height)
        self.scene.addLine(-max_x + fix, line_y, max_x +
                           fix, line_y, self.black_pen)

        previous = 0
        recomended_d = 35
        for i in range(0, int(self.max_contrib / self.unit) + 1):
            x = unit_pixels * i
            """grid lines"""
            self.scene.addLine(x + fix, first_y, x + fix,
                               line_y, self.light_gray_pen)
            self.scene.addLine(-x + fix, first_y, -x + fix,
                               line_y, self.light_gray_pen)

            self.scene.addLine(x + fix, line_y, x + fix, line_y +
                               marking_len, self.black_pen)
            self.scene.addLine(-x + fix, line_y, -x + fix, line_y +
                               marking_len, self.black_pen)
            """markings on the ruler"""
            if x + fix - previous > recomended_d:
                self.place_centered(self.format_marking(
                    i*self.unit), x + fix, line_y + marking_len + 5)
                if x > 0:
                    self.place_centered(
                        self.format_marking(-i*self.unit), -x + fix, line_y + marking_len + 5)
                previous = x + fix

    def format_marking(self, x, places=2):
        return QGraphicsSimpleTextItem(str(round(x, places)), None)

    def get_scale(self):
        """figures out on what scale is max score (1, .1, .01)
        TESTING NEEDED, maybe something more elegant.
        """
        if self.max_contrib > 10:
            return 10
        elif self.max_contrib > 1:
            return 1
        elif self.max_contrib > 0.1:
            return 0.1
        else:
            return 0.01

    def draw_attribute(self, y, atr_name, atr_val, atr_contrib, error):
        fix = (self.offset_left + self.atr_area_w)
        """vertical line where x = 0"""
        self.scene.addLine(0 + fix, y, 0 + fix, y +
                           self.rect_height, self.black_pen)
        """borders"""
        self.scene.addLine(self.offset_left,
                           y, fix + self.atr_area_w, y, self.gray_pen)
        self.scene.addLine(self.offset_left, y + self.rect_height,
                           fix + self.atr_area_w, y + self.rect_height, self.gray_pen)

        if atr_name is not None and atr_val is not None and atr_contrib is not None:
            atr_contrib_x = atr_contrib * self.scale + fix
            error_x = error * self.scale

            padded_rect = self.rect_height - 2 * self.offset_y
            len_rec = 2 * error_x
            graphed_rect = QGraphicsRectItem(
                atr_contrib_x - error_x, y + self.offset_y, len_rec, padded_rect)
            graphed_rect.setBrush(self.brush)
            graphed_rect.setPen(QPen(Qt.NoPen))
            self.scene.addItem(graphed_rect)
            """vertical line marks calculated contribution of attribute"""
            self.atr_line = self.scene.addLine(atr_contrib_x, y + self.offset_y + 2, atr_contrib_x,
                                               y + self.rect_height - self.offset_y - 2, self.blue_pen)

            """atr name and value on the left"""
            self.place_left(QGraphicsSimpleTextItem(
                atr_val, None), y + self.rect_height/2)
            self.place_left_edge(QGraphicsSimpleTextItem(
                atr_name, None), y + self.rect_height/2)

            """atr score on the right"""
            self.place_right(self.format_marking(
                atr_contrib), y + self.rect_height/2)

    def place_left(self, text, y):
        """places text to the left"""
        self.place_centered(text, 2 * self.space +
                            self.name_w + self.val_w/2, y)

    def place_left_edge(self, text, y):
        """places text more left than place_left"""
        self.scene.addLine(0, y, 0 - 10, y + 2, QPen(Qt.white, 0))
        self.place_centered(text, self.space + self.name_w/2, y)

    def place_right(self, text, y):
        x = self.offset_left + 2 * self.atr_area_w + self.graph_space + 15
        self.scene.addLine(x, y, x, y + 2, QPen(Qt.white, 0))
        self.place_centered(text, self.offset_left + 2 *
                            self.atr_area_w + self.graph_space, y)

    def place_centered(self, text, x, y):
        """centers the text around given coordinates"""
        to_center = text.boundingRect().width()/2
        text.setPos(x - to_center, y)
        self.scene.addItem(text)

    def split_boxes_area(self, h, num_boxes, header_h):
        """calculates y coordinates of boxes to be plotted, calculates rect_height
        Parameters
        ---------
        h : int
            height of area
        num_boxes : int
            number of boxes to fill our area
        header_h : int
            height of header
        Returns:
            list y_coordinates
        """
        return [(-h + i*self.rect_height) for i in range(num_boxes)]


def main():
    app = QApplication([])
    w = OWExplainPredictions()
    data = Orange.data.Table("iris.tab")
    data_subset = data[:20]
    w.set_data(data_subset)
    w.set_data(None)
    w.show()
    app.exec_()


if __name__ == "__main__":
    sys.exit(main())
