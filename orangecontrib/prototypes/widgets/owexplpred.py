import sys
import copy
import logging
import concurrent.futures
from functools import partial
import time

from AnyQt.QtWidgets import (
    QApplication, QFormLayout, QTableView,  QSplitter, QHeaderView)
from AnyQt.QtCore import Qt, QThread, pyqtSlot, QMetaObject, Q_ARG, QAbstractProxyModel
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
from Orange.data import DiscreteVariable, ContinuousVariable, StringVariable, Domain, Table
from Orange.widgets.utils.concurrent import (
    ThreadExecutor, FutureWatcher, methodinvoke
)


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

    def __init__(self, data, model, p_val=0.05, error=0.05, batch_size=500, max_iter=100000, min_iter=1000, seed=42):
        self.model = model
        self.data = data
        self.p_val = p_val
        self.error = error
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.atr_names = [var.name for var in data.domain.attributes]
        self.seed = seed
        #variables, saved for possible restart
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
        "somewhat ugly fix for printing values in column"
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
                        metas=[StringVariable(name="Feature"), StringVariable(name = "Value")])

        if update_prediction is not None:
            update_prediction(class_value)

        def create_res_table():
            nonzero = self.steps != 0
            expl_scaled = (self.expl[nonzero]/self.steps[nonzero]).reshape(1, -1)
            # creating return array
            ips = np.hstack((expl_scaled.T, np.sqrt(
                z_sq * self.var[nonzero] / self.steps[nonzero]).reshape(-1, 1)))
            table = Table.from_numpy(domain, ips,
                                     metas=np.hstack((np.asarray(self.atr_names)[nonzero[0]].reshape(-1, 1),
                                                        attr_values[nonzero[0]].reshape(-1,1))))
            return table

        while not(all(self.iterations_reached[0, :] > self.max_iter)):
            prog = 1 - np.sum(self.max_iter - self.iterations_reached)/worst_case
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

            # update variance
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


class OWExplainPred(OWWidget):

    name = "Explain Predictions"
    description = "Computes attribute contributions to the final prediction with an approximation algorithm for shapely value"
    icon = "icons/ExplainPredictions.svg"
    priority = 200
    gui_error = settings.Setting(0.05)
    gui_p_val = settings.Setting(0.05)

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

        self.dataview = QTableView(verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
                                   sortingEnabled=True,
                                   selectionMode=QTableView.NoSelection,
                                   focusPolicy=Qt.StrongFocus)

        self.dataview.sortByColumn(2, Qt.DescendingOrder)
        self.dataview.horizontalHeader().setResizeMode(QHeaderView.Stretch)

        domain = Domain([ContinuousVariable("Score"),
                         ContinuousVariable("Error")],
                        metas=[StringVariable(name="Feature"), StringVariable(name="Value")])
        self.placeholder_table_model = TableModel(
            Table.from_domain(domain), parent=None)

        self.dataview.setModel(self.placeholder_table_model)

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

        self.mainArea.layout().addWidget(self.dataview)

        self.resize(640, 480)

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        """Set input 'Data'"""
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
        """Set input 'Model'"""
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

        self.dataview.setModel(self.placeholder_table_model)
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
        model = TableModel(table, parent=None)
        header = self.dataview.horizontalHeader()
        model.sort(
            header.sortIndicatorSection(),
            header.sortIndicatorOrder())
        self.dataview.setModel(model)
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
        if self.stop :
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

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()


def main():
    app = QApplication([])
    w = OWExplainPred()
    data = Orange.data.Table("iris.tab")
    data_subset = data[:20]
    w.set_data(data_subset)
    w.set_data(None)
    w.show()
    app.exec_()


if __name__ == "__main__":
    sys.exit(main())
