import sys
import copy
import logging
import concurrent.futures
from functools import partial

from AnyQt.QtWidgets import (
    QApplication, QFormLayout, QTableView,  QSplitter, QSizePolicy)
from AnyQt.QtCore import Qt, QThread, pyqtSlot
from PyQt5.QtGui import QSizePolicy
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
        seed for the numpy.random generator, default is 667892

    Returns:
    -------
    class_value: float
        either index of predicted class or predicted value
    table: Orange.data.Table
        table containing atributes and corresponding contributions

    """

    def __init__(self, data, model, p_val=0.05, error=0.05, batch_size=100, max_iter=59000, min_iter=500, seed=667892):
        self.model = model
        self.data = data
        self.p_val = p_val
        self.error = error
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.atr_names = [var.name for var in data.domain.attributes]
        self.seed = seed

    def anytime_explain(self, instance, callback=None):
        data_rows, no_atr = self.data.X.shape
        class_value = self.model(instance)[0]
        prng = RandomState(self.seed)

        # placeholders
        steps = np.zeros((1, no_atr), dtype=float)
        mu = np.zeros((1, no_atr), dtype=float)
        M2 = np.zeros((1, no_atr), dtype=float)
        expl = np.zeros((1, no_atr), dtype=float)
        var = np.ones((1, no_atr), dtype=float)

        batch_mx_size = self.batch_size * no_atr
        z_sq = abs(st.norm.ppf(self.p_val/2))**2
        atr_err = np.zeros((1, no_atr), dtype=float)
        atr_err.fill(np.nan)

        tiled_x = np.tile(instance.X, (self.batch_size, 1))
        tiled_metas = np.tile(instance.metas, (self.batch_size, 1))
        tiled_w = np.tile(instance.W, (self.batch_size, 1))
        tiled_y = np.tile(instance.Y, (self.batch_size, 1))
        tiled_inst = Table.from_numpy(instance.domain, tiled_x, tiled_y, tiled_metas, tiled_w)

        inst1 = copy.deepcopy(tiled_inst)
        inst2 = copy.deepcopy(tiled_inst)
        iterations_reached = np.zeros((1, no_atr))

        while not(all(iterations_reached[0, :] > self.max_iter)):
            if not(any(iterations_reached[0, :] > self.max_iter)):
                a = np.argmax(prng.multinomial(1, pvals=(var[0, :]/(np.sum(var[0, :])))))
            else:
                a = np.argmin(iterations_reached[0, :])

            perm = (prng.random_sample(batch_mx_size).reshape(self.batch_size, no_atr)) > 0.5
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
            expl[0, a] += diff

            # update variance
            steps[0, a] +=self.batch_size
            iterations_reached[0, a] +=self.batch_size
            d = diff - mu[0, a]
            mu[0, a] += d / steps[0, a]
            M2[0, a] += d * (diff - mu[0, a])
            var[0, a] = M2[0, a] / (steps[0, a] - 1)

            if (callback()):
                break

            # exclude from sampling if necessary
            needed_iter = z_sq * var[0, a] / (self.error**2)
            if (needed_iter <= steps[0, a]) and (steps[0, a] >= self.min_iter) or (steps[0, a] > self.max_iter):
                iterations_reached[0, a] = self.max_iter + 1
                atr_err[0, a] = np.sqrt(z_sq * var[0, a] / steps[0, a])

        expl[0, :] = expl[0, :]/steps[0, :]

        # creating return array
        ordered = np.argsort(expl[0])[::-1]
        domain = Domain([], [ContinuousVariable('contributions'),
                         ContinuousVariable('max error')],
                        metas = [StringVariable(name = "attributes")])
        table = Table.from_numpy(domain, np.empty((no_atr, 0), dtype=np.float64),
                                Y = np.hstack((expl.T,np.sqrt(z_sq * var[0,:] / steps[0,:]).reshape(-1, 1))),
                                metas = np.asarray(self.atr_names).reshape(-1, 1))
        table.Y = table.Y[ordered]
        table.metas = table.metas[ordered]
        return class_value, table

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
    #icon = "iconImage.png"
    priority = 200
    gui_error = settings.Setting(5)
    gui_p_val = settings.Setting(5)

    class Inputs:
        data = Input("Data", Table, default=True)
        model = Input("Model", Model, multiple=False)
        sample = Input("Sample", Table)

    class Outputs:
        explanations = Output("Explanations", Table)

    class Error(OWWidget.Error):
        sample_too_big = widget.Msg("Too many samples to explain.")
        selection_not_matching = widget.Msg(
            "Data and Sample domains do not match.")

    def __init__(self):
        super().__init__()
        self.data = None
        self.model = None
        self.to_explain = None
        self.explanations = None

        self._task = None
        self._executor = ThreadExecutor()

        self.dataview = QTableView(verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
                                   sortingEnabled=True,
                                   selectionMode=QTableView.NoSelection,
                                   focusPolicy=Qt.StrongFocus)

        box = gui.vBox(self.controlArea, "Stopping criteria")
        self.error_spin = gui.spin(box,
                                   self,
                                   "gui_error",
                                   1,
                                   100,
                                   label="Max error: ",
                                   callback=self._update_error_spin,
                                   controlWidth=80,
                                   keyboardTracking=False)
        self.error_spin.setSuffix("%")
        self.p_val_spin = gui.spin(box,
                                   self,
                                   "gui_p_val",
                                   1,
                                   100,
                                   label="P-value: ",
                                   callback=self._update_p_val_spin,
                                   controlWidth=80, keyboardTracking=False)
        self.p_val_spin.setSuffix("%")

        gui.rubber(self.controlArea)

        cancel_button = gui.button(self.controlArea,
                                   self,
                                   "Stop computation",
                                   callback=self.cancel,
                                   tooltip="Displays results so far, may not be as accurate.")

        predictions_box = gui.vBox(self.mainArea, "Model prediction")
        self.predict_info = gui.widgetLabel(predictions_box, "")

        self.mainArea.layout().addWidget(self.dataview)

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        """Set input 'Data'"""
        self.data = data
        self.explanations = None
        if data is not None:
            model = TableModel(data, parent=None)

    @Inputs.model
    def set_predictor(self, model):
        """Set input 'Model'"""
        self.model = model
        self.explanations = None

    @Inputs.sample
    @check_sql_input
    def set_sample(self, sample):
        """Set input 'Sample', checks if size is appropriate"""
        self.to_explain = sample
        self.explanations = None
        self.Error.sample_too_big.clear()
        if sample is not None and len(sample.X) != 1:
            self.to_explain = None
            self.Error.sample_too_big()

    def handleNewSignals(self):
        if self._task is not None:
            self.cancel()
        assert self._task is None

        self.dataview.setModel(None)
        self.predict_info.setText("")
        self.Error.selection_not_matching.clear()
        if self.data is not None and self.to_explain is not None:

            self.to_explain = self.to_explain.transform(self.data.domain)

            if not(any(np.isnan(self.to_explain.X[0]))):
                if self.model is not None:
                    # calculate contributions
                    e = ExplainPredictions(self.data,
                                           self.model,
                                           batch_size=min(
                                               int(len(self.data.X) / 5), 100),
                                           p_val=self.gui_p_val / 100,
                                           error=self.gui_error / 100)
                    self._task = task = Task()
                    def callback():
                        nonlocal task
                        if task.canceled:
                            return True
                        return False
                    explain_func = partial(
                        e.anytime_explain, self.to_explain, callback=callback)
                    task.future = self._executor.submit(explain_func)
                    task.watcher = FutureWatcher(task.future)
                    task.watcher.done.connect(self._task_finished)
            else:
                self.Error.selection_not_matching()
        else:
            self.Outputs.explanations.send(None)

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

        try:
            results = f.result()
        except Exception as ex:
            log = logging.getLogger()
            log.exception(__name__, exc_info=True)
            self.error("Exception occured during evaluation: {!r}".format(ex))

            for key in self.results.keys():
                self.results[key] = None
        else:
            class_value = results[0]
            self.explanations = results[1]
            self._print_prediction(class_value)
            self.Outputs.explanations.send(self.explanations)
            model = TableModel(self.explanations, parent=None)
            self.dataview.setModel(model)

    def cancel(self):
        """
        Cancel the current task (if any).
        """
        if self._task is not None:
            self._task.cancel()
            assert self._task.future.done()
            # disconnect the `_task_finished` slot
            self._task.watcher.done.disconnect(self._task_finished)
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
        self.handleNewSignals()

    def _update_p_val_spin(self):
        self.cancel()
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
