from AnyQt.QtWidgets import QApplication, QFormLayout, QTableView,  QSplitter
from AnyQt.QtCore import Qt, QThread, pyqtSlot
from PyQt5.QtGui import QDoubleValidator

import numpy as np
import scipy.stats as st

import random
import sys
import copy
import logging
import concurrent.futures
from functools import partial

import Orange
import Orange. evaluation
from Orange.widgets.widget import OWWidget, Output, Input, Msg
from Orange.widgets import gui, widget, settings
from Orange.widgets.utils.itemmodels import TableModel
from Orange.widgets.utils.sql import check_sql_input
from Orange.base import Model
from Orange.data import DiscreteVariable, ContinuousVariable, Domain, Table
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

class ExplainPredictions(object):
    """
    Class used to explain individual predictions by determining the importance of attribute values. All interactions between atributes are accounted for by calculating Shapely value.
    Parameters:

    :param data: table with dataset 
    :type: Orange data table
    :param model: model to be used for prediction
    :type model: Orange model
    :param pError: p value of error
    :type pError: float
    :param error: desired max error of approximation algorithm
    :type error: float
    :param batchSize: size of batch used in prediction, bigger batch speeds up the calculations
    :type batchSize: int
    :param maxIter: maximum number of iterations
    :type maxIter: int
    :param minIter: minimum number of iterations
    :type minIter:int
    """

    def __init__(self, data, model, pError=0.05, error=0.05, batchSize=100, maxIter=59000, minIter=100):

        self.model = model
        self.data = data
        self.pError = pError
        self.error = error
        self.batchSize = batchSize
        self.maxIter = maxIter
        self.minIter = minIter
        self.atr_names = DiscreteVariable(name = 'attributes', values = [var.name for var in data.domain.attributes])


    def anytime_explain(self, instance, callback = None):
        dataRows, noAtr = self.data.X.shape
        classValue = self.model(instance)[0]

        # placeholders : steps, mean, sum of squared differences, calcuated contribuitons
        steps = np.zeros((1, noAtr), dtype=float)
        mu = np.zeros((1, noAtr), dtype=float)
        M2 = np.zeros((1, noAtr), dtype=float)
        expl = np.zeros((1, noAtr), dtype=float)
        var = np.ones((1, noAtr), dtype=float)

        atr_indices = np.asarray(range(noAtr)).reshape((1, noAtr))
        batchMxSize = self.batchSize * noAtr
        zSq = abs(st.norm.ppf(self.pError/2))**2

        tiled_inst = Table.from_numpy(instance.domain, np.tile(instance.X, (self.batchSize, 1)), np.full((self.batchSize, 1), instance.Y[0]))
        inst1 = copy.deepcopy(tiled_inst)
        inst2 = copy.deepcopy(tiled_inst)
        iterations_reached = np.zeros((1, noAtr))

        while not(all(iterations_reached[0,:] > self.maxIter)):
            if not(any(iterations_reached[0,:] > self.maxIter)):
                a = np.random.choice(atr_indices[0], p=(var[0,:]/(np.sum(var[0,:]))))
            else:
                a = np.argmin(iterations_reached[0,:])

            perm = np.random.choice([True, False], batchMxSize, replace=True)
            perm = np.reshape(perm, (self.batchSize, noAtr))
            rand_data = self.data.X[random.sample(
                range(dataRows), k=self.batchSize), :]
            inst1.X = np.copy(tiled_inst.X)
            inst1.X[perm] = rand_data[perm]
            inst2.X = np.copy(inst1.X)

            inst1.X[:, a] = tiled_inst.X[:, a]
            inst2.X[:, a] = rand_data[:, a]
            f1 = self._get_predictions(inst1, classValue)
            f2 = self._get_predictions(inst2, classValue)
            
            diff = np.sum(f1 - f2)
            expl[0, a] += diff

            # update variance
            steps[0, a] += self.batchSize
            iterations_reached[0,a] += self.batchSize
            d = diff - mu[0, a]
            mu[0, a] += d/steps[0, a]
            M2[0, a] += d*(diff - mu[0, a])
            var[0, a] = M2[0, a] / (steps[0, a]-1)

            if (callback()):
                break

            # exclude from sampling if necessary
            neededIter = zSq * var[0, a] / (self.error**2)    
            if (neededIter <= steps[0, a]) and (steps[0,a] >= self.minIter) or (steps[0, a] > self.maxIter):
                iterations_reached[0,a] = self.maxIter + 1
                
                    
        expl[0,:] = expl[0,:]/steps[0,:]

        #creating return array
        domain = Domain([self.atr_names], [ContinuousVariable('contributions')])
        table = Table.from_list(domain, np.asarray(self.atr_names.values).reshape(-1, 1))
        ordered = np.argsort(np.abs(expl[0]))[::-1]
        table.Y = expl.T[ordered]
        table.X = table.X[ordered]
        return classValue, table

    def _get_predictions(self, inst, classValue):
        if isinstance(self.data.domain.class_vars[0], ContinuousVariable):
            #regression
            return self.model(inst)
        else:
            #classification
            predictions =  (self.model(inst) == classValue) * 1
            #return self.model(inst, Model.ValueProbs)[1][:,int(classValue)]
            return predictions 



class OWExplainPred(OWWidget):

    name = "Explain Predictions"
    description = "Calculates attribute contributions to final prediction"
    #icon = "iconImage.png"
    priority = 200
    error = settings.Setting(5)
    p_val = settings.Setting(5)

    class Inputs:
        data = Input("Data", Table, default = True)
        model = Input("Model", Model, multiple = False)
        sample = Input("Sample", Table)      

    class Outputs:
        explanations = Output("Explanations", Table)
  
    class Warning(OWWidget.Warning):
        empty_data = widget.Msg("Empty dataset.")
        sample_too_big = widget.Msg("Too many samples to explain.")
        selection_not_matching = widget.Msg("Pick a sample to explain.")
    

    def __init__(self):
        super().__init__()
        self.data = None
        self.model = None
        self.toExplain = None
        self.explanations = None

        self._task = None
        self._executor = ThreadExecutor()  

        self.dataview = QTableView(
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
            horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
            horizontalScrollMode=QTableView.ScrollPerPixel,
            selectionMode=QTableView.NoSelection,
            focusPolicy=Qt.StrongFocus
        )

        box = gui.vBox(self.controlArea, "Stopping criteria")
        form = QFormLayout()
        box.layout().addLayout(form)

        self.error_spin = gui.spin(box, self, "error", 1, 100, callback = self._update_error_spin, 
                    controlWidth = 80, keyboardTracking=False)
        self.error_spin.setSuffix("%")

        self.p_val_spin = gui.spin(box, self, "p_val", 1, 100, callback = self._update_p_val_spin,
                    controlWidth = 80, keyboardTracking = False)
        self.p_val_spin.setSuffix("%")

        form.addRow("Max error: ", self.error_spin)
        form.addRow("P-value: ", self.p_val_spin)
 
        cancel_button = gui.button(self.controlArea, self, "This is taking too long, stop.", callback = self.cancel, tooltip = "Displays results so far, may not be as accurate.")
        

        predictions_box = gui.vBox(self.mainArea, "Explaining for prediction")
        self.predict_info = gui.widgetLabel(predictions_box, "")

        self.mainArea.layout().addWidget(self.dataview)



    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        """Set input 'Data'"""
        self.data = data
        self.explanations = None
        if data is not None:
            model = TableModel(data, parent = None)

        

    @Inputs.model  
    def set_predictor(self, model):
        """Set input 'Model'"""
        self.model = model
        self.explanations = None

    @Inputs.sample
    @check_sql_input
    def set_sample(self,sample):
        """Set input 'Sample', checks if size is appropriate"""
        self.toExplain = sample
        self.explanations = None
        self.Warning.selection_not_matching.clear()
        self.Warning.sample_too_big.clear()
        if sample is not None:
            if len(sample.X[0]) != len(self.data.X[0]):
                self.Warning.selection_not_matching()
            else:
                if len(sample.X) != 1:
                    print ("sample too big: " + str(self.toExplain))
                    self.toExplain = None
                    self.Warning.sample_too_big()

    def handleNewSignals(self):
        if self._task is not None:
            self.cancel()
        assert self._task is None

        self.dataview.setModel(None)
        self.predict_info.setText("")
        if self.data is not None and self.model is not None and self.toExplain is not None:
            #calculate contributions
            if self.explanations is None:
                e = ExplainPredictions(self.data, self.model, batchSize = min(int(len(self.data.X)/5), 100))
                self._task = task = Task()

                def callback():
                    nonlocal task
                    if task.canceled :
                        #TODO: return current results
                        return True
                    return False
                    #set_progress(finished*100)

                explain_func = partial(e.anytime_explain, self.toExplain, callback = callback)
                task.future = self._executor.submit(explain_func)
                task.watcher = FutureWatcher(task.future)
                task.watcher.done.connect(self._task_finished)

        else:
            self.Outputs.explanations.send(None)
 

    @pyqtSlot(concurrent.futures.Future)
    def _task_finished(self, f):
        """
        :param f: future instance holding the result of learner evaluation
        :type f: Future
        """
        assert self.thread() is QThread.currentThread()
        assert self._task is not None
        assert self._task.future is f
        assert f.done()

        self._task = None
        #TODOself.progressBarFinished()

        try:
            results = f.result()
        except Exception as ex:
            log = logging.getLogger()
            log.exception(__name__, exc_info=True)
            self.error("Exception occured during evaluation: {!r}".format(ex))

            for key in self.results.keys():
                self.results[key] = None
        else:
            classValue = results[0]
            self.explanations = results[1]
            self._print_prediction(classValue)
            self.Outputs.explanations.send(self.explanations)
            model = TableModel(self.explanations, parent = None)
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
            



    def _print_prediction(self,classValue):
        """
        :param classValue: Number representing either index of predicted class value, looked up in domain, or predicted value (regression)
        :type classValue: float
        """
        name = self.data.domain.class_vars[0].name
        if isinstance(self.data.domain.class_vars[0], ContinuousVariable):
            self.predict_info.setText(name + ":      " + str(classValue))
        else:
            self.predict_info.setText(name + ":      " + self.data.domain.class_vars[0].values[int(classValue)])


    def _update_error_spin(self):
        pass

    def _update_p_val_spin(self):
        pass

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()



def main():
    app = QApplication([])
    w = OWExplainPred()
    #data = Orange.data.Table("iris.tab")
    data = Orange.data.Table("iris.tab")
    data_subset = data[:20]
    w.set_data(data_subset)
    w.set_data(None)
    w.show()
    app.exec_()


if __name__ == "__main__":
    sys.exit(main())

