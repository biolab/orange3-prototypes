"""Confusion matrix widget"""

import Orange
import Orange.evaluation
import numpy as np
import sklearn.metrics as skl_metrics
from AnyQt.QtCore import Qt, QSize
from Orange.widgets import widget, settings, gui
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.widget import Msg, Input, Output

from orangecontrib.prototypes.widgets.contingency_table import ContingencyTable


def confusion_matrix(res, index):
    """
    Compute confusion matrix

    Args:
        res (Orange.evaluation.Results): evaluation results
        index (int): model index

    Returns: Confusion matrix
    """
    labels = np.arange(len(res.domain.class_var.values))
    if not res.actual.size:
        # scikit-learn will not return an zero matrix
        return np.zeros((len(labels), len(labels)))
    else:
        return skl_metrics.confusion_matrix(
            res.actual, res.predicted[index], labels=labels)


class OWConfusionMatrix(widget.OWWidget):
    """Confusion matrix widget"""

    name = "Confusion Matrix"
    description = "Display a confusion matrix constructed from " \
                  "the results of classifier evaluations."
    icon = "icons/ConfusionMatrix.svg"
    priority = 1001

    class Inputs:
        evaluation_results = Input("Evaluation Results", Orange.evaluation.Results)

    class Outputs:
        selected_data = Output("Selected Data", Orange.data.Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Orange.data.Table)

    quantities = ["Number of instances",
                  "Proportion of predicted",
                  "Proportion of actual"]

    settings_version = 1
    settingsHandler = settings.ClassValuesContextHandler()

    selected_learner = settings.Setting([0], schema_only=True)
    selection = settings.ContextSetting(set())
    selected_quantity = settings.Setting(0)
    append_predictions = settings.Setting(True)
    append_probabilities = settings.Setting(False)
    autocommit = settings.Setting(True)

    UserAdviceMessages = [
        widget.Message(
            "Clicking on cells or in headers outputs the corresponding "
            "data instances",
            "click_cell")]

    class Error(widget.OWWidget.Error):
        no_regression = Msg("Confusion Matrix cannot show regression results.")
        invalid_values = Msg("Evaluation Results input contains invalid values")

    def __init__(self):
        super().__init__()

        self.data = None
        self.results = None
        self.learners = []

        self.learners_box = gui.listBox(
            self.controlArea, self, "selected_learner", "learners", box=True,
            callback=self._learner_changed
        )

        self.outputbox = gui.vBox(self.controlArea, "Output")
        box = gui.hBox(self.outputbox)
        gui.checkBox(box, self, "append_predictions",
                     "Predictions", callback=self._invalidate)
        gui.checkBox(box, self, "append_probabilities",
                     "Probabilities",
                     callback=self._invalidate)

        gui.auto_commit(self.outputbox, self, "autocommit",
                        "Send Selected", "Send Automatically", box=False)

        self.mainArea.layout().setContentsMargins(0, 0, 0, 0)

        box = gui.vBox(self.mainArea, box=True)

        sbox = gui.hBox(box)
        gui.rubber(sbox)
        gui.comboBox(sbox, self, "selected_quantity",
                     items=self.quantities, label="Show: ",
                     orientation=Qt.Horizontal, callback=self._update)

        self.tableview = ContingencyTable(self)
        box.layout().addWidget(self.tableview)

        selbox = gui.hBox(box)
        gui.button(selbox, self, "Select Correct",
                   callback=self.select_correct, autoDefault=False)
        gui.button(selbox, self, "Select Misclassified",
                   callback=self.select_wrong, autoDefault=False)
        gui.button(selbox, self, "Clear Selection",
                   callback=self.select_none, autoDefault=False)

    def sizeHint(self):
        """Initial size"""
        return QSize(750, 340)

    @Inputs.evaluation_results
    def set_results(self, results):
        """Set the input results."""

        prev_sel_learner = self.selected_learner.copy()
        self.clear()
        self.warning()
        self.closeContext()

        data = None
        if results is not None and results.data is not None:
            data = results.data[results.row_indices]

        if data is not None and not data.domain.has_discrete_class:
            self.Error.no_regression()
            data = results = None
        else:
            self.Error.no_regression.clear()

        nan_values = False
        if results is not None:
            assert isinstance(results, Orange.evaluation.Results)
            if np.any(np.isnan(results.actual)) or \
                    np.any(np.isnan(results.predicted)):
                # Error out here (could filter them out with a warning
                # instead).
                nan_values = True
                results = data = None

        if nan_values:
            self.Error.invalid_values()
        else:
            self.Error.invalid_values.clear()

        self.results = results
        self.data = data

        if data is not None:
            class_values = data.domain.class_var.values
        elif results is not None:
            raise NotImplementedError

        if results is None:
            self.report_button.setDisabled(True)
        else:
            self.report_button.setDisabled(False)

            nmodels = results.predicted.shape[0]

            # NOTE: The 'learner_names' is set in 'Test Learners' widget.
            if hasattr(results, "learner_names"):
                self.learners = results.learner_names
            else:
                self.learners = ["Learner #{}".format(i + 1)
                                 for i in range(nmodels)]

            self.tableview.set_headers(class_values, class_values, "Actual", "Predicted")
            self.openContext(data.domain.class_var)
            if not prev_sel_learner or prev_sel_learner[0] >= len(self.learners):
                if self.learners:
                    self.selected_learner[:] = [0]
            else:
                self.selected_learner[:] = prev_sel_learner
            self._update()
            self.tableview.set_selection(self.selection)
            self.unconditional_commit()

    def clear(self):
        """Reset the widget, clear controls"""
        self.results = None
        self.data = None
        self.tableview.clear()
        # Clear learners last. This action will invoke `_learner_changed`
        self.learners = []

    def select_correct(self):
        """Select the diagonal elements of the matrix"""
        self.tableview.set_selection({(i, i) for i in range(len(self.tableview.classesv))})

    def select_wrong(self):
        """Select the off-diagonal elements of the matrix"""
        self.tableview.set_selection({(i, j)
                                      for i in range(len(self.tableview.classesv))
                                      for j in range(len(self.tableview.classesv)) if i != j})

    def select_none(self):
        """Reset selection"""
        self.tableview.selectionModel().clear()

    def _prepare_data(self):
        indices = self.tableview.selectedIndexes()
        indices = {(ind.row() - 2, ind.column() - 2) for ind in indices}
        actual = self.results.actual
        learner_name = self.learners[self.selected_learner[0]]
        predicted = self.results.predicted[self.selected_learner[0]]
        selected = [i for i, t in enumerate(zip(actual, predicted))
                    if t in indices]

        extra = []
        class_var = self.data.domain.class_var
        metas = self.data.domain.metas

        if self.append_predictions:
            extra.append(predicted.reshape(-1, 1))
            var = Orange.data.DiscreteVariable(
                "{}({})".format(class_var.name, learner_name),
                class_var.values
            )
            metas = metas + (var,)

        if self.append_probabilities and \
                        self.results.probabilities is not None:
            probs = self.results.probabilities[self.selected_learner[0]]
            extra.append(np.array(probs, dtype=object))
            pvars = [Orange.data.ContinuousVariable("p({})".format(value))
                     for value in class_var.values]
            metas = metas + tuple(pvars)

        domain = Orange.data.Domain(self.data.domain.attributes,
                                    self.data.domain.class_vars,
                                    metas)
        data = self.data.transform(domain)
        if len(extra):
            data.metas[:, len(self.data.domain.metas):] = \
                np.hstack(tuple(extra))
        data.name = learner_name

        if selected:
            annotated_data = create_annotated_table(data, selected)
            data = data[selected]
        else:
            annotated_data = create_annotated_table(data, [])
            data = None

        return data, annotated_data

    def commit(self):
        """Output data instances corresponding to selected cells"""
        if self.results is not None and self.data is not None \
                and self.selected_learner:
            data, annotated_data = self._prepare_data()
        else:
            data = None
            annotated_data = None

        self.Outputs.selected_data.send(data)
        self.Outputs.annotated_data.send(annotated_data)

    def _invalidate(self):
        indices = self.tableview.selectedIndexes()
        self.selection = {(ind.row() - 2, ind.column() - 2) for ind in indices}
        self.commit()

    def _learner_changed(self):
        self._update()
        self.commit()

    def _update(self):
        # Update the displayed confusion matrix
        if self.results is not None and self.selected_learner:
            cmatrix = confusion_matrix(self.results, self.selected_learner[0])
            colsum = cmatrix.sum(axis=0)
            rowsum = cmatrix.sum(axis=1)
            n = len(cmatrix)
            diag = np.diag_indices(n)

            colors = cmatrix.astype(np.double)
            colors[diag] = 0
            if self.selected_quantity == 0:
                normalized = cmatrix.astype(np.int)
                formatstr = "{}"
                div = np.array([colors.max()])
            else:
                if self.selected_quantity == 1:
                    normalized = 100 * cmatrix / colsum
                    div = colors.max(axis=0)
                else:
                    normalized = 100 * cmatrix / rowsum[:, np.newaxis]
                    div = colors.max(axis=1)[:, np.newaxis]
                formatstr = "{:2.1f} %"
            div[div == 0] = 1
            colors /= div
            colors[diag] = normalized[diag] / normalized[diag].max()

            def tooltip(i, j):
                return "actual: {}\npredicted: {}".format(self.tableview.classesv[i],
                                                          self.tableview.classesh[j])
            self.tableview.update_table(normalized, colsum=colsum, rowsum=rowsum, colors=colors, formatstr=formatstr,
                                        tooltip=tooltip)

    def send_report(self):
        """Send report"""
        if self.results is not None and self.selected_learner:
            self.report_table(
                "Confusion matrix for {} (showing {})".
                format(self.learners[self.selected_learner[0]],
                       self.quantities[self.selected_quantity].lower()),
                self.tableview)

    @classmethod
    def migrate_settings(cls, settings, version):
        if not version:
            # For some period of time the 'selected_learner' property was
            # changed from List[int] -> int
            # (commit 4e49bb3fd0e11262f3ebf4b1116a91a4b49cc982) and then back
            # again (commit 8a492d79a2e17154a0881e24a05843406c8892c0)
            if "selected_learner" in settings and \
                    isinstance(settings["selected_learner"], int):
                settings["selected_learner"] = [settings["selected_learner"]]


if __name__ == "__main__":
    from AnyQt.QtWidgets import QApplication

    APP = QApplication([])
    w = OWConfusionMatrix()
    w.show()
    IRIS = Orange.data.Table("iris")
    RES_CV = Orange.evaluation.CrossValidation(
        IRIS, [Orange.classification.TreeLearner(),
               Orange.classification.MajorityLearner()],
        store_data=True)
    w.set_results(RES_CV)
    APP.exec_()
