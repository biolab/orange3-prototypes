import concurrent.futures
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QApplication,
    QListView,
)
from AnyQt.QtGui import QBrush, QColor
from AnyQt.QtCore import Qt, QThread, Slot

from Orange.data import Table, DiscreteVariable
from Orange.data.filter import FilterDiscrete, Values
from Orange.widgets import widget, settings, gui
from Orange.widgets.utils.annotated_data import create_annotated_table
from Orange.widgets.utils.itemmodels import PyTableModel, DomainModel
from Orange.widgets.utils.concurrent import (
    ThreadExecutor, FutureWatcher, methodinvoke
)

from orangecontrib.prototypes.significance import (
    perm_test, hyper_test, chi2_test, t_test,
    fligner_killeen_test, mannwhitneyu_test,
    gumbel_min_test, gumbel_max_test,
    CORRECTED_LABEL,
)
from orangecontrib.prototypes.pandas_util import table_from_frame


log = logging.getLogger(__name__)


class OWSignificantGroups(widget.OWWidget):
    name = 'Significant Groups'
    description = "Test whether instances grouped by nominal values are " \
                  "significantly different from random samples or the "\
                  "dataset in whole."
    icon = 'icons/SignificantGroups.svg'
    priority = 200

    class Inputs(widget.OWWidget.Inputs):
        data = widget.Input('Data', Table)

    class Outputs(widget.OWWidget.Outputs):
        selected_data = widget.Output('Selected Data', Table, default=True)
        data = widget.Output('Data', Table)
        results = widget.Output('Test Results', Table)

    want_main_area = True
    want_control_area = True

    class Information(widget.OWWidget.Information):
        nothing_significant = widget.Msg('Chosen parameters reveal no significant groups')

    class Error(widget.OWWidget.Error):
        no_vars_selected = widget.Msg('No independent variables selected')
        no_class_selected = widget.Msg('No dependent variable selected')

    TEST_STATISTICS = OrderedDict((
        ('mean', np.nanmean),
        ('variance', np.nanvar),
        ('median', np.nanmedian),
        ('minimum', np.nanmin),
        ('maximum', np.nanmax),
    ))

    settingsHandler = settings.DomainContextHandler()

    chosen_X = settings.ContextSetting([])
    chosen_y = settings.ContextSetting(0)
    is_permutation = settings.Setting(False)
    test_statistic = settings.Setting(next(iter(TEST_STATISTICS)))
    min_count = settings.Setting(20)

    def __init__(self):
        self._task = None  # type: Optional[self.Task]
        self._executor = ThreadExecutor(self)

        self.data = None
        self.test_type = ''

        self.discrete_model = DomainModel(separators=False, valid_types=(DiscreteVariable,), parent=self)
        self.domain_model = DomainModel(valid_types=DomainModel.PRIMITIVE, parent=self)

        box = gui.vBox(self.controlArea, 'Hypotheses Testing')
        gui.listView(box, self, 'chosen_X', model=self.discrete_model,
                     box='Grouping Variables',
                     selectionMode=QListView.ExtendedSelection,
                     callback=self.Error.no_vars_selected.clear,
                     toolTip='Select multiple variables with Ctrl+ or Shift+Click.')
        target = gui.comboBox(box, self, 'chosen_y',
                              sendSelectedValue=True,
                              label='Test Variable',
                              callback=[self.set_test_type,
                                        self.Error.no_class_selected.clear])
        target.setModel(self.domain_model)

        gui.checkBox(box, self, 'is_permutation', label='Permutation test',
                     callback=self.set_test_type)
        gui.comboBox(box, self, 'test_statistic', label='Statistic:',
                     items=tuple(self.TEST_STATISTICS),
                     orientation=Qt.Horizontal,
                     sendSelectedValue=True,
                     callback=self.set_test_type)
        gui.label(box, self, 'Test: %(test_type)s')

        box = gui.vBox(self.controlArea, 'Filter')
        gui.spin(box, self, 'min_count', 5, 1000, 5,
                 label='Minimum group size (count):')

        self.btn_compute = gui.button(self.controlArea, self, '&Compute', callback=self.compute)
        gui.rubber(self.controlArea)

        class Model(PyTableModel):
            _n_vars = 0
            _BACKGROUND = [QBrush(QColor('#eee')),
                           QBrush(QColor('#ddd'))]

            def setHorizontalHeaderLabels(self, labels, n_vars):
                self._n_vars = n_vars
                super().setHorizontalHeaderLabels(labels)

            def data(self, index, role=Qt.DisplayRole):
                if role == Qt.BackgroundRole and index.column() < self._n_vars:
                    return self._BACKGROUND[index.row() % 2]
                if role == Qt.DisplayRole or role == Qt.ToolTipRole:
                    colname = self.headerData(index.column(), Qt.Horizontal)
                    if colname.lower() in ('count', 'count | class'):
                        row = self.mapToSourceRows(index.row())
                        return int(self[row] [index.column()])
                return super().data(index, role)

        owwidget = self

        class View(gui.TableView):
            _vars = None
            def set_vars(self, vars):
                self._vars = vars

            def selectionChanged(self, *args):
                super().selectionChanged(*args)

                rows = list({index.row() for index in self.selectionModel().selectedRows(0)})

                if not rows:
                    owwidget.Outputs.data.send(None)
                    return

                model = self.model().tolist()
                filters = [Values([FilterDiscrete(self._vars[col], {model[row][col]})
                                   for col in range(len(self._vars))])
                           for row in self.model().mapToSourceRows(rows)]
                data = Values(filters, conjunction=False)(owwidget.data)

                annotated = create_annotated_table(owwidget.data, data.ids)

                owwidget.Outputs.selected_data.send(data)
                owwidget.Outputs.data.send(annotated)

        self.view = view = View(self)
        self.model = Model(parent=self)
        view.setModel(self.model)
        view.horizontalHeader().setStretchLastSection(False)
        self.mainArea.layout().addWidget(view)

        self.set_test_type()

    @Inputs.data
    def set_data(self, data):
        self.data = data
        domain = None if data is None else data.domain

        self.closeContext()

        self.domain_model.set_domain(domain)
        self.discrete_model.set_domain(domain)
        if domain is not None:
            if domain.class_var:
                self.chosen_y = domain.class_var.name

        self.openContext(domain)

        self.set_test_type()

    def set_test_type(self):
        if self.data is None:
            return

        yvar = self.data.domain[self.chosen_y]

        self.controls.test_statistic.setEnabled(yvar.is_continuous)

        if self.is_permutation:
            test = 'Permutation '
            if yvar.is_discrete:
                test += 'χ² '
            else:
                test += str(self.test_statistic) + ' '
        else:
            test = ''
            if yvar.is_discrete:
                test += 'χ² ' if len(yvar.values) > 2 else 'Hypergeometric '
            else:
                if self.test_statistic == 'mean':
                    test += "Student's t-"
                elif self.test_statistic == 'variance':
                    test += "Fligner–Killeen "
                elif self.test_statistic == 'median':
                    test += "Mann–Whitney U "
                elif self.test_statistic in ('minimum', 'maximum'):
                    test += "Gumbel distribution "
                else:
                    assert False, self.test_statistic
        test += 'test'
        self.test_type = test

    def compute(self):
        if not self.chosen_X:
            self.Error.no_vars_selected()
            return

        if not self.chosen_y:
            self.Error.no_class_selected()
            return

        # If listview selection was a single item the list of items is not a list,
        # but futher below we expect it to be
        if not isinstance(self.chosen_X, (list, tuple)):
            self.chosen_X = [self.chosen_X]

        self.btn_compute.setEnabled(False)
        yvar = self.data.domain[self.chosen_y]

        def get_col(var, col):
            values = np.array(list(var.values) + [np.nan], dtype=object)
            pd.Categorical(col, list(var.values))
            col = pd.Series(col).fillna(-1).astype(int)
            return values[col]

        X = np.column_stack([get_col(var, self.data.get_column_view(var)[0])
                             for var in (self.data.domain[i]
                                         for i in self.chosen_X)])
        X = pd.DataFrame(X, columns=self.chosen_X)
        y = pd.Series(self.data.get_column_view(yvar)[0])

        test, args, kwargs = None, (X, y), dict(min_count=self.min_count)
        if self.is_permutation:
            statistic = 'chi2' if yvar.is_discrete else self.TEST_STATISTICS[self.test_statistic]
            test = perm_test
            kwargs.update(
                statistic=statistic, n_jobs=-2,
                callback=methodinvoke(self, "setProgressValue", (int, int)))
        else:
            if yvar.is_discrete:
                if len(yvar.values) > 2:
                    test = chi2_test
                else:
                    test = hyper_test
                    args = (X, y.astype(bool))
            else:
                test = {
                    'mean': t_test,
                    'variance': fligner_killeen_test,
                    'median': mannwhitneyu_test,
                    'minimum': gumbel_min_test,
                    'maximum': gumbel_max_test,
                }[self.test_statistic]

        self._task = task = self.Task()
        self.progressBarInit()
        task.future = self._executor.submit(test, *args, **kwargs)
        task.watcher = FutureWatcher(task.future)
        task.watcher.done.connect(self.on_computed)

    @Slot(int, int)
    def setProgressValue(self, n, N):
        assert self.thread() is QThread.currentThread()
        self.progressBarSet(n / (N + 1) * 100)

    class Task:
        future = ...  # type: concurrent.futures.Future
        watcher = ...  # type: FutureWatcher
        cancelled = False  # type: bool

        def cancel(self):
            self.cancelled = True
            # Cancel the future. Note this succeeds only if the execution has
            # not yet started (see `concurrent.futures.Future.cancel`) ..
            self.future.cancel()
            # ... and wait until computation finishes
            concurrent.futures.wait([self.future])

    @Slot(concurrent.futures.Future)
    def on_computed(self, future):
        assert self.thread() is QThread.currentThread()
        assert future.done()

        self._task = None
        self.progressBarFinished()

        df = future.result()
        # Only retain "significant" p-values
        df = df[df[CORRECTED_LABEL] < .2]

        columns = [var.name for var in df.index.name] + list(df.columns)
        lst = [list(i) + list(j)
               for i, j in zip(df.index, df.values)]

        results_table = table_from_frame(pd.DataFrame(lst, columns=columns),
                                         force_nominal=True)
        results_table.name = 'Significant Groups'
        self.Outputs.results.send(results_table)

        self.view.set_vars(list(df.index.name))
        self.model.setHorizontalHeaderLabels(columns, len(df.index.name))
        self.model.wrap(lst)
        self.view.sortByColumn(len(columns) - 1, Qt.AscendingOrder)

        self.Information.nothing_significant(shown=not lst)
        self.btn_compute.setEnabled(True)

    def send_report(self):
        self.report_items([
            ('Test Variable', self.chosen_y),
            ('Test', self.test_type),
            ('Min. group size', self.min_count),
        ])
        self.report_table('Significant Groups', self.view)


if __name__ == "__main__":
    a = QApplication([])
    log.setLevel(logging.DEBUG)
    ow = OWSignificantGroups()
    ow.show()
    ow.set_data(Table('heart_disease'))
    a.exec()
