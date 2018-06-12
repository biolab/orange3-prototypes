import numpy as np
from Orange.data import (ContinuousVariable, DiscreteVariable, StringVariable,
                         Domain, Table)
from Orange.data.filter import FilterDiscrete, Values
from Orange.statistics import contingency
from Orange.widgets import widget, gui
from Orange.widgets.settings import (Setting, ContextSetting,
                                     DomainContextHandler)
from Orange.widgets.utils.annotated_data import create_annotated_table, ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.visualize.owsieve import ChiSqStats
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from orangecontrib.prototypes.widgets.contingency_table import ContingencyTable


class OWContingencyTable(widget.OWWidget):
    name = "Contingency Table"
    description = "Construct a contingency table from given data."
    icon = "icons/Contingency.svg"
    priority = 2010

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        contingency = Output("Contingency Table", Table, default=True)
        selected_data = Output("Selected Data", Table)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    settingsHandler = DomainContextHandler(metas_in_res=True)
    rows = ContextSetting(None)
    columns = ContextSetting(None)
    selection = ContextSetting(set())
    auto_apply = Setting(True)

    want_main_area = True

    def __init__(self):
        super().__init__()

        self.data = None
        self.feature_model = DomainModel(valid_types=DiscreteVariable)
        self.table = None

        box = gui.vBox(self.controlArea, "Rows")
        gui.comboBox(box, self, 'rows', sendSelectedValue=True,
                     model=self.feature_model, callback=self._attribute_changed)

        box = gui.vBox(self.controlArea, "Columns")
        gui.comboBox(box, self, 'columns', sendSelectedValue=True,
                     model=self.feature_model, callback=self._attribute_changed)

        gui.rubber(self.controlArea)

        box = gui.vBox(self.controlArea, "Scores")
        self.scores = gui.widgetLabel(box, "\n\n")

        self.apply_button = gui.auto_commit(
            self.controlArea, self, "auto_apply", "&Apply", box=False)

        self.tableview = ContingencyTable(self)
        self.mainArea.layout().addWidget(self.tableview)

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        if self.feature_model:
            self.closeContext()
        self.data = data
        self.feature_model.set_domain(None)
        self.rows = None
        self.columns = None
        if self.data:
            self.feature_model.set_domain(self.data.domain)
            if self.feature_model:
                self.rows = self.feature_model[0]
                self.columns = self.feature_model[0]
                self.openContext(data)
                self.tableview.set_variables(self.rows, self.columns)
                self.table = contingency_table(self.data, self.columns, self.rows)
                self.tableview.update_table(self.table.X, formatstr="{:.0f}")
        else:
            self.tableview.clear()

    def handleNewSignals(self):
        self._attribute_changed()

    def commit(self):
        if len(self.selection):
            cells = []
            for ir, r in enumerate(self.rows.values):
                for ic, c in enumerate(self.columns.values):
                    if (ir, ic) in self.selection:
                        cells.append(Values([FilterDiscrete(self.rows, [r]), FilterDiscrete(self.columns, [c])]))
            selected_data = Values(cells, conjunction=False)(self.data)
            annotated_data = create_annotated_table(self.data,
                                                    np.where(np.in1d(self.data.ids, selected_data.ids, True)))
        else:
            selected_data = None
            annotated_data = create_annotated_table(self.data, [])
        self.Outputs.contingency.send(self.table)
        self.Outputs.selected_data.send(selected_data)
        self.Outputs.annotated_data.send(annotated_data)

    def _invalidate(self):
        self.selection = self.tableview.get_selection()
        self.commit()

    def _attribute_changed(self):
        self.tableview.set_selection(self.selection)
        self.table = None
        if self.data and self.rows and self.columns:
            self.tableview.set_variables(self.rows, self.columns)
            self.table = contingency_table(self.data, self.columns, self.rows)
            self.tableview.update_table(self.table.X, formatstr="{:.0f}")

            chi = ChiSqStats(self.data, self.rows, self.columns)
            vardata1 = self.data.get_column_view(self.rows.name)[0]
            vardata2 = self.data.get_column_view(self.columns.name)[0]
            self.scores.setText("ARI: {:.3f}\nAMI: {:.3f}\nχ²={:.2f}, p={:.3f}".format(
                adjusted_rand_score(vardata1, vardata2),
                adjusted_mutual_info_score(vardata1, vardata2),
                chi.chisq,
                chi.p))
        else:
            self.scores.setText("\n\n")
        self._invalidate()

    def send_report(self):
        rows = None
        columns = None
        if self.data is not None:
            rows = self.rows
            if rows in self.data.domain:
                rows = self.data.domain[rows]
            columns = self.columns
            if columns in self.data.domain:
                columns = self.data.domain[columns]
        self.report_items((
            ("Rows", rows),
            ("Columns", columns),
        ))


def contingency_table(data, columns, rows):
    ct = contingency.get_contingency(data, columns, rows)
    metavar = StringVariable(rows.name)
    metas = [[str(val)] for val in rows.values]
    domain = Domain([ContinuousVariable(val, number_of_decimals=0)
                     for val in columns.values], metas=[metavar])
    return Table(domain, ct, metas=metas)


def main():
    from AnyQt.QtWidgets import QApplication
    app = QApplication([])

    w = OWContingencyTable()
    data = Table("titanic")
    w.set_data(data)
    w.handleNewSignals()
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()
