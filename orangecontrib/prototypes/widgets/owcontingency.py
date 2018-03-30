from AnyQt.QtGui import QStandardItemModel
from Orange.data import (ContinuousVariable, DiscreteVariable, StringVariable,
                         Domain, Table)
from Orange.data.filter import FilterDiscrete, Values
from Orange.statistics import contingency
from Orange.widgets import widget, gui
from Orange.widgets.settings import (Setting, ContextSetting,
                                     DomainContextHandler)
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.sql import check_sql_input

from orangecontrib.prototypes.widgets.contingency_table import ContingencyTable


class OWContingencyTable(widget.OWWidget):
    name = "Contingency Table"
    description = "Construct a contingency table from given data."
    icon = "icons/Contingency.svg"
    priority = 2010

    inputs = [("Data", Table, "set_data", widget.Default)]
    outputs = [("Contingency Table", Table, ),
               ("Selected Data", Table, )]

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

        self.apply_button = gui.auto_commit(
            self.controlArea, self, "auto_apply", "&Apply", box=False)

        self.tablemodel = QStandardItemModel(self)
        view = self.tableview = ContingencyTable(self, self.tablemodel)
        self.mainArea.layout().addWidget(view)

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
                self.tableview.initialize(self.rows.values, self.columns.values)
                self.table = contingency_table(self.data, self.columns, self.rows)
                self.tableview.update_table(self.table.X, formatstr="{:.0f}")
        else:
            self.tablemodel.clear()

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
        else:
            selected_data = None
        self.send("Selected Data", selected_data)
        self.send("Contingency Table", self.table)

    def _invalidate(self):
        self.selection = self.tableview.get_selection()
        self.commit()

    def _attribute_changed(self):
        self.tableview.set_selection(self.selection)
        self.table = None
        if self.data and self.rows and self.columns:
            self.tableview.initialize(self.rows.values, self.columns.values)
            self.table = contingency_table(self.data, self.columns, self.rows)
            self.tableview.update_table(self.table.X, formatstr="{:.0f}")
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


def test():
    from AnyQt.QtWidgets import QApplication
    app = QApplication([])

    w = OWContingencyTable()
    data = Table("titanic")
    w.set_data(data)
    w.handleNewSignals()
    w.show()
    app.exec_()


if __name__ == "__main__":
    test()
