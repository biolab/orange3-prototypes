from Orange.data import (ContinuousVariable, DiscreteVariable, StringVariable,
                         Domain, Table)
from Orange.statistics import contingency
from Orange.widgets import widget, gui
from Orange.widgets.settings import (Setting, ContextSetting,
                                     DomainContextHandler)
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.sql import check_sql_input


class OWContingencyTable(widget.OWWidget):
    name = "Contingency Table"
    description = "Construct a contingency table from given data."
    icon = "icons/Contingency.svg"
    priority = 2010

    inputs = [("Data", Table, "set_data", widget.Default)]
    outputs = [("Contingency Table", Table, )]

    settingsHandler = DomainContextHandler(metas_in_res=True)
    rows = ContextSetting(None)
    columns = ContextSetting(None)
    auto_apply = Setting(True)

    want_main_area = False

    def __init__(self):
        super().__init__()

        self.data = None
        self.feature_model = DomainModel(valid_types=DiscreteVariable)

        box = gui.vBox(self.controlArea, "Rows")
        gui.comboBox(box, self, 'rows', sendSelectedValue=True,
                     model=self.feature_model, callback=self._invalidate)

        box = gui.vBox(self.controlArea, "Columns")
        gui.comboBox(box, self, 'columns', sendSelectedValue=True,
                     model=self.feature_model, callback=self._invalidate)

        self.apply_button = gui.auto_commit(
            self.controlArea, self, "auto_apply", "&Apply", box=False)

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

    def handleNewSignals(self):
        self._invalidate()

    def commit(self):
        table = None
        if self.data and self.rows and self.columns:
            table = contingency_table(self.data, self.columns, self.rows)
        self.send("Contingency Table", table)

    def _invalidate(self):
        self.commit()

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
