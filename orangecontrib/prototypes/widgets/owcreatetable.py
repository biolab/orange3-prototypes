from typing import Optional

import numpy as np
from AnyQt.QtCore import Qt, QAbstractTableModel, QModelIndex, QSize
from AnyQt.QtWidgets import QTableView, QItemDelegate, QLineEdit, QCompleter

from Orange.data import (
    Table, Domain, ContinuousVariable, DiscreteVariable, TimeVariable
)
from Orange.widgets import gui
from Orange.widgets.settings import (
    Setting, ContextSetting, PerfectDomainContextHandler
)
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Output, Input, Msg


DEFAULT_DATA = [[None for x in range(3)] for y in range(3)]


class EditableTableItemDelegate(QItemDelegate):
    def createEditor(self, parent, options, index: QModelIndex):
        model = index.model()  # type: EditableTableModel
        if not model.is_discrete(index.column()):
            return super().createEditor(parent, options, index)

        vals = model.discrete_vals(index.column())
        edit = QLineEdit(parent)
        edit.setCompleter(QCompleter(vals, edit, filterMode=Qt.MatchContains))

        def save():
            if edit.text():
                model.setData(index, edit.text())

        edit.editingFinished.connect(save)
        return edit

    def setEditorData(self, editor, index):
        current_val = index.model().data(index, Qt.DisplayRole)
        editor.setText(current_val)


class EditableTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._table = [[None]]
        self._domain = None

    def set_domain(self, domain: Optional[Domain]):
        self._domain = domain
        self.clear()
        n_columns = len(domain) if domain is not None else 1
        self.columns_changed_to(n_columns)

    def is_discrete(self, column):
        column_data = set(row[column] for row in self._table) - {None}
        return (
            self._domain is not None and self._domain[column].is_discrete
            or (
                column_data
                and not self.is_time_variable(column)
                and not all(map(lambda s: s.isdigit(), column_data))
            )
        )

    def is_time_variable(self, column):
        values = self.time_vals(column)
        return values and not all(e is None for e in values)

    def time_vals(self, column):
        column_data = [row[column] for row in self._table]
        try:
            tvar = TimeVariable('_')
            values = [
                tvar.parse_exact_iso(d) if d is not None
                else None for d in column_data
            ]
            return values
        except ValueError:
            return None

    def discrete_vals(self, column):
        if self._domain is not None and self._domain[column].is_discrete:
            return self._domain[column].values
        else:
            return list(set(row[column] for row in self._table) - {None})

    def rowCount(self, parent=None):
        return len(self._table)

    def columnCount(self, parent=None):
        return len(self._table[0])

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsEditable

    def data(self, index: QModelIndex, role=None):
        value = self._table[index.row()][index.column()]
        if role == Qt.DisplayRole and value is not None:
            return str(value)
        elif role == Qt.TextAlignmentRole:
            return Qt.AlignVCenter | Qt.AlignLeft

    def setData(self, index: QModelIndex, value: str, role=None):
        value = None if not value else value
        self._table[index.row()][index.column()] = value
        self.dataChanged.emit(index, index)
        return True

    def set_table(self, table):
        self._table = table
        self.dataChanged.emit(
            self.index(0, 0),
            self.index(len(table) - 1, len(table[0]))
        )

    def get_raw_table(self):
        return self._table

    def rows_changed_to(self, n_rows):
        while n_rows != self.rowCount():
            self.rows_changed(n_rows)

    def rows_changed(self, n_rows):
        if n_rows > self.rowCount():
            self.beginInsertRows(QModelIndex(), n_rows, n_rows)
            self._table.append([None for _ in range(self.columnCount())])
            self.endInsertRows()
        elif n_rows < self.rowCount():
            self.beginRemoveRows(QModelIndex(),
                                 self.rowCount() - 1, self.rowCount() - 1)
            self._table.pop()
            self.endRemoveRows()

    def columns_changed_to(self, n_columns):
        while n_columns != self.columnCount():
            self.columns_changed(n_columns)

    def columns_changed(self, n_columns):
        if n_columns > self.columnCount():
            self.beginInsertColumns(QModelIndex(), n_columns, n_columns)
            for row in self._table:
                row.append(None)
            self.endInsertColumns()
        elif n_columns < self.columnCount():
            self.beginRemoveColumns(QModelIndex(),
                                    self.columnCount() - 1,
                                    self.columnCount() - 1)
            for row in self._table:
                row.pop()
            self.endRemoveColumns()

    def headerData(self, section, orientation, role=None):
        if orientation == Qt.Vertical:
            return super().headerData(section, orientation, role)

        if role == Qt.DisplayRole:
            if self._domain is None:
                return str(section + 1)
            else:
                return self._domain[section].name

    def clear(self):
        self.rows_changed_to(3)
        self.columns_changed_to(3)
        self.set_table([[None for x in range(3)] for y in range(3)])

    def get_table(self):
        domain = self.get_domain()
        data = np.array(self._table)  # type:
        for ci in range(data.shape[1]):
            if isinstance(domain.variables[ci], TimeVariable):
                data[:, ci] = self.time_vals(ci)
        return Table.from_list(domain, data)

    def get_domain(self):
        if self._domain is not None:
            return self._domain

        vars = []
        for ci in range(self.columnCount()):
            if self.is_discrete(ci):
                values = set(
                    row[ci] for row in self._table if row[ci] is not None
                )
                var = DiscreteVariable(name=str(ci + 1), values=values)
            elif self.is_time_variable(ci):
                var = TimeVariable(name=str(ci + 1))
            else:
                var = ContinuousVariable(name=str(ci + 1))
            vars.append(var)
        return Domain(vars)


class OWCreateTable(OWWidget):
    name = "Create Table"
    icon = "icons/CreateTable.png"
    priority = 50
    keywords = []

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    class Error(OWWidget.Error):
        transform_err = Msg("Data does not fit to domain")

    settingsHandler = PerfectDomainContextHandler()

    n_rows = Setting(len(DEFAULT_DATA))
    n_columns = Setting(len(DEFAULT_DATA[0]))
    auto_commit = Setting(True)
    # since data is a small (at most 20x20) table we can afford to store it
    # as a context
    data = ContextSetting(DEFAULT_DATA.copy(), schema_only=True)

    def __init__(self):
        super().__init__()

        options = {"labelWidth": 100, "controlWidth": 50}
        box = gui.vBox(self.controlArea, "Control")
        self.r_spin = gui.spin(box, self, "n_rows", 1, 20, 1, **options,
                               label="Rows:", callback=self.rows_changed)
        self.c_spin = gui.spin(box, self, "n_columns", 1, 20, 1, **options,
                               label="Columns:", callback=self.columns_changed)
        box.setMinimumWidth(200)

        gui.rubber(self.controlArea)
        gui.auto_send(self.buttonsArea, self, "auto_commit")

        box = gui.vBox(self.mainArea, True, margin=0)
        self.table = QTableView(box)
        self.table.setItemDelegate(EditableTableItemDelegate())
        box.layout().addWidget(self.table)

        self.table_model = EditableTableModel()
        self.table.setModel(self.table_model)
        self.table_model.dataChanged.connect(self.commit)
        self.rows_changed()
        self.columns_changed()

    def rows_changed(self):
        self.table_model.rows_changed_to(self.n_rows)
        self.commit()

    def columns_changed(self):
        self.table_model.columns_changed_to(self.n_columns)
        self.commit()

    def commit(self):
        data = None
        try:
            data = self.table_model.get_table()
            self.Error.transform_err.clear()
        except Exception as ex:
            self.Error.transform_err()
        self.Outputs.data.send(data)

    @Inputs.data
    def set_dataset(self, data):
        self.closeContext()
        self.data = DEFAULT_DATA.copy()
        if data is not None:
            self.table_model.set_domain(data.domain)
            self.c_spin.setEnabled(False)
        else:
            self.table_model.set_domain(None)
            self.c_spin.setEnabled(True)
        self.r_spin.setValue(self.table_model.rowCount())
        self.c_spin.setValue(self.table_model.columnCount())
        print(self.data)
        self.openContext(data)
        self.unconditional_commit()

    @staticmethod
    def sizeHint():
        return QSize(800, 500)

    def closeContext(self):
        self.data = self.table_model.get_raw_table()
        print("here", DEFAULT_DATA)
        super(OWCreateTable, self).closeContext()

    def openContext(self, domain):
        print(self.data)
        super(OWCreateTable, self).openContext(domain)
        print(self.data)
        self.table_model.set_table(self.data)
        print(self.data)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWCreateTable).run(Table("iris"))
