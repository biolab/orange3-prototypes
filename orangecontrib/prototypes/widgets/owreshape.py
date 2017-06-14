import sys

import numpy as np

from AnyQt.QtWidgets import QApplication
from AnyQt.QtCore import Qt, QSize

import Orange.data

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels


class OWReshape(widget.OWWidget):
    name = "To Shopping List"
    description = "Reshape from a 'wide' records table to 'long' format."
    icon = "icons/ToShoppingList.svg"

    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Data", Orange.data.Table)]

    want_main_area = False
    resizing_enabled = False

    settingsHandler = settings.PerfectDomainContextHandler(metas_in_res=True)

    idvar = settings.ContextSetting(0)  # type: Orange.data.Variable

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = None  # type: Orange.data.Table

        self.idvar_model = itemmodels.VariableListModel(parent=self)
        self.item_var_name = "Item"
        self.value_var_name = "Rating"

        box = gui.widgetBox(self.controlArea, "Info")
        self.info_text = gui.widgetLabel(box, "No data")

        box = gui.widgetBox(self.controlArea, "Id var")
        self.var_cb = gui.comboBox(box, self, "idvar",
                                   callback=self._invalidate)
        self.var_cb.setMinimumContentsLength(16)
        self.var_cb.setModel(self.idvar_model)
        gui.lineEdit(self.controlArea, self, "item_var_name",
                     box="Item name", callback=self._invalidate)
        gui.lineEdit(self.controlArea, self, "value_var_name",
                     box="Value name", callback=self._invalidate)

    def sizeHint(self):
        return QSize(300, 50)

    def clear(self):
        self.data = None
        self.idvar_model[:] = []
        self.error("")

    def set_data(self, data):
        self.closeContext()
        self.clear()

        idvars = []
        if data is not None:
            domain = data.domain
            idvars = [var for var in domain.metas + domain.variables
                      if isinstance(var, (Orange.data.DiscreteVariable,
                                          Orange.data.StringVariable))]
            if not idvars:
                self.error("No suitable id columns.")
                data = None

        self.data = data

        if self.data is not None:
            self.idvar_model[:] = idvars
            self.openContext(data)
            self.info_text.setText("Data with {} instances".format(len(data)))
        else:
            self.info_text.setText("No data")
        self.commit()

    def _invalidate(self):
        self.commit()

    def commit(self):
        if self.data is None:
            self.send("Data", None)
            return

        self.error("")
        data, domain = self.data, self.data.domain
        idvar = self.idvar_model[self.idvar]

        itemvars = [var for var in domain.attributes if var is not idvar]
        item_names = [v.name for v in itemvars]
        if len(set(item_names)) != len(itemvars):
            self.error("Duplicate column names")
            self.send("Data", None)
            return

        item_var = Orange.data.DiscreteVariable(self.item_var_name,
                                                values=item_names)
        value_var = Orange.data.ContinuousVariable(self.value_var_name)
        try:
            outdata = reshape_long(data, idvar, item_var, value_var)

        except ValueError as err:
            self.error(str(err))
            outdata = None

        self.send("Data", outdata)


def reshape_long(table, idvar, itemvar, valuevar):
    assert isinstance(table, Orange.data.Table)
    assert isinstance(valuevar, Orange.data.ContinuousVariable)
    assert isinstance(itemvar, Orange.data.DiscreteVariable)

    var_indices = np.array(
        [i for i, v in enumerate(table.domain.attributes) if v is not idvar],
        dtype=np.intp)

    id_coldata, _ = table.get_column_view(idvar)
    id_uniq, id_index, id_inverse = np.unique(
        id_coldata, return_index=True, return_inverse=True
    )

    if not id_uniq.size == id_coldata.size:
        raise ValueError("{} column contains duplicate entries."
                         .format(idvar.name))

    if idvar.is_string:
        idvar = Orange.data.DiscreteVariable(idvar.name, list(id_uniq))
        id_coldata = id_inverse

    outdomain = Orange.data.Domain([idvar, itemvar], [valuevar])

    X_parts, Y_parts = [], []
    N = 0
    for id_, row in zip(id_coldata, table):
        vals = row.x[var_indices]
        defined = np.flatnonzero(~np.isnan(vals))
        
        idcol_ = np.full((defined.size,), id_, dtype=id_coldata.dtype)
        itemcol_ = defined
        valcol = vals[defined]
        X_parts.append([idcol_, itemcol_])
        Y_parts.append([valcol])
        N += defined.size

    X = np.empty((N, 2))
    Y = np.empty((N, 1))
    i = 0
    for (x_1, x_2), (y_p,) in zip(X_parts, Y_parts):
        X[i:i + x_1.size, 0] = x_1
        X[i:i + x_2.size, 1] = x_2
        Y[i:i + y_p.size, 0] = y_p
        i += y_p.size
    assert i == N

    table = Orange.data.Table.from_numpy(outdomain, X, Y)
    return table


def main(argv=None):
    app = QApplication(list(argv) if argv else [])
    argv = app.arguments()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "brown-selected"

    data = Orange.data.Table(filename)
    w = OWReshape()
    w.show()
    w.raise_()
    w.set_data(data)
    w.handleNewSignals()
    app.exec()
    w.set_data(None)
    w.handleNewSignals()
    w.onDeleteWidget()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
