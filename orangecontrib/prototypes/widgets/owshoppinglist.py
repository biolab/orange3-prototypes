from itertools import chain
from typing import Union, Optional

import numpy as np
from scipy import sparse as sp

from AnyQt.QtWidgets import QFormLayout

from orangewidget.utils.widgetpreview import WidgetPreview
from orangewidget.widget import Msg

from Orange.data import \
    Table, Domain, DiscreteVariable, StringVariable, ContinuousVariable
from Orange.widgets import widget, gui
from Orange.widgets.settings import \
    DomainContextHandler, Setting, ContextSetting
from Orange.widgets.utils import itemmodels


DEFAULT_ITEM_NAME = "Item"
DEFAULT_VALUE_NAME = "Value"


class OWShoppingList(widget.OWWidget):
    name = "Shopping List"
    description = "Reshape a table to a list of item-value pairs"
    icon = "icons/ToShoppingList.svg"

    class Inputs:
        data = widget.Input("Data", Table)

    class Outputs:
        data = widget.Output("Data", Table)

    class Warning(widget.OWWidget.Warning):
        no_suitable_features = Msg("No suitable columns for id")

    want_main_area = False
    resizing_enabled = False

    settingsHandler = DomainContextHandler()
    idvar: Union[DiscreteVariable, StringVariable] = ContextSetting(None)
    only_numeric = Setting(True)
    exclude_zeros = Setting(False)
    item_var_name = Setting("")
    value_var_name = Setting("")
    auto_apply = Setting(True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data: Optional[Table] = None

        box = gui.widgetBox(self.controlArea, "Identifier")
        self.idvar_model = itemmodels.VariableListModel()
        self.var_cb = gui.comboBox(
            box, self, "idvar", model=self.idvar_model,
            callback=self._invalidate, minimumContentsLength=16,
            tooltip="A column with identifier, like customer's id")

        box = gui.widgetBox(self.controlArea, "Filter")
        gui.checkBox(
            box, self, "only_numeric", "Treat only numeric columns as items",
            callback=self._invalidate)
        gui.checkBox(
            box, self, "exclude_zeros", "Exclude zero values",
            callback=self._invalidate,
            tooltip="Besides missing values, also omit items with zero values")

        form = QFormLayout()
        gui.widgetBox(
            self.controlArea, "Names for generated features", orientation=form)
        form.addRow("Item:",
                    gui.lineEdit(
                        None, self, "item_var_name",
                        callback=self._invalidate,
                        placeholderText=DEFAULT_ITEM_NAME,
                        styleSheet="padding-left: 3px"))
        form.addRow("Value:",
                    gui.lineEdit(
                        None, self, "value_var_name",
                        callback=self._invalidate,
                        placeholderText=DEFAULT_VALUE_NAME,
                        styleSheet="padding-left: 3px"))

        gui.auto_apply(self.controlArea, self)

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.Warning.clear()
        self.idvar_model[:] = []
        self.data = data
        if data is not None:
            self.idvar_model[:] = (
                var
                for var in chain(data.domain.variables, data.domain.metas)
                if isinstance(var, (DiscreteVariable, StringVariable))
                and self._is_unique(var))
            if not self.idvar_model:
                self.Warning.no_suitable_features()

        self.idvar = self.idvar_model[0] if self.idvar_model else None
        self.openContext(data)
        self.commit()

    def _is_unique(self, var):
        col = self.data.get_column_view(var)[0]
        col = col[self._notnan_mask(col)]
        return len(col) == len(set(col))

    @staticmethod
    def _notnan_mask(col):
        return np.isfinite(col) if col.dtype == float else col != ""

    def _invalidate(self):
        self.commit()

    def commit(self):
        self.Error.clear()
        if self.idvar is None:
            self.Outputs.data.send(None)
        else:
            self.Outputs.data.send(self._reshape_to_long())

    def _reshape_to_long(self):
        # Get a mask with columns used for data
        useful_vars = self._get_useful_vars()
        item_names = self._get_item_names(useful_vars)
        n_useful = len(item_names)

        # Get identifiers, remove rows with missing id data
        idvalues, _ = self.data.get_column_view(self.idvar)
        idmask = self._notnan_mask(idvalues)
        x = self.data.X[idmask]
        idvalues = idvalues[idmask]

        # For string ids, use indices and store names
        if self.idvar.is_string:
            id_names = idvalues
            idvalues = np.arange(len(idvalues))
        else:
            id_names = False

        # Prepare columns of the long list
        if sp.issparse(x):
            xcoo = x.tocoo()
            col_selection = useful_vars[xcoo.col]
            idcol = idvalues[xcoo.row[col_selection]]
            items = xcoo.col[col_selection]
            items = (np.cumsum(useful_vars) - 1)[items]  # renumerate
            values = xcoo.data[col_selection]
        else:
            idcol = np.repeat(idvalues, n_useful)
            items = np.tile(np.arange(n_useful), len(x))
            values = x[:, useful_vars].flatten()

        # Create a mask for removing long-list entries with missing or zero vals
        # There should be no zero values in sparse matrices, but not a lot of
        # code is required to remove them
        selected = self._notnan_mask(values)
        if self.exclude_zeros:
            included = values != 0
            if not self.only_numeric:
                disc_mask = np.array(
                    [var.is_discrete
                     for var, useful in zip(self.data.domain.attributes, useful_vars)
                     if useful])
                if sp.issparse(x):
                    included |= disc_mask[items]
                else:
                    included |= np.tile(disc_mask, len(x))
            selected &= included

        # Filter the long list
        idcol = idcol[selected]
        items = items[selected]
        values = values[selected]

        domain = self._prepare_domain(item_names, id_names)
        return Table.from_numpy(domain, np.vstack((idcol, items)).T, values)

    def _get_useful_vars(self):
        domain = self.data.domain

        if self.exclude_zeros or self.only_numeric:
            cont_vars = np.array([var.is_continuous for var in domain.attributes])
        if self.only_numeric:
            useful_vars = cont_vars
        else:
            useful_vars = np.full(len(domain.attributes), True)

        ididx = domain.index(self.idvar)
        if ididx >= 0:
            useful_vars[ididx] = False
        return useful_vars

    def _get_item_names(self, useful_vars):
        return tuple(
            var.name
            for var, useful in zip(self.data.domain.attributes, useful_vars)
            if useful)

    def _prepare_domain(self, item_names, idnames=()):
        item_var = DiscreteVariable(
            self.item_var_name or DEFAULT_ITEM_NAME,
            values=item_names)
        value_var = ContinuousVariable(
            self.value_var_name or DEFAULT_VALUE_NAME)
        idvar = self.idvar
        if self.idvar.is_string:
            idvar = DiscreteVariable(idvar.name, values=tuple(idnames))
        return Domain([idvar, item_var], [value_var])


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWShoppingList).run(Table("zoo")[50:])
