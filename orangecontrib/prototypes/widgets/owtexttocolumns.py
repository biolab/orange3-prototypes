from functools import partial

import numpy as np

from AnyQt.QtCore import Qt

from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler
from Orange.widgets.widget import OWWidget, Msg, Output, Input
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.data import Table, Domain, DiscreteVariable, StringVariable
from Orange.data.util import SharedComputeValue, get_unique_names

from orangewidget.settings import Setting


def get_substrings(values, delimiter):
    return sorted({ss.strip() for s in values for ss in s.split(delimiter)}
                  - {""})


class SplitColumn:
    def __init__(self, data, attr, delimiter):
        self.attr = attr
        self.delimiter = delimiter
        column = set(data.get_column(self.attr))
        self.new_values = tuple(get_substrings(column, self.delimiter))

    def __call__(self, data):
        column = data.get_column(self.attr)
        values = [{ss.strip() for ss in s.split(self.delimiter)}
                  for s in column]
        return {v: np.array([i for i, xs in enumerate(values) if v in xs])
                for v in self.new_values}

    def __eq__(self, other):
        return self.attr == other.attr \
               and self.delimiter == other.delimiter \
               and self.new_values == other.new_values

    def __hash__(self):
        return hash((self.attr, self.delimiter, self.new_values))


class OneHotStrings(SharedComputeValue):
    def __init__(self, fn, new_feature):
        super().__init__(fn)
        self.new_feature = new_feature

    def compute(self, data, shared_data):
        indices = shared_data[self.new_feature]
        col = np.zeros(len(data))
        col[indices] = 1
        return col

    def __eq__(self, other):
        return super().__eq__(other) and self.new_feature == other.new_feature

    def __hash__(self):
        return super().__hash__() ^ hash(self.new_feature)


class OneHotDiscrete:
    def __init__(self, variable, delimiter, value):
        self.variable = variable
        self.value = value
        self.delimiter = delimiter

    def __call__(self, data):
        column = data.get_column(self.variable).astype(float)
        col = np.zeros(len(column))
        col[np.isnan(column)] = np.nan
        for val_idx, value in enumerate(self.variable.values):
            if self.value in value.split(self.delimiter):
                col[column == val_idx] = 1
        return col

    def __eq__(self, other):
        return self.variable == other.variable \
               and self.value == other.value \
               and self.delimiter == other.delimiter

    def __hash__(self):
        return hash((self.variable, self.value, self.delimiter))


class OWTextToColumns(OWWidget):
    name = "Text to Columns"
    description = "Split text or categorical variables into binary indicators"
    icon = "icons/TextToColumns.svg"
    keywords = ["split"]
    priority = 700

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    class Warning(OWWidget.Warning):
        no_disc = Msg("Data contains only numeric variables.")

    want_main_area = False
    resizing_enabled = False

    settingsHandler = DomainContextHandler()
    attribute = ContextSetting(None)
    delimiter = ContextSetting(";")
    auto_apply = Setting(True)

    def __init__(self):
        super().__init__()
        self.data = None

        variable_select_box = gui.vBox(self.controlArea, "Variable")

        gui.comboBox(variable_select_box, self, "attribute",
                     orientation=Qt.Horizontal, searchable=True,
                     callback=self.apply.deferred,
                     model=DomainModel(valid_types=(StringVariable,
                                                    DiscreteVariable)))
        gui.lineEdit(
            variable_select_box, self, "delimiter",
            orientation=Qt.Horizontal, callback=self.apply.deferred)

        gui.auto_apply(self.buttonsArea, self, commit=self.apply)

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.data = data

        model = self.controls.attribute.model()
        model.set_domain(data.domain if data is not None else None)
        self.Warning.no_disc(shown=data is not None and not model)
        if not model:
            self.attribute = None
            self.data = None
            return
        self.attribute = model[0]
        self.openContext(data)
        self.apply.now()

    @gui.deferred
    def apply(self):
        if self.attribute is None:
            self.Outputs.data.send(None)
            return
        var = self.data.domain[self.attribute]

        if var.is_discrete:
            values = get_substrings(var.values, self.delimiter)
            computer = partial(OneHotDiscrete, var, self.delimiter)
        else:
            sc = SplitColumn(self.data, var, self.delimiter)
            values = sc.new_values
            computer = partial(OneHotStrings, sc)
        names = get_unique_names(self.data.domain, values, equal_numbers=False)

        new_columns = tuple(DiscreteVariable(
            name, values=("0", "1"), compute_value=computer(value)
        ) for value, name in zip(values, names))

        new_domain = Domain(
            self.data.domain.attributes + new_columns,
            self.data.domain.class_vars, self.data.domain.metas
        )
        extended_data = self.data.transform(new_domain)
        self.Outputs.data.send(extended_data)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWTextToColumns).run(Table.from_file(
        "tests/orange-in-education.tab"))
