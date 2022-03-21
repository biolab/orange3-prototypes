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


class SplitColumn:
    def __init__(self, data, attr, delimiter):
        self.attr = attr
        self.delimiter = delimiter

        column = self.get_string_values(data, self.attr)
        values = [s.split(self.delimiter) for s in column]
        self.new_values = sorted({val if val else "?" for vals in values for
                                  val in vals})

    def __call__(self, data):
        column = self.get_string_values(data, self.attr)
        values = [set(s.split(self.delimiter)) for s in column]
        shared_data = {v: [i for i, xs in enumerate(values) if v in xs] for v
                       in self.new_values}
        return shared_data

    @staticmethod
    def get_string_values(data, var):
        # turn discrete to string variable
        column = data.get_column_view(var)[0]
        if var.is_discrete:
            return [var.str_val(x) for x in column]
        return column


class OneHotStrings(SharedComputeValue):

    def __init__(self, fn, new_feature):
        super().__init__(fn)
        self.new_feature = new_feature

    def compute(self, data, shared_data):
        indices = shared_data[self.new_feature]
        col = np.zeros(len(data))
        col[indices] = 1
        return col


class OWSplit(OWWidget):
    name = "Split"
    description = "Split string variables to create discrete."
    icon = "icons/Split.svg"
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

        sc = SplitColumn(self.data, var, self.delimiter)

        new_columns = tuple(DiscreteVariable(
                get_unique_names(self.data.domain, v), values=("0", "1"),
                compute_value=OneHotStrings(sc, v)
            ) for v in sc.new_values)

        new_domain = Domain(
            self.data.domain.attributes + new_columns,
            self.data.domain.class_vars, self.data.domain.metas
        )
        extended_data = self.data.transform(new_domain)
        self.Outputs.data.send(extended_data)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWSplit).run(Table.from_file(
        "tests/orange-in-education.tab"))
