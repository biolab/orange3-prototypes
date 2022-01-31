import numpy as np
from AnyQt.QtWidgets import QFormLayout, QSizePolicy

from Orange.data import Table, DiscreteVariable
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME, \
    create_annotated_table
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.utils.widgetpreview import WidgetPreview
from orangewidget.utils.itemmodels import PyListModel
from orangewidget.widget import Msg


class OWQuickSelect(widget.OWWidget):
    name = 'Quick Select'
    icon = 'icons/QuickSelect.svg'
    description = 'Select instances with specific feature value.'

    class Inputs:
        data = widget.Input("Data", Table)

    class Outputs:
        matching = widget.Output("Matching Data", Table, default=True)
        unmatched = widget.Output("Unmatched Data", Table)
        annotated = widget.Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    class Error(widget.OWWidget.Error):
        no_categorical = Msg("No categorical variables")

    want_main_area = False
    resizing_enabled = False

    settingsHandler = settings.DomainContextHandler()
    variable = settings.ContextSetting(None)
    pending_value = settings.ContextSetting("")

    def __init__(self):
        super().__init__()

        # This is not a setting because openContext cannot retrieve it before
        # filling the combo. Settings store this as pending_value
        self.value = ""

        self.data = None
        self.n_matched = None

        form = QFormLayout()
        gui.widgetBox(self.controlArea, orientation=form)

        self.var_model = DomainModel(
            order=DomainModel.MIXED, valid_types=DiscreteVariable)
        var_combo = gui.comboBox(
            None, self, "variable", contentsLength=50,
            model=self.var_model, callback=self._on_variable_changed)

        self.value_model = PyListModel()
        value_combo = gui.comboBox(
            None, self, "value", label="Value: ",
            model=self.value_model, callback=self._on_value_changed,
            contentsLength=50,
            sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))

        form.addRow("Variable:", var_combo)
        form.addRow("Value:", value_combo)

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.data = data
        self.Error.no_categorical.clear()
        # TODO: Check that contexts are retrieved properly, also when removing
        # and re-adding a connection

        if data:
            self.var_model.set_domain(data.domain)
            self.info.set_input_summary(
                len(data), format_summary_details(data))
            if not self.var_model.rowCount():
                self.data = None
                self.Error.no_categorical()
        else:
            self.var_model.set_domain(None)
            self.info.set_input_summary(self.info.NoInput)

        self.variable = self.var_model[0] if self.data else None
        self.openContext(self.data)
        self.set_value_list()
        if self.variable and self.pending_value in self.variable.values:
            self.value = self.pending_value
        self.commit()

    def set_value_list(self):
        if self.variable is None:
            self.value_model.clear()
            self.value = ""
        else:
            self.value_model[:] = self.variable.values
            self.value = self.value_model[0] if self.variable.values else ""

    def _on_variable_changed(self):
        self.set_value_list()
        self.commit()

    def _on_value_changed(self):
        self.pending_value = self.value
        self.commit()

    def commit(self):
        if not (self.data and self.variable and self.value):
            annotated = matching = unmatched = None
            self.n_matched = None
        else:
            column = self.data.get_column_view(self.variable)[0]
            valind = self.variable.values.index(self.value)
            mask = column == valind
            annotated = create_annotated_table(self.data, np.flatnonzero(mask))
            matching = self.data[mask]
            unmatched = self.data[~mask]
            self.n_matched = len(matching)

        self.Outputs.matching.send(matching)
        self.Outputs.unmatched.send(unmatched)
        self.Outputs.annotated.send(annotated)

    def send_report(self):
        if not self.data:
            return

        self.report_items(
            "",
            [("Filter", f"{self.variable.name} = '{self.value}'"),
             ("Matching instances",
              f"{self.n_matched} (out of {len(self.data)})")]
        )


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWQuickSelect).run(Table("heart_disease"))

