from enum import IntEnum
from operator import attrgetter

import numpy as np
from Orange.widgets.utils.signals import Input, Output
from scipy.stats import spearmanr

from AnyQt.QtCore import Qt, QItemSelectionModel, QItemSelection, QSize
from AnyQt.QtGui import QStandardItem
from AnyQt.QtWidgets import QHeaderView

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.preprocess import SklImpute
from Orange.widgets import gui
from Orange.widgets.settings import Setting, ContextSetting, \
    DomainContextHandler
from Orange.widgets.visualize.utils import VizRankDialogAttrPair
from Orange.widgets.widget import OWWidget, AttributeList, Msg


class CorrelationType(IntEnum):
    PEARSON, SPEARMAN = 0, 1

    @staticmethod
    def items():
        return ["Pairwise Pearson correlation", "Pairwise Spearman correlation"]


class CorrelationRank(VizRankDialogAttrPair):
    def initialize(self):
        super().initialize()
        data = self.master.cont_data
        self.attrs = data and data.domain.attributes
        self.model_proxy.setFilterKeyColumn(-1)
        self.rank_table.horizontalHeader().setStretchLastSection(False)

    def compute_score(self, state):
        (a1, a2), corr_type = state, self.master.correlation_type
        if corr_type == CorrelationType.PEARSON:
            return -np.corrcoef(self.master.cont_data.X[:, [a1, a2]].T)[0, 1]
        else:
            return -spearmanr(self.master.cont_data.X[:, [a1, a2]])[0]

    def row_for_state(self, score, state):
        attrs = sorted((self.attrs[x] for x in state), key=attrgetter("name"))
        attr_1_item = QStandardItem(attrs[0].name)
        attr_2_item = QStandardItem(attrs[1].name)
        correlation_item = QStandardItem(str(round(-score, 3)))
        attr_1_item.setData(attrs, self._AttrRole)
        attr_2_item.setData(attrs, self._AttrRole)
        correlation_item.setData(attrs)
        correlation_item.setData(Qt.AlignCenter, Qt.TextAlignmentRole)
        return [attr_1_item, attr_2_item, correlation_item]

    def check_preconditions(self):
        return self.master.cont_data is not None


class OWCorrelations(OWWidget):
    name = "Correlations"
    description = "Compute all pairwise attribute correlations."
    icon = "icons/Correlations.svg"
    priority = 2000

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)
        features = Output("Features", AttributeList)
        correlations = Output("Correlations", Table)

    want_control_area = False

    settingsHandler = DomainContextHandler()
    selection = ContextSetting(())
    correlation_type = Setting(0)

    class Information(OWWidget.Information):
        not_enough_vars = Msg("Need at least two continuous features.")
        not_enough_inst = Msg("Need at least two instances.")

    def __init__(self):
        super().__init__()
        self.data = None
        self.cont_data = None

        # GUI
        box = gui.vBox(self.mainArea)
        self.correlation_combo = gui.comboBox(
            box, self, "correlation_type", items=CorrelationType.items(),
            orientation=Qt.Horizontal, callback=self._correlation_combo_changed)

        self.vizrank, _ = CorrelationRank.add_vizrank(
            None, self, None, self._vizrank_selection_changed)

        gui.separator(box)
        box.layout().addWidget(self.vizrank.filter)
        box.layout().addWidget(self.vizrank.rank_table)

        button_box = gui.hBox(self.mainArea)
        button_box.layout().addWidget(self.vizrank.button)

    def sizeHint(self):
        return QSize(350, 400)

    def _correlation_combo_changed(self):
        self.apply()

    def _vizrank_selection_changed(self, *args):
        self.selection = args
        self.commit()

    def _vizrank_select(self):
        model = self.vizrank.rank_table.model()
        selection = QItemSelection()
        for i in range(model.rowCount()):
            if model.data(model.index(i, 0)) == self.selection[0].name and \
                    model.data(model.index(i, 1)) == self.selection[1].name:
                selection.select(model.index(i, 0), model.index(i, 2))
                self.vizrank.rank_table.selectionModel().select(
                    selection, QItemSelectionModel.ClearAndSelect)
                break

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.clear_messages()
        self.data = data
        self.cont_data = None
        self.selection = ()
        if data is not None:
            cont_attrs = [a for a in data.domain.attributes if a.is_continuous]
            if len(cont_attrs) < 2:
                self.Information.not_enough_vars()
            elif len(data) < 2:
                self.Information.not_enough_inst()
            else:
                domain = data.domain
                cont_dom = Domain(cont_attrs, domain.class_vars, domain.metas)
                self.cont_data = SklImpute()(Table.from_table(cont_dom, data))
        self.apply()
        self.openContext(self.data)
        self._vizrank_select()

    def apply(self):
        self.vizrank.initialize()
        if self.cont_data is not None:
            # this triggers self.commit() by changing vizrank selection
            self.vizrank.toggle()
            header = self.vizrank.rank_table.horizontalHeader()
            header.setStretchLastSection(True)
            header.setSectionResizeMode(QHeaderView.ResizeToContents)
        else:
            self.commit()

    def commit(self):
        if self.data is None or self.cont_data is None:
            self.Outputs.data.send(self.data)
            self.Outputs.features.send(None)
            self.Outputs.correlations.send(None)
            return

        metas = [StringVariable("Feature 1"), StringVariable("Feature 2")]
        domain = Domain([ContinuousVariable("Correlation")], metas=metas)
        model = self.vizrank.rank_model
        x = np.array([[float(model.data(model.index(row, 2)))] for row
                      in range(model.rowCount())])
        m = np.array([[model.data(model.index(row, 0)),
                       model.data(model.index(row, 1))] for row
                      in range(model.rowCount())], dtype=object)
        corr_table = Table(domain, x, metas=m)
        corr_table.name = "Correlations"

        self.Outputs.data.send(self.data)
        # data has been imputed; send original attributes
        self.Outputs.features.send(AttributeList([attr.compute_value.variable
                                                  for attr in self.selection]))
        self.Outputs.correlations.send(corr_table)

    def send_report(self):
        self.report_table(CorrelationType.items()[self.correlation_type],
                          self.vizrank.rank_table)


if __name__ == "__main__":
    from AnyQt.QtWidgets import QApplication

    app = QApplication([])
    ow = OWCorrelations()
    iris = Table("iris")
    ow.set_data(iris)
    ow.show()
    app.exec_()
    ow.saveSettings()
