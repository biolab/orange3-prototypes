import numpy

from sklearn.decomposition import FactorAnalysis

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QTableView

from Orange.data import Table, Domain
from Orange.widgets import settings
from Orange.widgets.widget import OWWidget
from orangewidget.widget import Input, Output
from orangewidget.utils.widgetpreview import WidgetPreview
from orangewidget import gui

class OWFactorAnalysis(OWWidget):
    name = "Factor Analysis"
    description = "Randomly selects a subset of instances from the dataset."
    icon = "icons/DataSamplerB.svg"
    priority = 20

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        sample = Output("Sampled Data", Table)

    settingsHandler = settings.DomainContextHandler()   # TODO puciga
    n_components = settings.ContextSetting(1)
    commitOnChange = settings.Setting(0)        #TODO puciga
    autocommit = settings.Setting(True)

    def __init__(self): # __init__ je konstruktor (prva metoda ki se klice, ko se ustvari instance tega razreda)
        super().__init__()  # Ker je OWFactorAnalysis derivat OWWIdgeta (deduje iz OWWidgeta), najprej inicializiram njega
        self.dataset = None

        # Control area settings
        self.optionsBox = gui.widgetBox(self.controlArea, "Options")
        gui.spin(
            self.optionsBox,
            self,
            "n_components",
            minv=1,
            maxv=100,
            step=1,
            label="Number of components:",
            callback=[self.factor_analysis, self.commit.deferred], #deferred = zapoznelo
        )

        gui.auto_commit(
            self.controlArea, self, 'autocommit', 'Commit',
            orientation=Qt.Horizontal)

        gui.separator(self.controlArea)  # TODO tega mogoce ne potrebujem ker je to za zadnjim elementom v controlArea

        # Main area settings
        self.mainArea.setVisible(True)

        gui.separator(self.mainArea) # TODO tega mogoce ne potrebujem, ker je to pred prvim elementom v mainArea

        box = gui.vBox(self.mainArea, box = "Eigenvalue Scores")
        self.left_side.setContentsMargins(0,0,0,0)
        table = self.table_view = QTableView(self.mainArea)
        #table.setModel(self.table_model)
        table.setSelectionMode(QTableView.SingleSelection)
        table.setSelectionBehavior(QTableView.SelectRows)
        table.setItemDelegate(gui.ColoredBarItemDelegate(self, color=Qt.cyan))
        #table.selectionModel().selectionChanged.connect(self.select_row)
        table.setMaximumWidth(300)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().hide()
        table.setShowGrid(False)
        box.layout().addWidget(table)


    @Inputs.data
    def set_data(self, dataset):
        self.closeContext()
        if dataset is None:
            self.sample = None
        else:
            self.openContext(dataset.domain)  #Kaj je funkcija contexta

        self.dataset = dataset
        self.optionsBox.setDisabled(False)
        self.commit.now() # Takoj poklici metodo commit

    def factor_analysis(self):
        # Z izbranimi n_componentami izracunaj FA na self.dataset
        result = FactorAnalysis(self.n_components).fit(self.dataset.X)
        # Iz spremenljivke result (ki je instance nekega razreda) izlusci samo tabelo ki nas zanima
        self.components = result.components_

        # Pretvori tabelo nazaj v Orange.data.Table
        self.result = Table.from_numpy(Domain(self.dataset.domain.attributes),
                                       self.components)

    @gui.deferred
    def commit(self):
        if self.dataset is None:
            self.Outputs.sample.send(None)
        else:
            self.factor_analysis()
            # Poslji self.result v Outputs channel.
            self.Outputs.sample.send(self.result)


if __name__ == "__main__":
    WidgetPreview(OWFactorAnalysis).run(Table("iris"))