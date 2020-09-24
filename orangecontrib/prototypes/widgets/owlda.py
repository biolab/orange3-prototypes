import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from Orange.data import Table, Domain, StringVariable, ContinuousVariable
from Orange.preprocess import Impute, Continuize
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview


class OWLDA(OWWidget):
    name = "LDA"
    description = "LDA optimization of linear projections."
    icon = "icons/LDA.svg"

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        transformed = Output("Transformed data", Table, default=True)
        components = Output("Components", Table)

    want_main_area = False
    resizing_enabled = False

    learner_name = settings.Setting("LDA")
    autocommit = settings.Setting(True)

    class Error(OWWidget.Error):
        no_disc_class = Msg("Data with a discrete class variable expected.")
        no_pos_definite = Msg("Eigenvalues cannot be computed.")

    def __init__(self):
        super().__init__(self)
        self.data = None
        self.preprocessed_data = None

        gui.lineEdit(gui.widgetBox(self.controlArea, self.tr("Name")),
                     self, "learner_name")
        gui.auto_apply(self.controlArea, self, "autocommit")

    @Inputs.data
    def set_data(self, data):
        self.Error.no_disc_class.clear()
        self.Error.no_pos_definite.clear()
        if data and not data.domain.has_discrete_class:
            self.Error.no_disc_class()
            self.data = None
            return
        self.data = data
        self.commit()

    def commit(self):
        transformed = components = None

        if self.data is not None:
            self.preprocessed_data = self.data
            preprocessor = [Impute(), Continuize()]
            for pp in preprocessor:
                self.preprocessed_data = pp(self.preprocessed_data)

            lda = LinearDiscriminantAnalysis(solver='eigen', n_components=2)
            try:
                X = lda.fit_transform(self.preprocessed_data.X,
                                      self.preprocessed_data.Y)
            except np.linalg.LinAlgError:
                self.Error.no_pos_definite()
                self.Outputs.transformed.send(None)
                self.Outputs.components.send(None)
                return
            dom = Domain([ContinuousVariable('Component 1'),
                          ContinuousVariable('Component 2')],
                         self.data.domain.class_vars, self.data.domain.metas)
            transformed = Table.from_numpy(dom, X, self.data.Y, self.data.metas)

            transformed.name = self.data.name + ' (LDA)'
            dom = Domain(self.data.domain.attributes,
                         metas=[StringVariable(name='component')])
            metas = np.array([['Component {}'.format(i + 1)
                              for i in range(lda.scalings_.shape[1])]],
                             dtype=object).T
            components = Table.from_numpy(dom, lda.scalings_.T, metas=metas)
            components.name = 'components'

        self.Outputs.transformed.send(transformed)
        self.Outputs.components.send(components)


if __name__ == "__main__":
    WidgetPreview(OWLDA).run(Table("iris"))

