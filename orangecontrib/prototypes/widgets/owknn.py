from Orange.regression import KNNRegressionLearner
from PyQt4.QtCore import Qt

from Orange.data import Table
from Orange.classification import KNNLearner
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from orangecontrib.prototypes.widgets.utils.multiinput import (
    MultiInputMixin,
    InputTypes,
)


class OWKNNLearner(OWBaseLearner, MultiInputMixin):
    name = "Nearest Neighbors"
    description = "Predict according to the nearest training instances."
    icon = "icons/KNN.svg"
    priority = 20

    LEARNER = KNNLearner

    # TODO The MultiInput metaclass should take care of this
    handlers = {}
    trigger = 'set_data'

    weights = ["uniform", "distance"]
    metrics = ["euclidean", "manhattan", "chebyshev", "mahalanobis"]

    learner_name = Setting("kNN")
    n_neighbors = Setting(5)
    metric_index = Setting(0)
    weight_type = Setting(0)

    def __init__(self):
        super().__init__()
        MultiInputMixin.__init__(self)

    def add_main_layout(self):
        box = gui.vBox(self.controlArea, "Neighbors")
        self.n_neighbors_spin = gui.spin(
            box, self, "n_neighbors", 1, 100, label="Number of neighbors:",
            alignment=Qt.AlignRight, callback=self.settings_changed)
        self.metrics_combo = gui.comboBox(
            box, self, "metric_index", orientation=Qt.Horizontal,
            label="Metric:", items=[i.capitalize() for i in self.metrics],
            callback=self.settings_changed)
        self.weights_combo = gui.comboBox(
            box, self, "weight_type", orientation=Qt.Horizontal,
            label="Weight:", items=[i.capitalize() for i in self.weights],
            callback=self.settings_changed)

    def create_learner(self):
        return self.LEARNER(
            n_neighbors=self.n_neighbors,
            metric=self.metrics[self.metric_index],
            weights=self.weights[self.weight_type],
            preprocessors=self.preprocessors
        )

    def get_learner_parameters(self):
        return (("Number of neighbours", self.n_neighbors),
                ("Metric", self.metrics[self.metric_index].capitalize()),
                ("Weight", self.weights[self.weight_type].capitalize()))


@OWKNNLearner.data_handler(target_type=InputTypes.DISCRETE)
class KNNClassification:
    LEARNER = KNNLearner
    weights = ["uniform"]


@OWKNNLearner.data_handler(target_type=InputTypes.CONTINUOUS)
class KNNRegression:
    LEARNER = KNNRegressionLearner
    weights = ["distance"]


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication

    a = QApplication(sys.argv)
    ow = OWKNNLearner()
    d = Table('iris')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
