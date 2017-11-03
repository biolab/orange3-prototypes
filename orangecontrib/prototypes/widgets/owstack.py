from collections import OrderedDict

from Orange.data import Table
from Orange.base import Learner
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.widget import Msg, Input

from orangecontrib.prototypes.stack import StackedLearner


class OWStackedLearner(OWBaseLearner):
    name = "Stacking"
    description = "Stack multiple models."
    icon = "icons/Stacking.svg"
    priority = 100

    LEARNER = StackedLearner

    learner_name = Setting("Stack")

    class Inputs(OWBaseLearner.Inputs):
        learners = Input("Learners", Learner, multiple=True)
        aggregate = Input("Aggregate", Learner)

    def __init__(self):
        self.learners = OrderedDict()
        self.aggregate = None
        super().__init__()

    def add_main_layout(self):
        pass

    @Inputs.learners
    def set_learners(self, learner, id):
        if id in self.learners and learner is None:
            del self.learners[id]
        elif learner is not None:
            self.learners[id] = learner
        self.apply()

    @Inputs.aggregate
    def set_aggregate(self, aggregate):
        self.aggregate = aggregate
        self.apply()

    def create_learner(self):
        if not self.learners or not self.aggregate:
            return None
        return self.LEARNER(
            tuple(self.learners.values()), self.aggregate,
            preprocessors=self.preprocessors)

    def get_learner_parameters(self):
        return (("Base learners", [l.name for l in self.learners.values()]),
                ("Aggregator", self.aggregate.name))


if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    ow = OWStackedLearner()
    d = Table(sys.argv[1] if len(sys.argv) > 1 else 'iris')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
