from Orange.data.table import Table
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget
from PyQt4 import QtGui
from Orange.classification.random_forest import RandomForestClassifier
from Orange.classification.tree import TreeClassifier


class OWPythagoreanForest(OWWidget):
    name = 'Pythagorean forest'
    description = 'Pythagorean forest for visualising random forests.'
    priority = 100

    # Enable the save as feature
    graph_name = True

    inputs = [('Random forest', RandomForestClassifier, 'set_rf')]
    outputs = [('Tree', TreeClassifier)]

    # Settings
    depth_limit = settings.ContextSetting(10)
    target_class_index = settings.ContextSetting(0)
    size_calc_idx = settings.Setting(0)
    size_log_scale = settings.Setting(2)
    tooltips_enabled = settings.Setting(True)

    def __init__(self):
        super().__init__()
        # Instance variables
        # The raw skltree model that was passed to the input
        self.model = None
        self.dataset = None
        self.clf_dataset = None

        self.color_palette = None

        # CONTROL AREA
        # Tree info area
        box_info = gui.widgetBox(self.controlArea, 'Tree')
        self.info = gui.widgetLabel(box_info, label='No tree.')

        # Display controls area
        box_display = gui.widgetBox(self.controlArea, 'Display')

        # Stretch to fit the rest of the unsused area
        gui.rubber(self.controlArea)

        # Bottom options
        self.inline_graph_report()

        self.controlArea.setSizePolicy(
            QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)

        self.scene = QtGui.QGraphicsScene(self)

        self.view = TileableGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.mainArea.layout().addWidget(self.view)

        self.resize(800, 500)

    def set_rf(self, model=None):
        """When a different forest is given."""
        self.clear()
        self.model = model

        if model is not None:
            tree = model.skl_model.estimators_[0]
            clf = TreeClassifier(tree)
            clf.domain = model.domain
            clf.instances = model.instances
            self.send('Tree', clf)

    def clear(self):
        """Clear all relevant data from the widget."""
        self.model = None

    def onDeleteWidget(self):
        """When deleting the widget."""
        super().onDeleteWidget()
        self.clear()

    def commit(self):
        """Commit the selected tree to output."""
        pass

    def send_report(self):
        pass


class TileableGraphicsView(QtGui.QGraphicsView):
    pass


def main():
    import sys
    import Orange
    from Orange.classification.random_forest import RandomForestLearner

    argv = sys.argv
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = 'iris'

    app = QtGui.QApplication(argv)
    ow = OWPythagoreanForest()
    data = Orange.data.Table(filename)
    clf = RandomForestLearner()(data)
    ow.set_rf(clf)

    ow.show()
    ow.raise_()
    ow.handleNewSignals()
    app.exec_()

    sys.exit(0)


if __name__ == '__main__':
    main()
