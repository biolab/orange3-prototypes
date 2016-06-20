import numpy as np
from Orange.regression.random_forest import RandomForestRegressor
from Orange.regression.tree import TreeRegressor
from Orange.widgets.utils.colorpalette import ContinuousPaletteGenerator
from PyQt4 import QtGui

from orangecontrib.prototypes.widgets.owpythagoreanforest import \
    OWPythagoreanForest


class OWClassificationPythagoreanForest(OWPythagoreanForest):
    name = 'Regression Pythagorean forest'
    description = 'Pythagorean forest for visualising regression random ' \
                  'forests.'
    priority = 100

    inputs = [('Random forest', RandomForestRegressor, 'set_rf')]
    outputs = [('Tree', TreeRegressor)]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.REGRESSION_COLOR_CALC = [
            ('None', lambda _, __: QtGui.QColor(255, 255, 255)),
            ('Class mean', self._color_class_mean),
            ('Standard deviation', self._color_stddev),
        ]

    # MODEL CHANGED METHODS
    def _update_target_class_combo(self):
        self._clear_target_class_combo()
        self.ui_target_class_combo.addItems(
            list(zip(*self.REGRESSION_COLOR_CALC))[0])
        self.ui_target_class_combo.setCurrentIndex(self.target_class_index)

    # MODEL REMOVED CONTROL ELEMENTS CLEAR METHODS
    def _clear_target_class_combo(self):
        # Since the target classes are just the different coloring methods,
        # we can reuse the selected index when the tree changes, unlike with
        # the classification tree.
        self.ui_target_class_combo.clear()

    # HELPFUL METHODS
    def _get_color_palette(self):
        return ContinuousPaletteGenerator(
            *self.forest_adapter.domain.class_var.colors)

    def _get_node_color(self, adapter, tree_node):
        return self.REGRESSION_COLOR_CALC[self.target_class_index][1](
            adapter, tree_node
        )

    def _color_class_mean(self, adapter, tree_node):
        # calculate node colors relative to the mean of the node samples
        min_mean = np.min(self.clf_dataset.Y)
        max_mean = np.max(self.clf_dataset.Y)
        instances = adapter.get_instances_in_nodes(self.clf_dataset, tree_node)
        mean = np.mean(instances.Y)

        return self.color_palette[(mean - min_mean) / (max_mean - min_mean)]

    def _color_stddev(self, adapter, tree_node):
        # calculate node colors relative to the standard deviation in the node
        # samples
        min_mean, max_mean = 0, np.std(self.clf_dataset.Y)
        instances = adapter.get_instances_in_nodes(self.clf_dataset, tree_node)
        std = np.std(instances.Y)

        return self.color_palette[(std - min_mean) / (max_mean - min_mean)]


def main():
    import sys
    import Orange.widgets
    from Orange.regression.random_forest import RandomForestRegressionLearner

    argv = sys.argv
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = 'housing'

    app = QtGui.QApplication(argv)
    ow = OWClassificationPythagoreanForest()
    data = Orange.data.Table(filename)
    clf = RandomForestRegressionLearner(
        n_estimators=100, random_state=42)(data)
    clf.instances = data
    ow.set_rf(clf)

    ow.show()
    ow.raise_()
    ow.handleNewSignals()
    app.exec_()

    sys.exit(0)


if __name__ == '__main__':
    main()
