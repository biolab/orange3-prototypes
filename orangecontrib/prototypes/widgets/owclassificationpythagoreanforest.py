import numpy as np
from Orange.classification.random_forest import RandomForestClassifier
import Orange.widgets
from Orange.classification.tree import TreeClassifier
from PyQt4 import QtGui

from orangecontrib.prototypes.widgets.owpythagoreanforest import \
    OWPythagoreanForest


class OWClassificationPythagoreanForest(OWPythagoreanForest):
    name = 'Classification Pythagorean forest'
    description = 'Pythagorean forest for visualising classification random ' \
                  'forests.'
    priority = 100

    inputs = [('Random forest', RandomForestClassifier, 'set_rf')]
    outputs = [('Tree', TreeClassifier)]

    # MODEL CHANGED METHODS
    def _update_target_class_combo(self):
        self._clear_target_class_combo()
        self.ui_target_class_combo.addItem('None')
        values = [c.title() for c in
                  self.model.domain.class_vars[0].values]
        self.ui_target_class_combo.addItems(values)

    # MODEL REMOVED CONTROL ELEMENTS CLEAR METHODS
    def _clear_target_class_combo(self):
        self.ui_target_class_combo.clear()
        self.ui_target_class_index = 0
        self.ui_target_class_combo.setCurrentIndex(self.target_class_index)

    # HELPFUL METHODS
    def _get_node_color(self, adapter, tree_node):
        # this is taken almost directly from the existing classification tree
        # viewer
        colors = self.color_palette
        distribution = adapter.get_distribution(tree_node.label)[0]
        total = adapter.num_samples(tree_node.label)

        if self.target_class_index:
            p = distribution[self.target_class_index - 1] / total
            color = colors[self.target_class_index - 1].light(200 - 100 * p)
        else:
            modus = np.argmax(distribution)
            p = distribution[modus] / (total or 1)
            color = colors[int(modus)].light(400 - 300 * p)
        return color

    def _get_color_palette(self):
        if self.model.domain.class_var.is_discrete:
            colors = [QtGui.QColor(*c)
                      for c in self.model.domain.class_var.colors]
        else:
            colors = None
        return colors


def main():
    import sys
    import Orange.widgets
    from Orange.classification.random_forest import RandomForestLearner

    argv = sys.argv
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = 'iris'

    app = QtGui.QApplication(argv)
    ow = OWClassificationPythagoreanForest()
    data = Orange.data.Table(filename)
    clf = RandomForestLearner(n_estimators=100, random_state=42)(data)
    clf.instances = data
    ow.set_rf(clf)

    ow.show()
    ow.raise_()
    ow.handleNewSignals()
    app.exec_()

    sys.exit(0)


if __name__ == '__main__':
    main()
