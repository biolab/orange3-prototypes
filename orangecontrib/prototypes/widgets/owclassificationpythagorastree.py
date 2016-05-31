import Orange
import numpy as np
from Orange.classification.tree import TreeClassifier
from PyQt4 import QtGui

from orangecontrib.prototypes.widgets.owpythagorastree import OWPythagorasTree
from orangecontrib.prototypes.widgets.pythagorastreeviewer import \
    SklTreeAdapter


class OWClassificationPythagorasTree(OWPythagorasTree):
    name = 'Classification Pythagoras Tree'
    description = 'Generalized Pythagoras Tree for visualizing clasification' \
                  ' trees.'
    priority = 100

    inputs = [('Classification Tree', TreeClassifier, 'set_tree')]

    def _update_target_class_combo(self):
        self._clear_target_class_combo()
        self.target_class_combo.addItem('None')
        values = [c.title() for c in
                  self.tree_adapter.domain.class_vars[0].values]
        self.target_class_combo.addItems(values)

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

    def _get_tooltip(self, node):
        distribution = self.tree_adapter.get_distribution(node.label)[0]
        total = self.tree_adapter.num_samples(node.label)
        if self.target_class_index:
            samples = distribution[self.target_class_index - 1]
            text = ''
        else:
            modus = np.argmax(distribution)
            samples = distribution[modus]
            text = self.tree_adapter.domain.class_vars[0].values[modus] + \
                '<br>'
        ratio = samples / np.sum(distribution)

        rules = self.tree_adapter.rules(node.label)
        rules = '<br>'.join(
            '%s %s %s' % (n, s, v) for n, s, v in rules) \

        splitting_attr = self.tree_adapter.attribute(node.label)

        return '<p>' \
            + text \
            + '{}/{} samples ({:2.3f}%)'.format(
                int(samples), total, ratio * 100) \
            + '<br><br>Split by ' + splitting_attr.name \
            + '<hr>' \
            + rules \
            + '</p>'

    def _get_tree_adapter(self, model):
        return SklTreeAdapter(
            model.skl_model.tree_,
            model.domain,
            adjust_weight=self.SIZE_CALCULATION[self.size_calc_idx][1],
        )


def main():
    import sys
    from Orange.classification.tree import TreeLearner

    argv = sys.argv
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = 'iris'

    app = QtGui.QApplication(argv)
    ow = OWClassificationPythagorasTree()
    data = Orange.data.Table(filename)
    clf = TreeLearner(max_depth=1000)(data)
    clf.instances = data
    ow.set_tree(clf)

    ow.show()
    ow.raise_()
    ow.handleNewSignals()
    app.exec_()

    sys.exit(0)


if __name__ == '__main__':
    main()
