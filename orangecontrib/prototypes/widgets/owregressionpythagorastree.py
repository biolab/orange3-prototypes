from functools import lru_cache

import Orange
from Orange.regression.tree import TreeRegressor
from Orange.widgets.utils.colorpalette import ContinuousPaletteGenerator
from PyQt4 import QtGui

from orangecontrib.prototypes.widgets.owpythagorastree import OWPythagorasTree
from orangecontrib.prototypes.widgets.pythagorastreeviewer import \
    SklTreeAdapter

import numpy as np


class OWRegressionPythagorasTree(OWPythagorasTree):
    name = 'Regression Pythagoras Tree'
    description = 'Generalized Pythagoras Tree for visualizing regression' \
                  ' trees.'
    priority = 100

    inputs = [('Regression Tree', TreeRegressor, 'set_tree')]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.REGRESSION_COLOR_CALC = [
            ('None', lambda _, __: QtGui.QColor(255, 255, 255)),
            ('Instances in node', self._color_num_nodes),
            ('Class mean', self._color_class_mean),
            ('Standard deviation', self._color_stddev),
        ]

    def _update_target_class_combo(self):
        self._clear_target_class_combo()
        self.target_class_combo.addItems(
            list(zip(*self.REGRESSION_COLOR_CALC))[0])
        self.target_class_combo.setCurrentIndex(self.target_class_index)

    def _clear_target_class_combo(self):
        # Since the target classes are just the different coloring methods,
        # we can reuse the selected index when the tree changes, unlike with
        # the classification tree.
        self.target_class_combo.clear()

    def _get_color_palette(self):
        return ContinuousPaletteGenerator(
            *self.tree_adapter.domain.class_var.colors)

    def _get_node_color(self, adapter, tree_node):
        return self.REGRESSION_COLOR_CALC[self.target_class_index][1](
            adapter, tree_node
        )

    def _color_num_nodes(self, adapter, tree_node):
        # calculate node colors relative to the numbers of samples in the node
        total_samples = adapter.num_samples(adapter.root)
        num_samples = adapter.num_samples(tree_node.label)
        return self.color_palette[num_samples / total_samples]

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

    def _get_tooltip(self, node):
        total = self.tree_adapter.num_samples(self.tree_adapter.root)
        samples = self.tree_adapter.num_samples(node.label)
        ratio = samples / total
        impurity = self.tree_adapter.get_impurity(node.label)

        rules = self.tree_adapter.rules(node.label)
        rules = '<br>'.join(
            '%s %s %s' % (n, s, v) for n, s, v in rules) \

        splitting_attr = self.tree_adapter.attribute(node.label)

        return '<p>Impurity: {:2.3f}%'.format(impurity) \
            + '<br>{}/{} samples ({:2.3f}%)'.format(
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
    from Orange.regression.tree import TreeRegressionLearner

    argv = sys.argv
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = 'housing'

    app = QtGui.QApplication(argv)
    ow = OWRegressionPythagorasTree()
    data = Orange.data.Table(filename)
    reg = TreeRegressionLearner(max_depth=25)(data)
    reg.instances = data
    ow.set_tree(reg)

    ow.show()
    ow.raise_()
    ow.handleNewSignals()
    app.exec_()

    sys.exit(0)


if __name__ == '__main__':
    main()
