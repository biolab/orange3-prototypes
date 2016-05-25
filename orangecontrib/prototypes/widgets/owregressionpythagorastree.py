import Orange
from Orange.data.table import Table
from Orange.regression.tree import TreeRegressor
from Orange.widgets.utils.colorpalette import ContinuousPaletteGenerator
from PyQt4 import QtGui

from orangecontrib.prototypes.widgets.owpythagorastree import OWPythagorasTree
from orangecontrib.prototypes.widgets.pythagorastreeviewer import \
    SklTreeAdapter


class OWRegressionPythagorasTree(OWPythagorasTree):
    name = 'Regression Pythagoras Tree'
    description = 'Generalized Pythagoras Tree for visualizing regression' \
                  ' trees.'
    priority = 100

    inputs = [('Regression Tree', TreeRegressor, 'set_tree')]

    def _update_target_class_combo(self):
        self._clear_target_class_combo()
        values = ['Default', 'Instances in node', 'Impurity']
        self.target_class_combo.addItems(values)

    def _get_color_palette(self):
        return ContinuousPaletteGenerator(
            *self.tree_adapter.domain.class_var.colors)

    def _get_node_color(self, tree_node):
        # this is taken almost directly from the existing regression tree
        # viewer
        colors = self.color_palette
        total_samples = self.tree_adapter.num_samples(self.tree_adapter.root)
        max_impurity = self.tree_adapter.get_impurity(self.tree_adapter.root)

        li = [0.5,
              self.tree_adapter.num_samples(tree_node.label) / total_samples,
              self.tree_adapter.get_impurity(tree_node.label) / max_impurity]

        return QtGui.QBrush(colors[self.target_class_index].light(
            180 - li[self.target_class_index] * 150
        ))

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
            model,
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
    reg = TreeRegressionLearner(max_depth=5)(data)
    reg.instances = data
    ow.set_tree(reg)

    ow.show()
    ow.raise_()
    ow.handleNewSignals()
    app.exec_()

    sys.exit(0)


if __name__ == '__main__':
    main()
