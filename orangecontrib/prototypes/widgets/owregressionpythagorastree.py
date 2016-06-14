import Orange
import numpy as np
from Orange.regression.tree import TreeRegressor
from Orange.widgets.utils.colorpalette import ContinuousPaletteGenerator
from PyQt4 import QtGui

from orangecontrib.prototypes.utils.common.owlegend import (
    OWContinuousLegend,
    Anchorable
)
from orangecontrib.prototypes.utils.tree.skltreeadapter import SklTreeAdapter
from orangecontrib.prototypes.widgets.owpythagorastree import OWPythagorasTree


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
            ('Class mean', self._color_class_mean),
            ('Standard deviation', self._color_stddev),
        ]

    # MODEL CHANGED CONTROL ELEMENTS UPDATE METHODS
    def _update_target_class_combo(self):
        self._clear_target_class_combo()
        self.target_class_combo.addItems(
            list(zip(*self.REGRESSION_COLOR_CALC))[0])
        self.target_class_combo.setCurrentIndex(self.target_class_index)

    def _update_legend_colors(self):
        if self.legend is not None:
            self.scene.removeItem(self.legend)

        def _get_colors_domain(domain):
            class_var = domain.class_var
            start, end, pass_through_black = class_var.colors
            if pass_through_black:
                lst_colors = [QtGui.QColor(*c) for c
                              in [start, (0, 0, 0), end]]
            else:
                lst_colors = [QtGui.QColor(*c) for c in [start, end]]
            return lst_colors

        legend_options = {
            'corner': Anchorable.BOTTOM_RIGHT,
            'offset': (110, 180),
        }

        # Currently, the first index just draws the outline without any color
        if self.target_class_index == 0:
            self.legend = None
            return
        # The colors are the class mean
        elif self.target_class_index == 1:
            values = (np.min(self.clf_dataset.Y), np.max(self.clf_dataset.Y))
            colors = _get_colors_domain(self.model.domain)
            while len(values) != len(colors):
                values.insert(1, -1)

            self.legend = OWContinuousLegend(items=list(zip(values, colors)),
                                             **legend_options)
        # Colors are the stddev
        elif self.target_class_index == 2:
            values = (0, np.std(self.clf_dataset.Y))
            colors = _get_colors_domain(self.model.domain)
            while len(values) != len(colors):
                values.insert(1, -1)

            self.legend = OWContinuousLegend(items=list(zip(values, colors)),
                                             **legend_options)

        self.legend.setVisible(self.show_legend)
        self.scene.addItem(self.legend)

    # MODEL REMOVED CONTROL ELEMENTS CLEAR METHODS
    def _clear_target_class_combo(self):
        # Since the target classes are just the different coloring methods,
        # we can reuse the selected index when the tree changes, unlike with
        # the classification tree.
        self.target_class_combo.clear()

    # HELPFUL METHODS
    def _get_color_palette(self):
        return ContinuousPaletteGenerator(
            *self.tree_adapter.domain.class_var.colors)

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

    def _get_tooltip(self, node):
        total = self.tree_adapter.num_samples(self.tree_adapter.root)
        samples = self.tree_adapter.num_samples(node.label)
        ratio = samples / total

        instances = self.tree_adapter.get_instances_in_nodes(
            self.clf_dataset, node)
        mean = np.mean(instances.Y)
        std = np.std(instances.Y)

        rules = self.tree_adapter.rules(node.label)
        sorted_rules = sorted(rules[:-1], key=lambda rule: rule.attr_name)
        rules_str = ''
        if len(rules):
            rules_str = '<hr>'
            rules_str += '<br>'.join(str(rule) for rule in sorted_rules)
            rules_str += '<br><b>%s</b>' % rules[-1]

        splitting_attr = self.tree_adapter.attribute(node.label)

        return '<p>μ: {:2.3f}'.format(mean) \
            + '<br>σ: {:2.3f}'.format(std) \
            + '<br>{}/{} samples ({:2.3f}%)'.format(
                int(samples), total, ratio * 100) \
            + '<br><br>Split by ' + splitting_attr.name \
            + rules_str \
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
