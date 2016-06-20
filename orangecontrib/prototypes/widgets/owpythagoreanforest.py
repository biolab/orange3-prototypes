from math import log, sqrt

import numpy as np
from Orange.classification.random_forest import RandomForestClassifier
from Orange.classification.tree import TreeClassifier
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt

from orangecontrib.prototypes.utils.common.owgrid import OWGrid, \
    SelectableGridItem, ZoomableGridItem, PaddedGridItem
from orangecontrib.prototypes.utils.tree.skltreeadapter import SklTreeAdapter
from orangecontrib.prototypes.widgets.pythagorastreeviewer import \
    PythagorasTreeViewer


class OWPythagoreanForest(OWWidget):
    name = 'Pythagorean forest'
    description = 'Pythagorean forest for visualising random forests.'
    priority = 100

    # Enable the save as feature
    graph_name = 'scene'

    inputs = [('Random forest', RandomForestClassifier, 'set_rf')]
    outputs = [('Tree', TreeClassifier)]

    # Settings
    depth_limit = settings.ContextSetting(10)
    target_class_index = settings.ContextSetting(0)
    size_calc_idx = settings.Setting(0)
    size_log_scale = settings.Setting(2)
    tooltips_enabled = settings.Setting(True)
    zoom = settings.Setting(True)

    def __init__(self):
        super().__init__()
        # Instance variables
        # The raw skltree model that was passed to the input
        self.model = None
        self.forest_adapter = None
        self.dataset = None
        self.clf_dataset = None
        # We need to store refernces to the trees and grid items
        self.ptrees, self.grid_items = [], []

        self.color_palette = None

        # Different methods to calculate the size of squares
        self.SIZE_CALCULATION = [
            ('Normal', lambda x: x),
            ('Square root', lambda x: sqrt(x)),
            ('Logarithmic', lambda x: log(x * self.size_log_scale)),
        ]

        # CONTROL AREA
        # Tree info area
        box_info = gui.widgetBox(self.controlArea, 'Forest')
        self.ui_info = gui.widgetLabel(box_info, label='')

        # Display controls area
        box_display = gui.widgetBox(self.controlArea, 'Display')
        self.ui_depth_slider = gui.hSlider(
            box_display, self, 'depth_limit', label='Depth', ticks=False,
            callback=self.max_depth_changed)
        self.ui_target_class_combo = gui.comboBox(
            box_display, self, 'target_class_index', label='Target class',
            orientation='horizontal', items=[], contentsLength=8,
            callback=self.target_colors_changed)
        self.ui_size_calc_combo = gui.comboBox(
            box_display, self, 'size_calc_idx', label='Size',
            orientation='horizontal',
            items=list(zip(*self.SIZE_CALCULATION))[0], contentsLength=8,
            callback=self.size_calc_changed)
        self.ui_depth_slider = gui.hSlider(
            box_display, self, 'zoom', label='Zoom', ticks=False, minValue=10,
            maxValue=150, callback=self.zoom_changed, createLabel=False)

        # Stretch to fit the rest of the unsused area
        gui.rubber(self.controlArea)

        self.controlArea.setSizePolicy(
            QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)

        # MAIN AREA
        self.scene = QtGui.QGraphicsScene(self)
        self.scene.selectionChanged.connect(self.commit)
        self.grid = OWGrid()
        self.grid.geometryChanged.connect(self._update_scene_rect)
        self.scene.addItem(self.grid)

        self.view = QtGui.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.mainArea.layout().addWidget(self.view)

        self.resize(800, 500)

        self.clear()

    def set_rf(self, model=None):
        """When a different forest is given."""
        self.clear()
        self.model = model

        if model is not None:
            self.forest_adapter = self._get_forest_adapter(self.model)
            self.color_palette = self._get_color_palette()
            self._draw_trees()

            self._update_info_box()
            self._update_target_class_combo()
            self._update_depth_slider()

            self._update_main_area()

    def clear(self):
        """Clear all relevant data from the widget."""
        self.model = None
        self.forest_adapter = None
        self.ptrees = []

        self._clear_info_box()
        self._clear_target_class_combo()
        self._clear_depth_slider()

    # CONTROL AREA CALLBACKS
    def max_depth_changed(self):
        pass

    def target_colors_changed(self):
        pass

    def size_calc_changed(self):
        pass

    def zoom_changed(self):
        for item in self.grid_items:
            item.set_max_size(self.zoom * 5)

        if self.grid:
            width = (self.view.width() -
                     self.view.verticalScrollBar().width())

            self.grid.reflow(width)
            self.grid.setPreferredWidth(width)

    # MODEL CHANGED METHODS
    def _update_info_box(self):
        self.ui_info.setText(
            'Trees: {}'.format(self.forest_adapter.num_trees)
        )

    def _update_target_class_combo(self):
        self._clear_target_class_combo()
        self.ui_target_class_combo.addItem('None')
        values = [c.title() for c in
                  self.model.domain.class_vars[0].values]
        self.ui_target_class_combo.addItems(values)

    def _update_depth_slider(self):
        pass

    # MODEL CLEARED METHODS
    def _clear_info_box(self):
        self.ui_info.setText('No forest on input.')

    def _clear_target_class_combo(self):
        self.ui_target_class_combo.clear()
        self.ui_target_class_index = 0
        self.ui_target_class_combo.setCurrentIndex(self.target_class_index)

    def _clear_depth_slider(self):
        pass

    # HELPFUL METHODS
    def _update_main_area(self):
        pass

    def _get_color_palette(self):
        if self.model.domain.class_var.is_discrete:
            colors = [QtGui.QColor(*c)
                      for c in self.model.domain.class_var.colors]
        else:
            colors = None
        return colors

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

    def _get_forest_adapter(self, model):
        return SklRandomForestAdapter(
            model,
            adjust_weight=self.SIZE_CALCULATION[self.size_calc_idx][1],
        )

    def _draw_trees(self):
        for tree in self.forest_adapter.get_trees():
            ptree = PythagorasTreeViewer(
                None, tree, node_color_func=self._get_node_color,
                interactive=False)
            self.ptrees.append(ptree)
            self.grid_items.append(GridItem(ptree, self.grid, max_size=250))

        self.grid.set_items(self.grid_items)

        # TODO check that this is really necessary
        if self.grid:
            width = (self.view.width() -
                     self.view.verticalScrollBar().width())
            self.grid.reflow(width)
            self.grid.setPreferredWidth(width)

    def onDeleteWidget(self):
        """When deleting the widget."""
        super().onDeleteWidget()
        self.clear()

    def commit(self):
        """Commit the selected tree to output."""
        pass
        model = self.model
        tree = model.skl_model.estimators_[0]
        clf = TreeClassifier(tree)
        clf.domain = model.domain
        clf.instances = model.instances
        self.send('Tree', clf if self.model is not None else None)

    def send_report(self):
        self.report_plot()

    def _update_scene_rect(self):
        self.scene.setSceneRect(self.scene.itemsBoundingRect())

    def resizeEvent(self, ev):
        width = (self.view.width() - self.view.verticalScrollBar().width())
        self.grid.reflow(width)
        self.grid.setPreferredWidth(width)

        super().resizeEvent(ev)


class GridItem(SelectableGridItem, ZoomableGridItem, PaddedGridItem):
    pass


class SklRandomForestAdapter:
    def __init__(self, model, adjust_weight=lambda x: x):
        self._adapters = []

        self._trees = model.skl_model.estimators_
        self._domain = model.domain
        self._adjust_weight = adjust_weight

    def get_trees(self):
        if len(self._adapters) > 0:
            return self._adapters
        if len(self._trees) < 1:
            return self._adapters

        self._adapters = [
            SklTreeAdapter(tree.tree_, self._domain, self._adjust_weight)
            for tree in self._trees
        ]
        return self._adapters

    @property
    def num_trees(self):
        return len(self._adapters)


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
    clf = RandomForestLearner(n_estimators=500)(data)
    clf.instances = data
    ow.set_rf(clf)

    ow.show()
    ow.raise_()
    ow.handleNewSignals()
    app.exec_()

    sys.exit(0)


if __name__ == '__main__':
    main()
