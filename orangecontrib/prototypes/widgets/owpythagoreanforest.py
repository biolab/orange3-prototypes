from math import log, sqrt

import numpy as np
from Orange.base import RandomForest, Tree
from Orange.classification.tree import TreeClassifier
from Orange.data import Table
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget
from PyQt4 import QtGui
from PyQt4.QtCore import Qt

from orangecontrib.prototypes.utils.common.owgrid import (
    OWGrid,
    SelectableGridItem,
    ZoomableGridItem
)
from orangecontrib.prototypes.utils.tree.skltreeadapter import SklTreeAdapter
from orangecontrib.prototypes.widgets.pythagorastreeviewer import \
    PythagorasTreeViewer


class OWPythagoreanForest(OWWidget):
    name = 'Pythagorean forest'
    description = 'Pythagorean forest for visualising random forests.'
    icon = 'icons/PythagoreanForest.svg'

    priority = 100

    inputs = [('Random forest', RandomForest, 'set_rf')]
    outputs = [('Tree', Tree)]

    # Enable the save as feature
    graph_name = 'scene'

    # Settings
    depth_limit = settings.ContextSetting(10)
    target_class_index = settings.ContextSetting(0)
    size_calc_idx = settings.Setting(0)
    size_log_scale = settings.Setting(2)
    zoom = settings.Setting(50)
    selected_tree_index = settings.ContextSetting(-1)

    def __init__(self):
        super().__init__()
        # Instance variables
        # The raw skltree model that was passed to the input
        self.model = None
        self.forest_adapter = None
        self.dataset = None
        self.clf_dataset = None
        # We need to store refernces to the trees and grid items
        self.grid_items, self.ptrees = [], []

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
        self.ui_zoom_slider = gui.hSlider(
            box_display, self, 'zoom', label='Zoom', ticks=False, minValue=20,
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

            self.dataset = model.instances
            # this bit is important for the regression classifier
            if self.dataset is not None and \
                    self.dataset.domain != model.domain:
                self.clf_dataset = Table.from_table(
                    self.model.domain, self.dataset)
            else:
                self.clf_dataset = self.dataset

            self._update_info_box()
            self._update_target_class_combo()
            self._update_depth_slider()

            self.selected_tree_index = -1

    def clear(self):
        """Clear all relevant data from the widget."""
        self.model = None
        self.forest_adapter = None
        self.ptrees = []
        self.grid_items = []
        self.grid.clear()

        self._clear_info_box()
        self._clear_target_class_combo()
        self._clear_depth_slider()

    # CONTROL AREA CALLBACKS
    def max_depth_changed(self):
        for tree in self.ptrees:
            tree.set_depth_limit(self.depth_limit)

    def target_colors_changed(self):
        for tree in self.ptrees:
            tree.target_class_has_changed()

    def size_calc_changed(self):
        if self.model is not None:
            self.forest_adapter = self._get_forest_adapter(self.model)
            self.grid.clear()
            self._draw_trees()
            # Keep the selected item
            if self.selected_tree_index != -1:
                self.grid_items[self.selected_tree_index].setSelected(True)

    def zoom_changed(self):
        for item in self.grid_items:
            item.set_max_size(self._calculate_zoom(self.zoom))

        width = (self.view.width() - self.view.verticalScrollBar().width())
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
        self.depth_limit = self._get_max_depth()

        self.ui_depth_slider.setEnabled(True)
        self.ui_depth_slider.setMaximum(self.depth_limit)
        self.ui_depth_slider.setValue(self.depth_limit)

    # MODEL CLEARED METHODS
    def _clear_info_box(self):
        self.ui_info.setText('No forest on input.')

    def _clear_target_class_combo(self):
        self.ui_target_class_combo.clear()
        self.target_class_index = 0
        self.ui_target_class_combo.setCurrentIndex(self.target_class_index)

    def _clear_depth_slider(self):
        self.ui_depth_slider.setEnabled(False)
        self.ui_depth_slider.setMaximum(0)

    # HELPFUL METHODS
    def _get_max_depth(self):
        return max([tree.tree_adapter.max_depth for tree in self.ptrees])

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
            model.domain,
            adjust_weight=self.SIZE_CALCULATION[self.size_calc_idx][1],
        )

    def _draw_trees(self):
        self.grid_items, self.ptrees = [], []

        for tree in self.forest_adapter.get_trees():
            ptree = PythagorasTreeViewer(
                None, tree, node_color_func=self._get_node_color,
                interactive=False, padding=100)
            self.grid_items.append(GridItem(
                ptree, self.grid, max_size=self._calculate_zoom(self.zoom)
            ))
            self.ptrees.append(ptree)
        self.grid.set_items(self.grid_items)
        # This is necessary when adding items for the first time
        if self.grid:
            width = (self.view.width() -
                     self.view.verticalScrollBar().width())
            self.grid.reflow(width)
            self.grid.setPreferredWidth(width)

    @staticmethod
    def _calculate_zoom(zoom_level):
        """Calculate the max size for grid items from zoom level setting."""
        return zoom_level * 5

    def onDeleteWidget(self):
        """When deleting the widget."""
        super().onDeleteWidget()
        self.clear()

    def commit(self):
        """Commit the selected tree to output."""
        if len(self.scene.selectedItems()) == 0:
            self.send('Tree', None)
            # The selected tree index should only reset when model changes
            if self.model is None:
                self.selected_tree_index = -1
            return

        selected_item = self.scene.selectedItems()[0]
        self.selected_tree_index = self.grid_items.index(selected_item)
        tree = self.model.skl_model.estimators_[self.selected_tree_index]
        clf = TreeClassifier(tree)
        clf.domain = self.model.domain
        clf.instances = self.model.instances
        self.send('Tree', clf)

    def send_report(self):
        self.report_plot()

    def _update_scene_rect(self):
        self.scene.setSceneRect(self.scene.itemsBoundingRect())

    def resizeEvent(self, ev):
        width = (self.view.width() - self.view.verticalScrollBar().width())
        self.grid.reflow(width)
        self.grid.setPreferredWidth(width)

        super().resizeEvent(ev)


class GridItem(SelectableGridItem, ZoomableGridItem):
    pass


class SklRandomForestAdapter:
    def __init__(self, model, domain, adjust_weight=lambda x: x):
        self._adapters = []

        self._domain = domain

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

    @property
    def domain(self):
        return self._domain
