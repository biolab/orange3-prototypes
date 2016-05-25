from math import sqrt, log

import numpy as np
from Orange.data.table import Table
from Orange.widgets import gui, settings
from Orange.widgets.utils.colorpalette import DefaultRGBColors
from Orange.widgets.widget import OWWidget
from PyQt4 import QtGui
from PyQt4.QtCore import Qt

from orangecontrib.prototypes.widgets.pythagorastreeviewer import \
    PythagorasTreeViewer, SquareGraphicsItem


class OWPythagorasTree(OWWidget):
    # name = 'Pythagoras Tree'
    # description = 'Generalized Pythagoras Tree for visualizing trees.'
    # priority = 100

    # inputs = [('Tree', TreeAdapter, 'set_tree')]
    outputs = [('Selected Data', Table)]

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
        # The tree adapter instance which is passed from the outside
        self.tree_adapter = None

        self.color_palette = None

        # Different methods to calculate the size of squares
        self.SIZE_CALCULATION = [
            ('Normal', lambda x: x),
            ('Square root', lambda x: sqrt(x)),
            ('Logarithmic', lambda x: log(x * self.size_log_scale)),
        ]

        # CONTROL AREA
        # Tree info area
        box_info = gui.widgetBox(self.controlArea, 'Tree')
        self.info = gui.widgetLabel(box_info, label='No tree.')

        # Display controls area
        box_display = gui.widgetBox(self.controlArea, 'Display')
        self.depth_slider = gui.hSlider(
            box_display, self, 'depth_limit', label='Depth', ticks=False,
            callback=self.update_depth)
        self.target_class_combo = gui.comboBox(
            box_display, self, 'target_class_index', label='Target class',
            orientation='horizontal', items=[], contentsLength=8,
            callback=self.update_colors)
        self.size_calc_combo = gui.comboBox(
            box_display, self, 'size_calc_idx', label='Size',
            orientation='horizontal',
            items=list(zip(*self.SIZE_CALCULATION))[0], contentsLength=8,
            callback=self.update_size_calc)
        # the log slider needs its own box to be able to be completely hidden
        self.log_scale_box = gui.widgetBox(box_display)
        gui.hSlider(
            self.log_scale_box, self, 'size_log_scale', label='Log scale',
            minValue=1, maxValue=100, ticks=False,
            callback=self.invalidate_tree)
        # the log scale slider should only be visible if the calc method is log
        if self.SIZE_CALCULATION[self.size_calc_idx][0] != 'Logarithmic':
            self.log_scale_box.setEnabled(False)
            self.log_scale_box.setVisible(False)

        gui.checkBox(
            box_display, self, 'tooltips_enabled', label='Enable tooltips',
            callback=self._update_tooltip_enabled)

        # Stretch to fit the rest of the unsused area
        gui.rubber(self.controlArea)

        self.controlArea.setSizePolicy(
            QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)

        # GUI - MAIN AREA
        # The QGraphicsScene doesn't actually require a parent, but not linking
        # the widget to the scene causes errors and a segfault on close due to
        # the way Qt deallocates memory and deletes objects.
        self.scene = QtGui.QGraphicsScene(self)
        self.scene.selectionChanged.connect(self.commit)
        self.view = TreeGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.mainArea.layout().addWidget(self.view)

        self.ptree = PythagorasTreeViewer()
        self.ptree.set_node_color_func(self._get_node_color)
        if self.tooltips_enabled:
            self.ptree.set_tooltip_func(self._get_tooltip)
        else:
            self.ptree.set_tooltip_func(lambda _: None)
        self.scene.addItem(self.ptree)

        self.resize(800, 500)

    def set_tree(self, model=None):
        """When a different tree is given."""
        self.clear()
        self.model = model

        if model is not None:
            self.tree_adapter = self._get_tree_adapter(self.model)
            self.color_palette = self._get_color_palette()
            self.ptree.set_tree(self.tree_adapter)

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

            self._update_main_area()

    def update_depth(self):
        """This method should be called when the depth changes"""
        self.ptree.set_depth_limit(self.depth_limit)

    def update_colors(self):
        self.ptree.target_class_has_changed()

    def update_size_calc(self):
        """On calc method combo box changed."""
        if self.SIZE_CALCULATION[self.size_calc_idx][0] == 'Logarithmic':
            self.log_scale_box.setEnabled(True)
            self.log_scale_box.setVisible(True)
        else:
            self.log_scale_box.setEnabled(False)
            self.log_scale_box.setVisible(False)
        self.invalidate_tree()

    def invalidate_tree(self):
        """When the tree needs to be recalculated. E.g. change of size calc."""
        self.tree_adapter = self._get_tree_adapter(self.model)

        self.ptree.set_tree(self.tree_adapter)
        self.ptree.set_depth_limit(self.depth_limit)
        self._update_main_area()

    def clear(self):
        """Clear all relevant data from the widget."""
        self.model = None
        self.dataset = None
        self.clf_dataset = None
        self.tree_adapter = None

        self.ptree.clear()
        self._clear_info_box()
        self._clear_target_class_combo()
        self._clear_depth_slider()

    def _update_info_box(self):
        self.info.setText(
            '{} nodes, {} depth'.format(
                self.tree_adapter.num_nodes,
                self.tree_adapter.max_depth
            )
        )

    def _clear_info_box(self):
        self.info.setText('No tree')

    def _update_depth_slider(self):
        self.depth_slider.setEnabled(True)
        self.depth_slider.setMaximum(self.tree_adapter.max_depth)
        self._set_max_depth()

    def _clear_depth_slider(self):
        self.depth_slider.setEnabled(False)
        self.depth_slider.setMaximum(0)

    def _set_max_depth(self):
        """Set the depth to the max depth and update appropriate actors."""
        self.depth_limit = self.tree_adapter.max_depth
        self.depth_slider.setValue(self.depth_limit)

    def _update_target_class_combo(self):
        return []

    def _clear_target_class_combo(self):
        self.target_class_combo.clear()
        self.target_class_index = 0
        self.target_class_combo.setCurrentIndex(self.target_class_index)

    def _update_tooltip_enabled(self):
        if self.tooltips_enabled:
            self.ptree.set_tooltip_func(self._get_tooltip)
        else:
            self.ptree.set_tooltip_func(lambda _: None)
        self.ptree.tooltip_has_changed()

    def _get_color_palette(self):
        if self.model.domain.class_var.is_discrete:
            colors = [QtGui.QColor(*c)
                      for c in self.model.domain.class_var.colors]
        else:
            colors = None
        return colors

    def _get_node_color(self, tree_node):
        return self.color_palette[0]

    def _get_tree_adapter(self, model):
        return model

    def _get_tooltip(self, node):
        raise NotImplemented()

    def _update_main_area(self):
        # refresh the scene rect, cuts away the excess whitespace, and adds
        # padding for panning.
        self.scene.setSceneRect(self.scene.itemsBoundingRect()
                                .adjusted(-150, -150, 150, 150))
        # fits the scene into the viewport
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def onDeleteWidget(self):
        """When deleting the widget."""
        super().onDeleteWidget()
        self.clear()

    def commit(self):
        """Commit the selected data to output."""
        if self.dataset is None:
            self.send('Selected Data', None)
            return
        # this is taken almost directly from the owclassificationtreegraph.py
        ta = self.tree_adapter
        items = filter(lambda x: isinstance(x, SquareGraphicsItem),
                       self.scene.selectedItems())

        selected_leaves = [ta.leaves(item.tree_node.label) for item in items]
        if len(selected_leaves) > 0:
            # get the leaves of the selected tree node
            selected_leaves = np.unique(np.hstack(selected_leaves))

        all_leaves = ta.leaves(ta.root)

        if len(selected_leaves) > 0:
            indices = np.searchsorted(all_leaves, selected_leaves, side='left')
            # all the leaf samples for each leaf
            leaf_samples = ta.get_samples_in_leaves(self.clf_dataset.X)
            # filter out the leaf samples array that are not selected
            leaf_samples = [leaf_samples[i] for i in indices]
            indices = np.hstack(leaf_samples)
        else:
            indices = []

        data = self.dataset[indices] if len(indices) else None
        self.send('Selected Data', data)

    def send_report(self):
        pass


class ZoomableGraphicsView(QtGui.QGraphicsView):
    def __init__(self, *args, **kwargs):
        self.zoom = 1
        # zoomout limit prevents the zoom factor to become negative, which
        # results in the canvas being flipped over the x axis
        self._zoomout_limit_reached = False
        self._initial_zoom = 1
        super().__init__(*args, **kwargs)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.zoom == 1:
            self.fitInView(self.scene().itemsBoundingRect(),
                           Qt.KeepAspectRatio)
            self._initial_zoom = self.matrix().m11()
            self.zoom = self._initial_zoom

    def wheelEvent(self, ev):
        if self._zooming_in(ev):
            self._reset_zoomout_limit()
        if self._zoomout_limit_reached and self._zooming_out(ev):
            ev.accept()
            return

        self.zoom += np.sign(ev.delta()) * 1 / 16
        if self.zoom <= 0:
            self._zoomout_limit_reached = True
            self.zoom += 1
        else:
            self.setTransformationAnchor(self.AnchorUnderMouse)
            self.setTransform(QtGui.QTransform().scale(self.zoom, self.zoom))
        ev.accept()

    @staticmethod
    def _zooming_out(ev):
        return ev.delta() < 0

    def _zooming_in(self, ev):
        return not self._zooming_out(ev)

    def _reset_zoomout_limit(self):
        self._zoomout_limit_reached = False

    def reset_zoom(self):
        """Reset the zoom to the initial size."""
        self.zoom = self._initial_zoom
        self.setTransform(QtGui.QTransform().scale(self.zoom, self.zoom))


class PannableGraphicsView(QtGui.QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)

    def enterEvent(self, ev):
        super().enterEvent(ev)
        self.viewport().setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        self.viewport().setCursor(Qt.ArrowCursor)


class TreeGraphicsView(PannableGraphicsView, ZoomableGraphicsView):
    pass
