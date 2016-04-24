from math import sqrt, log

import Orange
import numpy as np
from Orange.data.table import Table
from Orange.widgets import gui, settings
from Orange.widgets.utils.colorpalette import DefaultRGBColors
from Orange.widgets.widget import OWWidget
from PyQt4 import QtGui
from PyQt4.QtCore import Qt

from orangecontrib.prototypes.widgets.pythagorastreeviewer import \
    PythagorasTreeViewer, TreeAdapter


class OWPythagorasTree(OWWidget):
    name = 'Pythagoras Tree'
    description = 'Generalized Pythagoras Tree for visualizing trees.'
    priority = 100

    # Enable the save as feature
    graph_name = True

    inputs = [('Tree', TreeAdapter, 'set_tree')]
    outputs = [('Selected Data', Table)]

    # Settings
    depth_limit = settings.ContextSetting(10)
    target_class_index = settings.ContextSetting(0)
    size_calc_idx = settings.Setting(0)
    size_log_scale = settings.Setting(2)
    auto_commit = settings.Setting(True)

    def __init__(self):
        super().__init__()
        # Instance variables
        # The domain is needed to identify target classes
        self.domain = None
        # The raw skltree model that was passed to the input
        self.model = None
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

        # Stretch to fit the rest of the unsused area
        gui.rubber(self.controlArea)

        # Bottom options
        gui.auto_commit(
            self.controlArea, self, value='auto_commit',
            label='Send selected instances', auto_label='Auto send is on')
        self.inline_graph_report()

        self.controlArea.setSizePolicy(
            QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)

        # GUI - MAIN AREA
        # The QGraphicsScene doesn't actually require a parent, but not linking
        # the widget to the scene causes errors and a segfault on close due to
        # the way Qt deallocates memory and deletes objects.
        self.scene = QtGui.QGraphicsScene(self)
        self.view = TreeGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.mainArea.layout().addWidget(self.view)

        self.ptree = PythagorasTreeViewer()
        self.ptree.set_node_color_func(self._get_node_color)
        self.scene.addItem(self.ptree)

        self.resize(800, 500)

    def set_tree(self, model=None):
        """When a different tree is given."""
        self.clear()
        self.model = model

        if model is not None:
            self.domain = model.domain
            self.tree_adapter = self._get_tree_adapter(self.model)
            self.color_palette = self._get_color_palette()
            self.ptree.set_tree(self.tree_adapter)

            self._update_info_box()
            self._update_target_class_combo()
            self._update_depth_slider()

    def update_depth(self):
        """This method should be called when the depth changes"""
        self.ptree.set_depth_limit(self.depth_limit)

    def update_colors(self):
        self.ptree.update_colors()

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
        self.domain = None
        self.model = None
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

    def _get_color_palette(self):
        return [QtGui.QColor(*c) for c in DefaultRGBColors]

    def _get_node_color(self, tree_node):
        return self.color_palette[0]

    def _get_tree_adapter(self, model):
        return model

    def _update_main_area(self):
        # refresh the scene rect, cuts away the excess whitespace
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        # fits the scene into the viewport
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def onDeleteWidget(self):
        """When deleting the widget."""
        super().onDeleteWidget()
        self.clear()

    def commit(self):
        """Commit the selected data to output."""
        pass

    def send_report(self):
        pass


class ZoomableGraphicsView(QtGui.QGraphicsView):
    def __init__(self, *args, **kwargs):
        self.zoom = 1
        # zoomout limit prevents the zoom factor to become negative, which
        # results in the canvas being flipped over the x axis
        self._zoomout_limit_reached = False
        self._size_was_reset = False
        super().__init__(*args, **kwargs)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # if not self._size_was_reset:
        if self.zoom == 1:
            self._size_was_reset = True
            self.fitInView(self.scene().itemsBoundingRect(),
                           Qt.KeepAspectRatio)

    def wheelEvent(self, ev):
        if self._zooming_in(ev):
            self._reset_zoomout_limit()
        if self._zoomout_limit_reached and self._zooming_out(ev):
            ev.accept()
            return

        self.zoom += np.sign(ev.delta()) * 1
        k = 0.0028 * (self.zoom ** 2) + 0.2583 * self.zoom + 1.1389
        if k <= 0:
            self._zoomout_limit_reached = True
            self.zoom += 1
        else:
            self.setTransformationAnchor(self.AnchorUnderMouse)
            self.setTransform(QtGui.QTransform().scale(k / 2, k / 2))
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
        self.zoom = 1
        self._size_was_reset = False
        k = 0.0028 * (self.zoom ** 2) + 0.2583 * self.zoom + 1.1389
        self.setTransform(QtGui.QTransform().scale(k / 2, k / 2))


class PannableGraphicsView(QtGui.QGraphicsView):
    def __init__(self, *args, **kwargs):
        self.panning = False
        super().__init__(*args, **kwargs)


class TreeGraphicsView(PannableGraphicsView, ZoomableGraphicsView):
    pass