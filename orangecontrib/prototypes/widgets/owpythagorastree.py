from math import sqrt, log

from Orange.data.table import Table
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget
from PyQt4 import QtGui

from orangecontrib.prototypes.utils.common.owlegend import (
    LegendBuilder,
    AnchorableGraphicsView
)
from orangecontrib.prototypes.utils.common.scene import \
    UpdateItemsOnSelectGraphicsScene
from orangecontrib.prototypes.utils.common.view import (
    PannableGraphicsView,
    ZoomableGraphicsView,
    PreventDefaultWheelEvent
)
from orangecontrib.prototypes.widgets.pythagorastreeviewer import (
    PythagorasTreeViewer,
    SquareGraphicsItem
)


class OWPythagorasTree(OWWidget):
    outputs = [('Selected Data', Table)]

    # Enable the save as feature
    graph_name = 'scene'

    # Settings
    depth_limit = settings.ContextSetting(10)
    target_class_index = settings.ContextSetting(0)
    size_calc_idx = settings.Setting(0)
    size_log_scale = settings.Setting(2)
    tooltips_enabled = settings.Setting(True)
    show_legend = settings.Setting(False)

    def __init__(self):
        super().__init__()
        # Instance variables
        # The raw skltree model that was passed to the input
        self.model = None
        self.dataset = None
        self.clf_dataset = None
        # The tree adapter instance which is passed from the outside
        self.tree_adapter = None
        self.legend = None

        self.color_palette = None

        # Different methods to calculate the size of squares
        self.SIZE_CALCULATION = [
            ('Normal', lambda x: x),
            ('Square root', lambda x: sqrt(x)),
            ('Logarithmic', lambda x: log(x * self.size_log_scale)),
        ]

        # CONTROL AREA
        # Tree info area
        box_info = gui.widgetBox(self.controlArea, 'Tree Info')
        self.info = gui.widgetLabel(box_info, label='')

        # Display settings area
        box_display = gui.widgetBox(self.controlArea, 'Display Settings')
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
        self.log_scale_box = gui.hSlider(
            box_display, self, 'size_log_scale',
            label='Log scale factor', minValue=1, maxValue=100, ticks=False,
            callback=self.invalidate_tree)

        # Plot properties area
        box_plot = gui.widgetBox(self.controlArea, 'Plot Properties')
        gui.checkBox(
            box_plot, self, 'tooltips_enabled', label='Enable tooltips',
            callback=self.update_tooltip_enabled)
        gui.checkBox(
            box_plot, self, 'show_legend', label='Show legend',
            callback=self.update_show_legend)

        # Stretch to fit the rest of the unsused area
        gui.rubber(self.controlArea)

        self.controlArea.setSizePolicy(
            QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)

        # MAIN AREA
        # The QGraphicsScene doesn't actually require a parent, but not linking
        # the widget to the scene causes errors and a segfault on close due to
        # the way Qt deallocates memory and deletes objects.
        self.scene = TreeGraphicsScene(self)
        self.scene.selectionChanged.connect(self.commit)
        self.view = TreeGraphicsView(self.scene, padding=(150, 150))
        self.view.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.mainArea.layout().addWidget(self.view)

        self.ptree = PythagorasTreeViewer()
        self.ptree.set_node_color_func(self._get_node_color)
        self.scene.addItem(self.ptree)
        self.view.set_central_widget(self.ptree)
        if self.tooltips_enabled:
            self.ptree.set_tooltip_func(self._get_tooltip)
        else:
            self.ptree.set_tooltip_func(lambda _: None)

        self.resize(800, 500)
        # Clear the widget to correctly set the intial values
        self.clear()

    def set_tree(self, model=None):
        """When a different tree is given."""
        self.clear()
        self.model = model

        if model is not None:
            self.tree_adapter = self._get_tree_adapter(self.model)
            self.color_palette = self._get_color_palette()

            self.dataset = model.instances
            # this bit is important for the regression classifier
            if self.dataset is not None and \
                    self.dataset.domain != model.domain:
                self.clf_dataset = Table.from_table(
                    self.model.domain, self.dataset)
            else:
                self.clf_dataset = self.dataset

            self.ptree.set_tree(self.tree_adapter)

            self._update_legend_colors()
            self._update_legend_visibility()

            self._update_info_box()
            self._update_depth_slider()
            self._update_target_class_combo()

            self._update_main_area()

    def clear(self):
        """Clear all relevant data from the widget."""
        self.model = None
        self.dataset = None
        self.clf_dataset = None
        self.tree_adapter = None

        if self.legend is not None:
            self.scene.removeItem(self.legend)
        self.legend = None

        self.ptree.clear()
        self._clear_info_box()
        self._clear_target_class_combo()
        self._clear_depth_slider()
        self._update_log_scale_slider()

    # CONTROL AREA CALLBACKS
    def update_depth(self):
        """This method should be called when the depth changes"""
        self.ptree.set_depth_limit(self.depth_limit)

    def update_colors(self):
        self.ptree.target_class_has_changed()
        self._update_legend_colors()

    def update_size_calc(self):
        self._update_log_scale_slider()
        self.invalidate_tree()

    def invalidate_tree(self):
        """When the tree needs to be recalculated. E.g. change of size calc."""
        if self.model is not None:
            self.tree_adapter = self._get_tree_adapter(self.model)

            self.ptree.set_tree(self.tree_adapter)
            self.ptree.set_depth_limit(self.depth_limit)
            self._update_main_area()

    def update_tooltip_enabled(self):
        if self.tooltips_enabled:
            self.ptree.set_tooltip_func(self._get_tooltip)
        else:
            self.ptree.set_tooltip_func(lambda _: None)
        self.ptree.tooltip_has_changed()

    def update_show_legend(self):
        self._update_legend_visibility()

    # MODEL CHANGED CONTROL ELEMENTS UPDATE METHODS
    def _update_info_box(self):
        self.info.setText('Nodes: {}\nDepth: {}'.format(
            self.tree_adapter.num_nodes,
            self.tree_adapter.max_depth
        ))

    def _update_depth_slider(self):
        self.depth_slider.setEnabled(True)
        self.depth_slider.setMaximum(self.tree_adapter.max_depth)
        self._set_max_depth()

    def _update_target_class_combo(self):
        return []

    def _update_legend_visibility(self):
        if self.legend is not None:
            self.legend.setVisible(self.show_legend)

    def _update_legend_colors(self):
        self.legend = LegendBuilder()(
            domain=self.model.domain,
            dataset=self.clf_dataset
        )
        self.scene.addItem(self.legend)

    def _update_log_scale_slider(self):
        """On calc method combo box changed."""
        if self.SIZE_CALCULATION[self.size_calc_idx][0] == 'Logarithmic':
            self.log_scale_box.setEnabled(True)
        else:
            self.log_scale_box.setEnabled(False)

    # MODEL REMOVED CONTROL ELEMENTS CLEAR METHODS
    def _clear_info_box(self):
        self.info.setText('No tree on input')

    def _clear_depth_slider(self):
        self.depth_slider.setEnabled(False)
        self.depth_slider.setMaximum(0)

    def _clear_target_class_combo(self):
        self.target_class_combo.clear()
        self.target_class_index = 0
        self.target_class_combo.setCurrentIndex(self.target_class_index)

    # HELPFUL METHODS
    def _set_max_depth(self):
        """Set the depth to the max depth and update appropriate actors."""
        self.depth_limit = self.tree_adapter.max_depth
        self.depth_slider.setValue(self.depth_limit)

    def _get_color_palette(self):
        if self.model.domain.class_var.is_discrete:
            colors = [QtGui.QColor(*c)
                      for c in self.model.domain.class_var.colors]
        else:
            colors = None
        return colors

    def _get_node_color(self, adapter, tree_node):
        return self.color_palette[0]

    def _get_tree_adapter(self, model):
        return model

    def _get_tooltip(self, node):
        raise NotImplemented()

    def _update_main_area(self):
        # refresh the scene rect, cuts away the excess whitespace, and adds
        # padding for panning.
        self.scene.setSceneRect(self.view.central_widget_rect())
        # reset the zoom level
        self.view.recalculate_and_fit()

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
        items = filter(lambda x: isinstance(x, SquareGraphicsItem),
                       self.scene.selectedItems())

        data = self.tree_adapter.get_instances_in_nodes(
            self.clf_dataset, [item.tree_node for item in items])
        self.send('Selected Data', data)

    def send_report(self):
        self.report_plot()


class TreeGraphicsView(
    PannableGraphicsView,
    ZoomableGraphicsView,
    AnchorableGraphicsView,
    PreventDefaultWheelEvent
):
    pass


class TreeGraphicsScene(UpdateItemsOnSelectGraphicsScene):
    pass
