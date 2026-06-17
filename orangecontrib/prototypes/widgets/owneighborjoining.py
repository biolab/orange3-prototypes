from itertools import chain
from contextlib import contextmanager

from typing import List, Optional, Union
import heapq
import numpy as np

from AnyQt.QtWidgets import (
    QGraphicsWidget, QGraphicsScene, QGridLayout, QSizePolicy,
    QAction, QComboBox, QGraphicsGridLayout, QGraphicsSceneMouseEvent, QLabel
)
from AnyQt.QtGui import QPen, QFont, QKeySequence, QPainterPath, QFontMetrics
from AnyQt.QtCore import (
    Qt, QObject, QSize, QPointF, QRectF, QLineF, QEvent
)
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

from Orange.widgets.utils.localization import pl
from orangewidget.utils.signals import LazyValue

import Orange.data
from Orange.data.domain import filter_visible
from Orange.data import DiscreteVariable, ContinuousVariable, \
    StringVariable, Table
import Orange.misc
from Orange.clustering.hierarchical import preorder, leaves, prune
from Orange.data.util import get_unique_names

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels, combobox
from Orange.widgets.utils.annotated_data import (lazy_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME,
                                                 domain_with_annotation_column,
                                                 add_columns,
                                                 create_annotated_table)
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.utils.plotutils import AxisItem
from Orange.widgets.widget import Input, Output, Msg

from Orange.widgets.utils.stickygraphicsview import StickyGraphicsView
from Orange.widgets.utils.graphicsview import GraphicsWidgetView

from orangecontrib.prototypes.neighbor_joining import (
    neighbor_joining_core, reorder_children
)
from orangecontrib.prototypes.neighbor_joining_adapter import (
    treenode_to_orange_tree
)
from orangecontrib.prototypes.dendrogram_nj import DendrogramWidget

__all__ = ["OWNeighborJoining"]


MAX_ITEMS = 1000
MAX_PRUNED_LABEL_WIDTH = 200


@contextmanager
def blocked(obj):
    old = obj.signalsBlocked()
    obj.blockSignals(True)
    try:
        yield obj
    finally:
        obj.blockSignals(old)


class GraphicsView(GraphicsWidgetView, StickyGraphicsView):
    def minimumSizeHint(self) -> QSize:
        msh = super().minimumSizeHint()
        w = self.centralWidget()
        if w is not None:
            width = w.minimumWidth() + 4 + self.verticalScrollBar().width()
            msh.setWidth(max(int(width), msh.width()))
        return msh

    def eventFilter(self, recv: QObject, event: QEvent) -> bool:
        ret = super().eventFilter(recv, event)
        if event.type() == QEvent.LayoutRequest and recv is self.centralWidget():
            self.updateGeometry()
        return ret


class OWNeighborJoining(widget.OWWidget):
    name = "Neighbor Joining"
    description = "Display a dendrogram of a neighbor joining " \
                  "constructed from the input distance matrix."
    icon = "icons/HierarchicalClustering.svg"
    priority = 2100
    keywords = "neighbor joining"

    class Inputs:
        distances = Input("Distances", Orange.misc.DistMatrix)
        data = Input("Data", Orange.data.Table)
        subset = Input("Data Subset", Orange.data.Table, explicit=True)

    class Outputs:
        selected_data = Output("Selected Data", Orange.data.Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Orange.data.Table)

    settings_version = 2
    settingsHandler = settings.DomainContextHandler()

    #: Index of the selected annotation item (variable, ...)
    annotation = settings.ContextSetting("Enumeration")
    #: Out-of-context setting for the case when the "Name" option is available
    annotation_if_names = settings.Setting("Name")
    #: Out-of-context setting for the case with just "Enumeration" and "None"
    annotation_if_enumerate = settings.Setting("Enumeration")
    #: Selected tree pruning (none/max depth)
    pruning = settings.Setting(0)
    #: Maximum depth when max depth pruning is selected
    max_depth = settings.Setting(10)

    #: Selected cluster selection method (none, cut distance, top n)
    selection_method = settings.Setting(0)
    #: Cut height ratio wrt root height
    cut_ratio = settings.Setting(75.0)
    #: Number of top clusters to select
    top_n = settings.Setting(3)
    #: Dendrogram zoom factor
    zoom_factor = settings.Setting(0)
    #: Show labels only for subset (if present)
    label_only_subset = settings.Setting(False)
    #: Color for label decoration
    color_by: Union[DiscreteVariable, ContinuousVariable, None] = \
        settings.ContextSetting(None)

    autocommit = settings.Setting(True)

    graph_name = "scene"  # QGraphicsScene

    basic_annotations = [None, "Enumeration"]

    class Error(widget.OWWidget.Error):
        empty_matrix = Msg("Distance matrix is empty.")
        not_finite_distances = Msg("Some distances are infinite")
        not_symmetric = widget.Msg("Distance matrix is not symmetric.")
        no_numeric_features = Msg("No numeric features for distance computation.")
        too_many_items = Msg(
            "Neighbor Joining is too slow for this many items "
            "({n}; maximum is {max_n})."
        )
        distance_computation_error = Msg("Error computing distances: {error}")
        tree_construction_error = Msg("Error constructing neighbor joining tree: {error}")

    class Warning(widget.OWWidget.Warning):
        subset_on_no_table = \
            Msg("Unused data subset: distances do not refer to data instances")
        subset_not_subset = \
            Msg("Some data from the subset does not appear in distance matrix")
        subset_wrong = \
            Msg("Subset data refers to a different table")
        pruning_disables_colors = \
            Msg("Pruned cluster doesn't show colors and indicate subset")
        many_clusters = \
            Msg("Variables with too many values may "
                "degrade the performance of downstream widgets.")

    def _set_valid_matrix(self, matrix):
        self.matrix = None
        if len(matrix) < 2:
            self.Error.empty_matrix()
        elif len(matrix) > MAX_ITEMS:
            self.Error.too_many_items(n=len(matrix), max_n=MAX_ITEMS)
        elif not matrix.is_symmetric():
            self.Error.not_symmetric()
        elif not np.all(np.isfinite(matrix)):
            self.Error.not_finite_distances()
        else:
            self.matrix = matrix

    def __init__(self):
        super().__init__()

        self.matrix = None
        self.data = None
        self.items = None
        self.subset = None
        self.subset_rows = set()
        self.root = None
        self._displayed_root = None
        self.cutoff_height = 0.0

        spin_width = QFontMetrics(self.font()).horizontalAdvance("M" * 7)

        model = itemmodels.VariableListModel(placeholder="None")
        model[:] = self.basic_annotations

        grid = QGridLayout()
        gui.widgetBox(self.controlArea, "Annotations", orientation=grid)
        self.label_cb = cb = combobox.ComboBoxSearch(
            minimumContentsLength=14,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        cb.setModel(model)
        cb.setCurrentIndex(cb.findData(self.annotation, Qt.EditRole))

        def on_annotation_activated():
            self.annotation = self.label_cb.currentData(Qt.EditRole)
            self._update_labels()
        cb.activated.connect(on_annotation_activated)

        def on_annotation_changed(value):
            self.label_cb.setCurrentIndex(
                self.label_cb.findData(value, Qt.EditRole))
        self.connect_control("annotation", on_annotation_changed)

        grid.addWidget(self.label_cb, 0, 0, 1, 2)

        cb = gui.checkBox(
            None, self, "label_only_subset", "Show labels only for subset",
            disabled=True,
            callback=self._update_labels, stateWhenDisabled=False)
        grid.addWidget(cb, 1, 0, 1, 2)

        model = itemmodels.DomainModel(
            valid_types=(DiscreteVariable, ContinuousVariable),
            placeholder="None")
        cb = gui.comboBox(
            None, self, "color_by", orientation=Qt.Horizontal,
            model=model, callback=self._update_labels,
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding,
                                   QSizePolicy.Fixed),
            contentsLength=10
        )
        self.color_by_label = QLabel("Color by:")
        grid.addWidget(self.color_by_label, 2, 0)
        grid.addWidget(cb, 2, 1)

        box = gui.radioButtons(
            self.controlArea, self, "pruning", box="Pruning",
            callback=self._invalidate_pruning)
        grid = QGridLayout()
        box.layout().addLayout(grid)
        grid.addWidget(
            gui.appendRadioButton(box, "None", addToLayout=False),
            0, 0
        )
        self.max_depth_spin = gui.spin(
            box, self, "max_depth", minv=1, maxv=100,
            controlWidth=spin_width, alignment=Qt.AlignRight,
            callback=self._max_depth_changed,
            keyboardTracking=False, addToLayout=False
        )
        self.max_depth_spin.lineEdit().returnPressed.connect(
            self._max_depth_return)

        grid.addWidget(
            gui.appendRadioButton(box, "Max depth:", addToLayout=False),
            1, 0)
        grid.addWidget(self.max_depth_spin, 1, 1)

        self.selection_box = gui.radioButtons(
            self.controlArea, self, "selection_method",
            box="Selection",
            callback=self._selection_method_changed)

        grid = QGridLayout()
        self.selection_box.layout().addLayout(grid)
        grid.addWidget(
            gui.appendRadioButton(
                self.selection_box, "Manual", addToLayout=False),
            0, 0
        )
        grid.addWidget(
            gui.appendRadioButton(
                self.selection_box, "Height ratio:", addToLayout=False),
            1, 0
        )
        self.cut_ratio_spin = gui.spin(
            self.selection_box, self, "cut_ratio", 0, 100, step=1e-1,
            controlWidth=spin_width, alignment = Qt.AlignRight,
            spinType=float, callback=self._cut_ratio_changed,
            addToLayout=False
        )
        self.cut_ratio_spin.setSuffix(" %")
        self.cut_ratio_spin.lineEdit().returnPressed.connect(
            self._cut_ratio_return)

        grid.addWidget(self.cut_ratio_spin, 1, 1)

        grid.addWidget(
            gui.appendRadioButton(
                self.selection_box, "Top N:", addToLayout=False),
            2, 0
        )
        self.top_n_spin = gui.spin(
            self.selection_box, self, "top_n", 1, 1000,
            controlWidth=spin_width, alignment=Qt.AlignRight,
            callback=self._top_n_changed, addToLayout=False)
        self.top_n_spin.lineEdit().returnPressed.connect(self._top_n_return)
        grid.addWidget(self.top_n_spin, 2, 1)

        self.zoom_slider = gui.hSlider(
            self.controlArea, self, "zoom_factor", box="Zoom",
            minValue=-6, maxValue=3, step=1, ticks=True, createLabel=False,
            callback=self.__update_font_scale)

        zoom_in = QAction(
            "Zoom in", self, shortcut=QKeySequence.ZoomIn,
            triggered=self.__zoom_in
        )
        zoom_out = QAction(
            "Zoom out", self, shortcut=QKeySequence.ZoomOut,
            triggered=self.__zoom_out
        )
        zoom_reset = QAction(
            "Reset zoom", self,
            shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_0),
            triggered=self.__zoom_reset
        )
        self.addActions([zoom_in, zoom_out, zoom_reset])

        self.controlArea.layout().addStretch()

        gui.auto_send(self.buttonsArea, self, "autocommit")

        self.scene = QGraphicsScene(self)
        self.view = GraphicsView(
            self.scene,
            horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOff,
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
            alignment=Qt.AlignLeft | Qt.AlignVCenter,
            widgetResizable=True,
        )
        # Disable conflicting action shortcuts. We define our own.
        for a in self.view.viewActions():
            a.setEnabled(False)
        self.mainArea.layout().setSpacing(1)
        self.mainArea.layout().addWidget(self.view)

        def axis_view(orientation):
            ax = AxisItem(orientation=orientation, maxTickLength=7)
            ax.mousePressed.connect(self._activate_cut_line)
            ax.mouseMoved.connect(self._activate_cut_line)
            ax.mouseReleased.connect(self._activate_cut_line)
            ax.setRange(1.0, 0.0)
            return ax

        self.top_axis = axis_view("top")
        self.bottom_axis = axis_view("bottom")

        self._main_graphics = QGraphicsWidget(
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        )
        scenelayout = QGraphicsGridLayout()
        scenelayout.setHorizontalSpacing(10)
        scenelayout.setVerticalSpacing(10)

        self._main_graphics.setLayout(scenelayout)
        self.scene.addItem(self._main_graphics)
        self.view.setCentralWidget(self._main_graphics)

        self.dendrogram = DendrogramWidget(pen_width=2, leaf_heights=True)
        self.dendrogram.setSizePolicy(QSizePolicy.MinimumExpanding,
                                      QSizePolicy.MinimumExpanding)
        self.dendrogram.selectionChanged.connect(self._invalidate_output)
        self.dendrogram.selectionEdited.connect(self._selection_edited)

        # DendrogramWidget paints NJ labels at branch ends. This empty column
        # only reserves space for labels that extend outside the dendrogram.
        self.labels = QGraphicsWidget()
        self.labels.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.labels.setMinimumWidth(0)
        self.labels.setMaximumWidth(0)

        scenelayout.addItem(self.top_axis, 0, 0,
                            alignment=Qt.AlignLeft | Qt.AlignVCenter)
        scenelayout.addItem(self.dendrogram, 1, 0,
                            alignment=Qt.AlignLeft | Qt.AlignVCenter)
        scenelayout.addItem(self.labels, 1, 1,
                            alignment=Qt.AlignLeft | Qt.AlignVCenter)
        scenelayout.addItem(self.bottom_axis, 2, 0,
                            alignment=Qt.AlignLeft | Qt.AlignVCenter)
        self.top_axis.setZValue(self.dendrogram.zValue() + 10)
        self.bottom_axis.setZValue(self.dendrogram.zValue() + 10)
        self.cut_line = SliderLine(self.top_axis,
                                   orientation=Qt.Horizontal)
        self.cut_line.valueChanged.connect(self._dendrogram_slider_changed)
        self.dendrogram.geometryChanged.connect(self._dendrogram_geom_changed)
        self._set_cut_line_visible(self.selection_method == 1)
        self.__update_font_scale()

    @Inputs.distances
    def set_distances(self, matrix):
        self.Error.clear()
        self.data = None
        self.matrix = None
        if matrix is not None:
            try:
                row_items = getattr(matrix, "row_items", None)
                if isinstance(row_items, Orange.data.Table) \
                        and not self._has_input_features(row_items):
                    self.Error.no_numeric_features()
                    return
                self._set_valid_matrix(matrix)
            except Exception as e:
                self.Error.distance_computation_error(error=str(e))

    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        self.data = data
        self.matrix = None
        if data is not None:
            try:
                from Orange import distance
                if not self._has_input_features(data):
                    self.Error.no_numeric_features()
                    return
                if data.domain.has_continuous_attributes():
                    matrix = distance.Euclidean(data, axis=1, impute=True)
                elif data.domain.has_discrete_attributes():
                    matrix = distance.Hamming(data, axis=1, impute=True)
                else:
                    self.Error.no_numeric_features()
                    return
                self._set_valid_matrix(matrix)
            except Exception as e:
                self.Error.distance_computation_error(error=str(e))

    @staticmethod
    def _has_input_features(data):
        return bool(data.domain.attributes)

    @Inputs.subset
    def set_subset(self, subset):
        self.subset = subset
        self.controls.label_only_subset.setDisabled(subset is None)

    def handleNewSignals(self):
        matrix = self.matrix
        if matrix is not None:
            self._set_items(matrix.row_items, matrix.axis)
        else:
            self._set_items(None)
        self._update()

        self.Warning.clear()
        rows = set()
        if self.subset and self.matrix is not None and self.root is not None:
            subsetids = set(self.subset.ids)
            if not isinstance(self.items, Orange.data.Table) \
                    or not self.matrix.axis:
                self.Warning.subset_on_no_table()
            elif (dataids := set(self.items.ids)) and not subsetids & dataids:
                self.Warning.subset_wrong()
            elif not subsetids <= dataids:
                self.Warning.subset_not_subset()
            else:
                indices = [leaf.value.index for leaf in leaves(self.root)]
                rows = {
                    row for row, rowid in enumerate(self.items.ids[indices])
                    if rowid in subsetids
                }
        elif self.subset:
            self.Warning.subset_on_no_table()

        self.subset_rows = rows
        self._update_labels()
        self.commit.now()

    def _set_items(self, items, axis=1):
        self.closeContext()
        self.items = items
        model = self.label_cb.model()
        color_model = self.controls.color_by.model()
        color_model.set_domain(None)
        self.color_by = None
        if len(model) == 2 and model[0] is None:
            if model[1] == "Name":
                self.annotation_if_names = self.annotation
            if model[1] == self.basic_annotations[1]:
                self.annotation_if_enumerate = self.annotation
        if isinstance(items, Orange.data.Table) and axis:
            metas_class = tuple(
                filter_visible(chain(items.domain.metas,
                                     items.domain.class_vars)))
            visible_attrs = tuple(filter_visible(items.domain.attributes))
            if not (metas_class or visible_attrs):
                model[:] = self.basic_annotations
            else:
                model[:] = (
                    (None, )
                    + metas_class
                    + (model.Separator, ) * bool(metas_class and visible_attrs)
                    + visible_attrs)
            for meta in items.domain.metas:
                if isinstance(meta, StringVariable):
                    self.annotation = meta
                    break
            else:
                if items.domain.class_vars:
                    # No string metas: show class
                    self.annotation = items.domain.class_vars[0]
                else:
                    # No string metas and no class: show the first option
                    # which is not None (in the worst case, Enumeration)
                    self.annotation = model[1]

            color_model.set_domain(items.domain)
            if items.domain.class_vars:
                self.color_by = items.domain.class_vars[0]
            self.openContext(items.domain)
        elif isinstance(items, Orange.data.Table) and not axis \
                or (isinstance(items, list) and \
                    all(isinstance(var, Orange.data.Variable)
                        for var in items)):
            model[:] = (None, "Name")
            self.annotation = self.annotation_if_names
        else:
            model[:] = self.basic_annotations
            self.annotation = self.annotation_if_enumerate

        no_colors = len(color_model) == 1
        self.controls.color_by.setDisabled(no_colors)
        self.color_by_label.setDisabled(no_colors)

    def _clear_plot(self):
        self.dendrogram.set_root(None)

    def _set_displayed_root(self, root):
        self._clear_plot()
        self._displayed_root = root
        # Pruned NJ trees can have a smaller displayed height. Keep the axis
        # scale tied to the full tree so pruning does not visually rescale it.
        height = self.root.value.height if self.root else None
        self.dendrogram.set_reference_height(height)
        self.dendrogram.set_root(root)
        self._update_labels()

    def _update(self):
        self._clear_plot()
        distances = self.matrix
        if distances is not None:
            try:
                D = np.asarray(distances, dtype=float)

                # Use stable, unique internal labels. Display labels can be
                # duplicated, while the adapter maps TreeNode names back to rows.
                labels = [str(i) for i in range(D.shape[0])]
                label_to_index = dict(zip(labels, range(len(labels))))

                nj_root = neighbor_joining_core(D, labels)
                reorder_children(nj_root, D, label_to_index)
                tree = treenode_to_orange_tree(nj_root, label_to_index)

                self.root = tree
                self.top_axis.setRange(tree.value.height, 0.0)
                self.bottom_axis.setRange(tree.value.height, 0.0)

                if self.pruning:
                    self._set_displayed_root(prune(tree, level=self.max_depth))
                else:
                    self._set_displayed_root(tree)
            except Exception as ex:
                self.root = None
                self._set_displayed_root(None)
                self.Error.tree_construction_error(error=str(ex))
        else:
            self.root = None
            self._set_displayed_root(None)

        self._apply_selection()


    def _update_labels(self):
        if not hasattr(self, "labels"):
            # This method can be called during widget initialization when
            # creating check box for label_only_subset, if it's value is
            # initially True.
            # See https://github.com/biolab/orange-widget-base/pull/213;
            # if it's merged, this check can be removed.
            return

        self.Warning.pruning_disables_colors(
            shown=self.pruning
                  and (self.subset_rows or self.color_by is not None))
        labels = []
        if self.root and self._displayed_root:
            indices = [leaf.value.index for leaf in leaves(self.root)]

            if self.annotation is None:
                if not self.pruning \
                        and self.subset_rows and self.color_by is None:
                    # Empty strings let the dendrogram still show subset bolding.
                    labels = [""] * len(indices)
                else:
                    labels = []
            elif self.annotation == "Enumeration":
                labels = [str(i+1) for i in indices]
            elif self.annotation == "Name":
                row_items = self.matrix.row_items
                labels = []
                for i in indices:
                    item = row_items[i]
                    if isinstance(item, Orange.data.Instance):
                        # Prefer class label if present
                        if item.domain.class_vars:
                            labels.append(str(item.get_class()))
                        else:
                            labels.append(str(item))
                    elif hasattr(item, "name"):
                        labels.append(item.name)
                    else:
                        labels.append(str(item))

            elif isinstance(self.annotation, Orange.data.Variable):
                col_data = self.items.get_column(self.annotation)
                labels = [self.annotation.str_val(val) for val in col_data]
                labels = [labels[idx] for idx in indices]
            else:
                labels = []
            if not self.pruning and \
                    labels and self.label_only_subset and self.subset_rows:
                labels = [label if row in self.subset_rows else ""
                          for row, label in enumerate(labels)]

            if labels and self._displayed_root is not self.root:
                index_to_label = {
                    leaf.value.index: labels[pos]
                    for pos, leaf in enumerate(leaves(self.root))
                }

                new_labels = []
                for node in leaves(self._displayed_root):
                    leaf_indices = node.value.members
                    new_labels.append(
                        ", ".join(index_to_label[i] for i in leaf_indices)
                    )

                labels = new_labels

        # Compute per-leaf colors (for the small rectangle) and subset bold mask
        colors = None
        if not self.pruning and self.color_by is not None and labels:
            col = self.items.get_column(self.color_by)
            colors = list(self.color_by.palette.values_to_qcolors(col[indices]))

        subset = set() if self.pruning else (self.subset_rows or set())
        bold_mask = [i in subset for i in range(len(labels))] if labels else []

        self.dendrogram.set_leaf_label_max_width(
            MAX_PRUNED_LABEL_WIDTH if self.pruning else None
        )

        # labels are rendered at branch ends
        # by DendrogramWidget so non-ultrametric leaves stay visually attached.
        if labels:
            selected_mask = bold_mask if subset else None
            self.dendrogram.set_leaf_labels(labels, colors=colors,
                                            selected=selected_mask,
                                            bold=bold_mask)
        else:
            self.dendrogram.clear_leaf_labels()

        self._update_label_column_width()

    def _update_label_column_width(self):
        width = int(self.dendrogram.leaf_label_width_hint())
        self.labels.setMinimumWidth(width)
        self.labels.setPreferredWidth(width)
        self.labels.setMaximumWidth(width)
        self.labels.updateGeometry()

    def _set_selected_nodes(self, selection):
        # type: (List[Tree]) -> None
        """
        Set the nodes in `selection` to be the current selected nodes.

        The selection nodes must be subtrees of the current `_displayed_root`.
        """
        self.dendrogram.selectionChanged.disconnect(self._invalidate_output)
        try:
            self.dendrogram.set_selected_clusters(selection)
        finally:
            self.dendrogram.selectionChanged.connect(self._invalidate_output)

    def _max_depth_return(self):
        if self.pruning != 1:
            self.pruning = 1
            self._invalidate_pruning()

    def _max_depth_changed(self):
        self.pruning = 1
        self._invalidate_pruning()

    def _invalidate_output(self):
        self.commit.deferred()

    def _invalidate_pruning(self):
        if self.root:
            selection = self.dendrogram.selected_nodes()
            ranges = [node.value.range for node in selection]
            if self.pruning:
                self._set_displayed_root(
                    prune(self.root, level=self.max_depth))
            else:
                self._set_displayed_root(self.root)
            selected = [node for node in preorder(self._displayed_root)
                        if node.value.range in ranges]

            self.dendrogram.set_selected_clusters(selected)

        self._apply_selection()

    @gui.deferred
    def commit(self):
        items = getattr(self.matrix, "items", self.items)
        self.Warning.many_clusters.clear()
        if not items or self.root is None:
            self.Outputs.selected_data.send(None)
            self.Outputs.annotated_data.send(None)
            return

        selection = self.dendrogram.selected_nodes()
        selection = sorted(selection, key=lambda c: c.value.first)

        indices = [leaf.value.index for leaf in leaves(self.root)]

        maps = [indices[node.value.first:node.value.last]
                for node in selection]
        if len(maps) > 20:
            self.Warning.many_clusters()

        selected_indices = list(chain(*maps))

        if not selected_indices:
            self.Outputs.selected_data.send(None)
            annotated_data = lazy_annotated_table(items, []) \
                if self.selection_method == 0 and self.matrix.axis else None
            self.Outputs.annotated_data.send(annotated_data)
            return

        selected_data = annotated_data = None

        if isinstance(items, Orange.data.Table) and self.matrix.axis == 1:
            # Select rows
            data, domain = items, items.domain

            c = np.full(self.matrix.shape[0], len(maps))
            for i, indices in enumerate(maps):
                c[indices] = i

            clust_name = get_unique_names(domain, "Cluster")
            values = [f"C{i + 1}" for i in range(len(maps))]

            sel_clust_var = Orange.data.DiscreteVariable(
                name=clust_name, values=values)
            sel_domain = add_columns(domain, metas=(sel_clust_var,))
            selected_data = LazyValue[Table](
                lambda: items.add_column(
                    sel_clust_var, c, to_metas=True)[c != len(maps)],
                domain=sel_domain, length=len(selected_indices))

            ann_clust_var = Orange.data.DiscreteVariable(
                name=clust_name, values=values + ["Other"]
            )
            ann_domain = add_columns(
                domain_with_annotation_column(data)[0], metas=(ann_clust_var, ))
            annotated_data = LazyValue[Table](
                lambda: create_annotated_table(
                    data=items.add_column(ann_clust_var, c, to_metas=True),
                    selected_indices=selected_indices),
                domain=ann_domain, length=len(items)
            )

        elif isinstance(items, Orange.data.Table) and self.matrix.axis == 0:
            # Select columns
            attrs = []
            unselected_indices = sorted(set(range(self.root.value.last)) -
                                        set(selected_indices))
            for clust, indices in chain(enumerate(maps, start=1),
                                        [(0, unselected_indices)]):
                for i in indices:
                    attr = items.domain[i].copy()
                    attr.attributes["cluster"] = clust
                    attrs.append(attr)
            all_domain = Orange.data.Domain(
                # len(unselected_indices) can be 0
                attrs[:len(attrs) - len(unselected_indices)],
                items.domain.class_vars, items.domain.metas)

            selected_data = LazyValue[Table](
                lambda: items.from_table(all_domain, items),
                domain=all_domain, length=len(items))

            sel_domain = Orange.data.Domain(
                attrs,
                items.domain.class_vars, items.domain.metas)
            annotated_data = LazyValue[Table](
                lambda: items.from_table(sel_domain, items),
                domain=sel_domain, length=len(items))

        self.Outputs.selected_data.send(selected_data)
        self.Outputs.annotated_data.send(annotated_data)

    @Slot(QPointF)
    def _activate_cut_line(self, pos: QPointF):
        """Activate cut line selection an set cut value to `pos.x()`."""
        self.selection_method = 1
        self.cut_line.setValue(pos.x())
        self._selection_method_changed()

    def onDeleteWidget(self):
        super().onDeleteWidget()
        self._clear_plot()
        self.dendrogram.clear()
        self.dendrogram.deleteLater()

    def _dendrogram_geom_changed(self):
        pos = self.dendrogram.pos_at_height(self.cutoff_height)
        dendro_geom = self.dendrogram.geometry()
        self._set_slider_value(pos.x(), dendro_geom.width())

        self.cut_line.setLength(
            self.bottom_axis.geometry().bottom()
            - self.top_axis.geometry().top()
        )

        geom = self._main_graphics.geometry()
        assert geom.topLeft() == QPointF(0, 0)

        def adjustLeft(rect):
            rect = QRectF(rect)
            rect.setLeft(geom.left())
            return rect
        margin = 3
        self.scene.setSceneRect(geom)
        self.view.setSceneRect(geom)

        def headerFooterRect(axis):
            rect = QRectF(axis.geometry())
            rect.setLeft(dendro_geom.left())
            rect.setWidth(dendro_geom.width())
            return rect

        self.view.setHeaderSceneRect(
            adjustLeft(headerFooterRect(self.top_axis)).adjusted(0, 0, 0, margin)
        )
        self.view.setFooterSceneRect(
            adjustLeft(headerFooterRect(self.bottom_axis)).adjusted(0, -margin, 0, 0)
        )

    def _dendrogram_slider_changed(self, value):
        p = QPointF(value, 0)
        cl_height = self.dendrogram.height_at(p)

        self.set_cutoff_height(cl_height)

    def _set_slider_value(self, value, span):
        with blocked(self.cut_line):
            self.cut_line.setRange(0, span)
            self.cut_line.setValue(value)

    def set_cutoff_height(self, height):
        self.cutoff_height = height
        if self.root and self.root.value.height:
            self.cut_ratio = 100 * height / self.root.value.height
        self.select_max_height(height)

    def _set_cut_line_visible(self, visible):
        self.cut_line.setVisible(visible)

    def select_top_n(self, n):
        root = self._displayed_root
        if not root:
            return

        # Orange's top_clusters is tailored to hierarchical clustering. Keep
        # splitting the highest visible NJ cluster until n clusters are shown.
        heap = [(-root.value.height, id(root), root)]

        while len(heap) < n:
            popped = []
            splittable = None
            while heap:
                pr, tie, cl = heapq.heappop(heap)
                if not cl.is_leaf:
                    splittable = (pr, tie, cl)
                    break
                popped.append((pr, tie, cl))

            for item in popped:
                heapq.heappush(heap, item)
            if splittable is None:
                break

            _, _, cl = splittable
            for child in cl.branches:
                heapq.heappush(heap, (-child.value.height, id(child), child))

        clusters = [cl for _, _, cl in heap]
        self.dendrogram.set_selected_clusters(clusters)

    def select_max_height(self, height):
        root = self._displayed_root
        if root:
            clusters = clusters_at_height(root, height)
            self.dendrogram.set_selected_clusters(clusters)

    def _cut_ratio_changed(self):
        self.selection_method = 1
        self._selection_method_changed()

    def _cut_ratio_return(self):
        if self.selection_method != 1:
            self.selection_method = 1
            self._selection_method_changed()

    def _top_n_changed(self):
        self.selection_method = 2
        self._selection_method_changed()

    def _top_n_return(self):
        if self.selection_method != 2:
            self.selection_method = 2
            self._selection_method_changed()

    def _selection_method_changed(self):
        self._set_cut_line_visible(self.selection_method == 1)
        if self.root:
            self._apply_selection()

    def _apply_selection(self):
        if not self.root:
            return

        if self.selection_method == 0:
            pass
        elif self.selection_method == 1:
            height = self.cut_ratio * self.root.value.height / 100
            self.set_cutoff_height(height)
            pos = self.dendrogram.pos_at_height(height)
            self._set_slider_value(pos.x(), self.dendrogram.size().width())
        elif self.selection_method == 2:
            self.select_top_n(self.top_n)

    def _selection_edited(self):
        # Selection was edited by clicking on a cluster in the
        # dendrogram view.
        self.selection_method = 0
        self._selection_method_changed()
        self._invalidate_output()

    def __zoom_in(self):
        def clip(minval, maxval, val):
            return min(max(val, minval), maxval)
        self.zoom_factor = clip(self.zoom_slider.minimum(),
                                self.zoom_slider.maximum(),
                                self.zoom_factor + 1)
        self.__update_font_scale()

    def __zoom_out(self):
        def clip(minval, maxval, val):
            return min(max(val, minval), maxval)
        self.zoom_factor = clip(self.zoom_slider.minimum(),
                                self.zoom_slider.maximum(),
                                self.zoom_factor - 1)
        self.__update_font_scale()

    def __zoom_reset(self):
        self.zoom_factor = 0
        self.__update_font_scale()

    def __update_font_scale(self):
        font = self.scene.font()
        factor = (1.25 ** self.zoom_factor)
        font = qfont_scaled(font, factor)
        self._main_graphics.setFont(font)
        self.dendrogram.setFont(font)
        self._update_label_column_width()

    def send_report(self):
        annot = self.label_cb.currentText()
        if isinstance(self.annotation, str):
            annot = annot.lower()
        if self.selection_method == 0:
            sel = "manual"
        elif self.selection_method == 1:
            sel = "at {:.1f} of height".format(self.cut_ratio)
        else:
            sel = f"top {self.top_n} {pl(self.top_n, 'cluster')}"
        self.report_items((
            ("Method", "Neighbor Joining"),
            ("Annotation", annot),
            ("Pruning",
            self.pruning != 0 and "{} levels".format(self.max_depth)),
            ("Selection", sel),
        ))
        self.report_plot()

    @classmethod
    def migrate_context(cls, context, version):
        if version < 2:
            if context.values["annotation"] == "None":
                context.values["annotation"] = None


def qfont_scaled(font, factor):
    scaled = QFont(font)
    if font.pointSizeF() != -1:
        scaled.setPointSizeF(font.pointSizeF() * factor)
    elif font.pixelSize() != -1:
        scaled.setPixelSize(int(font.pixelSize() * factor))
    return scaled


class AxisItem(AxisItem):
    mousePressed = Signal(QPointF, Qt.MouseButton)
    mouseMoved = Signal(QPointF, Qt.MouseButtons)
    mouseReleased = Signal(QPointF, Qt.MouseButton)

    #: \reimp
    def wheelEvent(self, event):
        event.ignore()  # ignore event to propagate to the view -> scroll

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.mousePressed.emit(event.pos(), event.button())
        super().mousePressEvent(event)
        event.accept()

    def mouseMoveEvent(self, event):
        self.mouseMoved.emit(event.pos(), event.buttons())
        super().mouseMoveEvent(event)
        event.accept()

    def mouseReleaseEvent(self, event):
        self.mouseReleased.emit(event.pos(), event.button())
        super().mouseReleaseEvent(event)
        event.accept()


class SliderLine(QGraphicsWidget):
    """A movable slider line."""
    valueChanged = Signal(float)

    def __init__(self, parent=None, orientation=Qt.Vertical, value=0.0,
                 length=10.0, **kwargs):
        self._orientation = orientation
        self._value = value
        self._length = length
        self._min = 0.0
        self._max = 1.0
        self._line: Optional[QLineF] = QLineF()
        self._pen: Optional[QPen] = None
        super().__init__(parent, **kwargs)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        if self._orientation == Qt.Vertical:
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.setCursor(Qt.SizeHorCursor)

    def pen(self) -> QPen:
        if self._pen is None:
            return QPen(self.palette().text(), 1.0, Qt.DashLine)
        else:
            return QPen(self._pen)

    def setValue(self, value: float):
        value = min(max(value, self._min), self._max)

        if self._value != value:
            self.prepareGeometryChange()
            self._value = value
            self._line = None
            self.valueChanged.emit(value)

    def value(self) -> float:
        return self._value

    def setRange(self, minval: float, maxval: float) -> None:
        maxval = max(minval, maxval)
        if minval != self._min or maxval != self._max:
            self._min = minval
            self._max = maxval
            self.setValue(self._value)

    def setLength(self, length: float):
        if self._length != length:
            self.prepareGeometryChange()
            self._length = length
            self._line = None

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        event.accept()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        if self._orientation == Qt.Vertical:
            self.setValue(pos.y())
        else:
            self.setValue(pos.x())
        event.accept()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._orientation == Qt.Vertical:
            self.setValue(event.pos().y())
        else:
            self.setValue(event.pos().x())
        event.accept()

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(self.boundingRect())
        return path

    def boundingRect(self) -> QRectF:
        if self._line is None:
            if self._orientation == Qt.Vertical:
                self._line = QLineF(0, self._value, self._length, self._value)
            else:
                self._line = QLineF(self._value, 0, self._value, self._length)
        r = QRectF(self._line.p1(), self._line.p2())
        penw = self.pen().width()
        return r.adjusted(-penw, -penw, penw, penw)

    def paint(self, painter, *args):
        if self._line is None:
            self.boundingRect()

        painter.save()
        painter.setPen(self.pen())
        painter.drawLine(self._line)
        painter.restore()

def clusters_at_height(root, height):
    """Return a list of clusters by cutting the tree at `height`.
    """
    selected = []
    covered = set()
    for cl in preorder(root):
        if cl in covered:
            continue
        if cl.value.height < height:
            selected.append(cl)
            covered.update(preorder(cl))

    for leaf in leaves(root):
        # NJ leaves can have positive heights; include leaves not already
        # covered by an internal cluster below the cut.
        if leaf not in covered:
            selected.append(leaf)

    return selected

def main():
    # pragma: no cover
    from Orange import distance
    data = Orange.data.Table("iris")
    matrix = distance.Euclidean(data)
    WidgetPreview(OWNeighborJoining).run(matrix)

if __name__ == "__main__":  # pragma: no cover
    main()