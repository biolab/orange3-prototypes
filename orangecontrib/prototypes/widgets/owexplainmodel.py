from typing import Tuple, Optional, List, Callable
from types import SimpleNamespace

import numpy as np

from AnyQt.QtCore import Qt, QRectF, QSizeF, QSize, pyqtSignal as Signal
from AnyQt.QtGui import QColor, QPen, QBrush, QPainter, QLinearGradient
from AnyQt.QtWidgets import QGraphicsItemGroup, QGraphicsLineItem, \
    QGraphicsScene, QGraphicsWidget, QGraphicsGridLayout, \
    QGraphicsEllipseItem, QGraphicsSimpleTextItem, QSizePolicy, \
    QGraphicsRectItem, QGraphicsSceneMouseEvent

import pyqtgraph as pg

from Orange.base import Model
from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.data.table import DomainTransformationError
from Orange.widgets import gui
from Orange.widgets.settings import Setting, ContextSetting, \
    ClassValuesContextHandler
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.utils.graphicslayoutitem import SimpleLayoutItem
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.utils.stickygraphicsview import StickyGraphicsView
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output, OWWidget, Msg

from orangecontrib.prototypes.explanation.explainer import \
    get_shap_values_and_colors, RGB_LOW, RGB_HIGH, temp_seed


class Results(SimpleNamespace):
    x = None  # type: Optional[List[np.ndarray]]
    colors = None  # type: Optional[List[np.ndarray]]
    names = None  # type: Optional[List[str]]
    mask = None  # type: Optional[List[np.ndarray]]


def run(data: Table, model: Model, state: TaskState) -> Results:
    if not data or not model:
        return None

    def callback(i: float, status=""):
        state.set_progress_value(i * 100)
        if status:
            state.set_status(status)
        if state.is_interruption_requested():
            raise Exception

    x, names, mask, colors = get_shap_values_and_colors(model, data, callback)
    return Results(x=x, colors=colors, names=names, mask=mask)


class Legend(QGraphicsWidget):
    WIDTH = 30
    BAR_WIDTH = 7

    def __init__(self, parent):
        super().__init__(parent)
        self.__offset = 2
        self.__group = QGraphicsItemGroup(self)
        self.__bar_height = ViolinItem.HEIGHT * 3
        self._add_bar()
        self._add_high_label()
        self._add_low_label()
        self._add_feature_label()

    def _add_bar(self):
        item = QGraphicsRectItem(0, 0, self.BAR_WIDTH, self.__bar_height)
        gradient = QLinearGradient(0, 0, 0, self.__bar_height)
        gradient.setColorAt(0, QColor(*RGB_HIGH))
        gradient.setColorAt(1, QColor(*RGB_LOW))
        item.setPen(QPen(Qt.NoPen))
        item.setBrush(gradient)
        self.__group.addToGroup(item)

    def _add_high_label(self):
        font = self.font()
        font.setPixelSize(9)
        item = QGraphicsSimpleTextItem("High")
        item.setFont(font)
        item.setX(self.BAR_WIDTH + self.__offset)
        item.setY(0)
        self.__group.addToGroup(item)

    def _add_low_label(self):
        font = self.font()
        font.setPixelSize(9)
        item = QGraphicsSimpleTextItem("Low")
        item.setFont(font)
        item.setX(self.BAR_WIDTH + self.__offset)
        item.setY(self.__bar_height - item.boundingRect().height())
        self.__group.addToGroup(item)

    def _add_feature_label(self):
        font = self.font()
        font.setPixelSize(11)
        item = QGraphicsSimpleTextItem("Feature value")
        item.setRotation(-90)
        item.setFont(font)
        item.setX(self.BAR_WIDTH + self.__offset * 2)
        item.setY(self.__bar_height / 2 + item.boundingRect().width() / 2)
        self.__group.addToGroup(item)

    def sizeHint(self, *_):
        return QSizeF(self.WIDTH, ViolinItem.HEIGHT)


class ViolinItem(QGraphicsWidget):
    HEIGHT = 50
    POINT_R = 6
    SCALE_FACTOR = 0.5
    selection_changed = Signal(float, float, str)

    class SelectionRect(QGraphicsRectItem):
        COLOR = [255, 255, 0]

        def __init__(self, parent, width: int):
            super().__init__(parent)
            self.parent_width = width

            color = QColor(*self.COLOR)
            color.setAlpha(100)
            self.setBrush(color)

            color = QColor(*self.COLOR)
            self.setPen(color)

    def __init__(self, parent, attr_name: str, x_range: Tuple[float],
                 width: int):
        super().__init__(parent)
        assert x_range[0] == -x_range[1]
        self.__attr_name = attr_name
        self.__width = width
        self.__range = x_range[1] if x_range[1] else 1
        self.__group = None  # type: Optional[QGraphicsItemGroup]
        self.__selection_rect = None  # type: Optional[QGraphicsRectItem]
        self.x_data = None  # type: Optional[np.ndarray]
        parent.selection_cleared.connect(self.__remove_selection_rect)

    @property
    def attr_name(self):
        return self.__attr_name

    def set_data(self, x_data: np.ndarray, color_data: np.ndarray):
        def put_point(_x, _y):
            item = QGraphicsEllipseItem()
            item.setX(_x)
            item.setY(_y)
            item.setRect(0, 0, self.POINT_R, self.POINT_R)
            color = QColor(*colors.pop())
            item.setPen(QPen(color))
            item.setBrush(QBrush(color))
            self.__group.addToGroup(item)

        self.x_data = x_data
        self.__group = QGraphicsItemGroup(self)

        x_data_unique, dist, x_data = self.prepare_data()
        for x, d in zip(x_data_unique, dist):
            colors = color_data[x_data == np.round(x, 3)]
            colors = list(colors[np.random.choice(len(colors), 11)])
            y = self.HEIGHT / 2 - self.POINT_R / 2
            self.plot_data(put_point, x, y, d)

    def prepare_data(self):
        x_data = self._values_to_pixels(self.x_data)
        x_data = x_data[~np.isnan(x_data)]
        if len(x_data) == 0:
            return

        x_data = np.round(x_data - self.POINT_R / 2, 3)

        # remove duplicates and get counts (distribution) to set y
        x_data_unique, counts = np.unique(x_data, return_counts=True)
        min_count, max_count = np.min(counts), np.max(counts)
        dist = (counts - min_count) / (max_count - min_count)
        if min_count == max_count:
            dist[:] = 1
        dist = dist ** 0.7

        # plot rarest values first
        indices = np.argsort(counts)
        return x_data_unique[indices], dist[indices], x_data

    @staticmethod
    def plot_data(func: Callable, x: float, y: float, d: float):
        func(x, y)  # y = 0
        if d > 0:
            offset = d * 10
            func(x, y + offset)  # y = (0, 10]
            func(x, y - offset)  # y = (-10, 0]
            for i in range(2, int(offset), 2):
                func(x, y + i)  # y = [2, 8]
                func(x, y - i)  # y = [-8, -2]

    def _values_to_pixels(self, x: np.ndarray) -> np.ndarray:
        # scale data to [-0.5, 0.5]
        x = x / self.__range * self.SCALE_FACTOR
        # round data to 3. decimal for sampling
        x = np.round(x, 3)
        # convert to pixels
        return x * self.__width + self.__width / 2

    def _values_from_pixels(self, p: np.ndarray) -> np.ndarray:
        # convert from pixels
        x = (p - self.__width / 2) / self.__width
        # rescale data from [-0.5, 0.5]
        return np.round(x * self.__range / self.SCALE_FACTOR, 3)

    def rescale(self, width):
        def move_point(_x, *_):
            item = next(points)
            item.setX(_x)

        self.__width = width
        self.updateGeometry()
        points = (item for item in self.__group.childItems())

        x_data_unique, dist, x_data = self.prepare_data()
        for x, d in zip(x_data_unique, dist):
            self.plot_data(move_point, x, 0, d)

        if self.__selection_rect is not None:
            old_width = self.__selection_rect.parent_width
            rect = self.__selection_rect.rect()
            x1 = self.__width * rect.x() / old_width
            x2 = self.__width * (rect.x() + rect.width()) / old_width
            rect = QRectF(x1, rect.y(), x2 - x1, rect.height())
            self.__selection_rect.setRect(rect)
            self.__selection_rect.parent_width = self.__width

    def sizeHint(self, *_):
        return QSizeF(self.__width, self.HEIGHT)

    def __remove_selection_rect(self):
        if self.__selection_rect is not None:
            self.__selection_rect.setParentItem(None)
            if self.scene() is not None:
                self.scene().removeItem(self.__selection_rect)
            self.__selection_rect = None

    def add_selection_rect(self, x1, x2):
        x1, x2 = self._values_to_pixels(np.array([x1, x2]))
        rect = QRectF(x1, 0, x2 - x1, self.HEIGHT)
        self.__selection_rect = ViolinItem.SelectionRect(self, self.__width)
        self.__selection_rect.setRect(rect)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        event.accept()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        if event.buttons() & Qt.LeftButton:
            if self.__selection_rect is None:
                self.__selection_rect = ViolinItem.SelectionRect(
                    self, self.__width)
            x = event.buttonDownPos(Qt.LeftButton).x()
            rect = QRectF(x, 0, event.pos().x() - x, self.HEIGHT).normalized()
            rect = rect.intersected(self.contentsRect())
            self.__selection_rect.setRect(rect)
            event.accept()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        x1 = event.buttonDownPos(Qt.LeftButton).x()
        x2 = event.pos().x()
        if x1 > x2:
            x2, x1 = x1, x2
        x1, x2 = self._values_from_pixels(np.array([x1, x2]))
        self.selection_changed.emit(x1, x2, self.__attr_name)
        event.accept()


class ViolinPlot(QGraphicsWidget):
    LABEL_COLUMN, VIOLIN_COLUMN, LEGEND_COLUMN = range(3)
    VIOLIN_COLUMN_WIDTH, OFFSET = 300, 250
    MAX_N_ITEMS = 100
    MAX_ATTR_LEN = 20
    selection_cleared = Signal()
    selection_changed = Signal(float, float, str)

    def __init__(self):
        super().__init__()
        self.__violin_column_width = self.VIOLIN_COLUMN_WIDTH  # type: int
        self.__range = None  # type: Optional[Tuple[float, float]]
        self.__violin_items = []  # type: List[ViolinItem]
        self.__bottom_axis = pg.AxisItem(parent=self, orientation="bottom",
                                         maxTickLength=7, pen=QPen(Qt.black))
        self.__bottom_axis.setLabel("Impact on model output")
        self.__vertical_line = QGraphicsLineItem(self.__bottom_axis)
        self.__vertical_line.setPen(QPen(Qt.gray))
        self.__legend = Legend(self)

        self.__layout = QGraphicsGridLayout()
        self.__layout.addItem(self.__legend, 0, ViolinPlot.LEGEND_COLUMN)
        self.__layout.setVerticalSpacing(0)
        self.setLayout(self.__layout)

    @property
    def violin_column_width(self):
        return self.__violin_column_width

    @violin_column_width.setter
    def violin_column_width(self, view_width: int):
        self.__violin_column_width = max(self.VIOLIN_COLUMN_WIDTH,
                                         view_width - self.OFFSET)

    @property
    def bottom_axis(self):
        return self.__bottom_axis

    def set_data(self, x: np.ndarray, colors: np.ndarray,
                 names: List[str], n_attrs: float, view_width: int):
        self.violin_column_width = view_width
        abs_max = np.max(np.abs(x)) * 1.05
        self.__range = (-abs_max, abs_max)
        self._set_violin_items(x, colors, names)
        self._set_labels(names)
        self._set_bottom_axis()
        self._set_vertical_line()
        self.set_n_visible(n_attrs)

    def set_n_visible(self, n: int):
        for i in range(len(self.__violin_items)):
            violin_item = self.__layout.itemAt(i, ViolinPlot.VIOLIN_COLUMN)
            violin_item.setVisible(i < n)
            text_item = self.__layout.itemAt(i, ViolinPlot.LABEL_COLUMN).item
            text_item.setVisible(i < n)

        x = self.__vertical_line.line().x1()
        n = min(n, len(self.__violin_items))
        self.__vertical_line.setLine(x, 0, x, -ViolinItem.HEIGHT * n)

    def rescale(self, view_width: int):
        self.violin_column_width = view_width
        with temp_seed(0):
            for item in self.__violin_items:
                item.rescale(self.violin_column_width)

        self.__bottom_axis.setWidth(self.violin_column_width)
        x = self.violin_column_width / 2
        self.__vertical_line.setLine(x, 0, x, self.__vertical_line.line().y2())

    def show_legend(self, show: bool):
        self.__legend.setVisible(show)
        self.__bottom_axis.setWidth(self.violin_column_width)
        x = self.violin_column_width / 2
        self.__vertical_line.setLine(x, 0, x, self.__vertical_line.line().y2())

    def _set_violin_items(self, x: np.ndarray, colors: np.ndarray,
                          labels: List[str]):
        with temp_seed(0):
            for i in range(x.shape[1]):
                item = ViolinItem(self, labels[i], self.__range,
                                  self.violin_column_width)
                item.set_data(x[:, i], colors[:, i])
                item.selection_changed.connect(self.select)
                self.__violin_items.append(item)
                self.__layout.addItem(item, i, ViolinPlot.VIOLIN_COLUMN)
                if i == self.MAX_N_ITEMS:
                    break

    def _set_labels(self, labels: List[str]):
        for i, (label, _) in enumerate(zip(labels, self.__violin_items)):
            short = f"{label[:self.MAX_ATTR_LEN - 1]}..." \
                if len(label) > self.MAX_ATTR_LEN else label
            text = QGraphicsSimpleTextItem(short, self)
            text.setToolTip(label)
            item = SimpleLayoutItem(text)
            item.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.__layout.addItem(item, i, ViolinPlot.LABEL_COLUMN,
                                  Qt.AlignRight | Qt.AlignVCenter)

    def _set_bottom_axis(self):
        self.__bottom_axis.setRange(*self.__range)
        self.__layout.addItem(self.__bottom_axis,
                              len(self.__violin_items),
                              ViolinPlot.VIOLIN_COLUMN)

    def _set_vertical_line(self):
        x = self.violin_column_width / 2
        n = len(self.__violin_items)
        self.__vertical_line.setLine(x, 0, x, -ViolinItem.HEIGHT * n)

    def deselect(self):
        self.selection_cleared.emit()

    def select(self, *args):
        self.selection_changed.emit(*args)

    def select_from_settings(self, x1: float, x2: float, attr_name: str):
        point_r_diff = 2 * self.__range[1] / (self.violin_column_width / 2)
        for item in self.__violin_items:
            if item.attr_name == attr_name:
                item.add_selection_rect(x1 - point_r_diff, x2 + point_r_diff)
                break
        self.select(x1, x2, attr_name)


class GraphicsScene(QGraphicsScene):
    mouse_clicked = Signal(object)

    def mousePressEvent(self, event):
        self.mouse_clicked.emit(event)
        super().mousePressEvent(event)


class GraphicsView(StickyGraphicsView):
    resized = Signal()

    def resizeEvent(self, ev):
        if ev.size().width() != ev.oldSize().width():
            self.resized.emit()
        return super().resizeEvent(ev)


class OWExplainModel(OWWidget, ConcurrentWidgetMixin):
    name = "Explain Model"
    description = "Model explanation widget."
    icon = "icons/ExplainModel.svg"
    priority = 100

    class Inputs:
        data = Input("Data", Table, default=True)
        model = Input("Model", Model)

    class Outputs:
        selected_data = Output("Selected Data", Table)
        scores = Output("Scores", Table)

    class Error(OWWidget.Error):
        domain_transform_err = Msg("{}")
        unknown_err = Msg("{}")

    class Information(OWWidget.Information):
        data_sampled = Msg("Data has been sampled.")

    settingsHandler = ClassValuesContextHandler()
    target_index = ContextSetting(0)
    n_attributes = Setting(10)
    show_legend = Setting(True)
    selection = Setting((), schema_only=True)  # type: Tuple[str, List[int]]
    auto_send = Setting(True)

    graph_name = "scene"

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.__results = None  # type: Optional[Results]
        self.data = None  # type: Optional[Table]
        self.model = None  # type: Optional[Model]
        self._violin_plot = None  # type: Optional[ViolinPlot]
        self.setup_gui()
        self.__pending_selection = self.selection

    def setup_gui(self):
        self._add_controls()
        self._add_plot()
        self.info.set_input_summary(self.info.NoInput)
        self.info.set_output_summary(self.info.NoOutput)

    def _add_plot(self):
        self.scene = GraphicsScene()
        self.view = GraphicsView(self.scene)
        self.view.resized.connect(self.update_plot)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.mainArea.layout().addWidget(self.view)

    def _add_controls(self):
        box = gui.vBox(self.controlArea, "Target class")
        self._target_combo = gui.comboBox(
            box, self, "target_index",
            callback=self.__target_combo_changed,
            contentsLength=12)

        box = gui.hBox(self.controlArea, "Display features")
        gui.label(box, self, "Best ranked: ")
        gui.spin(box, self, "n_attributes", 1, ViolinPlot.MAX_N_ITEMS,
                 controlWidth=80, callback=self.__n_spin_changed)
        box = gui.hBox(self.controlArea, True)
        gui.checkBox(box, self, "show_legend", "Show legend",
                     callback=self.__show_check_changed)

        gui.rubber(self.controlArea)
        box = gui.vBox(self.controlArea, box=True)
        gui.auto_send(box, self, "auto_send", box=False)

    def __target_combo_changed(self):
        self.update_scene()
        self.clear_selection()

    def __n_spin_changed(self):
        if self._violin_plot is not None:
            self._violin_plot.set_n_visible(self.n_attributes)

    def __show_check_changed(self):
        if self._violin_plot is not None:
            self._violin_plot.show_legend(self.show_legend)

    @Inputs.data
    @check_sql_input
    def set_data(self, data: Optional[Table]):
        self.data = data
        summary = len(data) if data else self.info.NoInput
        details = format_summary_details(data) if data else ""
        self.info.set_input_summary(summary, details)

    @Inputs.model
    def set_model(self, model: Optional[Model]):
        self.closeContext()
        self.model = model
        self.setup_controls()
        self.openContext(self.model.domain.class_var if self.model else None)

    def setup_controls(self):
        self._target_combo.clear()
        self._target_combo.setEnabled(True)
        if self.model is not None:
            if self.model.domain.has_discrete_class:
                self._target_combo.addItems(self.model.domain.class_var.values)
                self.target_index = 0
            elif self.model.domain.has_continuous_class:
                self.target_index = -1
                self._target_combo.setEnabled(False)
            else:
                raise NotImplementedError

    def handleNewSignals(self):
        self.clear()
        self.start(run, self.data, self.model)

    def clear(self):
        self.__results = None
        self.cancel()
        self.clear_selection()
        self.clear_scene()
        self.clear_messages()

    def clear_selection(self):
        if self.selection:
            self.selection = ()
            self.commit()

    def clear_scene(self):
        self.scene.clear()
        self.scene.setSceneRect(QRectF())
        self.view.setSceneRect(QRectF())
        self.view.setHeaderSceneRect(QRectF())
        self.view.setFooterSceneRect(QRectF())
        self._violin_plot = None

    def commit(self):
        if not self.selection or not self.selection[1]:
            self.info.set_output_summary(self.info.NoOutput)
            self.Outputs.selected_data.send(None)
        else:
            data = self.data[self.selection[1]]
            detail = format_summary_details(data)
            self.info.set_output_summary(len(data), detail)
            self.Outputs.selected_data.send(data)

    def update_scene(self):
        self.clear_scene()
        scores = None
        if self.__results is not None:
            assert isinstance(self.__results.x, list)
            x = self.__results.x[self.target_index]
            scores_x = np.mean(np.abs(x), axis=0)
            indices = np.argsort(scores_x)[::-1]
            colors = self.__results.colors
            names = [self.__results.names[i] for i in indices]
            if x.shape[1]:
                self.setup_plot(x[:, indices], colors[:, indices], names)
            scores = self.create_scores_table(scores_x, self.__results.names)
        self.Outputs.scores.send(scores)

    def setup_plot(self, x: np.ndarray, colors: np.ndarray, names: List[str]):
        width = int(self.view.viewport().rect().width())
        self._violin_plot = ViolinPlot()
        self._violin_plot.set_data(x, colors, names, self.n_attributes, width)
        self._violin_plot.show_legend(self.show_legend)
        self._violin_plot.selection_cleared.connect(self.clear_selection)
        self._violin_plot.selection_changed.connect(self.update_selection)
        self._violin_plot.layout().activate()
        self._violin_plot.geometryChanged.connect(self.update_scene_rect)
        self.scene.addItem(self._violin_plot)
        self.scene.mouse_clicked.connect(self._violin_plot.deselect)
        self.update_scene_rect()

    def update_plot(self):
        if self._violin_plot is not None:
            width = int(self.view.viewport().rect().width())
            self._violin_plot.rescale(width)

    def update_selection(self, min_val: float, max_val: float, attr_name: str):
        assert self.__results is not None
        x = self.__results.x[self.target_index]
        column = self.__results.names.index(attr_name)
        mask = self.__results.mask.copy()
        mask[self.__results.mask] = np.logical_and(x[:, column] <= max_val,
                                                   x[:, column] >= min_val)
        if not self.selection and not any(mask):
            return
        self.selection = (attr_name, list(np.flatnonzero(mask)))
        self.commit()

    def update_scene_rect(self):
        def extend_horizontal(rect):
            rect = QRectF(rect)
            rect.setLeft(geom.left())
            rect.setRight(geom.right())
            return rect

        geom = self._violin_plot.geometry()
        self.scene.setSceneRect(geom)
        self.view.setSceneRect(geom)

        footer_geom = self._violin_plot.bottom_axis.geometry()
        footer = extend_horizontal(footer_geom.adjusted(0, -3, 0, 10))
        self.view.setFooterSceneRect(footer)

    @staticmethod
    def create_scores_table(scores: np.ndarray, names: List[str]):
        domain = Domain([ContinuousVariable("Score")],
                        metas=[StringVariable("Feature")])
        scores_table = Table(domain, scores[:, None],
                             metas=np.array(names)[:, None])
        scores_table.name = "Feature Scores"
        return scores_table

    def on_partial_result(self, _):
        pass

    def on_done(self, results: Optional[Results]):
        self.__results = results
        if self.data and results is not None and not all(results.mask):
            self.Information.data_sampled()
        self.update_scene()
        self.select_pending()

    def select_pending(self):
        if not self.__pending_selection or not self.__pending_selection[1] \
                or self.__results is None:
            return

        attr_name, row_indices = self.__pending_selection
        names = self.__results.names
        if not names or attr_name not in names:
            return
        col_index = names.index(attr_name)
        mask = np.zeros(self.__results.mask.shape, dtype=bool)
        mask[row_indices] = True
        mask = np.logical_and(self.__results.mask, mask)
        row_indices = np.flatnonzero(mask[self.__results.mask])
        column = self.__results.x[self.target_index][row_indices, col_index]
        x1, x2 = np.min(column), np.max(column)
        self._violin_plot.select_from_settings(x1, x2, attr_name)
        self.__pending_selection = ()
        self.unconditional_commit()

    def on_exception(self, ex: Exception):
        if isinstance(ex, DomainTransformationError):
            self.Error.domain_transform_err(ex)
        else:
            self.Error.unknown_err(ex)

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    def sizeHint(self):
        sh = self.controlArea.sizeHint()
        return sh.expandedTo(QSize(800, 520))

    def send_report(self):
        if not self.data or not self.model:
            return
        items = {"Target class": "None"}
        if self.model.domain.has_discrete_class:
            class_var = self.model.domain.class_var
            items["Target class"] = class_var.values[self.target_index]
        self.report_items(items)
        self.report_plot()


if __name__ == "__main__":  # pragma: no cover
    from Orange.classification import RandomForestLearner
    from Orange.regression import RandomForestRegressionLearner

    table = Table("heart_disease")
    if table.domain.has_continuous_class:
        rf_model = RandomForestRegressionLearner(random_state=42)(table)
    else:
        rf_model = RandomForestLearner(random_state=42)(table)
    WidgetPreview(OWExplainModel).run(set_data=table, set_model=rf_model)
