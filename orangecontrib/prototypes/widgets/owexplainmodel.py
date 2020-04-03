from typing import Tuple, Optional, List
from types import SimpleNamespace

import numpy as np

from AnyQt.QtCore import Qt, QRectF, QSizeF, QSize
from AnyQt.QtGui import QColor, QPen, QBrush, QPainter, QLinearGradient
from AnyQt.QtWidgets import QGraphicsItemGroup, QGraphicsLineItem, \
    QGraphicsScene, QGraphicsWidget, QGraphicsGridLayout, \
    QGraphicsEllipseItem, QGraphicsSimpleTextItem, QSizePolicy, \
    QGraphicsRectItem

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
    get_shap_values_and_colors, RGB_LOW, RGB_HIGH


class Results(SimpleNamespace):
    x = None  # type: Optional[List[np.ndarray]]
    colors = None  # type: Optional[List[np.ndarray]]
    names = None  # type: Optional[List[str]]


def run(data: Table, model: Model, state: TaskState) -> Results:
    if not data or not model:
        return None

    def callback(i: float, status=""):
        state.set_progress_value(i * 100)
        if status:
            state.set_status(status)
        if state.is_interruption_requested():
            raise Exception

    x, names, colors = get_shap_values_and_colors(model, data, callback)
    return Results(x=x, colors=colors, names=names)


class Legend(QGraphicsWidget):
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
        return QSizeF(30, ViolinItem.HEIGHT)


class ViolinItem(QGraphicsWidget):
    WIDTH, HEIGHT = 400, 40
    POINT_R = 4

    def __init__(self, parent):
        super().__init__(parent)
        self.__group = None  # type: Optional[QGraphicsItemGroup]

    def set_data(self, x_data: np.ndarray, color_data: np.ndarray):
        def put_point(_x, _y, _c):
            item = QGraphicsEllipseItem(_x, _y, self.POINT_R, self.POINT_R)
            color = QColor(*_c)
            item.setPen(QPen(color))
            item.setBrush(QBrush(color))
            self.__group.addToGroup(item)

        self.__group = QGraphicsItemGroup(self)
        x_data = np.round(x_data, 3)[~np.isnan(x_data)]
        if len(x_data) == 0:
            return

        x_data_unique, counts = np.unique(x_data, return_counts=True)
        min_count, max_count = np.min(counts), np.max(counts)
        dist = (counts - min_count) / (max_count - min_count)
        if min_count == max_count:
            dist[:] = 1
        dist = dist ** 0.7
        indices = np.argsort(counts)
        for x, d in zip(x_data_unique[indices], dist[indices]):
            colors = color_data[x_data == np.round(x, 3)]
            x = x * self.WIDTH + self.WIDTH / 2 - self.POINT_R / 2
            y = self.HEIGHT / 2 - self.POINT_R / 2

            np.random.seed(0)
            colors = list(colors[np.random.choice(len(colors), 11)])
            put_point(x, y, colors.pop())  # y = 0
            if d > 0:
                offset = d * 10
                put_point(x, y + offset, colors.pop())  # y = (0, 10]
                put_point(x, y - offset, colors.pop())  # y = (-10, 0]
                for i in range(2, int(offset), 2):
                    put_point(x, y + i, colors.pop())  # y = [2, 8]
                    put_point(x, y - i, colors.pop())  # y = [-8, -2]

    def sizeHint(self, *_):
        return QSizeF(self.WIDTH, self.HEIGHT)


class ViolinPlot(QGraphicsWidget):
    LABEL_COLUMN, VIOLIN_COLUMN, LEGEND_COLUMN = range(3)
    MAX_N_ITEMS = 100
    MAX_ATTR_LEN = 20

    def __init__(self):
        super().__init__()
        self.__range = None  # type: Optional[Tuple[float, float]]
        self.__violin_items = []  # type: List[ViolinItem]
        self.__bottom_axis = pg.AxisItem(parent=self, orientation="bottom",
                                         maxTickLength=7, pen=QPen(Qt.black))
        self.__vertical_line = QGraphicsLineItem(self.__bottom_axis)
        self.__vertical_line.setPen(QPen(Qt.gray))
        self.__legend = Legend(self)

        self.__layout = QGraphicsGridLayout()
        self.__layout.addItem(self.__legend, 0, ViolinPlot.LEGEND_COLUMN)
        self.__layout.setVerticalSpacing(0)
        self.setLayout(self.__layout)

    @property
    def bottom_axis(self):
        return self.__bottom_axis

    def set_data(self, x: np.ndarray, colors: np.ndarray,
                 names: List[str], n_attrs: float):
        abs_max = np.max(np.abs(x)) * 1.05
        self.__range = (-abs_max, abs_max)
        self._set_violin_items(x, colors)
        self._set_labels(names)
        self._set_bottom_axis()
        self.set_n_visible(n_attrs)

    def set_n_visible(self, n: int):
        for i in range(len(self.__violin_items)):
            violin_item = self.__layout.itemAt(i, ViolinPlot.VIOLIN_COLUMN)
            violin_item.setVisible(i < n)
            text_item = self.__layout.itemAt(i, ViolinPlot.LABEL_COLUMN).item
            text_item.setVisible(i < n)

        x = ViolinItem.WIDTH / 2
        n = min(n, len(self.__violin_items))
        self.__vertical_line.setLine(x, 0, x, -ViolinItem.HEIGHT * n)

    def _set_violin_items(self, x: np.ndarray, colors: np.ndarray):
        x = x / self.__range[1] * 0.5  # scale data to [-0.5, 0.5]
        for i in range(x.shape[1]):
            item = ViolinItem(self)
            item.set_data(x[:, i], colors[:, i])
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


class OWExplainModel(OWWidget, ConcurrentWidgetMixin):
    name = "Explain Model"
    description = "Model explanation widget."
    icon = "icons/ExplainModel.svg"
    priority = 100

    class Inputs:
        data = Input("Data", Table, default=True)
        model = Input("Model", Model)

    class Outputs:
        scores = Output("Scores", Table)

    class Error(OWWidget.Error):
        domain_transform_err = Msg("{}")
        unknown_err = Msg("{}")

    class Info(OWWidget.Information):
        data_sampled = Msg("Data has been sampled.")

    settingsHandler = ClassValuesContextHandler()
    target_index = ContextSetting(0)
    n_attributes = Setting(10)

    graph_name = "scene"

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.__results = None  # type: Optional[Results]
        self.data = None  # type: Optional[Table]
        self.model = None  # type: Optional[Model]
        self._violin_plot = None  # type: Optional[ViolinPlot]
        self.setup_gui()

    def setup_gui(self):
        self._add_controls()
        self._add_plot()
        self.info.set_input_summary(self.info.NoInput)

    def _add_plot(self):
        self.scene = QGraphicsScene()
        self.view = StickyGraphicsView(self.scene)
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
        self.n_spin = gui.spin(
            box, self, "n_attributes", 1, ViolinPlot.MAX_N_ITEMS,
            controlWidth=80, callback=self.__n_spin_changed)
        gui.rubber(self.controlArea)

    def __target_combo_changed(self):
        self.update_scene()

    def __n_spin_changed(self):
        if self._violin_plot is not None:
            self._violin_plot.set_n_visible(self.n_attributes)

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
        self.clear_scene()
        self.clear_messages()

    def clear_scene(self):
        self.scene.clear()
        self.scene.setSceneRect(QRectF())
        self.view.setSceneRect(QRectF())
        self.view.setHeaderSceneRect(QRectF())
        self.view.setFooterSceneRect(QRectF())
        self._violin_plot = None

    def update_scene(self):
        self.clear_scene()
        scores = None
        if self.__results is not None:
            x = self.__results.x
            x = x[self.target_index] if isinstance(x, list) else x
            scores_x = np.mean(np.abs(x), axis=0)
            indices = np.argsort(scores_x)[::-1]
            colors = self.__results.colors
            names = [self.__results.names[i] for i in indices]
            self.setup_plot(x[:, indices], colors[:, indices], names)
            scores = self.create_scores_table(scores_x, self.__results.names)
        self.Outputs.scores.send(scores)

    def setup_plot(self, x: np.ndarray, colors: np.ndarray, names: List[str]):
        self._violin_plot = ViolinPlot()
        self._violin_plot.set_data(x, colors, names, self.n_attributes)
        self._violin_plot.layout().activate()
        self._violin_plot.geometryChanged.connect(self.update_scene_rect)
        self.scene.addItem(self._violin_plot)
        self.update_scene_rect()

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
        footer = extend_horizontal(footer_geom.adjusted(0, -3, 0, 0))
        self.view.setFooterSceneRect(footer)

    @staticmethod
    def create_scores_table(scores: np.ndarray, names: List[str]):
        domain = Domain([ContinuousVariable("Score (mean SHAP value)")],
                        metas=[StringVariable("Feature")])
        scores_table = Table(domain, scores[:, None],
                             metas=np.array(names)[:, None])
        scores_table.name = "Feature Scores"
        return scores_table

    def on_partial_result(self, _):
        pass

    def on_done(self, results: Optional[Results]):
        self.__results = results
        if self.data and results is not None and results.x is not None \
                and len(self.data) != len(results.x[0]):
            self.Info.data_sampled()
        self.update_scene()

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
        return sh.expandedTo(QSize(900, 500))

    def send_report(self):
        if not self.data or not self.model:
            return
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
