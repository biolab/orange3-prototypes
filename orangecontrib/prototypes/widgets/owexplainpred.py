from typing import Tuple, Optional, List, Type, Callable
from types import SimpleNamespace

import numpy as np

from AnyQt.QtCore import Qt, QRectF, QSizeF, QSize, QPointF
from AnyQt.QtGui import QColor, QPen, QBrush, QPainter, QPainterPath, QPolygonF
from AnyQt.QtWidgets import QGraphicsItemGroup, QGraphicsLineItem, \
    QGraphicsScene, QGraphicsWidget, QGraphicsLinearLayout, QGraphicsView, \
    QGraphicsSimpleTextItem, QGraphicsRectItem, QGraphicsPathItem, \
    QGraphicsPolygonItem

import pyqtgraph as pg

from Orange.base import Model
from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.data.table import DomainTransformationError
from Orange.widgets import gui
from Orange.widgets.settings import Setting, ContextSetting, \
    ClassValuesContextHandler
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.state_summary import format_multiple_summaries
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output, OWWidget, Msg

from orangecontrib.prototypes.explanation.explainer import RGB_LOW, RGB_HIGH, \
    explain_predictions, prepare_force_plot_data


class RunnerResults(SimpleNamespace):
    values = None  # type: Optional[List[np.ndarray]]
    predictions = None  # type: Optional[np.ndarray]
    transformed_data = None  # type: Optional[Table]
    base_value = None  # type: Optional[float]


def run(data: Table, background_data: Table, model: Model, state: TaskState) \
        -> RunnerResults:
    if not data or not background_data or not model:
        return None

    def callback(i: float, status=""):
        state.set_progress_value(i * 100)
        if status:
            state.set_status(status)
        if state.is_interruption_requested():
            raise Exception

    values, pred, data, base_value = explain_predictions(
        model, data, background_data, callback)
    return RunnerResults(values=values, predictions=pred,
                         transformed_data=data, base_value=base_value)


def _collides(ind_y1: float, ind_y2: float, y1: float, y2: float, d=4) -> bool:
    return ind_y1 - d <= y2 <= ind_y2 + d or ind_y1 - d <= y1 <= ind_y2 + d


def _str(x: float, n_dec=2) -> str:
    _max = 10000
    if not np.isnan(x) and x == int(x) and abs(x) < _max:
        return str(int(x))
    elif 10 / _max < abs(x) < _max:
        return str(round(x, max(n_dec, 1)))
    else:
        return "%.*e" % (max(n_dec, 1), x)


class PartItem(QGraphicsPathItem):
    COLOR = NotImplemented
    TIP_LEN = 13

    def __init__(self, value: float, label: Tuple[str, float],
                 norm_value: float):
        super().__init__()
        color = QColor(*self.light_rgb)
        self.value = value
        self.norm_value = norm_value
        self.setPath(self.get_path())
        pen = QPen(color)
        pen.setWidth(2)
        self.setPen(pen)

        value = np.abs(value)
        self.value_item = item = QGraphicsSimpleTextItem(_str(value))
        item.setToolTip(_str(value, 3))
        font = item.font()
        font.setPixelSize(11)
        item.setFont(font)
        width = item.boundingRect().width()
        item.setX(StripeItem.WIDTH / 2 - width / 2)
        item.setPen(color)
        item.setBrush(color)

        self.label_item = QGraphicsSimpleTextItem(
            f"{label[0]} = {_str(label[1])}")
        self.label_item.setToolTip(f"{label[0]} = {_str(label[1], 3)}")
        self.label_item.setX(StripeItem.WIDTH + StripePlot.SPACING)

    @property
    def light_rgb(self) -> List[int]:
        rgb = np.array(self.COLOR)
        return list((rgb + (255 - rgb) * 0.7).astype(int))

    @property
    def value_height(self) -> float:
        return self.value_item.boundingRect().height()

    @property
    def label_height(self) -> float:
        return self.label_item.boundingRect().height()

    def get_path(self) -> QPainterPath:
        raise NotImplementedError


class HighPartItem(PartItem):
    COLOR = RGB_HIGH

    def get_path(self) -> QPainterPath:
        path = QPainterPath()
        path.lineTo(StripeItem.WIDTH / 2, -self.TIP_LEN)
        path.lineTo(StripeItem.WIDTH, 0)
        return path


class LowPartItem(PartItem):
    COLOR = RGB_LOW

    def get_path(self) -> QPainterPath:
        path = QPainterPath()
        path.lineTo(StripeItem.WIDTH / 2, self.TIP_LEN)
        path.lineTo(StripeItem.WIDTH, 0)
        return path


class CoverItem(QGraphicsPolygonItem):
    COLOR = QColor(Qt.white)
    ADJUST = 2

    def __init__(self):
        super().__init__()
        self.setPolygon(self._get_polygon())
        self.setPen(self.COLOR)
        self.setBrush(self.COLOR)
        self.setPos(0, 0)

    def _get_polygon(self) -> QPolygonF:
        return QPolygonF([QPointF(x, y) for x, y in self.get_points()])

    def get_points(self) -> List[Tuple[float]]:
        raise NotImplementedError


class LowCoverItem(CoverItem):
    def get_points(self) -> List[Tuple[int, int]]:
        width = StripeItem.WIDTH
        return [(0, 0), (0, -self.ADJUST), (width, -self.ADJUST),
                (width, 0), (int(width / 2), PartItem.TIP_LEN), (0, 0)]


class HighCoverItem(CoverItem):
    def get_points(self) -> List[Tuple[int, int]]:
        width = StripeItem.WIDTH
        return [(0, 0), (0, self.ADJUST), (width, self.ADJUST),
                (width, 0), (int(width / 2), -PartItem.TIP_LEN), (0, 0)]


class IndicatorItem(QGraphicsSimpleTextItem):
    COLOR = QColor(*(100, 100, 100))
    PADDING = 2
    MARGIN = 10

    def __init__(self, tooltip_prefix):
        super().__init__()
        self.__tooltip_prefix = tooltip_prefix
        self.setPen(QPen(Qt.NoPen))
        self.setBrush(QColor(Qt.white))

    def set_text(self, value: float, range_: float):
        try:
            n_dec = max(int(np.ceil(-np.log10(range_/10))), 2)
        except:
            n_dec = 2
        self.setText(_str(value, n_dec))
        self.setToolTip(self.__tooltip_prefix.format(_str(value, n_dec + 1)))
        width = self.boundingRect().width()
        self.setX(-width - self.MARGIN - self.PADDING - StripePlot.SPACING)

    def paint(self, painter: QPainter, option, widget):
        painter.setBrush(self.COLOR)
        painter.setPen(self.COLOR)
        painter.drawRect(self.boundingRect().adjusted(
            -self.PADDING, -self.PADDING, self.PADDING, self.PADDING))
        super().paint(painter, option, widget)


class PlotData(SimpleNamespace):
    high_values = None  # type: Optional[List[float]]
    low_values = None  # type: Optional[List[float]]
    high_labels = None  # type: Optional[List[Tuple[str, float]]]
    low_labels = None  # type: Optional[List[Tuple[str, float]]]
    value_range = None  # type: Optional[Tuple[float]]
    model_output = None  # type: Optional[float]
    base_value = None  # type: Optional[float]


class StripeItem(QGraphicsWidget):
    WIDTH = 70

    def __init__(self, parent):
        super().__init__(parent)
        self.__range = None  # type: Tuple[float]
        self.__value_range = None  # type: Tuple[float]
        self.__model_output = None  # type: float
        self.__base_value = None  # type: float

        self.__group = QGraphicsItemGroup(self)
        low_color, high_color = QColor(*RGB_LOW), QColor(*RGB_HIGH)

        self.__low_item = QGraphicsRectItem()
        self.__low_item.setPen(QPen(low_color))
        self.__low_item.setBrush(QBrush(low_color))

        self.__high_item = QGraphicsRectItem()
        self.__high_item.setPen(QPen(high_color))
        self.__high_item.setBrush(QBrush(high_color))

        self.__low_cover_item = LowCoverItem()
        self.__high_cover_item = HighCoverItem()

        pen = QPen(IndicatorItem.COLOR)
        pen.setStyle(Qt.DashLine)
        pen.setWidth(1)
        self.__model_output_line = QGraphicsLineItem()
        self.__model_output_line.setPen(pen)
        self.__base_value_line = QGraphicsLineItem()
        self.__base_value_line.setPen(pen)

        self.__model_output_ind = IndicatorItem("Model prediction: {}")
        self.__base_value_ind = IndicatorItem("Base value: {}\nThe average "
                                              "prediction for selected class.")

        self.__group.addToGroup(self.__low_item)
        self.__group.addToGroup(self.__high_item)
        self.__group.addToGroup(self.__low_cover_item)
        self.__group.addToGroup(self.__high_cover_item)
        self.__group.addToGroup(self.__model_output_line)
        self.__group.addToGroup(self.__base_value_line)
        self.__group.addToGroup(self.__model_output_ind)
        self.__group.addToGroup(self.__base_value_ind)

        self.__low_parts = []  # type: List[LowPartItem]
        self.__high_parts = []  # type: List[HighPartItem]

    @property
    def total_width(self):
        widths = [part.label_item.boundingRect().width()
                  for part in self.__low_parts + self.__high_parts] + [0]
        return self.WIDTH + StripePlot.SPACING + max(widths)

    @property
    def model_output_ind(self) -> IndicatorItem:
        return self.__model_output_ind

    @property
    def base_value_ind(self) -> IndicatorItem:
        return self.__base_value_ind

    def set_data(self, data: PlotData, y_range: Tuple[float, float],
                 height: float):
        self.__range = y_range
        self.__value_range = data.value_range
        self.__model_output = data.model_output
        self.__base_value = data.base_value

        r = self.__range[1] - self.__range[0]
        self.__model_output_ind.set_text(self.__model_output, r)
        self.__base_value_ind.set_text(self.__base_value, r)

        for value, label in zip(data.low_values, data.low_labels):
            self.__add_part(value, label, value / sum(data.low_values),
                            self.__low_parts, LowPartItem)
        for value, label in zip(data.high_values, data.high_labels):
            self.__add_part(value, label, value / sum(data.high_values),
                            self.__high_parts, HighPartItem)
        if self.__low_parts:
            self.__low_parts[-1].setVisible(False)
        else:
            self.__low_item.setVisible(False)
            self.__low_cover_item.setVisible(False)
        if self.__high_parts:
            self.__high_parts[-1].setVisible(False)
        else:
            self.__high_item.setVisible(False)
            self.__high_cover_item.setVisible(False)

        self.set_z_values()
        self.set_height(height)

    def __add_part(self, value: float, label: Tuple[str, str], norm_val: float,
                   list_: List[PartItem], cls_: Type[PartItem]):
        item = cls_(value, label, norm_val)
        list_.append(item)
        self.__group.addToGroup(item)
        self.__group.addToGroup(item.value_item)
        self.__group.addToGroup(item.label_item)

    def set_z_values(self):
        if len(self.__high_parts) < len(self.__low_parts):
            self.__high_cover_item.setZValue(-2)
            self.__high_item.setZValue(-3)
            for i, item in enumerate(self.__high_parts):
                item.setZValue(-1)
        else:
            self.__low_cover_item.setZValue(-2)
            self.__low_item.setZValue(-3)
            for i, item in enumerate(self.__low_parts):
                item.setZValue(-1)

    def set_height(self, height: float):
        if self.__range[1] == self.__range[0]:
            return
        height = height / (self.__range[1] - self.__range[0])

        y_top = height * (self.__range[1] - self.__value_range[1])
        h_top = height * (self.__value_range[1] - self.__model_output)
        y_bot = height * (self.__range[1] - self.__model_output)
        h_bot = height * (self.__model_output - self.__value_range[0])

        self.__low_item.setRect(QRectF(0, y_top, self.WIDTH, h_top))
        self.__high_item.setRect(QRectF(0, y_bot, self.WIDTH, h_bot))
        self.__low_cover_item.setY(y_top)
        self.__high_cover_item.setY(y_bot + h_bot)

        self._set_indicators_pos(height)

        def adjust_y_text_low(i):
            # adjust label y according to TIP_LEN
            # adjust for 0.8 * TIP_LEN, because label is not 1 pixel wide
            k = 0.4 if i == len(self.__low_parts) - 1 else 0.8
            return PartItem.TIP_LEN * k

        def adjust_y_text_high(i):
            k = 0.4 if i == 0 else 0.8
            return -PartItem.TIP_LEN * k

        self._set_parts_pos(height, y_top, h_top / height,
                            self.__low_parts, adjust_y_text_low)
        self._set_parts_pos(height, y_bot, h_bot / height,
                            self.__high_parts, adjust_y_text_high)

    def _set_parts_pos(self, height: float, y: float, diff: float,
                       parts: List[PartItem], adjust_y: Callable):
        for i, item in enumerate(parts):
            y_delta = height * item.norm_value * diff

            y_text = y + y_delta / 2 - item.value_height / 2
            visible = y_delta > item.value_height + 8

            y_test_adj = y_text + adjust_y(i)
            y_mid = height * (self.__range[1] - self.__model_output)
            collides = _collides(y_mid, y_mid, y_test_adj, y_test_adj +
                                 item.value_item.boundingRect().height())
            item.value_item.setVisible(visible and not collides)
            item.value_item.setY(y_test_adj)

            item.label_item.setVisible(visible)
            item.label_item.setY(y_text)

            y = y + y_delta
            item.setY(y)

    def _set_indicators_pos(self, height: float):
        mo_y = height * (self.__range[1] - self.__model_output)
        mo_h = self.__model_output_ind.boundingRect().height()
        self.__model_output_ind.setY(mo_y - mo_h / 2)
        self.__model_output_line.setLine(
            0, mo_y, -StripePlot.SPACING - IndicatorItem.MARGIN, mo_y)

        bv_y = height * (self.__range[1] - self.__base_value)
        bv_h = self.__base_value_ind.boundingRect().height()
        self.__base_value_ind.setY(bv_y - bv_h / 2)
        self.__base_value_line.setLine(
            -StripePlot.SPACING, bv_y,
            -StripePlot.SPACING - IndicatorItem.MARGIN, bv_y)
        collides = _collides(mo_y, mo_y + mo_h, bv_y, bv_y + bv_h, d=6)
        self.__base_value_ind.setVisible(not collides)


class AxisItem(pg.AxisItem):
    def __init__(self, indicators: List[IndicatorItem], **kwargs):
        super().__init__(**kwargs)
        self.__plot_indicators = indicators

    def drawPicture(self, p: QPainter, axis_spec: Tuple, tick_specs: List,
                    text_specs: List):
        new_text_specs = []
        for rect, flags, text in text_specs:
            if self.__collides_with_indicator(rect):
                continue
            new_text_specs.append((rect, flags, text))
        super().drawPicture(p, axis_spec, tick_specs, new_text_specs)

    def __collides_with_indicator(self, rect: QRectF) -> bool:
        y1 = rect.y()
        y2 = y1 + rect.height()
        for indicator in self.__plot_indicators:
            if not indicator.isVisible():
                continue
            ind_y1 = indicator.y()
            ind_y2 = ind_y1 + indicator.boundingRect().height()
            if _collides(ind_y1, ind_y2, y1, y2):
                return True
        return False


class StripePlot(QGraphicsWidget):
    HEIGHT = 400
    SPACING = 20
    HMARGIN = 30
    VMARGIN = 20

    def __init__(self):
        super().__init__()
        self.__height = None  # type: int
        self.__range = None  # type: Tuple[float, float]

        self.__layout = QGraphicsLinearLayout()
        self.__layout.setOrientation(Qt.Horizontal)
        self.__layout.setSpacing(self.SPACING)
        self.__layout.setContentsMargins(self.HMARGIN, self.VMARGIN,
                                         self.HMARGIN, self.VMARGIN)
        self.setLayout(self.__layout)

        self.__stripe_item = StripeItem(self)
        self.__left_axis = AxisItem([self.__stripe_item.model_output_ind,
                                     self.__stripe_item.base_value_ind],
                                    parent=self, orientation="left",
                                    maxTickLength=7, pen=QPen(Qt.black))

        self.__layout.addItem(self.__left_axis)
        self.__layout.addItem(self.__stripe_item)

    @property
    def height(self) -> float:
        return self.HEIGHT + 10 * self.__height

    def set_data(self, data: PlotData, height: float):
        diff = (data.value_range[1] - data.value_range[0]) * 0.01
        self.__range = (data.value_range[0] - diff, data.value_range[1] + diff)
        self.__left_axis.setRange(*self.__range)
        self.__height = height
        self.__stripe_item.set_data(data, self.__range, self.height)

    def set_height(self, height: float):
        self.__height = height
        self.__stripe_item.set_height(self.height)
        self.updateGeometry()

    def sizeHint(self, *_) -> QSizeF:
        return QSizeF(self.__left_axis.boundingRect().width() +
                      self.__stripe_item.total_width + self.HMARGIN * 2,
                      self.height + self.VMARGIN * 2)


class OWExplainPrediction(OWWidget, ConcurrentWidgetMixin):
    name = "Explain Prediction"
    description = "Prediction explanation widget."
    icon = "icons/ExplainPred.svg"
    priority = 110

    class Inputs:
        model = Input("Model", Model)
        background_data = Input("Background Data", Table)
        data = Input("Data", Table)

    class Outputs:
        scores = Output("Scores", Table)

    class Error(OWWidget.Error):
        domain_transform_err = Msg("{}")
        unknown_err = Msg("{}")

    class Information(OWWidget.Information):
        multiple_instances = Msg("Explaining prediction for the first "
                                 "instance in 'Data'.")

    settingsHandler = ClassValuesContextHandler()
    target_index = ContextSetting(0)
    stripe_len = Setting(10)

    graph_name = "scene"

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.__results = None  # type: Optional[Results]
        self.model = None  # type: Optional[Model]
        self.background_data = None  # type: Optional[Table]
        self.data = None  # type: Optional[Table]
        self._stripe_plot = None  # type: Optional[StripePlot]
        self.mo_info = ""
        self.bv_info = ""
        self.setup_gui()

    def setup_gui(self):
        self._add_controls()
        self._add_plot()
        self.info.set_input_summary(self.info.NoInput)

    def _add_plot(self):
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.mainArea.layout().addWidget(self.view)

    def _add_controls(self):
        box = gui.vBox(self.controlArea, "Target class")
        self._target_combo = gui.comboBox(box, self, "target_index",
                                          callback=self.__target_combo_changed,
                                          contentsLength=12)

        box = gui.hBox(self.controlArea, "Zoom")
        gui.hSlider(box, self, "stripe_len", None, minValue=1, maxValue=500,
                    createLabel=False, callback=self.__size_slider_changed)

        gui.rubber(self.controlArea)

        box = gui.vBox(self.controlArea, "Prediction info")
        gui.label(box, self, "%(mo_info)s")  # type: QLabel
        bv_label = gui.label(box, self, "%(bv_info)s")  # type: QLabel
        bv_label.setToolTip("The average prediction for selected class.")

    def __target_combo_changed(self):
        self.update_scene()

    def __size_slider_changed(self):
        if self._stripe_plot is not None:
            self._stripe_plot.set_height(self.stripe_len)

    @Inputs.data
    @check_sql_input
    def set_data(self, data: Optional[Table]):
        self.data = data

    @Inputs.background_data
    @check_sql_input
    def set_background_data(self, data: Optional[Table]):
        self.background_data = data

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
        self.check_inputs()
        data = self.data and self.data[:1]
        self.start(run, data, self.background_data, self.model)

    def clear(self):
        self.mo_info = ""
        self.bv_info = ""
        self.__results = None
        self.cancel()
        self.clear_scene()
        self.clear_messages()

    def check_inputs(self):
        if self.data and len(self.data) > 1:
            self.Information.multiple_instances()

        summary, details, kwargs = self.info.NoInput, "", {}
        if self.data or self.background_data:
            n_data = len(self.data) if self.data else 0
            n_background_data = len(self.background_data) \
                if self.background_data else 0
            summary = f"{self.info.format_number(n_background_data)}, " \
                      f"{self.info.format_number(n_data)}"
            kwargs = {"format": Qt.RichText}
            details = format_multiple_summaries([
                ("Background data", self.background_data),
                ("Data", self.data)
            ])
        self.info.set_input_summary(summary, details, **kwargs)

    def clear_scene(self):
        self.scene.clear()
        self.scene.setSceneRect(QRectF())
        self.view.setSceneRect(QRectF())
        self._stripe_plot = None

    def update_scene(self):
        self.clear_scene()
        self.mo_info = ""
        self.bv_info = ""
        scores = None
        if self.__results is not None:
            data = self.__results.transformed_data
            pred = self.__results.predictions
            base = self.__results.base_value
            values, _, labels, ranges = prepare_force_plot_data(
                self.__results.values, data, pred, self.target_index)

            index = 0
            HIGH, LOW = 0, 1
            plot_data = PlotData(high_values=values[index][HIGH],
                                 low_values=values[index][LOW][::-1],
                                 high_labels=labels[index][HIGH],
                                 low_labels=labels[index][LOW][::-1],
                                 value_range=ranges[index],
                                 model_output=pred[index][self.target_index],
                                 base_value=base[self.target_index])
            self.setup_plot(plot_data)

            self.mo_info = f"Model prediction: {_str(plot_data.model_output)}"
            self.bv_info = f"Base value: {_str(plot_data.base_value)}"

            assert isinstance(self.__results.values, list)
            scores = self.__results.values[self.target_index][0, :]
            names = [a.name for a in data.domain.attributes]
            scores = self.create_scores_table(scores, names)
        self.Outputs.scores.send(scores)

    def setup_plot(self, plot_data: PlotData):
        self._stripe_plot = StripePlot()
        self._stripe_plot.set_data(plot_data, self.stripe_len)
        self._stripe_plot.layout().activate()
        self._stripe_plot.geometryChanged.connect(self.update_scene_rect)
        self.scene.addItem(self._stripe_plot)
        self.update_scene_rect()

    def update_scene_rect(self):
        geom = self._stripe_plot.geometry()
        self.scene.setSceneRect(geom)
        self.view.setSceneRect(geom)

    @staticmethod
    def create_scores_table(scores: np.ndarray, names: List[str]) -> Table:
        domain = Domain([ContinuousVariable("Score")],
                        metas=[StringVariable("Feature")])
        scores_table = Table(domain, scores[:, None],
                             metas=np.array(names)[:, None])
        scores_table.name = "Feature Scores"
        return scores_table

    def on_partial_result(self, _):
        pass

    def on_done(self, results: Optional[RunnerResults]):
        self.__results = results
        self.update_scene()

    def on_exception(self, ex: Exception):
        if isinstance(ex, DomainTransformationError):
            self.Error.domain_transform_err(ex)
        else:
            self.Error.unknown_err(ex)

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    def sizeHint(self) -> QSizeF:
        sh = self.controlArea.sizeHint()
        return sh.expandedTo(QSize(700, 700))

    def send_report(self):
        if not self.data or not self.background_data or not self.model:
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
    WidgetPreview(OWExplainPrediction).run(set_background_data=table,
                                           set_data=table[:1],
                                           set_model=rf_model)
