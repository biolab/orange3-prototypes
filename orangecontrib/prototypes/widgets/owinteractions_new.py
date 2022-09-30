import copy
from itertools import chain
from threading import Lock, Timer
from typing import Callable, Optional, Iterable
import numpy as np

from AnyQt.QtGui import QColor, QPainter, QPen
from AnyQt.QtCore import QModelIndex, Qt, QLineF
from AnyQt.QtWidgets import QTableView, QHeaderView, \
    QStyleOptionViewItem, QApplication, QStyle

from Orange.data import Table, Variable
from Orange.preprocess import Discretize, Remove
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, AttributeList, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.settings import Setting, ContextSetting, DomainContextHandler

from orangecontrib.prototypes.ranktablemodel import RankModel
from orangecontrib.prototypes.interactions import InteractionScorer


class ModelQueue:
    """
    Another queueing object, similar to ``queue.Queue``.
    The main difference is that ``get()`` returns all its
    contents at the same time, instead of one by one.
    """
    def __init__(self):
        self.lock = Lock()
        self.model = []
        self.state = None

    def put(self, row, state):
        with self.lock:
            self.model.append(row)
            self.state = state

    def get(self):
        with self.lock:
            model, self.model = self.model, []
            state, self.state = self.state, None
        return model, state


def run(compute_score: Callable, row_for_state: Callable,
        iterate_states: Callable, saved_state: Optional[Iterable],
        progress: int, state_count: int, task: TaskState):
    """
    Replaces ``run_vizrank``, with some minor adjustments.
        - ``ModelQueue`` replaces ``queue.Queue``
        - `row_for_state` parameter added
        - `scores` parameter removed
    """
    task.set_status("Getting combinations...")
    task.set_progress_value(0.1)
    states = iterate_states(saved_state)

    task.set_status("Getting scores...")
    queue = ModelQueue()
    can_set_partial_result = True

    def do_work(st, next_st):
        try:
            score = compute_score(st)
            if score is not None:
                queue.put(row_for_state(score, st), next_st)
        except Exception:
            pass

    def reset_flag():
        nonlocal can_set_partial_result
        can_set_partial_result = True

    state = None
    next_state = next(states)
    try:
        while True:
            if task.is_interruption_requested():
                return queue.get()
            task.set_progress_value(progress * 100 // state_count)
            progress += 1
            state = copy.copy(next_state)
            next_state = copy.copy(next(states))
            do_work(state, next_state)
            # for simple scores (e.g. correlations widget) and many feature
            # combinations, the 'partial_result_ready' signal (emitted by
            # invoking 'task.set_partial_result') was emitted too frequently
            # for a longer period of time and therefore causing the widget
            # being unresponsive
            if can_set_partial_result:
                task.set_partial_result(queue.get())
                can_set_partial_result = False
                Timer(0.05, reset_flag).start()
    except StopIteration:
        do_work(state, None)
        task.set_partial_result(queue.get())
    return queue.get()


class Heuristic:
    RANDOM, INFO_GAIN = 0, 1
    type = {RANDOM: "Random Search",
            INFO_GAIN: "Information Gain Heuristic"}

    def __init__(self, weights, type=None):
        self.n_attributes = len(weights)
        self.attributes = np.arange(self.n_attributes)
        if type == self.RANDOM:
            np.random.shuffle(self.attributes)
        if type == self.INFO_GAIN:
            self.attributes = self.attributes[np.argsort(weights)]

    def generate_states(self):
        # prioritize two mid ranked attributes over highest first
        for s in range(1, self.n_attributes * (self.n_attributes - 1) // 2):
            for i in range(max(s - self.n_attributes + 1, 0), (s + 1) // 2):
                yield self.attributes[i], self.attributes[s - i]

    def get_states(self, initial_state):
        states = self.generate_states()
        if initial_state is not None:
            while next(states) != initial_state:
                pass
            return chain([initial_state], states)
        return states


class InteractionItemDelegate(gui.TableBarItem):
    def paint(self, painter: QPainter, option: QStyleOptionViewItem,
              index: QModelIndex) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = option.widget
        style = QApplication.style() if widget is None else widget.style()
        pen = QPen(QColor("#46befa"), 5, Qt.SolidLine, Qt.RoundCap)
        line = QLineF()
        self.__style = style
        text = opt.text
        opt.text = ""
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)
        textrect = style.subElementRect(
            QStyle.SE_ItemViewItemText, opt, widget)

        interaction = self.cachedData(index, Qt.EditRole)
        # only draw bars for first column
        if index.column() == 0 and interaction is not None:
            rect = option.rect
            pw = self.penWidth
            textoffset = pw + 2
            baseline = rect.bottom() - textoffset / 2
            origin = rect.left() + 3 + pw / 2  # + half pen width for the round line cap
            width = rect.width() - 3 - pw

            def draw_line(start, length):
                line.setLine(origin + start, baseline, origin + start + length, baseline)
                painter.drawLine(line)

            scorer = index.model().scorer
            attr1 = self.cachedData(index.siblingAtColumn(2), Qt.EditRole)
            attr2 = self.cachedData(index.siblingAtColumn(3), Qt.EditRole)
            l_bar = scorer.normalize(scorer.information_gain[int(attr1)])
            r_bar = scorer.normalize(scorer.information_gain[int(attr2)])
            # negative information gains stem from issues in interaction
            # calculation and may cause bars reaching out of intended area
            l_bar, r_bar = width * max(l_bar, 0), width * max(r_bar, 0)
            interaction *= width

            pen.setWidth(pw)
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(pen)
            draw_line(0, l_bar)
            draw_line(l_bar + interaction, r_bar)
            pen.setColor(QColor("#aaf22b") if interaction >= 0 else QColor("#ffaa7f"))
            painter.setPen(pen)
            draw_line(l_bar, interaction)
            painter.restore()
            textrect.adjust(0, 0, 0, -textoffset)

        opt.text = text
        self.drawViewItemText(style, painter, opt, textrect)


class OWInteractions(OWWidget, ConcurrentWidgetMixin):
    name = "Interactions New"
    description = "Compute all pairwise attribute interactions."
    icon = "icons/Interactions.svg"
    category = "Unsupervised"

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        features = Output("Features", AttributeList)

    settingsHandler = DomainContextHandler()
    selection = ContextSetting([])
    filter_text: str
    filter_text = ContextSetting("")
    feature: Variable
    feature = ContextSetting(None)
    heuristic_type: int
    heuristic_type = Setting(0)

    want_main_area = False
    want_control_area = True

    class Information(OWWidget.Information):
        removed_cons_feat = Msg("Constant features have been removed.")

    class Warning(OWWidget.Warning):
        not_enough_vars = Msg("At least two features are needed.")
        not_enough_inst = Msg("At least two instances are needed.")
        no_class_var = Msg("Target feature missing")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.keep_running = True
        self.saved_state = None
        self.progress = 0

        self.data = None  # type: Table
        self.pp_data = None  # type: Table
        self.n_attrs = 0

        self.scorer = None
        self.heuristic = None
        self.feature_index = None

        gui.comboBox(self.controlArea, self, "heuristic_type",
                     items=Heuristic.type.values(),
                     callback=self.on_heuristic_combo_changed,)

        self.feature_model = DomainModel(order=DomainModel.ATTRIBUTES,
                                         separators=False,
                                         placeholder="(All combinations)")
        gui.comboBox(self.controlArea, self, "feature",
                     callback=self.on_feature_combo_changed,
                     model=self.feature_model, searchable=True)

        self.filter = gui.lineEdit(self.controlArea, self, "filter_text",
                                   callback=self.on_filter_changed,
                                   callbackOnType=True)
        self.filter.setPlaceholderText("Filter ...")

        self.model = RankModel()
        self.model.setHorizontalHeaderLabels((
            "Interaction", "Information Gain", "Feature 1", "Feature 2"
        ))
        self.rank_table = view = QTableView(selectionBehavior=QTableView.SelectRows,
                                            selectionMode=QTableView.SingleSelection,
                                            showGrid=False,
                                            editTriggers=gui.TableView.NoEditTriggers)
        view.setSortingEnabled(True)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        view.setItemDelegate(InteractionItemDelegate())
        view.setModel(self.model)
        view.selectionModel().selectionChanged.connect(self.on_selection_changed)
        self.controlArea.layout().addWidget(view)

        self.button = gui.button(self.controlArea, self, "Start", callback=self.toggle)
        self.button.setEnabled(False)

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.clear_messages()
        self.selection = []
        self.data = data
        self.pp_data = None
        self.n_attrs = 0
        if data is not None:
            if len(data) < 2:
                self.Warning.not_enough_inst()
            elif data.Y.size == 0:
                self.Warning.no_class_var()
            else:
                remover = Remove(Remove.RemoveConstant)
                pp_data = Discretize()(remover(data))
                if remover.attr_results["removed"]:
                    self.Information.removed_cons_feat()
                if len(pp_data.domain.attributes) < 2:
                    self.Warning.not_enough_vars()
                else:
                    self.pp_data = pp_data
                    self.n_attrs = len(pp_data.domain.attributes)
                    self.scorer = InteractionScorer(pp_data)
                    self.heuristic = Heuristic(self.scorer.information_gain, self.heuristic_type)
                    self.model.set_domain(pp_data.domain, scorer=self.scorer)
        self.feature_model.set_domain(self.pp_data and self.pp_data.domain)
        self.openContext(self.pp_data)
        self.initialize()

    def initialize(self):
        if self.task is not None:
            self.keep_running = False
            self.cancel()
        self.keep_running = True
        self.saved_state = None
        self.progress = 0
        self.progressBarFinished()
        self.model.clear()
        self.filter.setText("")
        self.button.setText("Start")
        if self.pp_data is not None:
            self.toggle()

    def commit(self):
        if self.data is None:
            self.Outputs.features.send(None)
            return

        self.Outputs.features.send(AttributeList(
            [self.data.domain[attr] for attr in self.selection]))

    def toggle(self):
        self.keep_running = not self.keep_running
        if not self.keep_running:
            self.button.setText("Pause")
            self.button.repaint()
            self.progressBarInit()
            self.filter.setText("")
            self.filter.setEnabled(False)
            self.start(run, self.compute_score, self.row_for_state,
                       self.iterate_states, self.saved_state,
                       self.progress, self.state_count())
        else:
            self.button.setText("Continue")
            self.button.repaint()
            self.cancel()
            self.progressBarFinished()
            self.filter.setEnabled(True)

    def on_selection_changed(self, selected):
        self.selection = [self.model.data(ind) for ind in selected.indexes()[-2:]]
        self.commit()

    def on_filter_changed(self):
        self.model.filter(self.filter_text)

    def on_feature_combo_changed(self):
        self.feature_index = self.feature and self.pp_data.domain.index(self.feature)
        self.initialize()

    def on_heuristic_combo_changed(self):
        if self.pp_data is not None:
            self.heuristic = Heuristic(self.scorer.information_gain, self.heuristic_type)
        self.initialize()

    def compute_score(self, state):
        scores = (self.scorer(*state),
                  self.scorer.information_gain[state[0]],
                  self.scorer.information_gain[state[1]])
        return tuple(self.scorer.normalize(score) for score in scores)

    @staticmethod
    def row_for_state(score, state):
        return [score[0], sum(score)] + list(state)

    def iterate_states(self, initial_state):
        if self.feature is not None:
            return self._iterate_by_feature(initial_state)
        if self.heuristic is not None:
            return self.heuristic.get_states(initial_state)
        return self._iterate_all(initial_state)

    def _iterate_all(self, initial_state):
        i0, j0 = initial_state or (0, 0)
        for i in range(i0, self.n_attrs):
            for j in range(j0, i):
                yield i, j
            j0 = 0

    def _iterate_by_feature(self, initial_state):
        _, j0 = initial_state or (0, 0)
        for j in range(j0, self.n_attrs):
            if j != self.feature_index:
                yield self.feature_index, j

    def state_count(self):
        if self.feature is None:
            return self.n_attrs * (self.n_attrs - 1) // 2
        return self.n_attrs

    def on_partial_result(self, result):
        add_to_model, latest_state = result
        if add_to_model:
            self.saved_state = latest_state
            self.model.append(add_to_model)
            self.progress = len(self.model)
            self.progressBarSet(self.progress * 100 // self.state_count())

    def on_done(self, result):
        self.button.setText("Finished")
        self.button.setEnabled(False)
        self.filter.setEnabled(True)

    def send_report(self):
        self.report_table("Interactions", self.rank_table)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWInteractions).run(Table("iris"))
