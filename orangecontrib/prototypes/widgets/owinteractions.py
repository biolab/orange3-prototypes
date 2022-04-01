"""
Interactions widget
"""
from enum import IntEnum
from operator import attrgetter
from itertools import chain

import numpy as np

from AnyQt.QtCore import Qt, QSortFilterProxyModel
from AnyQt.QtCore import QLineF
from AnyQt.QtGui import QStandardItem, QPainter, QColor, QPen
from AnyQt.QtWidgets import QHeaderView
from AnyQt.QtCore import QModelIndex
from AnyQt.QtWidgets import QStyleOptionViewItem, QApplication, QStyle

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, AttributeList, Msg
from Orange.widgets.visualize.utils import VizRankDialogAttrPair
from Orange.preprocess import Discretize, Remove
import Orange.widgets.data.owcorrelations


SIZE_LIMIT = 1000000


class HeuristicType(IntEnum):
	"""
	Heuristic type enumerator. Possible choices: low Information gain first, random.
	"""
	INFOGAIN, RANDOM = 0, 1

	@staticmethod
	def items():
		"""
		Text for heuristic types. Can be used in gui controls (eg. combobox).
		"""
		return ["InfoGain Heuristic", "Random Search"]


class Interaction:
	def __init__(self, disc_data):
		self.data = disc_data
		self.n_attrs = len(self.data.domain.attributes)
		self.class_h = self.entropy(self.data.Y)
		self.attr_h = np.zeros(self.n_attrs)
		self.gains = np.zeros(self.n_attrs)
		self.removed_h = np.zeros((self.n_attrs, self.n_attrs))

		# Precompute information gain of each attribute for faster overall
		# computation and to create heuristic. Only removes necessary NaN values
		# to keep as much data as possible and keep entropies and information gains
		# invariant of third attribute.
		# In certain situations this can cause unexpected results i.e. negative
		# information gains or negative interactions lower than individual
		# attribute information.
		self.compute_gains()

	@staticmethod
	def distribution(ar):
		nans = np.isnan(ar)
		if nans.any():
			if len(ar.shape) == 1:
				ar = ar[~nans]
			else:
				ar = ar[~nans.any(axis=1)]
		_, counts = np.unique(ar, return_counts=True, axis=0)
		return counts / len(ar)

	def entropy(self, ar):
		p = self.distribution(ar)
		return -np.sum(p * np.log2(p))

	def compute_gains(self):
		for attr in range(self.n_attrs):
			self.attr_h[attr] = self.entropy(self.data.X[:, attr])
			self.gains[attr] = self.attr_h[attr] + self.class_h \
				- self.entropy(np.c_[self.data.X[:, attr], self.data.Y])

	def __call__(self, attr1, attr2):
		attrs = np.c_[self.data.X[:, attr1], self.data.X[:, attr2]]
		self.removed_h[attr1, attr2] = self.entropy(attrs) + self.class_h - self.entropy(np.c_[attrs, self.data.Y])
		score = self.removed_h[attr1, attr2] - self.gains[attr1] - self.gains[attr2]
		return score


class Heuristic:
	def __init__(self, weights, heuristic_type=None):
		self.n_attributes = len(weights)
		self.attributes = np.arange(self.n_attributes)
		if heuristic_type == HeuristicType.INFOGAIN:
			self.attributes = self.attributes[np.argsort(weights)]
		elif heuristic_type == HeuristicType.RANDOM:
			np.random.shuffle(self.attributes)

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
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.r = QColor(255, 170, 127)
		self.g = QColor(170, 242, 43)
		self.b = QColor(70, 190, 250)
		self.__line = QLineF()
		self.__pen = QPen(self.b, 5, Qt.SolidLine, Qt.RoundCap)

	def paint(
			self, painter: QPainter, option: QStyleOptionViewItem,
			index: QModelIndex
	) -> None:
		opt = QStyleOptionViewItem(option)
		self.initStyleOption(opt, index)
		widget = option.widget
		style = QApplication.style() if widget is None else widget.style()
		pen = self.__pen
		line = self.__line
		self.__style = style
		text = opt.text
		opt.text = ""
		style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)
		textrect = style.subElementRect(
			QStyle.SE_ItemViewItemText, opt, widget)

		# interaction is None for attribute items ->
		# only draw bars for first column
		interaction = self.cachedData(index, InteractionRank.IntRole)
		if interaction is not None:
			rect = option.rect
			pw = self.penWidth
			textoffset = pw + 2
			baseline = rect.bottom() - textoffset / 2
			origin = rect.left() + 3 + pw / 2 # + half pen width for the round line cap
			width = rect.width() - 3 - pw

			def draw_line(start, length):
				line.setLine(origin + start, baseline, origin + start + length, baseline)
				painter.drawLine(line)

			# negative information gains stem from issues in interaction calculation
			# may cause bars reaching out of intended area
			l_bar, r_bar = self.cachedData(index, InteractionRank.GainRole)
			l_bar, r_bar = width * max(l_bar, 0), width * max(r_bar, 0)
			interaction *= width

			pen.setColor(self.b)
			pen.setWidth(pw)
			painter.save()
			painter.setRenderHint(QPainter.Antialiasing)
			painter.setPen(pen)
			draw_line(0, l_bar)
			draw_line(l_bar + interaction, r_bar)
			pen.setColor(self.g if interaction >= 0 else self.r)
			painter.setPen(pen)
			draw_line(l_bar, interaction)
			painter.restore()
			textrect.adjust(0, 0, 0, -textoffset)

		opt.text = text
		self.drawViewItemText(style, painter, opt, textrect)


class SortProxyModel(QSortFilterProxyModel):
	def lessThan(self, left, right):
		role = self.sortRole()
		l_score = left.data(role)
		r_score = right.data(role)
		if l_score[-1] == "%":
			l_score, r_score = float(l_score[:-1]), float(r_score[:-1])
		return l_score < r_score


class InteractionRank(Orange.widgets.data.owcorrelations.CorrelationRank):
	IntRole = next(gui.OrangeUserRole)
	GainRole = next(gui.OrangeUserRole)

	def __init__(self, *args):
		VizRankDialogAttrPair.__init__(self, *args)
		self.interaction = None
		self.heuristic = None
		self.use_heuristic = False
		self.sel_feature_index = None

		self.model_proxy = SortProxyModel(self)
		self.model_proxy.setSourceModel(self.rank_model)
		self.rank_table.setModel(self.model_proxy)
		self.rank_table.selectionModel().selectionChanged.connect(self.on_selection_changed)
		self.rank_table.setItemDelegate(InteractionItemDelegate())
		self.rank_table.setSortingEnabled(True)
		self.rank_table.sortByColumn(0, Qt.DescendingOrder)
		self.rank_table.horizontalHeader().setStretchLastSection(False)
		self.rank_table.horizontalHeader().show()

	def initialize(self):
		VizRankDialogAttrPair.initialize(self)
		data = self.master.disc_data
		self.attrs = data and data.domain.attributes
		self.model_proxy.setFilterKeyColumn(-1)
		self.rank_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
		self.rank_model.setHorizontalHeaderLabels(["Interaction", "Info Gain", "Feature 1", "Feature 2"])
		self.heuristic = None
		self.use_heuristic = False
		self.sel_feature_index = self.master.feature and data.domain.index(self.master.feature)
		if data:
			if self.interaction is None or self.interaction.data != data:
				self.interaction = Interaction(data)
			self.use_heuristic = len(data) * len(self.attrs) ** 2 > SIZE_LIMIT
			if self.use_heuristic and not self.sel_feature_index:
				self.heuristic = Heuristic(self.interaction.gains, self.master.heuristic_type)

	def compute_score(self, state):
		attr1, attr2 = state
		h = self.interaction.class_h
		score = self.interaction(attr1, attr2) / h
		gain1 = self.interaction.gains[attr1] / h
		gain2 = self.interaction.gains[attr2] / h
		return score, gain1, gain2

	def row_for_state(self, score, state):
		attrs = sorted((self.attrs[x] for x in state), key=attrgetter("name"))
		attr_items = []
		for attr in attrs:
			item = QStandardItem(attr.name)
			item.setToolTip(attr.name)
			attr_items.append(item)
		score_items = [
			QStandardItem("{:+.1f}%".format(100 * score[0])),
			QStandardItem("{:.1f}%".format(100 * sum(score)))
		]
		score_items[0].setData(score[0], self.IntRole)
		# arrange bars to match columns
		gains = [x[1] for x in sorted(enumerate(score[1:]), key=lambda x: self.attrs[state[x[0]]].name)]
		score_items[0].setData(gains, self.GainRole)
		score_items[0].setToolTip("{}: {:+.1f}%\n{}: {:+.1f}%".format(attrs[0], 100*gains[0], attrs[1], 100*gains[1]))
		for item in score_items + attr_items:
			item.setData(attrs, self._AttrRole)
			item.setData(Qt.AlignLeft + Qt.AlignCenter, Qt.TextAlignmentRole)
		return score_items + attr_items

	def check_preconditions(self):
		return self.master.disc_data is not None


class OWInteractions(Orange.widgets.data.owcorrelations.OWCorrelations):
	# todo: make parent class for OWInteractions and OWCorrelations
	name = "Interactions"
	description = "Compute all pairwise attribute interactions."
	category = None
	icon = "icons/Interactions.svg"

	class Inputs:
		data = Input("Data", Table)

	class Outputs:
		features = Output("Features", AttributeList)
		interactions = Output("Interactions", Table)

	# feature and selection set by parent
	heuristic_type: int
	heuristic_type = Setting(0)

	class Warning(OWWidget.Warning):
		not_enough_vars = Msg("At least two features are needed.")
		not_enough_inst = Msg("At least two instances are needed.")
		no_class_var = Msg("Target feature missing")

	def __init__(self):
		OWWidget.__init__(self)
		self.data = None  # type: Table
		self.disc_data = None  # type: Table

		# GUI
		box = gui.vBox(self.controlArea)
		self.heuristic_combo = gui.comboBox(
			box, self, "heuristic_type", items=HeuristicType.items(),
			orientation=Qt.Horizontal, callback=self._heuristic_combo_changed
		)

		self.feature_model = DomainModel(
			order=DomainModel.ATTRIBUTES, separators=False,
			placeholder="(All combinations)")
		gui.comboBox(
			box, self, "feature", callback=self._feature_combo_changed,
			model=self.feature_model, searchable=True
		)

		self.vizrank, _ = InteractionRank.add_vizrank(
			None, self, None, self._vizrank_selection_changed)
		self.vizrank.button.setEnabled(False)
		self.vizrank.threadStopped.connect(self._vizrank_stopped)

		box.layout().addWidget(self.vizrank.filter)
		box.layout().addWidget(self.vizrank.rank_table)
		box.layout().addWidget(self.vizrank.button)

	def _heuristic_combo_changed(self):
		self.apply()

	@Inputs.data
	def set_data(self, data):
		self.closeContext()
		self.clear_messages()
		self.data = data
		self.disc_data = None
		self.selection = []
		if data is not None:
			if len(data) < 2:
				self.Warning.not_enough_inst()
			elif data.Y.size == 0:
				self.Warning.no_class_var()
			else:
				remover = Remove(Remove.RemoveConstant)
				data = remover(data)
				disc_data = Discretize()(data)
				if remover.attr_results["removed"]:
					self.Information.removed_cons_feat()
				if len(disc_data.domain.attributes) < 2:
					self.Warning.not_enough_vars()
				else:
					self.disc_data = disc_data
		self.feature_model.set_domain(self.disc_data and self.disc_data.domain)
		self.openContext(self.disc_data)
		self.apply()
		self.vizrank.button.setEnabled(self.disc_data is not None)

	def apply(self):
		self.vizrank.initialize()
		if self.disc_data is not None:
			# this triggers self.commit() by changing vizrank selection
			self.vizrank.toggle()
		else:
			self.commit()

	def commit(self):
		if self.data is None or self.disc_data is None:
			self.Outputs.features.send(None)
			self.Outputs.interactions.send(None)
			return

		attrs = [ContinuousVariable("Interaction")]
		metas = [StringVariable("Feature 1"), StringVariable("Feature 2")]
		domain = Domain(attrs, metas=metas)
		model = self.vizrank.rank_model
		x = np.array([
			[float(model.data(model.index(row, 0), InteractionRank.IntRole))]
			for row in range(model.rowCount())])
		m = np.array(
			[[a.name for a in model.data(model.index(row, 0), InteractionRank._AttrRole)]
				for row in range(model.rowCount())], dtype=object)
		int_table = Table(domain, x, metas=m)
		int_table.name = "Interactions"

		# data has been imputed; send original attributes
		self.Outputs.features.send(AttributeList(
			[self.data.domain[var.name] for var in self.selection]))
		self.Outputs.interactions.send(int_table)


if __name__ == "__main__":  # pragma: no cover
	WidgetPreview(OWInteractions).run(Table("iris"))
