"""

TODO:
  - Sorting by standard deviation: Use coefficient of variation (std/mean)
    or quartile coefficient of dispersion (Q3 - Q1) / (Q3 + Q1)
  - Standard deviation for nominal: try out Variation ratio (1 - n_mode/N)
"""
import locale
from enum import IntEnum
from functools import partial
from typing import Any, Optional  # pylint: disable=unused-import

import numpy as np
import scipy.stats as ss
from AnyQt.QtCore import Qt, QSize, QRectF, QVariant, \
    QModelIndex, pyqtSlot, QRegExp
from AnyQt.QtGui import QPainter, QColor, QKeySequence
from AnyQt.QtWidgets import QStyleOptionViewItem, QShortcut
from AnyQt.QtWidgets import QStyledItemDelegate, QGraphicsScene, \
    QTableView, QHeaderView, QStyle

import Orange.statistics.util as ut
from Orange.canvas.report import plural
from Orange.data import Table, StringVariable, DiscreteVariable, \
    ContinuousVariable, TimeVariable, Domain, Variable
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting, ContextSetting, \
    DomainContextHandler
from Orange.widgets.utils.itemmodels import DomainModel, AbstractSortTableModel
from Orange.widgets.utils.signals import Input
from orangecontrib.prototypes.widgets.utils.histogram import Histogram


class FeatureStatisticsTableModel(AbstractSortTableModel):
    DistributionRole = next(gui.OrangeUserRole)

    CLASS_VAR, META, ATTRIBUTE = range(3)
    COLOR_FOR_ROLE = {
        CLASS_VAR: QColor(160, 160, 160),
        META: QColor(220, 220, 200),
        ATTRIBUTE: QColor(255, 255, 255),
    }

    class Columns(IntEnum):
        ICON, NAME, DISTRIBUTION, CENTER, DISPERSION, MIN, MAX, MISSING = range(8)

        @property
        def name(self):
            return {self.ICON: '',
                    self.NAME: 'Name',
                    self.DISTRIBUTION: 'Distribution',
                    self.CENTER: 'Center',
                    self.DISPERSION: 'Dispersion',
                    self.MIN: 'Min.',
                    self.MAX: 'Max.',
                    self.MISSING: 'Missing values',
                    }[self.value]

        @property
        def index(self):
            return self.value

        @classmethod
        def from_index(cls, index):
            return cls(index)

    def __init__(self, data=None, parent=None):
        """

        Parameters
        ----------
        data : Table
        parent
        """
        super().__init__(parent)
        self._data = data
        self._domain = domain = data.domain  # type: Domain

        self._attributes = domain.attributes + domain.class_vars + domain.metas
        self.n_attributes = len(self._attributes)
        self.n_instances = len(data)

        self.__compute_statistics()

    @staticmethod
    def _attr_indices(attrs):
        # type: (List) -> Tuple[List[int], List[int], List[int], List[int]]
        """Get the indices of different attribute types eg. discrete."""
        disc_var_idx = [i for i, attr in enumerate(attrs) if isinstance(attr, DiscreteVariable)]
        cont_var_idx = [i for i, attr in enumerate(attrs)
                        if isinstance(attr, ContinuousVariable)
                        and not isinstance(attr, TimeVariable)]
        time_var_idx = [i for i, attr in enumerate(attrs) if isinstance(attr, TimeVariable)]
        string_var_idx = [i for i, attr in enumerate(attrs) if isinstance(attr, StringVariable)]
        return disc_var_idx, cont_var_idx, time_var_idx, string_var_idx

    def __compute_statistics(self):
        # We will compute statistics over all data at once
        matrices = [self._data.X, self._data._Y, self._data.metas]

        # Since data matrices can of mixed sparsity, we need to compute
        # attributes separately for each of them.
        matrices = zip([
            self._domain.attributes, self._domain.class_vars, self._domain.metas
        ], matrices)
        # Filter out any matrices with size 0, filter the zipped matrices to 
        # eliminate variables in a single swoop
        matrices = list(filter(lambda tup: tup[1].size, matrices))

        def _apply_to_types(attrs_x_pair, discrete_f=None, continuous_f=None,
                            time_f=None, string_f=None, default_val=np.nan):
            """Apply functions to variable types e.g. discrete_f to discrete 
            variables. Default value is returned if there is no function 
            defined for specific variable types."""
            attrs, x = attrs_x_pair
            result = np.full(len(attrs), default_val)
            disc_var_idx, cont_var_idx, time_var_idx, str_var_idx = self._attr_indices(attrs)
            if discrete_f and x[:, disc_var_idx].size:
                result[disc_var_idx] = discrete_f(x[:, disc_var_idx].astype(np.float64))
            if continuous_f and x[:, cont_var_idx].size:
                result[cont_var_idx] = continuous_f(x[:, cont_var_idx].astype(np.float64))
            if time_f and x[:, time_var_idx].size:
                result[time_var_idx] = time_f(x[:, time_var_idx].astype(np.float64))
            if string_f and x[:, str_var_idx].size:
                result[str_var_idx] = string_f(x[:, str_var_idx].astype(np.object))
            return result

        self._variable_types = [type(var) for var in self._attributes]
        self._variable_names = [var.name.lower() for var in self._attributes]

        # Compute the center
        _center = partial(
            _apply_to_types,
            discrete_f=lambda x: ss.mode(x)[0],
            continuous_f=lambda x: ut.nanmean(x, axis=0),
        )
        self._center = np.hstack(map(_center, matrices))

        # Compute the dispersion
        def _entropy(x):
            p = [ut.bincount(row)[0] for row in x.T]
            p = [pk / np.sum(pk) for pk in p]
            return np.fromiter((ss.entropy(pk) for pk in p), dtype=np.float64)
        _dispersion = partial(
            _apply_to_types,
            discrete_f=lambda x: _entropy(x),
            continuous_f=lambda x: ut.nanvar(x, axis=0),
        )
        self._dispersion = np.hstack(map(_dispersion, matrices))

        # Compute minimum values
        _max = partial(
            _apply_to_types,
            discrete_f=lambda x: ut.nanmax(x, axis=0),
            continuous_f=lambda x: ut.nanmax(x, axis=0),
        )
        self._max = np.hstack(map(_max, matrices))

        # Compute maximum values
        _min = partial(
            _apply_to_types,
            discrete_f=lambda x: ut.nanmin(x, axis=0),
            continuous_f=lambda x: ut.nanmin(x, axis=0),
        )
        self._min = np.hstack(map(_min, matrices))

        # Compute # of missing values
        _missing = partial(
            _apply_to_types,
            discrete_f=lambda x: ut.countnans(x, axis=0),
            continuous_f=lambda x: ut.countnans(x, axis=0),
            string_f=lambda x: (x == StringVariable.Unknown).sum(axis=0),
            time_f=lambda x: ut.countnans(x, axis=0),
        )
        self._missing = np.hstack(map(_missing, matrices))

    def sortColumnData(self, column):
        if column == self.Columns.ICON:
            return self._variable_types
        elif column == self.Columns.NAME:
            return self._variable_names
        elif column == self.Columns.DISTRIBUTION:
            # TODO Implement some form of sorting over the histograms
            return self._variable_names
        elif column == self.Columns.CENTER:
            return self._center
        elif column == self.Columns.DISPERSION:
            # Since absolute values of dispersion aren't very helpful when the
            # variables occupy different ranges, use the coefficient of
            # variation instead for a more reasonable sorting
            dispersion = np.array(self._dispersion)
            _, cont_var_indices, *_ = self._attr_indices(self._attributes)
            dispersion[cont_var_indices] /= self._center[cont_var_indices]
            return dispersion
        elif column == self.Columns.MIN:
            return self._min
        elif column == self.Columns.MAX:
            return self._max
        elif column == self.Columns.MISSING:
            return self._missing

    def headerData(self, section, orientation, role):
        # type: (int, Qt.Orientation, Qt.ItemDataRole) -> Any
        if orientation == Qt.Horizontal:
            if role == Qt.DisplayRole:
                return self.Columns.from_index(section).name

    def data(self, index, role):
        # type: (QModelIndex, Qt.ItemDataRole) -> Any
        if not index.isValid():
            return

        row, column = self.mapToSourceRows(index.row()), index.column()
        # Make sure we're not out of range
        if not 0 <= row <= self.n_attributes:
            return QVariant()

        output = None
        attribute = self._attributes[row]

        if column == self.Columns.ICON:
            if role == Qt.DecorationRole:
                return gui.attributeIconDict[attribute]
        elif column == self.Columns.NAME:
            if role == Qt.DisplayRole:
                output = attribute.name
        elif column == self.Columns.DISTRIBUTION:
            if role == self.DistributionRole:
                if isinstance(attribute, (DiscreteVariable, ContinuousVariable)):
                    return attribute, self._data
        elif column == self.Columns.CENTER:
            if role == Qt.DisplayRole:
                if isinstance(attribute, DiscreteVariable):
                    output = self._center[row]
                    if not np.isnan(output):
                        output = attribute.str_val(self._center[row])
                else:
                    output = self._center[row]
        elif column == self.Columns.DISPERSION:
            if role == Qt.DisplayRole:
                output = self._dispersion[row]
        elif column == self.Columns.MIN:
            if role == Qt.DisplayRole:
                if isinstance(attribute, DiscreteVariable):
                    if attribute.ordered:
                        output = attribute.str_val(self._min[row])
                else:
                    output = self._min[row]
        elif column == self.Columns.MAX:
            if role == Qt.DisplayRole:
                if isinstance(attribute, DiscreteVariable):
                    if attribute.ordered:
                        output = attribute.str_val(self._max[row])
                else:
                    output = self._max[row]
        elif column == self.Columns.MISSING:
            if role == Qt.DisplayRole:
                output = '%d (%d%%)' % (
                    self._missing[row],
                    100 * self._missing[row] / self.n_instances
                )

        if role == Qt.BackgroundRole:
            if attribute in self._domain.attributes:
                return self.COLOR_FOR_ROLE[self.ATTRIBUTE]
            elif attribute in self._domain.metas:
                return self.COLOR_FOR_ROLE[self.META]
            elif attribute in self._domain.class_vars:
                return self.COLOR_FOR_ROLE[self.CLASS_VAR]

        elif role == Qt.TextAlignmentRole:
            if column == self.Columns.NAME:
                return Qt.AlignLeft | Qt.AlignVCenter
            return Qt.AlignRight | Qt.AlignVCenter

        # Consistently format the text inside the table cells
        # The easiest way to check for NaN is to compare with itself
        if output != output:
            output = 'NaN'
        # Format ∞ properly
        elif output in (np.inf, -np.inf):
            output = '%s∞' % ['', '-'][output < 0]
        elif isinstance(output, int):
            output = locale.format('%d', output, grouping=True)
        elif isinstance(output, float):
            output = locale.format('%.2f', output, grouping=True)

        return output

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.n_attributes

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.Columns)


class NoFocusRectDelegate(QStyledItemDelegate):
    """Removes the light blue background and border on a focused item."""

    def paint(self, painter, option, index):
        # type: (QPainter, QStyleOptionViewItem, QModelIndex) -> None
        option.state &= ~QStyle.State_HasFocus
        super().paint(painter, option, index)


class DistributionDelegate(NoFocusRectDelegate):
    def __init__(self, parent=None):
        self.color_attribute = None
        super().__init__(parent)
        self.__cache = {}

    def clear(self):
        self.__cache.clear()

    def set_color_attribute(self, variable):
        assert variable is None or isinstance(variable, Variable)
        self.color_attribute = variable
        self.__cache.clear()

    def paint(self, painter, option, index):
        # type: (QPainter, QStyleOptionViewItem, QModelIndex) -> None
        data = index.data(FeatureStatisticsTableModel.DistributionRole)
        if data is None:
            return super().paint(painter, option, index)

        row = index.model().mapToSourceRows(index.row())

        if row not in self.__cache:
            scene = QGraphicsScene(self)
            attribute, data = data
            histogram = Histogram(
                data=data,
                variable=attribute,
                color_attribute=self.color_attribute,
                border=(0, 0, 2, 0),
                border_color='#ccc',
            )
            scene.addItem(histogram)
            self.__cache[row] = scene

        painter.setRenderHint(QPainter.HighQualityAntialiasing)

        background_color = index.data(Qt.BackgroundRole)
        self.__cache[row].setBackgroundBrush(background_color)

        self.__cache[row].render(
            painter,
            target=QRectF(option.rect),
            mode=Qt.IgnoreAspectRatio,
        )


class OWFeatureStatistics(widget.OWWidget):
    HISTOGRAM_ASPECT_RATIO = (7, 3)
    MINIMUM_HISTOGRAM_HEIGHT = 50
    MAXIMUM_HISTOGRAM_HEIGHT = 80

    name = 'Feature Statistics'
    description = 'Show basic statistics for data features.'
    icon = 'icons/FeatureStatistics.svg'

    class Inputs:
        data = Input('Data', Table, default=True)

    want_main_area = True
    buttons_area_orientation = Qt.Vertical

    settingsHandler = DomainContextHandler()

    color_var = ContextSetting(None)  # type: Optional[Variable]
    filter_string = ContextSetting('')

    def __init__(self):
        super().__init__()

        self.data = None  # type: Optional[Table]
        self.model = None  # type: Optional[FeatureStatisticsTableModel]

        # Information panel
        info_box = gui.vBox(self.controlArea, 'Info')
        info_box.setMinimumWidth(200)
        self.info_summary = gui.widgetLabel(info_box, wordWrap=True)
        self.info_attr = gui.widgetLabel(info_box, wordWrap=True)
        self.info_class = gui.widgetLabel(info_box, wordWrap=True)
        self.info_meta = gui.widgetLabel(info_box, wordWrap=True)
        self.set_info()

        # TODO: Implement filtering on the model
        # filter_box = gui.vBox(self.controlArea, 'Filter')
        # self.filter_text = gui.lineEdit(
        #     filter_box, self, value='filter_string',
        #     placeholderText='Filter variables by name',
        #     callback=self._filter_table_variables, callbackOnType=True,
        # )
        # shortcut = QShortcut(QKeySequence('Ctrl+f'), self, self.filter_text.setFocus)
        # shortcut.setWhatsThis('Filter variables by name')

        self.color_var_model = DomainModel(
            valid_types=(ContinuousVariable, DiscreteVariable),
            placeholder='None',
        )
        box = gui.vBox(self.controlArea, 'Histogram')
        self.cb_color_var = gui.comboBox(
            box, master=self, value='color_var',
            model=self.color_var_model, label='Color:', orientation=Qt.Horizontal,
        )
        self.cb_color_var.currentIndexChanged.connect(self.__color_var_changed)

        gui.rubber(self.controlArea)

        # Main area
        self.view = QTableView(
            showGrid=False,
            cornerButtonEnabled=False,
            sortingEnabled=True,
            selectionMode=QTableView.NoSelection,
            horizontalScrollMode=QTableView.ScrollPerPixel,
            verticalScrollMode=QTableView.ScrollPerPixel,
        )

        hheader = self.view.horizontalHeader()
        hheader.setStretchLastSection(False)
        # Contents precision specifies how many rows should be taken into
        # account when computing the sizes, 0 being the visible rows. This is
        # crucial, since otherwise the `ResizeToContents` section resize mode
        # would call `sizeHint` on every single row in the data before first
        # render. However this, this cannot be used here, since this only
        # appears to work properly when the widget is actually shown. When the
        # widget is not shown, size `sizeHint` is called on every row.
        hheader.setResizeContentsPrecision(5)
        # Set a nice default size so that headers have some space around titles
        hheader.setDefaultSectionSize(120)
        # Set individual column behaviour in `set_data` since the logical
        # indices must be valid in the model, which requires data.
        hheader.setSectionResizeMode(QHeaderView.Interactive)

        vheader = self.view.verticalHeader()
        vheader.setVisible(False)
        vheader.setSectionResizeMode(QHeaderView.Fixed)

        def bind_histogram_aspect_ratio(logical_index, _, new_size):
            """Force the horizontal and vertical header to maintain the defined
            aspect ratio specified for the histogram."""
            # Prevent function being exectued more than once per resize
            if logical_index is not self.model.Columns.DISTRIBUTION.index:
                return
            ratio_width, ratio_height = self.HISTOGRAM_ASPECT_RATIO
            unit_width = new_size / ratio_width
            new_height = unit_width * ratio_height
            effective_height = max(new_height, self.MINIMUM_HISTOGRAM_HEIGHT)
            effective_height = min(effective_height, self.MAXIMUM_HISTOGRAM_HEIGHT)
            vheader.setDefaultSectionSize(effective_height)

        def keep_row_centered(logical_index, old_size, new_size):
            """When resizing the widget when scrolled further down, the
            positions of rows changes. Obviously, the user resized in order to
            better see the row of interest. This keeps that row centered."""
            # TODO: This does not work properly
            # Prevent function being exectued more than once per resize
            if logical_index is not self.model.Columns.DISTRIBUTION.index:
                return
            top_row = self.view.indexAt(self.view.rect().topLeft()).row()
            bottom_row = self.view.indexAt(self.view.rect().bottomLeft()).row()
            middle_row = top_row + (bottom_row - top_row) // 2
            self.view.scrollTo(self.model.index(middle_row, 0), QTableView.PositionAtCenter)

        hheader.sectionResized.connect(bind_histogram_aspect_ratio)
        hheader.sectionResized.connect(keep_row_centered)

        self.distribution_delegate = DistributionDelegate()
        self.view.setItemDelegate(self.distribution_delegate)

        self.mainArea.layout().addWidget(self.view)

    def sizeHint(self):
        return QSize(900, 500)

    def _filter_table_variables(self):
        regex = QRegExp(self.filter_string)
        # If the user explicitly types different cases, we assume they know
        # what they are searching for and account for letter case in filter
        different_case = (
            any(c.islower() for c in self.filter_string) and
            any(c.isupper() for c in self.filter_string)
        )
        if not different_case:
            regex.setCaseSensitivity(Qt.CaseInsensitive)

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.data = data

        if data is not None:
            self.model = FeatureStatisticsTableModel(data, parent=self)
            self.color_var_model.set_domain(data.domain)
            # Set the selected index to 1 if any target classes, otherwise 0
            if data.domain.class_vars:
                self.color_var = data.domain.class_vars[0]
            self.openContext(self.data)
        else:
            self.model = None
            self.color_var_model.set_domain(None)

        self.view.setModel(self.model)
        self._filter_table_variables()

        self.distribution_delegate.clear()
        self.set_info()

        # The resize modes for individual columns must be set here, because
        # the logical index must be valid in `setSectionResizeMode`. It is not
        # valid when there is no data in the model.
        if self.model:
            columns, hheader = self.model.Columns, self.view.horizontalHeader()
            hheader.setSectionResizeMode(columns.ICON.index, QHeaderView.ResizeToContents)
            hheader.setSectionResizeMode(columns.DISTRIBUTION.index, QHeaderView.Stretch)

    @pyqtSlot(int)
    def __color_var_changed(self, new_index):
        attribute = None if new_index < 1 else self.cb_color_var.model()[new_index]
        self.distribution_delegate.set_color_attribute(attribute)

        if self.model:
            for row_idx in range(self.model.rowCount()):
                index = self.model.index(
                    row_idx,
                    self.model.Columns.DISTRIBUTION.index)
                self.view.update(index)

    @staticmethod
    def _format_variables_string(variables):
        agg = []
        for var_type_name, var_type in [
            ('categorical', DiscreteVariable),
            ('numeric', ContinuousVariable),
            ('time', TimeVariable),
            ('string', StringVariable)
        ]:
            var_type_list = [v for v in variables if isinstance(v, var_type)]
            if var_type_list:
                agg.append((
                    '%d %s' % (len(var_type_list), var_type_name),
                    len(var_type_list)
                ))

        if not agg:
            return 'No variables'

        attrs, counts = list(zip(*agg))
        if len(attrs) > 1:
            var_string = ', '.join(attrs[:-1]) + ' and ' + attrs[-1]
        else:
            var_string = attrs[0]
        return plural('%s variable{s}' % var_string, sum(counts))

    def set_info(self):
        if self.data is not None:
            self.info_summary.setText('<b>%s</b> contains %s with %s' % (
                self.data.name,
                plural('{number} instance{s}', self.model.n_instances),
                plural('{number} feature{s}', self.model.n_attributes)
            ))

            self.info_attr.setText(
                '<b>Attributes:</b><br>%s' %
                self._format_variables_string(self.data.domain.attributes)
            )
            self.info_class.setText(
                '<b>Class variables:</b><br>%s' %
                self._format_variables_string(self.data.domain.class_vars)
            )
            self.info_meta.setText(
                '<b>Metas:</b><br>%s' %
                self._format_variables_string(self.data.domain.metas)
            )
        else:
            self.info_summary.setText('No data on input.')
            self.info_attr.setText('')
            self.info_class.setText('')
            self.info_meta.setText('')

    def send_report(self):
        pass


if __name__ == '__main__':
    from AnyQt.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    ow = OWFeatureStatistics()

    ow.set_data(Table(sys.argv[1] if len(sys.argv) > 1 else 'iris'))
    ow.show()
    app.exec_()
