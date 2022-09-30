from numbers import Number, Integral
from typing import Iterable, Union
import numpy as np

from AnyQt.QtCore import QModelIndex, Qt, QAbstractTableModel

from Orange.data import Variable
from Orange.data.domain import Domain

from Orange.widgets import gui
from Orange.widgets.utils.itemmodels import DomainModel


MAX_ROWS = int(1e9)  # limits how many rows model will display


def _argsort(data: np.ndarray, order: Qt.SortOrder):
    # same as ``_argsortData`` in AbstractSortModel, might combine?
    if data.ndim == 1:
        indices = np.argsort(data, kind="mergesort")
    else:
        indices = np.lexsort(data.T[::-1])
    if order == Qt.DescendingOrder:
        indices = indices[::-1]
    return indices


class ArrayTableModel(QAbstractTableModel):
    """
    A proxy table model that stores and sorts its data with `numpy`,
    thus providing higher speeds and better scaling.

    TODO: Could extend ``AbstractSortTableModel`` or  ``PyTableModel``?
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__sortInd = ...  # type: np.ndarray
        self.__sortColumn = -1
        self.__sortOrder = Qt.AscendingOrder

        self._data = None  # type: np.ndarray
        self._columns = 0
        self._rows = 0
        self._max_display_rows = self._max_data_rows = MAX_ROWS
        self._headers = {}

    def columnData(self, column: Union[int, slice], apply_sort=False):
        if apply_sort:
            return self._data[:self._rows, column][self.__sortInd]
        return self._data[:self._rows, column]

    def sortColumn(self):
        return self.__sortColumn

    def sortOrder(self):
        return self.__sortOrder

    def mapToSourceRows(self, rows: Union[int, slice, list, np.ndarray]):
        if isinstance(self.__sortInd, np.ndarray) \
                and (isinstance(rows, (Integral, type(Ellipsis)))
                     or len(rows)):
            rows = self.__sortInd[rows]
        return rows

    def resetSorting(self):
        self.sort(-1)

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder):
        if self._data is None:
            return

        indices = self._sort(column, order)
        self.__sortColumn = column
        self.__sortOrder = order

        self.setSortIndices(indices)

    def setSortIndices(self, indices: np.ndarray):
        self.layoutAboutToBeChanged.emit([], QAbstractTableModel.VerticalSortHint)
        self.__sortInd = indices
        self.layoutChanged.emit([], QAbstractTableModel.VerticalSortHint)

    def _sort(self, column: int, order: Qt.SortOrder):
        if column < 0:
            return ...

        data = self.columnData(column)
        return _argsort(data, order)

    def extendSortFrom(self, sorted_rows: int):
        data = self.columnData(self.__sortColumn)
        ind = np.arange(sorted_rows, self._rows)
        order = 1 if self.__sortOrder == Qt.AscendingOrder else -1
        loc = np.searchsorted(data[:sorted_rows],
                              data[sorted_rows:self._rows],
                              sorter=self.__sortInd[::order])
        indices = np.insert(self.__sortInd[::order], loc, ind)[::order]
        self.setSortIndices(indices)

    def rowCount(self, parent=QModelIndex(), *args, **kwargs):
        return 0 if parent.isValid() else min(self._rows, self._max_display_rows)

    def columnCount(self, parent=QModelIndex(), *args, **kwargs):
        return 0 if parent.isValid() else self._columns

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return

        row, column = self.mapToSourceRows(index.row()), index.column()

        try:
            value = self._data[row, column]
        except IndexError:
            return
        match role:
            case Qt.EditRole:
                return value
            case Qt.DisplayRole:
                if isinstance(value, Number) and not \
                        (np.isnan(value) or np.isinf(value) or
                         isinstance(value, Integral)):
                    absval = abs(value)
                    strlen = len(str(int(absval)))
                    value = '{:.{}{}}'.format(value,
                                              2 if absval < .001 else
                                              3 if strlen < 2 else
                                              1 if strlen < 5 else
                                              0 if strlen < 6 else
                                              3,
                                              'f' if (absval == 0 or
                                                      absval >= .001 and
                                                      strlen < 6)
                                              else 'e')
                return str(value)
            case Qt.DecorationRole if isinstance(value, Variable):
                return gui.attributeIconDict[value]
            case Qt.ToolTipRole:
                return str(value)

    def setHorizontalHeaderLabels(self, labels: Iterable[str]):
        self._headers[Qt.Horizontal] = tuple(labels)

    def setVertcalHeaderLabels(self, labels: Iterable[str]):
        self._headers[Qt.Vertical] = tuple(labels)

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.DisplayRole):
        headers = self._headers.get(orientation)

        if headers and section < len(headers):
            if orientation == Qt.Vertical:
                section = self.mapToSourceRows(section)
            if role in {Qt.DisplayRole, Qt.ToolTipRole}:
                return headers[section]

        return super().headerData(section, orientation, role)

    def __len__(self):
        return self._rows

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def initialize(self, data: list[list[float]]):
        self.beginResetModel()
        self._data = np.array(data)
        self._rows, self._columns = self._data.shape
        self.resetSorting()
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self._data = None
        self._rows = self._columns = 0
        self.resetSorting()
        self.endResetModel()

    def append(self, rows: list[list[float]]):
        if not isinstance(self._data, np.ndarray):
            return self.initialize(rows)

        n_rows = len(rows)
        if n_rows == 0:
            print("nothing to add")
            return
        n_data = len(self._data)
        insert = self._rows < self._max_display_rows

        if insert:
            self.beginInsertRows(QModelIndex(), self._rows, min(self._max_display_rows, self._rows + n_rows) - 1)

        if self._rows + n_rows >= n_data:
            n_data = min(max(n_data + n_rows, 2 * n_data), self._max_data_rows)
            ar = np.full((n_data, self._columns), np.nan)
            ar[:self._rows] = self._data[:self._rows]
            self._data = ar

        self._data[self._rows:self._rows + n_rows] = rows
        self._rows += n_rows

        if self.__sortColumn >= 0:
            old_rows = self._rows - n_rows
            self.extendSortFrom(old_rows)

        if insert:
            self.endInsertRows()


class RankModel(ArrayTableModel):
    """
    Extends ``ArrayTableModel`` with filtering and other specific
    features for ``VizRankDialog`` type widgets, to display scores for
    combinations of attributes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__filterInd = ...  # type: np.ndarray
        self.__filterStr = ""

        self.domain = None  # type: Domain
        self.domain_model = DomainModel(DomainModel.ATTRIBUTES)

    def set_domain(self, domain: Domain, **kwargs):
        self.__dict__.update(kwargs)
        self.domain = domain
        self.domain_model.set_domain(domain)
        n_attrs = len(domain.attributes)
        self._max_data_rows = n_attrs * (n_attrs - 1) // 2

    def mapToSourceRows(self, rows):
        if isinstance(self.__filterInd, np.ndarray) \
                and (isinstance(rows, (Integral, type(Ellipsis)))
                     or len(rows)):
            rows = self.__filterInd[rows]
        return super().mapToSourceRows(rows)

    def resetFiltering(self):
        self.filter("")

    def filter(self, text: str):
        if self._data is None:
            return

        if not text:
            self.__filterInd = indices = ...
            self.__filterStr = ""
            self._max_display_rows = MAX_ROWS
        else:
            self.__filterStr = text
            indices = self._filter(text)

        self.setFilterIndices(indices)

    def setFilterIndices(self, indices: np.ndarray):
        self.layoutAboutToBeChanged.emit([])
        if isinstance(indices, np.ndarray):
            self.__filterInd = indices
            self._max_display_rows = len(indices)
        self.layoutChanged.emit([])

    def setSortIndices(self, indices: np.ndarray):
        super().setSortIndices(indices)

        # sorting messes up the filter indices, so they
        # must also be updated
        self.layoutAboutToBeChanged.emit([])
        if isinstance(self.__filterInd, np.ndarray):
            filter_indices = self._filter(self.__filterStr)
            self.__filterInd = filter_indices
            self._max_display_rows = len(filter_indices)
        self.layoutChanged.emit([])

    def _filter(self, text: str):
        attr = [i for i, attr in enumerate(self.domain.attributes)
                if str(text).lower() in attr.name.lower()]

        attr_data = self.columnData(slice(-2, None), apply_sort=True)
        valid = np.isin(attr_data, attr).any(axis=1)

        return valid.nonzero()[0]

    def append(self, rows):
        super().append(rows)

        if isinstance(self.__filterInd, np.ndarray):
            self.resetFiltering()

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return

        row, column = self.mapToSourceRows(index.row()), index.column()
        try:
            value = self._data[row, column]
        except IndexError:
            return

        if column >= self.columnCount() - 2 and role != Qt.EditRole:
            return self.domain_model.data(self.domain_model.index(int(value)), role)

        return super().data(index, role)
