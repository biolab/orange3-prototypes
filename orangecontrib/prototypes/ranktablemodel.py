import numpy as np

from AnyQt.QtCore import QModelIndex, Qt

from Orange.data.domain import Domain
from Orange.widgets.utils.itemmodels import DomainModel, PyTableModel


MAX_ROWS = int(1e9)  # limits how many rows model will display


class ArrayTableModel(PyTableModel):
    """
    A model for displaying 2-dimensional numpy arrays in ``QTableView`` objects.

    This model extends ``PyTableModel`` to gain access to the following methods:
    ``_roleData``, ``flags``, ``setData``, ``data``, ``setHorizontalHeaderLabels``,
    ``setVerticalHeaderLabels``, and ``headerData``.
    Other, unlisted methods aren't guaranteed to work and should be used with care.

    Also requires access to private members of ``AbstractSortTableModel`` directly;
    ``__sortInd`` is needed to append new unsorted data, and ``__init__`` is used
    because the parent implementation immediately wraps a list, which this model
    does not have.
    """
    def __init__(self, *args, **kwargs):
        super(PyTableModel, self).__init__(*args, **kwargs)

        self._headers = {}
        self._roleData = {}
        self._editable = kwargs.get("editable")

        self._data = None  # type: np.ndarray
        self._columns = 0
        self._rows = 0  # current number of rows containing data
        self._max_view_rows = MAX_ROWS  # maximum number of rows the model/view will display
        self._max_data_rows = MAX_ROWS  # maximum allowed size for the `_data` array
        # ``__len__`` returns _rows: amount of existing data in the model
        # ``rowCount`` returns the lowest of `_rows` and `_max_view_rows`:
        # how large the model/view thinks it is

    @property
    def __sortInd(self):
        return self._AbstractSortTableModel__sortInd

    def sortColumnData(self, column):
        return self._data[:self._rows, column]

    def extendSortFrom(self, sorted_rows: int):
        data = self.sortColumnData(self.sortColumn())
        new_ind = np.arange(sorted_rows, self._rows)
        order = 1 if self.sortOrder() == Qt.AscendingOrder else -1
        sorter = self.__sortInd[::order]
        new_sorter = np.argsort(data[sorted_rows:])
        loc = np.searchsorted(data[:sorted_rows],
                              data[sorted_rows:][new_sorter],
                              sorter=sorter)
        indices = np.insert(sorter, loc, new_ind[new_sorter])[::order]
        self.setSortIndices(indices)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else min(self._rows, self._max_view_rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._columns

    def __len__(self):
        return self._rows

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def initialize(self, data: list[list[float]]):
        self.beginResetModel()
        self._data = np.asarray(data)
        self._rows, self._columns = self._data.shape
        self._roleData = self._RoleData()
        self.resetSorting()
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self._data = None
        self._rows = self._columns = 0
        self._roleData.clear()
        self.resetSorting()
        self.endResetModel()

    def extend(self, rows: list[list[float]]):
        if not isinstance(self._data, np.ndarray):
            self.initialize(rows)
            return

        n_rows = len(rows)
        if n_rows == 0:
            return

        n_data = len(self._data)
        insert = self._rows < self._max_view_rows

        if insert:
            self.beginInsertRows(QModelIndex(), self._rows,
                                 min(self._max_view_rows, self._rows + n_rows) - 1)

        if self._rows + n_rows >= n_data:
            n_data = min(max(n_data + n_rows, 2 * n_data), self._max_data_rows)
            ar = np.full((n_data, self._columns), np.nan)
            ar[:self._rows] = self._data[:self._rows]
            self._data = ar

        self._data[self._rows:self._rows + n_rows] = rows
        self._rows += n_rows

        if insert:
            self.endInsertRows()

        if self.sortColumn() >= 0:
            old_rows = self._rows - n_rows
            self.extendSortFrom(old_rows)


class RankModel(ArrayTableModel):
    """
    Extends ``ArrayTableModel`` for ``VizRankDialog`` type widgets,
    to display scores for combinations of attributes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.domain_model = DomainModel(DomainModel.ATTRIBUTES)

    def set_domain(self, domain: Domain):
        self.domain_model.set_domain(domain)
        n_attrs = len(domain.attributes)
        self._max_data_rows = n_attrs * (n_attrs - 1) // 2

    def resetSorting(self):
        if self._data is None:
            self.sort(-1)
        else:
            self.sort(0, Qt.DescendingOrder)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        column = index.column()
        if column >= self.columnCount() - 2 and role != Qt.EditRole:
            # use domain model for all data (except editrole) in last two columns
            try:
                row = self.mapToSourceRows(index.row())
                value = self.domain_model.index(int(self._data[row, column]))
                return self.domain_model.data(value, role)
            except IndexError:
                return None

        return super().data(index, role)
