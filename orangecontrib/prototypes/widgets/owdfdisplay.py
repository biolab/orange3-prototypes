"""
A generic lightweight data display widget
"""

import sys
import math
import abc

from collections import OrderedDict
from itertools import islice, chain
from operator import attrgetter

from xml.sax.saxutils import escape
from functools import lru_cache, singledispatch, reduce

from typing import Optional, Tuple, List, Dict, Callable, Any, Iterable, \
    Sequence, Mapping, MutableMapping

from PyQt5.QtCore import (
    Qt, QSize, QAbstractItemModel, QAbstractTableModel, QModelIndex,
    QObject
)

from PyQt5.QtGui import (
    QIcon, QColor, QPainter, QStaticText, QTransform
)
from PyQt5.QtWidgets import (
    QTableView, QStyle, QStyledItemDelegate, QStyleOptionViewItem, QVBoxLayout,
    QApplication,
)

import numpy as np
import scipy.sparse as sp
import pandas as pd
import pandas.core.dtypes.dtypes

import Orange.data

from Orange.widgets import widget
from Orange.widgets.utils import itemmodels
from Orange.widgets.data.owtable import BlockSelectionModel
from Orange.widgets.utils.textimport import StampIconEngine


def table_view_compact(view):
    # type: (QTableView) -> None
    vheader = view.verticalHeader()
    option = view.viewOptions()
    option.text = "X"
    option.features |= QStyleOptionViewItem.HasDisplay
    size = view.style().sizeFromContents(
        QStyle.CT_ItemViewItem, option,
        QSize(20, 20), view)
    vheader.ensurePolished()
    vheader.setDefaultSectionSize(
        max(size.height(), vheader.minimumSectionSize())
    )


class DisplayData(abc.ABC):
    """
    An ABC for all data types that support display
    """
    pass


DisplayData.register(pd.DataFrame)
DisplayData.register(np.ndarray)
DisplayData.register(sp.spmatrix)
DisplayData.register(Orange.data.Table)


@singledispatch
def display_model(data):
    # type: (DisplayData) -> QAbstractItemModel
    """
    Return an QAbstractItemModel for a generic 2D table like instance

    This is an abstract generic method.
    """
    raise NotImplementedError


@display_model.register(pd.DataFrame)
def display_model_data_frame(data):
    # type: (pd.DataFrame) -> QAbstractItemModel
    model = DataFrameModel(data)
    return model


@display_model.register(Orange.data.Table)
def display_model_table(data):
    # type: (Orange.data.Table) -> QAbstractItemModel
    return itemmodels.TableModel(data)


@display_model.register(np.ndarray)
def display_model_ndarray(data):
    # type: (np.ndarray) -> QAbstractItemModel
    dtype = data.dtype
    if dtype.fields:
        # dispatch to recarray implementation instead
        return display_model(data.view(np.recarray))

    assert data.ndim == 2
    con = converter(data.dtype)
    role_dispatch = {
        Qt.DisplayRole: lambda i, j: con(data[i, j])
    }
    column_header_dispatch = {
        Qt.DecorationRole: lambda i: icon_for_dtype(dtype)
    }
    return TableModel(
        data.shape, role_dispatch, column_header_dispatch=column_header_dispatch
    )


@display_model.register(np.recarray)
def display_model_recarray(data):
    # type: (np.recarray) -> QAbstractItemModel
    dtype = data.dtype
    fields_ = dtype.fields  # type: Dict[str, Tuple[np.dtype, int]]
    descr = dtype.descr  # type: List[Tuple[str, Any]]  # desc has proper order
    colnames = [name for name, _ in descr]
    fields = [(name, fields_[name][0]) for name in colnames]
    colaux = [
        (name, converter(dtype)) for name, dtype in fields
    ]
    shape = (len(data), len(colaux))

    def data_(row, col):
        # take the column view
        name, converter = colaux[col]
        coldata = data[colnames[col]]
        return converter(coldata[row])

    dispatch = {
        Qt.DisplayRole: data_
    }

    def tooltip(column):
        # type: (int) -> str
        name, dtype = fields[column]
        header = "<b>{name}</b> : {dtype}".format(
            name=escape(name), dtype=escape(dtype.name)
        )
        if dtype.metadata:
            metaitems = ["{} : <i>{}</i>".format(escape(str(key)), escape(str(val)))
                         for key, val in dtype.metadata.items()]
            text = "<hr/>".join([header, "<br/>".join(metaitems)])
        else:
            text = header
        return text

    column_dispatch = {
        Qt.DisplayRole: lambda column: colnames[column],
        Qt.DecorationRole: lambda column: icon_for_dtype(fields[column][1]),
        Qt.ToolTipRole: lambda column: tooltip(column),
    }
    return TableModel(shape, dispatch, column_dispatch)


@display_model.register(sp.spmatrix)
def display_model_sparse(data):
    con = converter(data.dtype)
    dispatch = {
        Qt.DisplayRole: lambda i, j: con(data[i, j])
    }
    return TableModel(data.shape, dispatch)


@singledispatch
def select_ix(data, rows, columns):
    # type: (Sequence[Sequence[Any]], Sequence[int], Sequence[int]) -> Any
    """
    Select a subset of rows and columns from a 2D table `data`

    This is an abstract generic method.

    Parameters
    ----------
    data: Any
    rows: Sequence[int]
    columns: Sequence[int]

    Returns
    -------
    data: Any
        Data indexed by `rows` and `columns`. The returned value might be
        a view of the original data.
    """
    raise NotImplementedError


@select_ix.register(pd.DataFrame)
def select_ix_dataframe(data, rows, columns):
    # type: (pd.DataFrame, Sequence[int], Sequence[int]) -> pd.DataFrame
    return data.iloc[rows, columns]


@select_ix.register(np.ndarray)
def select_ix_ndarray(data, rows, columns):
    # type: (np.ndarray, Sequence[int], Sequence[int]) -> pd.DataFrame
    if data.dtype.fields:
        return select_ix_recarray(
            data.view(np.recarray), rows, columns).view(np.ndarray)
    return data[rows][:, columns]


@select_ix.register(np.recarray)
def select_ix_recarray(data, rows, columns):
    # type: (np.recarray, Sequence[int], Sequence[int]) -> np.recarray
    dtype = data.dtype  # type: np.dtype
    assert dtype.fields is not None
    fields_ = dtype.fields  # type: Mapping[str, Tuple[np.dtype, int]]
    descr = dtype.descr  # type: List[Tuple[str, Any]]  # desc has proper order
    colnames = [name for name, _ in descr]
    fields = [(name, fields_[name][0]) for name in colnames]
    fields_out = [fields[j] for j in columns]
    dtype_out = np.dtype(fields)
    out_size = len(rows)
    out = np.empty(out_size, dtype_out).view(np.recarray)  # type: np.recarray
    row_indices = np.asarray(rows, dtype=np.intp)
    for name_, _ in fields_out:
        np.take(data[name_], row_indices, out=out[name_])
    return out


@select_ix.register(Orange.data.Table)
def select_ix_orange_table(data, rows, columns):
    # type: (Orange.data.Table, Sequence[int], Sequence[int]) -> Orange.data.Table
    domain = data.domain
    vars = domain.variables + domain.metas
    x_vars, y_vars, meta_vars = [], [], []
    for i in columns:
        if 0 <= i < len(domain.attributes):
            x_vars.append(vars[i])
        elif len(domain.attributes) <= i < len(domain.variables):
            y_vars.append(vars[i])
        elif len(domain.variables) <= i < len(domain.variables) + len(domain.metas):
            meta_vars.append(vars[i])
        elif -len(domain.metas) <= i < 0:
            meta_vars.append(vars[i])
        else:
            assert False
    domain = Orange.data.Domain(x_vars, y_vars, meta_vars)
    return data.from_table(domain, data, rows)


class OWDataFrameDisplay(widget.OWWidget):
    name = "Data Frame Display"

    class Inputs:
        data = widget.Input("Data Frame", DisplayData,)

    class Outputs:
        data = widget.Output("Data Frame", DisplayData,)

    want_basic_layout = False

    def __init__(self):
        super().__init__()
        self.__header_state = b''
        self.data = None  # type: Optional[pd.DataFrame]
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.view = QTableView(
            editTriggers=QTableView.NoEditTriggers
        )
        table_view_compact(self.view)
        self.view.setItemDelegate(DataDelegate(self.view))
        self.layout().addWidget(self.view)

        self.info.set_input_summary(self.info.NoInput)
        self.info.set_output_summary(self.info.NoOutput)

    @Inputs.data
    def set_data_frame(self, df):
        # type: (Optional[DisplayData]) -> None
        self.clear()
        self.data = df
        if df is not None:
            self._setup_view(df)
        else:
            self.clear()

    def clear(self):
        self.data = None
        self.view.setModel(None)
        self.info.set_input_summary(self.info.NoInput)

    def _setup_view(self, df):
        if self.view.model() is not None:
            self.view.model().deleteLater()
            sel = self.view.selectionModel()
            sel.selectionChanged.disconnect(self._on_selectionChanged)

        model = display_model(df)

        self.view.setModel(model)
        self.view.setSelectionModel(BlockSelectionModel(model, self.view))
        sel = self.view.selectionModel()
        sel.selectionChanged.connect(self._on_selectionChanged)
        self.info.set_input_summary(len(df))

    def selection(self):
        view = self.view
        sel = view.selectionModel()
        selection = sel.selection()
        row_ranges, col_ranges = [], []

        for srange in selection:
            row_ranges.append(range(srange.top(), srange.bottom() + 1))
            col_ranges.append(range(srange.left(), srange.right() + 1))
        return merge_ranges(row_ranges), merge_ranges(col_ranges)

    def _on_selectionChanged(self, selected, deselected):
        self.commit()

    def commit(self):
        if self.data is None:
            self.info.set_output_summary(self.info.NoOutput)
            self.Outputs.data.send(None)
            return
        data = self.data
        rowsel, colsel = self.selection()

        def indices(ranges: Iterable[range]) -> np.ndarray:
            return np.fromiter(
                chain.from_iterable(ranges), dtype=np.intp,
                count=sum(len(r) for r in ranges)
            )

        rowindices = indices(rowsel)
        colindices = indices(colsel)
        model = self.view.model()

        if isinstance(model, itemmodels.TableModel):
            # TableModel has a builtin reordering of columns
            domain = model.domain
            colindices_ = []
            for i in colindices:
                i_ = domain.index(
                    model.headerData(
                        i, Qt.Horizontal, itemmodels.TableModel.VariableRole)
                )
                colindices_.append(i_)
            colindices = colindices_
        data = select_ix(data, rowindices, colindices)
        self.info.set_output_summary(len(data))
        self.Outputs.data.send(data)


def merge_ranges(ranges: 'Iterable[range]') -> 'Sequence[range]':
    def merge_range_seq_accum(accum: 'List[range]', r: range):
        last = accum[-1]
        assert r.step == 1
        assert last.step == 1
        assert last.start <= r.start
        if r.start <= last.stop:
            # merge into last
            accum[-1] = range(last.start, max(last.stop, r.stop))
        else:
            # push a new (disconnected) range interval
            accum.append(r)
        return accum

    ranges = sorted(ranges, key=attrgetter("start"))
    if ranges:
        return reduce(merge_range_seq_accum, islice(ranges, 1, None),
                      [ranges[0]])
    else:
        return []


def converter(dtype):
    if isinstance(dtype, np.dtype):
        if np.issubdtype(dtype, np.integer):
            return int
        elif np.issubdtype(dtype, np.bool_):
            return bool
        elif np.issubdtype(dtype, np.complexfloating):
            return complex
        elif np.issubdtype(dtype, np.floating):
            return float  # lambda val: None if math.isnan(val) else None
        elif np.issubdtype(dtype, np.str_):
            return str
        elif np.issubdtype(dtype, np.bytes_):
            return bytes
        elif np.issubdtype(dtype, np.datetime64):
            return str
        elif np.issubdtype(dtype, np.timedelta64):
            return str
        else:
            return str
    else:
        if isinstance(dtype, pd.core.dtypes.dtypes.CategoricalDtype):
            return str


@lru_cache(typed=True)
def icon_for_dtype(dtype):
    if isinstance(dtype, np.generic):
        dtype = np.dtype(dtype)

    if isinstance(dtype, np.dtype):
        if np.issubdtype(dtype, np.integer):
            return QIcon(StampIconEngine("I", QColor("red")))
        elif np.issubdtype(dtype, np.floating):
            return QIcon(StampIconEngine("N", QColor("red")))
        elif np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.object_):
            return QIcon(StampIconEngine("S", QColor("black")))
        elif np.issubdtype(dtype, np.datetime64):
            return QIcon(StampIconEngine("T", QColor("deepskyblue")))
        elif np.issubdtype(dtype, np.void):
            return QIcon(StampIconEngine("V", QColor("gray")))
        else:
            return QIcon(StampIconEngine("?", QColor("gray")))
    else:
        if isinstance(dtype, pd.core.dtypes.dtypes.CategoricalDtype):
            return QIcon(StampIconEngine("C", QColor("green")))
        elif isinstance(dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            return QIcon(StampIconEngine("T", QColor("deepskyblue")))

    return QIcon()


_Real = (float, np.float32, np.float64, np.float16), # numbers.Real,
_Integral = (int, np.integer)
_Number = _Integral + _Real
_String = (str, np.str_)

isnan = math.isnan


class DataDelegate(QStyledItemDelegate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        @lru_cache(maxsize=100 * 200)
        def sttext(text: str) -> QStaticText:
            return QStaticText(text)
        self.__static_text_cache = sttext
        self.__static_text_lru_cache = LRUCache(100 * 200)

    def displayText(self, value, locale):
        if isinstance(value, _Integral):
            return super().displayText(int(value), locale)
        elif isinstance(value, _Real):
            if isnan(value):
                return "N/A"
            else:
                super().displayText(float(value), locale)
        elif isinstance(value, _String):
            return str(value)
        return super().displayText(value, locale)

    def initStyleOption(self, option, index):
        # type: (QStyleOptionViewItem, QModelIndex) -> None
        super().initStyleOption(option, index)
        model = index.model()
        v = model.data(index, Qt.DisplayRole)
        if isinstance(v, _Number):
            option.displayAlignment = \
                (option.displayAlignment & ~Qt.AlignHorizontal_Mask) | \
                Qt.AlignRight

    def paint(self, painter, option, index):
        # type: (QPainter, QStyleOptionViewItem, QModelIndex) -> None
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = option.widget
        if widget is not None:
            style = widget.style()
        else:
            style = QApplication.style()

        text = opt.text
        opt.text = ""
        trect = style.subElementRect(QStyle.SE_ItemViewItemText, opt, widget)
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)
        # text margin (as in QCommonStylePrivate::viewItemDrawText)
        margin = style.pixelMetric(QStyle.PM_FocusFrameHMargin, None, widget) + 1
        trect = trect.adjusted(margin, 0, -margin, 0)
        opt.text = text
        if opt.textElideMode != Qt.ElideNone:
            st = self.__static_text_elided_cache(opt, trect.width())
        else:
            st = self.__static_text_cache(text)
        tsize = st.size()
        textalign = opt.displayAlignment
        text_pos_x = text_pos_y = 0.0

        if textalign & Qt.AlignLeft:
            text_pos_x = trect.left()
        elif textalign & Qt.AlignRight:
            text_pos_x = trect.x() + trect.width() - tsize.width()
        elif textalign & Qt.AlignHCenter:
            text_pos_x = trect.center().x() - tsize.width() / 2

        if textalign & Qt.AlignTop:
            text_pos_y = trect.top()
        elif textalign & Qt.AlignBottom:
            text_pos_y = trect.top() + trect.height() - tsize.height()
        elif textalign & Qt.AlignVCenter:
            text_pos_y = trect.center().y() - tsize.height() / 2

        painter.setFont(opt.font)
        painter.drawStaticText(text_pos_x, text_pos_y, st)

    def __static_text_elided_cache(
            self, option: QStyleOptionViewItem, width: int) -> QStaticText:
        """
        Return a QStaticText instance for depicting the text of the `option`
        item.
        """
        key = option.text, option.font.key(), option.textElideMode, width
        try:
            st = self.__static_text_lru_cache[key]
        except KeyError:
            fm = option.fontMetrics
            text = fm.elidedText(option.text, option.textElideMode, width)
            st = QStaticText(text)
            st.prepare(QTransform(), option.font)
            self.__static_text_lru_cache[key] = st
        return st


class LRUCache(MutableMapping):
    __slots__ = ("__dict", "__maxlen")

    def __init__(self, maxlen=100):
        self.__dict = OrderedDict()
        self.__maxlen = maxlen

    def __setitem__(self, key, value):
        self.__dict[key] = value
        self.__dict.move_to_end(key)
        if len(self.__dict) > self.__maxlen:
            self.__dict.popitem(last=False)

    def __getitem__(self, key):
        return self.__dict[key]

    def __delitem__(self, key):
        del self.__dict[key]

    def __contains__(self, key):
        return key in self.__dict

    def __delete__(self, key):
        del self.__dict[key]

    def __iter__(self):
        return iter(self.__dict)

    def __len__(self):
        return len(self.__dict)


class TableModel(QAbstractTableModel):
    __slots__ = (
        "shape", "__data", "__column_header", "__row_header",
        "__row_count", "__col_count",
    )

    Dispatch = Dict[int, Callable[[int, int], Any]]
    HeaderDispatch = Dict[int, Callable[[int], Any]]

    def __init__(
            self,
            shape: Tuple[int, int],
            datadispatch: Dispatch,
            column_header_dispatch: HeaderDispatch = None,
            row_header_dispatch: HeaderDispatch = None,
            parent=None,
            **kwargs
    ) -> None:
        super().__init__(parent, **kwargs)
        self.shape = shape
        self.__row_count = shape[0]
        self.__col_count = shape[1]
        self.__data = datadispatch
        self.__column_header = column_header_dispatch or {}
        self.__row_header = row_header_dispatch or {}

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return self.__row_count

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return self.__col_count

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None

        row, column = index.row(), index.column()
        N, M = self.shape
        if not 0 <= row < N and 0 <= column < M:
            return None

        delegate = self.__data.get(role, None)
        if delegate is not None:
            return delegate(row, column)
        else:
            return None

    def headerData(self, section: int, orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole) -> Any:
        if orientation == Qt.Horizontal:
            delegate = self.__column_header.get(role, None)
            if delegate is not None:
                return delegate(section)
            elif role == Qt.DisplayRole:
                return section + 1
        elif orientation == Qt.Vertical:
            delegate = self.__row_header.get(role, None)
            if delegate is not None:
                return delegate(section)
            elif role == Qt.DisplayRole:
                return section + 1


class DataFrameModel(QAbstractTableModel):
    __slots__ = (
        "__source", "__converters", "__shape", "__df_iat", "__df_columns",
        "__df_index", "__row_count", "__col_count"
    )

    def __init__(self, df: pd.DataFrame, parent: Optional[QObject] = None,
                 **kwargs):
        super().__init__(parent, **kwargs)
        self.__source = df
        self.__converters = [converter(dt) for dt in df.dtypes]
        self.__shape = tuple(df.shape)
        self.__row_count = self.__shape[0]
        self.__col_count = self.__shape[1]

        @lru_cache(maxsize=60 * 200)
        def iat(i, j):
            return df.iat[i, j]
        self.__df_iat = iat
        self.__df_columns = df.columns  # type: pd.Index
        self.__df_index = df.index      # type: pd.Index

    def rowCount(self, parent=QModelIndex()):
        # type: (QModelIndex) -> int
        if parent.isValid():
            return 0
        else:
            return self.__row_count

    def columnCount(self, parent=QModelIndex()):
        # type: (QModelIndex) -> int
        if parent.isValid():
            return 0
        else:
            return self.__col_count

    def data(self, index, role=Qt.DisplayRole):
        # type: (QModelIndex, Qt.ItemDataRole) -> Any
        if not index.isValid():
            return None

        row = index.row()
        column = index.column()
        N, M = self.__shape
        if not 0 <= row < N and 0 <= column < M:
            return None

        if role == Qt.DisplayRole:
            val = self.__df_iat(row, column)
            return self.__converters[column](val)
        elif role == Qt.EditRole:
            val = self.__df_iat(row, column)
            return val
        else:
            return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        # type: (int, Qt.Orientation, Qt.ItemDataRole) -> Any
        df = self.__source
        N, M = df.shape
        if (orientation == Qt.Horizontal and section >= M) \
                or (orientation == Qt.Vertical and section >= N):
            return None

        index = self.__df_columns if orientation == Qt.Horizontal \
            else self.__df_index
        if role == Qt.DisplayRole:
            val = index[section]
            return str(val)
        elif role == Qt.DecorationRole and orientation == Qt.Horizontal:
            return icon_for_dtype(df.dtypes[section])
        elif role == Qt.ToolTipRole and orientation == Qt.Horizontal:
            data = self.__source.iloc[:, section]  # type: pd.Series
            return summary_vector(data)


def summary_vector(data: pd.Series) -> str:
    text = ""
    dtype = data.dtype
    if isinstance(dtype, np.generic):
        dtype = np.dtype(dtype)

    if isinstance(dtype, np.dtype):
        if np.issubdtype(data.dtype, np.number):
            summary = data.describe(percentiles=[.25, .5, .75])
            style = "th { text-align: right; };"
            text = \
                f"<style>{style}</style>" \
                f"<table>" \
                f"<tr><th>Count</th><td>{int(summary['count']):d}</td></tr>" \
                f"<tr><th>Mean</th><td>{summary['mean']:.6g}</td></tr>" \
                f"<tr><th>Std</th><td>{summary['std']:.6g}</td></tr>" \
                f"<tr><th>Min</th><td>{summary['min']:.6g}</td></tr>" \
                f"<tr><th>25%</th><td>{summary['25%']:.6g}</td></tr>" \
                f"<tr><th>50%</th><td>{summary['50%']:.6g}</td></tr>" \
                f"<tr><th>75%</th><td>{summary['75%']:.6g}</td></tr>" \
                f"<tr><th>Max</th><td>{summary['max']:.6g}</td></tr>" \
                f"</table>"
    elif isinstance(dtype, pd.core.dtypes.dtypes.CategoricalDtype):
        dscr = pd.Categorical(data).describe()
        maxitems = 5
        if len(dscr) > maxitems:
            head = dscr[:maxitems]
            rest = dscr[maxitems:]
            dscr = pd.DataFrame({
                "counts": head.counts.tolist() + [rest.counts.sum()],
                "freqs": head.freqs.tolist() + [rest.freqs.sum()],
            },
                index=pd.Categorical(head.index.tolist() + ["..."])
            )
        return df_rich_text(dscr)
    elif isinstance(dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        return df_rich_text(data.describe())
    return text


def df_rich_text(df: pd.DataFrame) -> str:
    style = "th { text-align: right; };"
    def th(text): return f"<th>{escape(str(text))}</th>"
    def td(text): return f"<td>{escape(str(text))}</td>"
    header = [th(t) for t in ["", *df.columns]]
    rows = []
    for index, *rest in df.itertuples(name=None):
        row = [th(index), *map(td, rest)]
        rows.append(row)

    rows = (f"<tr>{''.join(r)}</tr>" for r in [header, *rows])
    text = f"<style>{style}</style>" \
           f"<table>" + "".join(rows) + "</table>"
    return text


def df_rich_text(df: pd.DataFrame) -> str:
    return rich_text_table(
        df.itertuples(index=False),
        column_headers=df.columns,
        row_headers=df.index
    )


def rich_text_table(
        table: Iterable[Iterable[str]],
        column_headers: Optional[Iterable[str]] = None,
        row_headers: Optional[Iterable[str]] = None
) -> str:
    style = "th { text-align: right; };"
    def th(text: str) -> str: return f"<th>{escape(str(text))}</th>"
    def td(text: str) -> str: return f"<td>{escape(str(text))}</td>"

    rows: List[List[str]] = []
    if column_headers is not None:
        header = [th(t) for t in ["", *column_headers]]
        rows.append(header)

    if row_headers is not None:
        rowsiter = ([th(h), *map(td, row)] for h, row in zip(row_headers, table))
    else:
        rowsiter = ([*map(td, row)] for row in table)

    rows.extend(rowsiter)

    rows_ = [f"<tr>{''.join(r)}</tr>" for r in rows]
    text = f"<style>{style}</style>" \
           f"<table>" + "".join(rows_) + "</table>"
    return text


def main(argv=None):
    app = QApplication(argv or [])

    argv = app.arguments()
    if len(argv) > 1:
        fname = argv[1]
        df = pd.read_csv(fname)
    else:
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    x = np.array([(1.0, 2), (3.0, 4)], dtype=[('x', float), ('y', np.dtype("i8", metadata={"a": 1}))])
    w = OWDataFrameDisplay()
    w.set_data_frame(df)
    w.show()
    w.raise_()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
