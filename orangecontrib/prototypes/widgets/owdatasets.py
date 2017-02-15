import sys
import os
import logging
import traceback
import enum
import numbers

from xml.sax.saxutils import escape
from concurrent.futures import ThreadPoolExecutor, Future

from types import SimpleNamespace as namespace
from typing import Optional, Dict, Tuple

from AnyQt.QtWidgets import (
    QLabel, QTextBrowser, QSplitter, QTreeView,
    QStyleOptionViewItem, QStyledItemDelegate, QApplication
)

from AnyQt.QtGui import QStandardItemModel, QStandardItem
from AnyQt.QtCore import (
    Qt, QSize, QObject, QThread, QModelIndex, QSortFilterProxyModel,
    QItemSelectionModel,
    pyqtSlot as Slot, pyqtSignal as Signal
)

from serverfiles import LocalFiles, ServerFiles, sizeformat

import Orange.data

from Orange.misc.environ import data_dir
from Orange.widgets import widget, settings, gui

INDEX_URL = "http://butler.fri.uni-lj.si/datasets/"


def local_cache_path():
    return os.path.join(data_dir(), "datasets")


def ensure_local(domain, filename, progress_advance=None):
    localfiles = LocalFiles(local_cache_path(),
                            serverfiles=ServerFiles(server=INDEX_URL))
    return localfiles.localpath_download(
        domain, filename, callback=progress_advance)


def list_remote():
    # type: () -> Dict[Tuple[str, str], dict]
    client = ServerFiles(server=INDEX_URL)
    return client.allinfo()


def list_local():
    # type: () -> Dict[Tuple[str, str], dict]
    return LocalFiles(local_cache_path()).allinfo()


def format_exception(error):
    # type: (BaseException) -> str
    return "\n".join(traceback.format_exception_only(type(error), error))


# Model header
class Header(enum.IntEnum):
    Local = 0
    Name = 1
    Size = 2
    Instances = 3
    Variables = 4
    Target = 5
    Tags = 6

HEADER = ["", "Name", "Size", "Instances", "Variables", "Class", "Tags"]


class SizeDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        # type: (QStyleOptionViewItem, QModelIndex) -> None
        super().initStyleOption(option, index)
        value = index.data(Qt.DisplayRole)
        if isinstance(value, numbers.Integral):
            option.text = sizeformat(int(value))


class NumericalDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        # type: (QStyleOptionViewItem, QModelIndex) -> None
        super().initStyleOption(option, index)
        data = index.data(Qt.DisplayRole)
        align = index.data(Qt.TextAlignmentRole)
        if align is None and isinstance(data, numbers.Number):
            # Right align if the model does not specify otherwise
            option.displayAlignment = Qt.AlignRight | Qt.AlignVCenter


class OWDataSets(widget.OWWidget):
    name = "Data Sets"
    description = ""
    icon = "icons/DataSets.svg"

    outputs = [("Data", Orange.data.Table)]

    #: Selected data set id
    selected_id = settings.Setting(None)   # type: Optional[Tuple[str, str]]

    auto_commit = settings.Setting(False)  # type: bool

    #: main area splitter state
    splitter_state = settings.Setting(b'')  # type: bytes
    header_state = settings.Setting(b'')    # type: bytes

    def __init__(self):
        super().__init__()

        self.__awaiting_state = None  # type: Optional[_FetchState]

        box = gui.widgetBox(self.controlArea, "Info")

        self.infolabel = QLabel(text="Initializing\n\n")
        box.layout().addWidget(self.infolabel)

        self.splitter = QSplitter(orientation=Qt.Vertical)

        self.view = QTreeView(
            sortingEnabled=True,
            selectionMode=QTreeView.SingleSelection,
            alternatingRowColors=True,
            rootIsDecorated=False,
            editTriggers=QTreeView.NoEditTriggers,
        )
        box = gui.widgetBox(self.splitter, "Description", addToLayout=False)
        self.descriptionlabel = QLabel(
            wordWrap=True,
            textFormat=Qt.RichText,
        )
        self.descriptionlabel = QTextBrowser(
            openExternalLinks=True,
            textInteractionFlags=Qt.TextSelectableByMouse |
                                 Qt.LinksAccessibleByMouse
        )
        self.descriptionlabel.setFrameStyle(QTextBrowser.NoFrame)
        # no (white) text background
        self.descriptionlabel.viewport().setAutoFillBackground(False)

        box.layout().addWidget(self.descriptionlabel)
        self.splitter.addWidget(self.view)
        self.splitter.addWidget(box)

        self.splitter.setSizes([300, 200])
        self.splitter.splitterMoved.connect(
            lambda:
                setattr(self, "splitter_state",
                        bytes(self.splitter.saveState()))
        )
        self.mainArea.layout().addWidget(self.splitter)
        self.controlArea.layout().addStretch(10)
        gui.auto_commit(self.controlArea, self, "auto_commit", "Commit")

        model = QStandardItemModel(self)
        model.setHorizontalHeaderLabels(HEADER)
        proxy = QSortFilterProxyModel()
        proxy.setSourceModel(model)
        self.view.setModel(model)

        if self.splitter_state:
            self.splitter.restoreState(self.splitter_state)

        self.view.setItemDelegateForColumn(
            Header.Size, SizeDelegate(self))
        self.view.setItemDelegateForColumn(
            Header.Local, gui.IndicatorItemDelegate(self, role=Qt.DisplayRole))
        self.view.setItemDelegateForColumn(
            Header.Instances, NumericalDelegate(self))
        self.view.setItemDelegateForColumn(
            Header.Variables, NumericalDelegate(self))

        self.view.resizeColumnToContents(Header.Local)

        if self.header_state:
            self.view.header().restoreState(self.header_state)

        self.setBlocking(True)
        self.setStatusMessage("Initializing")

        self._executor = ThreadPoolExecutor(max_workers=1)
        f = self._executor.submit(list_remote)
        w = FutureWatcher(f, parent=self)
        w.done.connect(self.__set_index)

    @Slot(object)
    def __set_index(self, f):
        # type: (Future) -> None
        # set results from `list_remote` query.
        assert QThread.currentThread() is self.thread()
        assert f.done()
        self.setBlocking(False)
        self.setStatusMessage("")
        allinfolocal = list_local()
        try:
            res = f.result()
        except Exception as er:
            log = logging.getLogger(__name__)
            log.exception("Error while fetching updated index")
            if not allinfolocal:
                self.error("Could not fetch data set list")
            else:
                self.warning("Could not fetch data sets list, only local "
                             "cached data sets are shown")
            res = {}

        allinforemote = res  # type: Dict[Tuple[str, str], dict]
        allkeys = set(allinfolocal)
        if allinforemote is not None:
            allkeys = allkeys | set(allinforemote)
        allkeys = sorted(allkeys)

        def info(prefix, filename):
            if (prefix, filename) in allinforemote:
                info = allinforemote[prefix, filename]
            else:
                info = allinfolocal[prefix, filename]
            islocal = (prefix, filename) in allinfolocal

            return namespace(
                prefix=prefix, filename=filename,
                title=info.get("title", filename),
                datetime=info.get("datetime", None),
                description=info.get("description", None),
                reference=info.get("reference", None),
                instances=info.get("instances", None),
                variables=info.get("variables", None),
                target=info.get("target", None),
                missing=info.get("missing", None),
                tags=info.get("tags", []),
                size=info.get("size", None),
                islocal=islocal
            )

        model = QStandardItemModel(self)
        model.setHorizontalHeaderLabels(HEADER)

        current_index = -1
        for i, (prefix, filename) in enumerate(allkeys):
            datainfo = info(prefix, filename)
            item1 = QStandardItem()
            item1.setData(" " if datainfo.islocal else "", Qt.DisplayRole)
            item1.setData(datainfo, Qt.UserRole)
            item2 = QStandardItem(datainfo.title)
            item3 = QStandardItem()
            item3.setData(datainfo.size, Qt.DisplayRole)
            item4 = QStandardItem()
            item4.setData(datainfo.instances, Qt.DisplayRole)
            item5 = QStandardItem()
            item5.setData(datainfo.variables, Qt.DisplayRole)
            item6 = QStandardItem()
            item6.setData(datainfo.target, Qt.DisplayRole)
            item6.setIcon(variable_icon(datainfo.target))
            item7 = QStandardItem()
            item7.setData(", ".join(datainfo.tags), Qt.DisplayRole)
            row = [item1, item2, item3, item4, item5, item6, item7]
            model.appendRow(row)

            if (prefix, filename) == self.selected_id:
                current_index = i

        hs = self.view.header().saveState()
        model_ = self.view.model()
        self.view.setModel(model)
        self.view.header().restoreState(hs)
        model_.deleteLater()
        model_.setParent(None)
        self.view.selectionModel().selectionChanged.connect(
            self.__on_selection
        )
        # Update the info text
        self.infolabel.setText(
            "{} datasets \n{} datasets cached"
            .format(model.rowCount(), len(allinfolocal)))

        if current_index != -1:
            selmodel = self.view.selectionModel()
            selmodel.select(
                model.index(current_index, 0),
                QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)

    def __update_cached_state(self):
        model = self.view.model()
        localinfo = list_local()
        assert isinstance(model, QStandardItemModel)
        allinfo = []
        for i in range(model.rowCount()):
            item = model.item(i, 0)
            info = item.data(Qt.UserRole)
            info.islocal = (info.prefix, info.filename) in localinfo
            item.setData(" " if info.islocal else "", Qt.DisplayRole)
            allinfo.append(info)

        self.infolabel.setText(
            "{} datasets\n{} datasets cached"
            .format(model.rowCount(), sum(info.islocal for info in allinfo))
        )

    def selected_dataset(self):
        """
        Return the current selected data set info or None if not selected

        Returns
        -------
        info : Optional[namespace]
        """
        rows = self.view.selectionModel().selectedRows(0)
        assert 0 <= len(rows) <= 1
        current = rows[0] if rows else None  # type: Optional[QModelIndex]
        if current is not None:
            info = current.data(Qt.UserRole)
            assert isinstance(info, namespace)
        else:
            info = None
        return info

    def __on_selection(self):
        # Main data sets view selection has changed
        rows = self.view.selectionModel().selectedRows(0)
        assert 0 <= len(rows) <= 1
        current = rows[0] if rows else None  # type: Optional[QModelIndex]
        if current is not None:
            di = current.data(Qt.UserRole)
            text = description_html(di)
            self.descriptionlabel.setText(text)
            self.selected_id = (di.prefix, di.filename)
        else:
            self.descriptionlabel.setText("")
            self.selected_id = None

        self.commit()

    def commit(self):
        """
        Commit a dataset to the output immediately (if available locally) or
        schedule download background and an eventual send.

        During the download the widget is in blocking state
        (OWWidget.isBlocking)
        """
        di = self.selected_dataset()
        if di is not None:
            self.error("")

            if self.__awaiting_state is not None:
                # disconnect from the __commit_complete
                self.__awaiting_state.watcher.done.disconnect(
                    self.__commit_complete)
                # .. and connect to update_cached_state
                # self.__awaiting_state.watcher.done.connect(
                #     self.__update_cached_state)
                # TODO: There are possible pending __progress_advance queued
                self.__awaiting_state.pb.advance.disconnect(
                    self.__progress_advance)
                self.progressBarFinished(processEvents=None)
                self.__awaiting_state = None

            if not di.islocal:
                pr = progress()
                callback = lambda pr=pr: pr.advance.emit()
                pr.advance.connect(self.__progress_advance, Qt.QueuedConnection)

                self.progressBarInit(processEvents=None)
                self.setStatusMessage("Fetching...")
                self.setBlocking(True)

                f = self._executor.submit(
                    ensure_local, di.prefix, di.filename,
                    progress_advance=callback)
                w = FutureWatcher(f, parent=self)
                w.done.connect(self.__commit_complete)
                self.__awaiting_state = _FetchState(f, w, pr)
            else:
                self.setStatusMessage("")
                self.setBlocking(False)
                self.commit_cached(di.prefix, di.filename)
        else:
            self.send("Data", None)

    @Slot(object)
    def __commit_complete(self, f):
        # complete the commit operation after the required file has been
        # downloaded
        assert QThread.currentThread() is self.thread()
        assert self.__awaiting_state is not None
        assert self.__awaiting_state.future is f

        if self.isBlocking():
            self.progressBarFinished(processEvents=None)
            self.setBlocking(False)
            self.setStatusMessage("")

        self.__awaiting_state = None

        try:
            path = f.result()
        except Exception as ex:
            log = logging.getLogger(__name__)
            log.exception("Error:")
            self.error(format_exception(ex))
            path = None

        self.__update_cached_state()

        if path is not None:
            data = Orange.data.Table(path)
        else:
            data = None
        self.send("Data", data)

    def commit_cached(self, prefix, filename):
        path = LocalFiles(local_cache_path()).localpath(prefix, filename)
        self.send("Data", Orange.data.Table(path))

    @Slot()
    def __progress_advance(self):
        assert QThread.currentThread() is self.thread()
        self.progressBarAdvance(1, processEvents=None)

    def onDeleteWidget(self):
        super().onDeleteWidget()
        if self.__awaiting_state is not None:
            self.__awaiting_state.watcher.done.disconnect(self.__commit_complete)
            self.__awaiting_state.pb.advance.disconnect(self.__progress_advance)
            self.__awaiting_state = None

    def sizeHint(self):
        return QSize(900, 600)

    def closeEvent(self, event):
        self.splitter_state = bytes(self.splitter.saveState())
        self.header_state = bytes(self.view.header().saveState())
        super().closeEvent(event)


class FutureWatcher(QObject):
    done = Signal(object)
    _p_done_notify = Signal(object)

    def __init__(self, future, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__future = future
        self._p_done_notify.connect(self.__on_done, Qt.QueuedConnection)
        future.add_done_callback(self._p_done_notify.emit)

    @Slot(object)
    def __on_done(self, f):
        assert f is self.__future
        self.done.emit(self.__future)


class progress(QObject):
    advance = Signal()


class _FetchState(object):
    def __init__(self, future, watcher, pb):
        self.future = future
        self.watcher = watcher
        self.pb = pb


def variable_icon(name):
    if name == "categorical":
        return gui.attributeIconDict[Orange.data.DiscreteVariable()]
    elif name == "numeric":  # ??
        return gui.attributeIconDict[Orange.data.ContinuousVariable()]
    else:
        return gui.attributeIconDict[-1]


def description_html(datainfo):
    # type: (namespace) -> str
    """
    Summarize a datainfo as a html fragment.
    """
    parts = [
        ("Name", datainfo.title),
        ("Description", datainfo.description),
        ("Reference", datainfo.reference),
    ]

    parts = [(t, d) for t, d in parts if d]
    dttemplate = '<dt><b>{}</b></dt><dd>{}</dd>'

    def format_dt(t, d):
        return dttemplate.format(escape(t), escape(d))

    html = ["<dl>"] + list(format_dt(*p) for p in parts) + ["</dl>"]
    return "\n".join(html)


def main(args=None):
    if args is None:
        args = sys.argv

    app = QApplication(list(args))
    w = OWDataSets()
    w.show()
    w.raise_()
    rv = app.exec_()
    w.saveSettings()
    w.onDeleteWidget()
    return rv

if __name__ == "__main__":
    sys.exit(main())
