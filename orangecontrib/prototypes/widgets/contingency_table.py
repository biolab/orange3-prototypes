from AnyQt.QtCore import Qt
from AnyQt.QtGui import QStandardItem, QColor, QFont
from AnyQt.QtWidgets import QTableView, QSizePolicy, QHeaderView, QStyledItemDelegate
from Orange.widgets import gui


BorderRole = next(gui.OrangeUserRole)
BorderColorRole = next(gui.OrangeUserRole)


class BorderedItemDelegate(QStyledItemDelegate):
    """Item delegate that paints border at the specified sides

    Data for `BorderRole` is a string containing letters t, r, b and/or l,
    which defines the sides at which the border is drawn.

    Role `BorderColorRole` sets the color for the cell. If not color is given,
    `self.color` is used as default.

    Args:
        color (QColor): default color (default default is black)
    """
    def __init__(self, color=Qt.black):
        super().__init__()
        self.color = color

    def paint(self, painter, option, index):
        """Overloads `paint` to draw borders"""
        QStyledItemDelegate.paint(self, painter, option, index)
        borders = index.data(BorderRole)
        if borders:
            color = index.data(BorderColorRole) or self.color
            painter.save()
            painter.setPen(color)
            rect = option.rect
            for side, p1, p2 in (("t", rect.topLeft(), rect.topRight()),
                                 ("r", rect.topRight(), rect.bottomRight()),
                                 ("b", rect.bottomLeft(), rect.bottomRight()),
                                 ("l", rect.topLeft(), rect.bottomLeft())):
                if side in borders:
                    painter.drawLine(p1, p2)
            painter.restore()


class ContingencyTable(QTableView):
    def __init__(self, parent, tablemodel, click_event=None):
        super().__init__(editTriggers=QTableView.NoEditTriggers)

        self.parent = parent
        self.tablemodel = tablemodel

        self.setModel(self.tablemodel)
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.horizontalHeader().setMinimumSectionSize(60)
        self.selectionModel().selectionChanged.connect(self.parent._invalidate)
        self.setShowGrid(False)
        self.setItemDelegate(BorderedItemDelegate(Qt.white))
        self.setSizePolicy(QSizePolicy.MinimumExpanding,
                           QSizePolicy.MinimumExpanding)
        if click_event:
            self.clicked.connect(click_event)

    def _item(self, i, j):
        return self.tablemodel.item(i, j) or QStandardItem()

    def _set_item(self, i, j, item):
        self.tablemodel.setItem(i, j, item)

    def initialize(self, headersv, headersh=None):
        # Maybe merge with __init__.
        # Maybe add unicodedata.lookup("N-ARY SUMMATION") automatically.
        headersh = headersh or headersv
        nclassesv = len(headersv) - 1
        nclassesh = len(headersh) - 1

        # TODO: Change "Predicted" and "Actual" to attribute names when headersh isn't None.
        item = self._item(0, 2)
        item.setData("Predicted", Qt.DisplayRole)
        item.setTextAlignment(Qt.AlignCenter)
        item.setFlags(Qt.NoItemFlags)

        self._set_item(0, 2, item)
        item = self._item(2, 0)
        item.setData("Actual", Qt.DisplayRole)
        item.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        item.setFlags(Qt.NoItemFlags)
        self.setItemDelegateForColumn(0, gui.VerticalItemDelegate())
        self._set_item(2, 0, item)
        self.setSpan(0, 2, 1, nclassesh)
        self.setSpan(2, 0, nclassesv, 1)

        font = self.tablemodel.invisibleRootItem().font()
        bold_font = QFont(font)
        bold_font.setBold(True)

        for i in (0, 1):
            for j in (0, 1):
                item = self._item(i, j)
                item.setFlags(Qt.NoItemFlags)
                self._set_item(i, j, item)

        for headers, nclasses, ix in ((headersv, nclassesv, lambda p: (p + 2, 1)),
                                      (headersh, nclassesh, lambda p: (1, p + 2))):
            for p, label in enumerate(headers):
                i, j = ix(p)
                item = self._item(i, j)
                item.setData(label, Qt.DisplayRole)
                item.setFont(bold_font)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item.setFlags(Qt.ItemIsEnabled)
                if p < nclasses:
                    item.setData("br"[j == 1], BorderRole)
                    item.setData(QColor(192, 192, 192), BorderColorRole)
                self._set_item(i, j, item)

        hor_header = self.horizontalHeader()
        if len(' '.join(headersh)) < 120:
            hor_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        else:
            hor_header.setDefaultSectionSize(60)
        self.tablemodel.setRowCount(nclassesv + 3)
        self.tablemodel.setColumnCount(nclassesh + 3)
