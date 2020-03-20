from operator import itemgetter

import numpy as np

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QListView, QSizePolicy, QComboBox

from Orange.data import Table
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.widgetpreview import WidgetPreview


class OWUnique(widget.OWWidget):
    name = 'Unique'
    icon = 'icons/Unique.svg'
    description = 'Filter instances unique by specified key attribute(s).'

    class Inputs:
        data = widget.Input("Data", Table)

    class Outputs:
        data = widget.Output("Data", Table)

    want_main_area = False

    TIEBREAKERS = {'Last instance': itemgetter(-1),
                   'First instance': itemgetter(0),
                   'Middle instance': lambda seq: seq[len(seq) // 2],
                   'Random instance': np.random.choice,
                   'Discard instances with non-unique keys':
                   lambda seq: seq[0] if len(seq) == 1 else None}

    settingsHandler = settings.DomainContextHandler()
    grouping_attrs = settings.ContextSetting([])
    tiebreaker = settings.Setting(next(iter(TIEBREAKERS)))
    autocommit = settings.Setting(True)

    def __init__(self):
        # Commit is thunked because autocommit redefines it
        # pylint: disable=unnecessary-lambda
        super().__init__()
        self.data = None

        list_args = dict(
            alternatingRowColors=True,
            dragEnabled=True, dragDropMode=QListView.DragDrop, acceptDrops=True,
            defaultDropAction=Qt.MoveAction, showDropIndicator=True,
            selectionMode=QListView.ExtendedSelection,
            selectionBehavior=QListView.SelectRows)

        hbox = gui.hBox(self.controlArea)

        listview_avail = QListView(self, **list_args)
        self.model_avail = VariableListModel(parent=self, enable_dnd=True)
        listview_avail.setModel(self.model_avail)
        gui.vBox(hbox, 'Available Variables').layout().addWidget(listview_avail)

        listview_key = QListView(self, **list_args)
        self.model_key = VariableListModel(parent=self, enable_dnd=True)
        listview_key.setModel(self.model_key)
        self.model_key.rowsInserted.connect(lambda: self.commit())
        self.model_key.rowsRemoved.connect(lambda: self.commit())
        gui.vBox(hbox, 'Group by Variables').layout().addWidget(listview_key)

        gui.comboBox(
            self.controlArea, self, 'tiebreaker', box="Item Selection",
            label='Instance to select in each group:', orientation=Qt.Horizontal,
            items=tuple(self.TIEBREAKERS),
            callback=lambda: self.commit(), sendSelectedValue=True,
            sizeAdjustPolicy=QComboBox.AdjustToContents,
            minimumContentsLength=20,
            sizePolicy=(QSizePolicy.Minimum, QSizePolicy.Preferred))
        gui.auto_commit(
            self.controlArea, self, 'autocommit', 'Commit',
            orientation=Qt.Horizontal)

    def storeSpecificSettings(self):
        self.grouping_attrs = list(self.model_key)

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.data = data
        if data:
            self.openContext(data.domain)
            self.model_key[:] = self.grouping_attrs
            self.model_avail[:] = \
                [var for var in data.domain.variables + data.domain.metas
                 if var not in self.model_key]
        else:
            self.grouping_attrs = []
            self.model_key.clear()
            self.model_avail.clear()
        self.unconditional_commit()

    def commit(self):
        if self.data is None:
            self.Outputs.data.send(None)
        else:
            self.Outputs.data.send(self._compute_unique_data())

    def _compute_unique_data(self):
        uniques = {}
        keys = zip(*[self.data.get_column_view(attr)[0]
                     for attr in self.model_key])
        for i, key in enumerate(keys):
            uniques.setdefault(key, []).append(i)

        choose = self.TIEBREAKERS[self.tiebreaker]
        selection = sorted(
            x for x in (choose(inds) for inds in uniques.values())
            if x is not None)
        if selection:
            return self.data[selection]
        else:
            return None


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWUnique).run(Table("iris"))
