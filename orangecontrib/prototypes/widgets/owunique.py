from collections import OrderedDict
from operator import itemgetter

import numpy as np

from AnyQt.QtCore import Qt, QTimer
from AnyQt.QtWidgets import QApplication, QListView

from Orange.data import Table
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.itemmodels import VariableListModel


class DnDListView(QListView):
    def __init__(self, callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._callback = callback

    def dropEvent(self, event):
        super().dropEvent(event)
        QTimer.singleShot(1, self._callback)


class OWUnique(widget.OWWidget):
    name = 'Unique'
    icon = 'icons/Unique.svg'
    description = 'Filter instances unique by specified key attribute(s).'

    inputs = [('Data', Table, 'set_data')]
    outputs = [('Unique Data', Table)]

    want_main_area = False

    settingsHandler = settings.DomainContextHandler()

    TIEBREAKERS = OrderedDict([('last', itemgetter(-1)),
                               ('first', itemgetter(0)),
                               ('middle', lambda seq: seq[len(seq) // 2]),
                               ('random', np.random.choice),
                               ('none (discard all instances with non-unique keys)',
                                lambda seq: seq[0] if len(seq) == 1 else None)])

    model_attrs = settings.ContextSetting(([], []))
    tiebreaker = settings.Setting(next(iter(TIEBREAKERS)))
    autocommit = settings.Setting(True)

    def __init__(self):
        hbox = gui.hBox(self.controlArea)
        _properties = dict(alternatingRowColors=True,
                           defaultDropAction=Qt.MoveAction,
                           dragDropMode=QListView.DragDrop,
                           dragEnabled=True,
                           selectionMode=QListView.ExtendedSelection,
                           selectionBehavior=QListView.SelectRows,
                           showDropIndicator=True,
                           acceptDrops=True)
        listview_avail = DnDListView(lambda: self.commit(), self, **_properties)
        self.model_avail = model = VariableListModel(parent=self, enable_dnd=True)
        listview_avail.setModel(model)

        listview_key = DnDListView(lambda: self.commit(), self, **_properties)
        self.model_key = model = VariableListModel(parent=self, enable_dnd=True)
        listview_key.setModel(model)

        box = gui.vBox(hbox, 'Available Variables')
        box.layout().addWidget(listview_avail)
        box = gui.vBox(hbox, 'Group-By Key')
        box.layout().addWidget(listview_key)

        gui.comboBox(self.controlArea, self, 'tiebreaker',
                     label='Which instance to select in each group:',
                     items=tuple(self.TIEBREAKERS.keys()),
                     callback=lambda: self.commit(),
                     sendSelectedValue=True)
        gui.auto_commit(self.controlArea, self, 'autocommit', 'Commit',
                        orientation=Qt.Horizontal)

    def set_data(self, data):
        self.data = data
        if data is None:
            self.model_avail.wrap([])
            self.model_key.wrap([])
            self.commit()
            return

        self.closeContext()
        self.model_attrs = (list(data.domain) + list(data.domain.metas), [])
        self.openContext(data.domain)

        self.model_avail.wrap(self.model_attrs[0])
        self.model_key.wrap(self.model_attrs[1])
        self.commit()

    def commit(self):
        if self.data is None:
            self.send('Unique Data', None)
            return

        uniques = OrderedDict()
        keys = zip(*[self.data.get_column_view(attr)[0]
                     for attr in self.model_key])
        for i, key in enumerate(keys):
            uniques.setdefault(key, []).append(i)

        choose = self.TIEBREAKERS[self.tiebreaker]
        selection = sorted([x for x in (choose(inds)
                                        for inds in uniques.values())
                            if x is not None])
        self.send('Unique Data', self.data[selection] if selection else None)


if __name__ == '__main__':
    app = QApplication([])
    w = OWUnique()
    w.show()
    w.set_data(Table('iris'))
    app.exec()
