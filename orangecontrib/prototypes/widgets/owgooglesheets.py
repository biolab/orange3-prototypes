import logging

from PyQt4.QtCore import QTimer, Qt
from PyQt4.QtGui import QApplication, QComboBox, QLabel

from Orange.util import try_
from Orange.widgets.utils.itemmodels import PyListModel
from Orange.widgets import widget, gui, settings
from Orange.data.table import Table

log = logging.getLogger(__name__)


class URLComboBox(QComboBox):

    class Model(PyListModel):
        TitleRole = Qt.UserRole + 1

        def data(self, index, role=Qt.DisplayRole):
            super_data = super().data(index, role)
            if role == Qt.DisplayRole:
                title = super().data(index, self.TitleRole)
                if title:
                    return '{} ({})'.format(title, super_data)
            return super_data

    def __init__(self, parent, model_list, **kwargs):
        super().__init__(parent, **kwargs)
        self.setModel(self.Model(iterable=model_list, parent=self))

    def setTitleFor(self, i, title):
        self.model().setData(self.model().index(i, 0), title, self.Model.TitleRole)


class OWGoogleSheets(widget.OWWidget):
    name = "Google Sheets"
    description = "Read data from a Google Sheets spreadsheet."
    icon = "icons/GoogleSheets.svg"
    priority = 20
    outputs = [("Data", Table)]

    want_main_area = False
    resizing_enabled = False

    recent = settings.Setting([])
    autocommit = settings.Setting(True)

    def __init__(self):
        super().__init__()
        self.table = None

        timer = QTimer(self,
                       singleShot=True,
                       timeout=self.load_url)
        hb = gui.widgetBox(self.controlArea, 'Google Sheets', orientation='horizontal')
        self.combo = combo = URLComboBox(
            hb, self.recent, editable=True, minimumWidth=400,
            insertPolicy=QComboBox.InsertAtTop,
            editTextChanged=lambda: timer.start(500),
            currentIndexChanged=lambda: (timer.stop(), self.load_url()))
        hb.layout().addWidget(QLabel('URL:', hb))
        hb.layout().addWidget(combo)
        hb.layout().setStretch(1, 2)
        box = gui.widgetBox(self.controlArea, "Info", addSpace=True)
        info = self.data_info = gui.widgetLabel(box, '')
        info.setWordWrap(True)
        self.controlArea.layout().addStretch(1)
        gui.auto_commit(self.controlArea, self, 'autocommit', label='Commit')

        self.set_info()

    def set_combo_items(self):
        self.combo.clear()
        for sheet in self.recent:
            self.combo.addItem(sheet.name, sheet.url)

    def commit(self):
        self.send('Data', self.table)

    class Error(widget.OWWidget.Error):
        error = widget.Msg("Couldn't load spreadsheet: {}. Ensure correct "
                           "access permissions; rectangular, top-left-aligned "
                           "sheet data ...")

    def load_url(self):
        url = self.combo.currentText()
        if not url:
            return
        try:
            with self.progressBar(3) as progress:
                progress.advance()
                table = Table.from_url(url)
                progress.advance()
        except Exception as e:
            log.exception("Couldn't load data from: %s", url)
            self.Error.error(try_(lambda: e.args[0], ''))
            self.table = None
        else:
            self.Error.clear()
            self.table = table
            self.combo.setTitleFor(self.combo.currentIndex(), table.name)
        self.set_info()
        self.commit()

    def set_info(self):
        data = self.table
        if data is None:
            self.data_info.setText('No spreadsheet loaded.')
            return
        text = "{}\n\n{} instance(s), {} feature(s), {} meta attribute(s)\n".format(
            data.name, len(data), len(data.domain.attributes), len(data.domain.metas))
        text += try_(lambda: '\nFirst entry: {}'
                             '\nLast entry: {}'.format(data[0, 'Timestamp'],
                                                       data[-1, 'Timestamp']), '')
        self.data_info.setText(text)


if __name__ == "__main__":
    a = QApplication([])
    ow = OWGoogleSheets()
    ow.show()
    a.exec_()
