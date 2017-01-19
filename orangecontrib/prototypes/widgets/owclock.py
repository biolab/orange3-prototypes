import numpy as np

from PyQt4.QtCore import Qt, QPoint, QDateTime, QSize, QRegExp
from PyQt4.QtGui import QApplication, QCalendarWidget, QBrush, \
    QWidget, QColor, QPainter, QLCDNumber, QPolygon, QSizePolicy, \
    QAbstractButton

from Orange.data import Table
from Orange.widgets import widget, gui
from Orange.widgets.utils.colorpalette import GradientPaletteGenerator

from orangecontrib.timeseries import Timeseries


class Calendar(QCalendarWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent,
                         dateEditEnabled=False,
                         gridVisible=True,
                         navigationBarVisible=True,
                         selectionMode=self.NoSelection,
                         verticalHeaderFormat=self.NoVerticalHeader,
                         **kwargs)
        # Hide internal prev/next buttons
        for button in self.findChildren(QAbstractButton,
                                        QRegExp('prev|next'))[:2]:
            button.setHidden(True)
        # Prevent default highlighting of selectedDate; we color highlighted
        # date *range* on our own terms in paintCell() below
        self.setStyleSheet('''
            QTableView {
                selection-background-color: white;
                selection-color: black;
            }
        ''')
        # Make header row a bit grayer
        view = self.findChild(QWidget, 'qt_calendar_calendarview')
        if view:
            pal = view.palette()
            pal.setColor(pal.AlternateBase, QColor(Qt.gray))
            view.setPalette(pal)

    def sizeHint(self):
        return QSize(300, 200)

    def setDateRange(self, date_from, date_to):
        super().setDateRange(date_from, date_to)
        self.setSelectedDate(date_to)

    def paintCell(self, painter, rect, date,
                  _HIGHLIGHTED_BRUSH=QBrush(QColor(0xFF, 0xFF, 0, 0xAA),
                                            Qt.DiagCrossPattern)):
        super().paintCell(painter, rect, date)
        if self.minimumDate() <= date <= self.maximumDate():
            painter.fillRect(rect, _HIGHLIGHTED_BRUSH)


class AnalogClock(QWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._time_from = self._time_to = None

    def sizeHint(self):
        return QSize(200, 200)

    def setTimeSpan(self, time_values):
        self._time_from = self._time_to = None
        if time_values is not None and len(time_values):
            self._time_to = time_to = QDateTime.fromMSecsSinceEpoch(1000 * time_values[-1]).toUTC()
            if len(time_values) > 1:
                time_from = QDateTime.fromMSecsSinceEpoch(1000 * time_values[0]).toUTC()
                if time_from.secsTo(time_to) < 23 * 3600:  # timespan < 23 hours
                    self._time_from = time_from
        self.update()

    def paintEvent(self, event):
        """Adapted from http://doc.qt.io/qt-5/qtwidgets-widgets-analogclock-example.html"""
        HOURHAND = QPolygon([QPoint(7, 8), QPoint(-7, 8), QPoint(0, -55)])
        MINUTEHAND = QPolygon([QPoint(7, 8), QPoint(-7, 8), QPoint(0, -87)])
        HOURCOLOR = QColor(Qt.black)
        MINUTECOLOR = QColor(0x11, 0x11, 0x11, 0xAA)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        SIDE = 200
        side = min(self.width(), self.height())
        painter.scale(side / SIDE, side / SIDE)

        # Background (night/day)
        if self._time_to is not None:
            time = self._time_to.time()
            hour_offset = time.hour() + time.minute() / 60
            DAY, NIGHT = QColor(Qt.white), QColor('#5555ff')
            if 7 <= hour_offset <= 19:
                background = DAY
            elif 6 <= hour_offset <= 7:
                palette = GradientPaletteGenerator(NIGHT, DAY)
                background = palette[(hour_offset - 6) / (7 - 6)]
            elif 19 <= hour_offset <= 20:
                palette = GradientPaletteGenerator(DAY, NIGHT)
                background = palette[(hour_offset - 19) / (20 - 19)]
            else:
                assert hour_offset < 7 or hour_offset > 20
                background = NIGHT
            painter.setBrush(QBrush(background))
            painter.setPen(HOURCOLOR)
            painter.drawEllipse(-SIDE / 2, -SIDE / 2, SIDE, SIDE)

        # Minute tickmarks
        painter.save()
        painter.setPen(MINUTECOLOR)
        for j in range(60):
            painter.drawLine(94, 0, 97, 0)
            painter.rotate(6)
        painter.restore()

        # Hour tickmarks
        painter.save()
        painter.setPen(HOURCOLOR)
        for _ in range(12):
            painter.drawLine(88, 0, 98, 0)
            painter.rotate(30)
        painter.restore()

        # Hour span
        if self._time_from is not None:
            time_from = self._time_from.time()
            time_to = self._time_to.time()
            if time_from.secsTo(time_to) / 3600 > .2:  # Don't draw really small intervals
                hour_from = (time_from.hour() + time_from.minute() / 60) % 12 - 3
                hour_to = (time_to.hour() + time_to.minute() / 60) % 12 - 3
                startAngle = -hour_to * 30 * 16
                spanAngle = -hour_from * 30 * 16 - startAngle
                color = QColor(0xFF, 0xFF, 0, 0xAA)
                painter.save()
                painter.setBrush(QBrush(color, Qt.DiagCrossPattern))
                painter.setPen(color.darker(180))
                painter.drawPie(-SIDE / 2, -SIDE / 2, SIDE, SIDE, startAngle, spanAngle)
                painter.restore()

        # Hour and minute hand
        if self._time_to is not None:
            time = self._time_to.time()

            painter.setPen(Qt.NoPen)

            painter.save()
            painter.setBrush(HOURCOLOR)
            painter.rotate(30 * (time.hour() + time.minute() / 60))
            painter.drawConvexPolygon(HOURHAND)
            painter.restore()

            painter.save()
            painter.setBrush(MINUTECOLOR)
            painter.rotate(6 * (time.minute() + time.second() / 60))
            painter.drawConvexPolygon(MINUTEHAND)
            painter.restore()


class DigitalClock(QLCDNumber):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, digitCount=8, **kwargs)

    def sizeHint(self):
        return QSize(200, 50)

    def setTime(self, time_value):
        text = '--:--:--'
        if time_value is not None:
            time = QDateTime.fromMSecsSinceEpoch(1000 * time_value).toUTC().time()
            text = time.toString("hh:mm:ss")
        self.display(text)



class OWClock(widget.OWWidget):
    name = 'Clock'
    icon = 'icons/Clock.svg'
    description = 'Calendar and clock showing a time (span).'

    inputs = [('Data', Table, 'set_data')]

    want_main_area = False

    def __init__(self):
        hbox = gui.hBox(self.controlArea)

        vbox = gui.vBox(hbox)
        self.calendar = Calendar(self)
        vbox.layout().addWidget(self.calendar)

        vbox = gui.vBox(hbox)
        EXPANDING = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.digital_clock = DigitalClock(self, sizePolicy=EXPANDING)
        self.analog_clock = AnalogClock(self, sizePolicy=EXPANDING)
        vbox.layout().addWidget(self.digital_clock, 1)
        vbox.layout().addWidget(self.analog_clock, 4)
        self.clear()

    def set_data(self, data):
        if data is None or not len(data):
            self.clear()
            return
        data = Timeseries.from_data_table(data)
        if not data.time_variable:
            self.clear()
            return
        time_values = np.sort(data.time_values)

        date_to = date_from = QDateTime.fromMSecsSinceEpoch(1000 * time_values[-1]).toUTC().date()
        if len(time_values) > 1:
            date_from = QDateTime.fromMSecsSinceEpoch(1000 * time_values[0]).toUTC().date()
        self.calendar.setDateRange(date_from, date_to)
        self.calendar.update()

        self.analog_clock.setTimeSpan(time_values)
        self.digital_clock.setTime(time_values[-1])

        self.calendar.setDisabled(False)
        self.digital_clock.setDisabled(False)

    def clear(self):
        self.analog_clock.setTimeSpan(None)
        self.digital_clock.setTime(None)
        self.calendar.setDisabled(True)
        self.digital_clock.setDisabled(True)


if __name__ == '__main__':
    app = QApplication([])
    w = OWClock()
    w.show()
    w.set_data(Table('/tmp/tmp2.tab')[:600])
    app.exec()
