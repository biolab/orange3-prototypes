"""
Spinner widget
--------------

An indeterminate progress indicator widget.
"""
import sys
import math
from typing import Dict, Union

from AnyQt.QtWidgets import (
    QWidget, QApplication, QStyle, QStyleOptionProgressBar, QSizePolicy
)
from AnyQt.QtGui import (
    QConicalGradient, QColor, QBrush, QPen, QPainter, QPicture, QPixmap
)
from AnyQt.QtCore import Qt, QSize, QPointF, QVariantAnimation, QEvent, QRectF


class Spinner(QWidget):
    def __init__(self, parent=None, **kwargs):
        kwargs.setdefault(
            "sizePolicy", QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        )
        super().__init__(parent, **kwargs)
        self.__picture = {}  # type: Dict[int, Union[QPicture, QPixmap]]
        self.__animation = QVariantAnimation(
            self, objectName="spinner-animation",
            startValue=0, endValue=360 // 12, duration=2000, loopCount=-1
        )
        self.__animation.valueChanged.connect(self.update)

    def showEvent(self, event):
        super().showEvent(event)
        self.__animation.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.__animation.stop()

    def sizeHint(self):
        style = self.style()
        option = QStyleOptionProgressBar()
        option.initFrom(self)
        option.minimum = option.maximum = option.progress = 0
        option.text = ""
        option.textAlignment = Qt.AlignCenter
        option.textVisible = False
        option.bottomToTop = False
        option.invertedAppearance = False
        option.orientation = Qt.Horizontal

        basesize = QSize(21, 21).expandedTo(QApplication.globalStrut())
        size = style.sizeFromContents(
            QStyle.CT_ProgressBar, option, basesize, self)
        size = size.height()
        return QSize(size, size)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return width

    def paintEvent(self, event):
        value = self.__animation.currentValue()
        rect = self.contentsRect()
        if value not in self.__picture:
            penwidth = max(4, math.ceil(math.sqrt(rect.width() * rect.height()) / 15))
            angle = value * 12
            basecolor = self.palette().highlight().color()
            pic = self.__createPicture(rect, penwidth, basecolor, -angle)
            if rect.width() * rect.height() < 4096:
                # use cached pixmaps for small sizes
                pix = QPixmap(rect.size())
                pix.fill(Qt.transparent)
                painter = QPainter(pix)
                pic.play(painter)
                painter.end()
                self.__picture[value] = pix
            else:
                self.__picture[value] = pic
        picture = self.__picture[value]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if isinstance(picture, QPicture):
            picture.play(painter)
        elif isinstance(picture, QPixmap):
            painter.drawPixmap(rect.topLeft(), picture)
        painter.end()

    def changeEvent(self, event):
        if event.type() == QEvent.ContentsRectChange:
            self.__picture = {}
        elif event.type() == QEvent.PaletteChange:
            self.__picture = {}
        elif event.type() == QEvent.EnabledChange or \
                event.type() == QEvent.ActivationChange:
            self.__picture = {}
        super().changeEvent(event)

    def resizeEvent(self, event):
        self.__picture = {}
        super().resizeEvent(event)
        self.update()

    def __createPicture(self, rect, penwidth, color, angle):
        pic = QPicture()
        painter = QPainter(pic)
        painter.setRenderHint(QPainter.Antialiasing)
        draw_spinner(painter, rect, penwidth, color, angle)
        painter.end()
        return pic


def draw_spinner(
        painter: QPainter, rect: QRectF, penwidth: float, color: QColor,
        angle: float
) -> None:
    gradient = QConicalGradient()
    color2 = QColor(color)
    color2.setAlpha(0)

    stops = [
        (0.0, color),
        (1.0, color2),
    ]
    gradient.setStops(stops)
    gradient.setCoordinateMode(QConicalGradient.ObjectBoundingMode)
    gradient.setCenter(QPointF(0.5, 0.5))
    gradient.setAngle(angle)
    pen = QPen()
    pen.setWidthF(penwidth)
    pen.setBrush(QBrush(gradient))
    painter.setPen(pen)
    painter.drawEllipse(
        rect.adjusted(penwidth / 2, penwidth / 2,
                      -penwidth / 2, -penwidth / 2))


def main(argv=None):
    app = QApplication(argv or [])
    w = Spinner()
    w.show()
    w.raise_()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
