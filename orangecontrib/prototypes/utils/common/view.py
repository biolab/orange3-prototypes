import numpy as np
from PyQt4 import QtGui
from PyQt4.QtCore import Qt


class ZoomableGraphicsView(QtGui.QGraphicsView):
    """Zoomable graphics view.

    Composable graphics view that adds zoom functionality.

    It also handles automatic resizing of content whenever the window is
    resized.

    Right click will reset the zoom to a factor where the entire scene is
    visible.

    Notes
    -----
      - This view will consume wheel scrolling and right mouse click events.

    """

    def __init__(self, *args, **kwargs):
        self.zoom = 1
        self.scale_factor = 1 / 16
        # zoomout limit prevents the zoom factor to become negative, which
        # results in the canvas being flipped over the x axis
        self._zoomout_limit_reached = False
        # Does the view need to recalculate the initial scale factor
        self._needs_to_recalculate_initial = True
        self._initial_zoom = -1
        super().__init__(*args, **kwargs)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._needs_to_recalculate_initial = True
        if self.zoom == -1:
            self.recalculate_and_fit()

    def wheelEvent(self, ev):
        if self.__zooming_in(ev):
            self.__reset_zoomout_limit()
        if self._zoomout_limit_reached and self.__zooming_out(ev):
            ev.accept()
            return

        self.zoom += np.sign(ev.delta()) * self.scale_factor
        if self.zoom <= 0:
            self._zoomout_limit_reached = True
            self.zoom += self.scale_factor
        else:
            self.setTransformationAnchor(self.AnchorUnderMouse)
            self.setTransform(QtGui.QTransform().scale(self.zoom, self.zoom))
        ev.accept()

    def mousePressEvent(self, ev):
        # right click resets the zoom factor
        if ev.button() == Qt.RightButton:
            self.reset_zoom()
            ev.accept()
        else:
            super().mousePressEvent(ev)

    @staticmethod
    def __zooming_out(ev):
        return ev.delta() < 0

    def __zooming_in(self, ev):
        return not self.__zooming_out(ev)

    def __reset_zoomout_limit(self):
        self._zoomout_limit_reached = False

    def recalculate_and_fit(self):
        """Recalculate the optimal zoom and fits the content into view.

        Should be called if the scene contents change, so that the optimal zoom
        can be recalculated.

        Returns
        -------

        """
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        self._initial_zoom = self.matrix().m11()
        self.zoom = self._initial_zoom

    def reset_zoom(self):
        """Reset the zoom to the optimal factor."""
        self.zoom = self._initial_zoom
        self._zoomout_limit_reached = False
        self.setTransform(QtGui.QTransform().scale(self.zoom, self.zoom))
        if self._needs_to_recalculate_initial:
            self.recalculate_and_fit()


class PannableGraphicsView(QtGui.QGraphicsView):
    """Pannable graphics view.

    Enables panning the graphics view.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)

    def enterEvent(self, ev):
        super().enterEvent(ev)
        self.viewport().setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        self.viewport().setCursor(Qt.ArrowCursor)
