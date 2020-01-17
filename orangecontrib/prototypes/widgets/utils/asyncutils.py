import os
import asyncio

from AnyQt.QtCore import QCoreApplication, QThread


def get_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get the asyncio.AbstractEventLoop for the main Qt application thread.

    The QCoreApplication instance must already have been created.
    Must only be called from the main Qt application thread.
    """
    try:
        # Python >= 3.7
        get_running_loop = asyncio.get_running_loop
    except AttributeError:
        get_running_loop = asyncio._get_running_loop
    app = QCoreApplication.instance()
    if app is None:
        raise RuntimeError("QCoreApplication is not running")
    if app.thread() is not QThread.currentThread():
        raise RuntimeError("Called from non-main thread")
    try:
        loop = get_running_loop()
    except RuntimeError:
        loop = None
    else:
        if loop is not None:
            return loop

    if os.environ.get("QT_API") == "pyqt5":
        os.environ["QT_API"] = "PyQt5"
    import qasync

    class EventLoop(qasync.QEventLoop):
        def __init__(self, app, *args, **kwargs):
            super().__init__(app, *args, **kwargs)
            app.__qasync_loop = self
            app.aboutToQuit.connect(self.close)

        def close(self):
            app = QCoreApplication.instance()
            if app is not None:
                app.__qasync_loop = None
                try:
                    app.aboutToQuit.disconnect(loop.close)
                except (RuntimeError, TypeError):
                    pass
            super().close()
    if loop is None:
        loop = EventLoop(app)
    return loop
