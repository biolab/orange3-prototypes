import sys
from contextlib import contextmanager
from typing import Mapping, Any, Sequence

from AnyQt.QtCore import QSize
from AnyQt.QtWidgets import QApplication

from ipykernel.inprocess import ipkernel

from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import (
    QtInProcessKernelManager, QtInProcessKernelClient
)

from orangewidget import widget


def swap_dict(a, b):
    """Swap the contents of dict `a` and `b` inplace."""
    tmp = a.copy()
    a.clear()
    a.update(b)
    b.clear()
    b.update(tmp)


class InProcessKernel(ipkernel.InProcessKernel):
    """
    An 'in process' kernel that sets up the shell's user namespace and
    display hooks to fake a separate execution environment.
    """
    # `InteractiveShell` (self.shell) is a singleton instance
    def __init__(self, *args, user_ns=None, **kwargs):
        super().__init__(*args, **kwargs)
        # assigning `self.user_ns` updates/initializes the self.shell.user_ns
        # with extra 'hidden' variables that are necessary for its
        # functionality
        if user_ns is None:
            self.user_ns = {}
        else:
            self.user_ns = user_ns
        # store the shells initialized user_ns for later execution setup
        self._user_ns = self.shell.user_ns.copy()
        self._user_ns_hidden = self.shell.user_ns_hidden.copy()
        # clone session (separate digest_history)
        self.session = self.session.clone()
        self.__execution_count = 1

    @property
    def execution_count(self):
        return self.__execution_count

    @execution_count.setter
    def execution_count(self, count):
        pass

    def push_user_namespace(self, ns: Mapping[str, Any]) -> None:
        """Push the `ns` namespace to available user namespace."""
        self._user_ns.update(ns)

    def get_user_namespace(self, names):
        return {name: self._user_ns.get(name) for name in names}

    def delete_user_namespace(self, which: Sequence[str]):
        """Remove the specified names from user namespace."""
        for n in which:
            self._user_ns.pop(n, None)

    def execute_request(self, stream, ident, parent):
        """Reimplemented.

        Set up the (local) execution environment/context then execute the
        request.
        """
        with self._setup_execution_context(), \
                self._redirect_display_hooks():
            super().execute_request(stream, ident, parent)

    @contextmanager
    def _setup_execution_context(self):
        """
        Patch `self.shell` to use this instances user namespace and
        display hook
        """
        user_ns = self.shell.user_ns
        user_ns_hidden = self.shell.user_ns_hidden
        kernel = self.shell.kernel
        execution_count = self.shell.execution_count
        self.shell.user_ns = self._user_ns.copy()
        self.shell.user_ns_hidden = self._user_ns_hidden.copy()
        self.shell.kernel = self
        self.shell.execution_count = self.__execution_count
        try:
            yield
        finally:
            # record changes to shell namespace
            self._user_ns = self.shell.user_ns.copy()
            self._user_ns_hidden = self.shell.user_ns_hidden.copy()
            self.__execution_count = self.shell.execution_count
            # restore the state at entry
            self.shell.kernel = kernel
            self.shell.user_ns = user_ns
            self.shell.user_ns_hidden = user_ns_hidden
            self.shell.execution_count = execution_count

            user_ns = {k: v for k, v in self._user_ns.items()
                       if k not in self._user_ns_hidden}
            # update `self.user_ns` inplace
            swap_dict(self.user_ns, user_ns)

    @contextmanager
    def _redirect_display_hooks(self):
        shell = self.shell
        session = shell.displayhook.session
        iopub_socket = shell.displayhook.pub_socket
        topic = shell.displayhook.topic
        try:
            shell.displayhook.session = self.session
            shell.displayhook.pub_socket = self.iopub_socket
            shell.displayhook.topic = self._topic('execute_result')
            shell.display_pub.session = self.session
            shell.display_pub.pub_socket = self.iopub_socket
            yield
        finally:
            shell.displayhook.session = session
            shell.displayhook.pub_socket = iopub_socket
            shell.displayhook.topic = topic
            shell.display_pub.session = session
            shell.display_pub.pub_socket = iopub_socket


class KernelManager(QtInProcessKernelManager):
    def start_kernel(self, **kwds):
        self.kernel = InProcessKernel(parent=self, session=self.session)


class OWIPythonConsole(widget.OWBaseWidget):
    name = "IPython Console"
    description = "An in process IPython console/script executor"
    icon = "icons/IPython.svg"
    inputs = [
        ("object", object, "set_input")
    ]
    outputs = [
        ("object", object)
    ]
    want_control_area = False

    def __init__(self):
        super().__init__()
        self._input = []
        self.user_namespace = {"self": self}
        kernel_manager = KernelManager()
        kernel_manager.start_kernel(
            show_banner=False, user_ns=self.user_namespace
        )
        kernel = kernel_manager.kernel
        kernel.gui = 'qt'

        kernel_client = kernel_manager.client()
        kernel_client.start_channels()

        ipython_widget = RichJupyterWidget()
        ipython_widget.kernel_manager = kernel_manager
        ipython_widget.kernel_client = kernel_client
        self.client = kernel_client  # type: QtInProcessKernelClient
        self.kernel = kernel  # type: InProcessKernel
        self.kernel.push_user_namespace({"in_object": None})
        self.mainArea.layout().addWidget(ipython_widget)

    def set_input(self, obj):
        self.kernel.push_user_namespace({"in_object": obj})

    def send_output(self, obj):
        self.send("object", obj)

    def sizeHint(self):
        return super().sizeHint().expandedTo(QSize(800, 600))

    def handleNewSignals(self):
        script = "out_object = in_object"
        self.client.execute(
            script, silent=True, store_history=False, allow_stdin=False
        )
        ns = self.kernel.get_user_namespace(["in_object"])
        self.send("object", ns.get("out_object"))

    def onDeleteWidget(self):
        self.kernel.push_user_namespace({"in_object":  None})
        super().onDeleteWidget()


def main(argv=None):
    app = QApplication(argv or [])
    w = OWIPythonConsole()
    w.setWindowTitle("Console 1")
    w.show()
    w.raise_()
    w.set_input("W1")
    w1 = OWIPythonConsole()
    w1.setWindowTitle("Console 2")
    w1.show()
    w1.raise_()
    w1.set_input("W2")
    w1.handleNewSignals()
    app.exec()


if __name__ == "__main__":
    main(sys.argv)
