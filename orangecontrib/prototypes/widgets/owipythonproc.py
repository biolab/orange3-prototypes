import sys
import asyncio

from concurrent.futures import Future
from typing import List, Awaitable, TypeVar, Optional, Dict
from typing_extensions import TypedDict

from AnyQt.QtCore import QSize, Qt
from AnyQt.QtGui import QKeySequence, QStandardItem
from AnyQt.QtWidgets import QApplication, QSplitter, QAction

from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.client import QtKernelClient
from qtconsole.manager import QtKernelManager

from orangewidget.widget import OWBaseWidget, Msg, Input, Output
from orangewidget.settings import Setting

from orangecontrib.prototypes.widgets.utils import remove_ansi_control
from orangecontrib.prototypes.widgets.utils.editor import (
    TextEditShortcutFilter, PythonCodeEditor, qshortcut
)
from orangecontrib.prototypes.widgets.utils.spinner import Spinner
from orangecontrib.prototypes.widgets.utils.asyncutils import get_event_loop
from orangecontrib.prototypes.widgets.utils.kernelutils import (
    on_reply, on_message, get_namespace, push_namespace
)


DEFAULT_SCRIPT = '''\
# The script namespace is populated with the widget's input as in_object
print("Hello World")
print("Got", in_object, "on input")
# assign results to out_object. This is extracted at the end and sent
# on widget's output.
out_object = in_object
# Can also run ipython magic commands, e.g.
%pylab inline
plot([0, 1], [0, 1])
'''


class ScriptItem(QStandardItem):
    Data = TypedDict("Data", {
        "name": str, "script": str, "path": Optional[str]
    })
    ScriptContentRole = Qt.UserRole + 40
    PathRole = ScriptContentRole + 1

    def todict(self) -> 'Data':
        return {
            "name": self.text(), "script": self.script(), "path": self.path()
        }

    @staticmethod
    def fromdict(state: 'Data') -> 'ScriptItem':
        item = ScriptItem()
        item.setText(state["name"])
        item.setScript(state["script"])
        item.setData(state["path"], ScriptItem.PathRole)
        return item

    def setScript(self, contents: str):
        self.setData(contents, ScriptItem.ScriptContentRole)

    def script(self) -> str:
        return self.data(ScriptItem.ScriptContentRole) or ""

    def setPath(self, path: str):
        self.setData(path, ScriptItem.PathRole)

    def path(self) -> str:
        return self.data(ScriptItem.PathRole) or ""


class OWIPythonConsole(OWBaseWidget):
    name = "IPython Console (Process)"
    icon = "icons/IPython.svg"
    description = "An out of process IPython console."

    class Inputs:
        object_ = Input("object", object)

    class Outputs:
        object_ = Output("object", object)

    want_control_area = False

    #: Current script content
    contents: str = Setting(DEFAULT_SCRIPT)
    #: The main area splitter state
    splitter_state: bytes = Setting(b'')
    #: The script "library" a list of recent scripts
    script_library: 'ScriptItem.Data' = Setting({}, schema_only=True)

    class Error(OWBaseWidget.Error):
        #: Error/exception running the script
        run_error = Msg("{}\n{}")

    def __init__(self):
        super().__init__()
        self._inputs = [None]
        self._loop = get_event_loop()
        # setup the kernel manager
        kernel_manager = QtKernelManager()
        # start the kernel
        kernel_manager.start_kernel(
            extra_arguments=[
                "--IPKernelApp.kernel_class="
                "orangecontrib.prototypes.widgets.kernel.IpyKernelPushGet"
            ]
        )
        self.kernel_manager = kernel_manager
        # setup and install client
        self.client: QtKernelClient = kernel_manager.client()
        self.client.start_channels()

        ipython_widget = RichJupyterWidget()
        ipython_widget.kernel_manager = kernel_manager
        ipython_widget.kernel_client = self.client
        self.script_edit = PythonCodeEditor()
        self.script_edit.setPlainText(self.contents)

        kseq = QKeySequence(Qt.ControlModifier | Qt.ShiftModifier | Qt.Key_R)
        action = QAction("Run", shortcut=kseq)
        action.triggered.connect(self.run)
        self.addAction(action)

        shf = TextEditShortcutFilter(kseq, parent=self)
        shf.activated.connect(self.run)
        self.script_edit.installEventFilter(shf)
        ipython_widget.installEventFilter(shf)

        shf = TextEditShortcutFilter(
            qshortcut("ctrl+c", macos_command_map=False), parent=self)
        shf.activated.connect(self.interrupt_kernel)
        ipython_widget.installEventFilter(shf)

        self.splitter = splitter = QSplitter(Qt.Vertical, parent=self)
        splitter.addWidget(self.script_edit)
        splitter.addWidget(ipython_widget)
        self.splitter.restoreState(self.splitter_state)

        sb = self.statusBar()
        self.__spinner = Spinner(sb, objectName="status-spinner", visible=False)

        sb.insertPermanentWidget(0, self.__spinner)

        layout = self.mainArea.layout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.layout().setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.client.iopub_channel.message_received.connect(
            self._handle_iopub_message
        )

        self.info.set_input_summary(self.info.NoInput)
        self.info.set_output_summary(self.info.NoOutput)
        self.__run_task = None  # type: Optional[asyncio.Task]

        self.settingsAboutToBePacked.connect(self.save_state)

    def _handle_iopub_message(self, msg):
        msg_type = msg["msg_type"]
        handler = getattr(self, f"_handle_{msg_type}_message", None)
        if handler is not None:
            handler(msg)

    def _handle_status_message(self, msg):
        state = msg["content"]["execution_state"]
        self.set_execution_state(state)

    def set_execution_state(self, state):
        running = state != "idle"
        self.__spinner.setVisible(running)

    def interrupt_kernel(self):
        if self.__run_task is not None:
            self.kernel_manager.interrupt_kernel()
            self.cancel()

    def sizeHint(self):
        return super().sizeHint().expandedTo(QSize(800, 600))

    @Inputs.object_
    def set_input(self, obj):
        self._inputs[0] = obj
        if obj is None:
            self.info.set_input_summary(None)
        else:
            self.info.set_input_summary(str(type(obj)))

    def handleNewSignals(self):
        self.run()

    def run(self):
        """Run the script with the current inputs."""
        self.cancel()
        ns = {"in_object": self._inputs[0]}
        script = self.script_edit.toPlainText()
        self.script_edit.document().setModified(False)

        client = self.client
        self.__run_task = task = self._loop.create_task(
            run_script(client, ns, script, ["out_object"], loop=self._loop)
        )
        task.add_done_callback(self._on_complete)
        self.setInvalidated(True)
        self.progressBarInit()

    def _on_complete(self, f: 'Future'):
        assert f is self.__run_task
        self.__run_task = None
        self.setInvalidated(False)
        self.progressBarFinished()
        try:
            ns = f.result()
            obj = ns.get("out_object", None)
        except RunScriptException as err:
            self.Error.run_error("Error", err.details or "")
            obj = None
        except Exception as err:
            self.error(str(err), )
            obj = None
            self.Outputs.object_.send(None)
            self.info.set_output_summary(None)

        self.Outputs.object_.send(obj)
        self.info.set_output_summary(str(type(obj)))

    def cancel(self):
        if self.__run_task is not None:
            self.__run_task.cancel()
            self.__run_task.remove_done_callback(self._on_complete)
            self.__run_task = None

    def onDeleteWidget(self):
        self.cancel()
        self.client.shutdown(restart=False)
        super().onDeleteWidget()

    def save_state(self) -> Dict[str, object]:
        self.splitter_state = self.splitter.saveState()
        self.contents = self.script_edit.toPlainText()

        return {
            "main-area-splitter-state": self.splitter_state,
            "current-edit-contents": self.contents
        }


T = TypeVar("T")


async def run_script(
        client: QtKernelClient, ns, code, collect_results: List[str] = [], *,
        loop=None
):
    """
    Run the `code` source in an updated namespace `ns`.

    When done collect all objects named in `collect_results` and
    extract/return them to the client.
    """
    def wrap_f(future: 'Future[T]') -> 'Awaitable[T]':
        return asyncio.wrap_future(future, loop=loop)
    await wrap_f(push_namespace(client, ns))

    msg_id = client.execute(code, silent=False, store_history=False)
    f = collect_execute_output(client, msg_id)
    res = await wrap_f(f)

    execute_reply = res["execute_reply"]
    execute_reply_content = execute_reply["content"]
    if execute_reply_content["status"] == "error":
        # script exception
        tb = remove_ansi_control(
            "".join(execute_reply_content.get('traceback', []))
        )
        raise RunScriptException(str(execute_reply), tb)

    return await wrap_f(get_namespace(client, collect_results))


def collect_execute_output(client: QtKernelClient, msg_id: str) -> Future:
    output = {
        "execute_reply": None,
        "execute_result": None,
        "display_data": [],
        "stream": [],
    }
    f = Future()

    @on_reply(client, "execute_reply", msg_id)
    def handle_reply(msg):
        output["execute_reply"] = msg
        # The reply is the last message for the execute request (?).
        # disconnect the streaming handlers.
        handle_display_data.disconnect()
        handle_stream.disconnect()
        if f.set_running_or_notify_cancel():
            f.set_result(output)

    @on_reply(client, "execute_result", msg_id)
    def handle_result(msg):
        output["result"] = msg

    @on_message(client, "display_data", msg_id, channel='iopub_channel')
    def handle_display_data(msg):
        output["display_data"].append(msg)

    @on_message(client, "stream", msg_id, channel='iopub_channel')
    def handle_stream(msg):
        output["stream"].append(msg)

    return f


class RunScriptException(Exception):
    def __init__(self, msg, details: Optional[str] = None):
        super().__init__(msg, details)
        self.details = details


def main(argv=None):
    app = QApplication(argv or [])
    w = OWIPythonConsole()
    w.setWindowTitle(w.windowTitle())
    w.show()
    w.raise_()
    w.set_input("This 1")
    w.handleNewSignals()
    app.exec()
    w.onDeleteWidget()


if __name__ == "__main__":
    main(sys.argv)
