from typing import Optional

from AnyQt.QtCore import QEvent, Qt, QObject, QSize
from AnyQt.QtGui import QIcon
from AnyQt.QtWidgets import QPlainTextEdit, QLineEdit, QTextEdit, \
    QToolButton, QSizePolicy

import openai
import tiktoken

from Orange.data import Table, StringVariable
from Orange.widgets import gui
from Orange.widgets.credentials import CredentialManager
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.settings import Setting, DomainContextHandler, \
    ContextSetting
from Orange.widgets.widget import OWWidget, Input, Msg

MODELS = ["gpt-3.5-turbo", "gpt-4"]


def run_gpt(
        api_key: str,
        model: str,
        text: str,
        prompt_start: str,
        prompt_end: str
) -> str:
    openai.api_key = api_key
    enc = tiktoken.encoding_for_model(model)

    text = enc.decode(enc.encode(text)[:3500])
    content = f"{prompt_start}\n{text}.\n{prompt_end}"
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]
    )
    return response.choices[0].message.content


class OWChatGPT(OWWidget):
    name = "Ask"
    description = "Ask AI language model a question."
    icon = "icons/chatgpt.svg"
    priority = 10
    keywords = ["text", "chat"]

    settingsHandler = DomainContextHandler()
    access_key = ""
    model_index = Setting(0)
    text_var = ContextSetting(None)
    prompt_start = Setting("")
    prompt_end = Setting("")

    class Inputs:
        data = Input("Data", Table)

    class Warning(OWWidget.Warning):
        missing_key = Msg("The Access key is missing.")
        missing_str_var = Msg("Data has no text variables.")

    class Error(OWWidget.Error):
        unknown_error = Msg("An error occurred while creating an answer.\n{}")

    def __init__(self):
        super().__init__()
        self.__data: Optional[Table] = None
        self.__text_var_model = DomainModel(valid_types=(StringVariable,))
        self.__start_text_edit: QTextEdit = None
        self.__end_text_edit: QTextEdit = None
        self.__answer_text_edit: QPlainTextEdit = None

        self.__cm = CredentialManager("Ask")
        self.access_key = self.__cm.access_key or ""

        self.setup_gui()
        self.setFocus()

    def setup_gui(self):
        box = gui.vBox(self.controlArea, "Model")
        edit = gui.lineEdit(box, self, "access_key", "API Key:",
                            orientation=Qt.Horizontal,
                            callback=self.__on_access_key_changed)
        edit.setEchoMode(QLineEdit.Password)
        gui.comboBox(box, self, "model_index", label="Model:",
                     orientation=Qt.Horizontal, items=MODELS)

        gui.comboBox(self.controlArea, self, "text_var", "Data",
                     "Text variable:", model=self.__text_var_model,
                     orientation=Qt.Horizontal)

        box = gui.hBox(self.controlArea, "Prompt")
        vbox = gui.vBox(box)
        vbox.setContentsMargins(0, 0, 0, 0)
        self.__start_text_edit = QTextEdit(tabChangesFocus=True)
        self.__start_text_edit.setText(self.prompt_start)
        self.__start_text_edit.setMinimumSize(200, 70)
        self.__start_text_edit.setSizePolicy(QSizePolicy.Expanding,
                                             QSizePolicy.Expanding)
        self.__start_text_edit.textChanged.connect(
            self.__on_start_text_changed)
        self.__start_text_edit.installEventFilter(self)
        self.setFocusProxy(self.__start_text_edit)

        self.__end_text_edit = QTextEdit(tabChangesFocus=True)
        self.__end_text_edit.setText(self.prompt_end)
        self.__end_text_edit.setMinimumSize(200, 70)
        self.__end_text_edit.setSizePolicy(QSizePolicy.Expanding,
                                           QSizePolicy.Expanding)
        self.__end_text_edit.textChanged.connect(self.__on_end_text_changed)
        self.__end_text_edit.installEventFilter(self)

        gui.label(vbox, self, "Start:")
        vbox.layout().addWidget(self.__start_text_edit)
        gui.label(vbox, self, "End:")
        vbox.layout().addWidget(self.__end_text_edit)

        vbox = gui.vBox(box)
        vbox.setContentsMargins(0, 0, 0, 0)
        style = self.style()
        icon = QIcon(style.standardIcon(style.SP_CommandLink))
        button = QToolButton()
        button.setIcon(icon)
        button.clicked.connect(self.__on_button_clicked)
        gui.rubber(vbox)
        vbox.layout().addWidget(button)

        gui.rubber(self.controlArea)

        box = gui.vBox(self.mainArea, "Answer")
        self.__answer_text_edit = QPlainTextEdit(readOnly=True)
        box.layout().addWidget(self.__answer_text_edit)

    def __on_start_text_changed(self):
        self.prompt_start = self.__start_text_edit.toPlainText()

    def __on_end_text_changed(self):
        self.prompt_end = self.__end_text_edit.toPlainText()

    def __on_access_key_changed(self):
        self.__cm.access_key = self.access_key

    def __on_button_clicked(self):
        self.ask()

    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:
        if ev.type() == QEvent.KeyPress:
            if ev.key() == Qt.Key_Return:
                if ev.modifiers() != Qt.ShiftModifier:
                    self.ask()
                    return True
        return False

    @Inputs.data
    def set_data(self, data: Table):
        self.closeContext()
        self.clear_messages()
        self.__data = data
        self.__text_var_model.set_domain(data.domain if data else None)
        self.text_var = self.__text_var_model[0] if self.__text_var_model \
            else None
        if data and not self.__text_var_model:
            self.Warning.missing_str_var()
        self.openContext(data)

    def ask(self):
        self.Warning.missing_key.clear()
        if self.access_key == "":
            self.Warning.missing_key()
        self.__answer_text_edit.setPlainText(self._get_answer())

    def _get_answer(self) -> str:
        self.Error.unknown_error.clear()
        if not self.__data or not self.text_var or not self.access_key:
            return ""

        texts = self.__data.get_column(self.text_var)
        text = "\n".join(texts)
        try:
            answer = run_gpt(self.access_key, MODELS[self.model_index],
                             text, self.prompt_start, self.prompt_end)
        except Exception as ex:
            answer = ""
            self.Error.unknown_error(ex)
        return answer

    def sizeHint(self):
        return QSize(800, 450)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWChatGPT).run(set_data=Table("zoo"))
