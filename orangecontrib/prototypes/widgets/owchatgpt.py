from AnyQt.QtWidgets import QPlainTextEdit

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from orangecontrib.prototypes.widgets.owchatgptbase import OWChatGPTBase, \
    run_gpt, MODELS


class OWChatGPT(OWChatGPTBase):
    name = "ChatGPT Summarize"
    description = "Summarize content using a ChatGPT."
    icon = "icons/chatgpt.svg"
    priority = 10
    keywords = ["text", "gpt"]

    auto_apply = Setting(True)

    def __init__(self):
        self.__answer_text_edit: QPlainTextEdit = None
        super().__init__()

    def setup_gui(self):
        super().setup_gui()
        box = gui.vBox(self.mainArea, "Answer")
        self.__answer_text_edit = QPlainTextEdit(readOnly=True)
        box.layout().addWidget(self.__answer_text_edit)

    def set_data(self, data: Table):
        super().set_data(data)
        self.commit.now()

    def on_done(self, answer: str):
        self.__answer_text_edit.setPlainText(answer)

    def ask_gpt(self, state) -> str:
        if not self._data or not self.text_var or not self.access_key:
            return ""

        texts = self._data.get_column(self.text_var)
        text = "\n".join(texts)

        state.set_progress_value(4)
        state.set_status("Thinking...")
        if state.is_interruption_requested():
            raise Exception

        return run_gpt(self.access_key, MODELS[self.model_index],
                       text, self.prompt_start, self.prompt_end)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWChatGPT).run(set_data=Table("zoo"))
