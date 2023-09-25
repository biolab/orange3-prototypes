from Orange.data import Table, StringVariable
from Orange.data.util import get_unique_names
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Output

from orangecontrib.prototypes.widgets.owchatgptbase import OWChatGPTBase, \
    run_gpt, MODELS


class OWChatGPTConstructor(OWChatGPTBase):
    name = "ChatGPT Constructor"
    description = "Construct a text field using a ChatGPT."
    icon = "icons/chatgpt.svg"
    priority = 11
    keywords = ["text", "gpt"]

    cache = Setting({})
    want_main_area = False

    class Outputs:
        data = Output("Data", Table)

    def set_data(self, data: Table):
        super().set_data(data)
        self.commit.deferred()

    @gui.deferred
    def commit(self):
        super().commit()

        answers = self._get_answers()
        data = self._data
        if data is not None:
            name = get_unique_names(data.domain, "Text")
            var = StringVariable(name)
            data = data.add_column(var, answers, to_metas=True)

        self.Outputs.data.send(data)

    def _get_answers(self) -> str:
        self.Error.unknown_error.clear()
        if not self._data or not self.text_var or not self.access_key:
            return ""

        texts = self._data.get_column(self.text_var)
        answers = []
        for text in texts:
            args = (text.strip(),
                    self.prompt_start.strip(),
                    self.prompt_end.strip())
            if args in self.cache:
                answer = self.cache[args]
            else:
                try:
                    answer = run_gpt(self.access_key, MODELS[self.model_index],
                                     *args)
                    self.cache[args] = answer
                except Exception as ex:
                    answer = ex
                    self.Error.unknown_error(ex)
            answers.append(answer)
        return answers


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWChatGPTConstructor).run(set_data=Table("zoo"))
