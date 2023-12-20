from typing import List

from Orange.data import Table, StringVariable
from Orange.data.util import get_unique_names
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

    def on_done(self, answers: List[str]):
        data = self._data
        if len(answers) > 0:
            name = get_unique_names(data.domain, "Text")
            var = StringVariable(name)
            data = data.add_column(var, answers, to_metas=True)
        self.Outputs.data.send(data)

    def ask_gpt(self, state) -> List:
        if not self._data or not self.text_var or not self.access_key:
            return []

        state.set_status("Thinking...")

        texts = self._data.get_column(self.text_var)
        answers = []
        for i, text in enumerate(texts):

            state.set_progress_value(i / len(texts) * 100)
            if state.is_interruption_requested():
                raise Exception

            args = (MODELS[self.model_index],
                    text.strip(),
                    self.prompt_start.strip(),
                    self.prompt_end.strip())
            if args in self.cache:
                answer = self.cache[args]
            else:
                try:
                    answer = run_gpt(self.access_key, *args)
                    self.cache[args] = answer
                except Exception as ex:
                    answer = ex
            answers.append(answer)
        return answers


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWChatGPTConstructor).run(set_data=Table("zoo"))
