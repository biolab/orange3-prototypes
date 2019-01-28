from AnyQt.QtWidgets import QListWidgetItem
from AnyQt.QtCore import Qt, QSize
from Orange.widgets import widget, gui


class OWHub(widget.OWWidget):
    name = "Hub"
    description = "Universal hub widget"
    icon = "icons/Hub.svg"

    inputs = [("Object", object, "get_input", widget.Default | widget.Multiple)]
    outputs = [("Object", object, widget.Dynamic)]

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        self.objects = {}
        self.lb_objects = gui.listBox(self.controlArea, self, box="Inputs",
                                      callback=self._on_selection_change,
                                      sizeHint=QSize(300, 300))
        self.lb_objects.setFocusPolicy(Qt.NoFocus)

    def get_input(self, obj, key):
        lb = self.lb_objects
        for item_index in range(lb.count()):
            if lb.item(item_index).data(Qt.UserRole) == key:
                break
        else:
            item_index = None

        if obj is None:
            if item_index is None:
                return
            del self.objects[key]
            lb.takeItem(item_index)
            if not self.objects:
                self.send("Object", None)
                return
        else:
            self.objects[key] = obj
            item_desc = "{} {}".format(
                type(obj).__name__, getattr(obj, "name", ""))
            if item_index is None:
                item = QListWidgetItem(item_desc)
                item.setData(Qt.UserRole, key)
                lb.addItem(item)
                lb.setCurrentItem(item)
            else:
                lb.item(item_index).setText(item_desc)
                self._on_selection_change()

    def _on_selection_change(self):
        if self.lb_objects.count() == 0:
            self.send("Object", None)
        else:
            key = self.lb_objects.currentItem().data(Qt.UserRole)
            self.send("Object", self.objects[key])
