from PyQt4.QtGui import QSizePolicy
from Orange.widgets import widget, gui


class OWHub(widget.OWWidget):
    name = "Hub"
    description = "Universal hub widget"
    icon = "icons/Hub.svg"

    inputs = [("Object", object, "get_input", widget.Default)]
    outputs = [("Object", object, widget.Dynamic)]

    NOTHING = "Nothing on input"

    def __init__(self):
        super().__init__()
        self.obj_type = self.NOTHING
        gui.label(self.controlArea, self, "Hubbing: %(obj_type)s")
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)

    def get_input(self, obj):
        self.obj_type = self.NOTHING if obj is None else type(obj).__name__
        self.send("Object", obj)
