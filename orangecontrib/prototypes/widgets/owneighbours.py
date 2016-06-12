import numpy as np
from sklearn.neighbors import DistanceMetric
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget


class OWNeighbours(OWWidget):
    name = "Neighbours"
    description = "Compute a matrix of pairwise distances."
    icon = "icons/Neighbours.svg"

    inputs = [("Data", Table, "set_data"), ("Seed", Table, "set_seed")]
    outputs = [("Data", Table)]

    k = Setting(10)
    autocommit = Setting(True)

    want_main_area = False
    buttons_area_orientation = Qt.Vertical

    data_info_default = "No data on input."
    seed_info_default = "No seed on input."

    def __init__(self):
        super().__init__()

        self.data = None
        self.seed = None
        box = gui.vBox(self.controlArea, "Info")
        self.data_info_label = gui.widgetLabel(box, self.data_info_default)
        self.seed_info_label = gui.widgetLabel(box, self.seed_info_default)

        box = gui.vBox(self.controlArea, "Outputing k instances")
        self.k_spin = gui.spin(box, self, "k", label="k:", step=1,
                               spinType=int, minv=0, maxv=100,
                               callback=self.k_changed)

        box = gui.auto_commit(self.buttonsArea, self, "autocommit", "Apply",
                              box=False, checkbox_label="Apply automatically")
        box.layout().insertSpacing(1, 8)
        self.layout().setSizeConstraint(self.layout().SetFixedSize)

    def set_data(self, data):
        text = self.data_info_default if data is None \
            else "{} data instances on input.".format(len(data))
        self.data = data
        self.data_info_label.setText(text)
        self.commit()

    def set_seed(self, seed):
        text = self.seed_info_default if seed is None \
            else "{} seed instances on input.".format(len(seed))
        self.seed = seed
        self.seed_info_label.setText(text)
        self.commit()

    def k_changed(self):
        self.commit()

    def commit(self):
        if self.data is None or self.seed is None:
            self.send("Data", None)
            return
        dist = DistanceMetric.get_metric('euclidean')
        new = dist.pairwise(np.vstack((self.data, self.seed)))[:len(self.data),
              len(self.data):]
        l = list(np.argsort(new.flatten()))[::-1]
        s = set()
        while len(l) > 0 and len(s) < self.k:
            s.add(int(l.pop() / len(self.seed)))
        print(s)
        neighbours = self.data[list(s)]
        self.send("Data", neighbours)


if __name__ == "__main__":
    a = QApplication([])
    ow = OWNeighbours()
    ow.show()
    ow.set_data(Table("iris"))
    ow.set_seed(Table("iris")[:10])
    ow.raise_()
    a.exec_()
    ow.saveSettings()
