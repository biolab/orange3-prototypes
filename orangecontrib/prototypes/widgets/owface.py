import os
import tempfile

import atexit
import cv2

from Orange.data import Table, Domain, StringVariable
from Orange.widgets import widget
from Orange.widgets.settings import Setting
from Orange.widgets import gui


face_cascade_classifier = cv2.CascadeClassifier(
    os.path.join(os.path.dirname(__file__), 'data', 'haarcascade_frontalface_default.xml'))


class OWFace(widget.OWWidget):
    name = "Face Detector"
    description = "Detect and extract a face from an image."
    icon = "icons/Face.svg"
    priority = 123

    inputs = [("Data", Table, "set_data")]
    outputs = [("Data", Table)]

    auto_run = Setting(True)

    def __init__(self):
        super().__init__()
        self.data = None
        self.faces = None

        gui.auto_commit(self.controlArea, self, "auto_run", "Run",
                        checkbox_label="Run after any change",
                        orientation="horizontal")

    def find_face(self, file_path, face_path):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        faces = face_cascade_classifier.detectMultiScale(img)
        if len(faces) != 1:
            return False
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        cv2.imwrite(face_path, face)
        return True

    @staticmethod
    def cleanup(filenames):
        for fname in filenames:
            os.unlink(fname)

    def commit(self):
        face_var = StringVariable("face")
        face_var.attributes["type"] = "image"
        domain = Domain([], metas=[face_var])
        faces_list = []
        tmp_files = []
        for row in self.data:
            file_abs = str(row["image"])
            file_path, file_name = os.path.split(file_abs)
            file_name, file_ext = os.path.splitext(file_name)
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as f:
                face_abs = f.name
                tmp_files.append(face_abs)
            if self.find_face(file_abs, face_abs):
                faces_list.append([face_abs])
            else:
                faces_list.append([""])
        atexit.register(self.cleanup, tmp_files)
        self.faces = Table.from_list(domain, faces_list)
        self.send_data()

    def send_data(self):
        comb = Table.concatenate([self.data, self.faces])
        self.send("Data", comb)

    def set_data(self, data):
        self.data = data
        self.faces = None
        if not self.data:
            self.send("Data", None)
        elif self.auto_run:
            self.commit()
