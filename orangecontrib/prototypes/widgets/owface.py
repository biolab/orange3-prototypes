import os
import tempfile

import atexit
import urllib

import cv2
import numpy as np
import sys

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
        self.img_attr = None
        self.faces = None

        haarcascade = os.path.join(os.path.dirname(__file__), 'data/haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(haarcascade)

        box = gui.vBox(self.controlArea, "Info")
        self.info = gui.widgetLabel(box, "No data.")

        gui.auto_commit(self.controlArea, self, "auto_run", "Run",
                        checkbox_label="Run after any change",
                        orientation="horizontal")

    def get_ext(self, file_path):
        """Find the extension of a file or url."""
        if not os.path.isfile(file_path):
            file_path = urllib.parse.urlparse(file_path).path
        return os.path.splitext(file_path)[1].strip().lower()

    def read_img(self, file_path):
        """Read an image from file or url and convert it to grayscale."""
        try:
            if os.path.isfile(file_path):
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            else:
                res = urllib.request.urlopen(file_path)
                arr = np.asarray(bytearray(res.read()), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            return img
        except:
            return None

    def find_face(self, file_path, face_path):
        """Find the face in image file_path and store it in face_path."""
        img = self.read_img(file_path)
        if img is None:
            return False
        # downscale to a reasonable size (long edge <= 1024)
        f = min(1024/img.shape[0], 1024/img.shape[1], 1)
        img = cv2.resize(img, None, fx=f, fy=f)
        faces = self.face_cascade.detectMultiScale(img)
        if len(faces) == 0:
            return False
        x, y, w, h = max(faces, key=lambda xywh: xywh[2] * xywh[3])
        face = img[y:y+h, x:x+w]
        cv2.imwrite(face_path, face)
        return True

    @staticmethod
    def cleanup(filenames):
        for fname in filenames:
            os.unlink(fname)

    def commit(self):
        if self.img_attr is None:
            self.send("Data", self.data)
            return
        face_var = StringVariable("face")
        face_var.attributes["type"] = "image"
        domain = Domain([], metas=[face_var])
        faces_list = []
        tmp_files = []
        n_faces = 0
        for row in self.data:
            file_abs = str(row[self.img_attr])
            file_ext = self.get_ext(file_abs)
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as f:
                face_abs = f.name
                tmp_files.append(face_abs)
            if self.find_face(file_abs, face_abs):
                faces_list.append([face_abs])
                n_faces += 1
            else:
                faces_list.append([""])
        atexit.register(self.cleanup, tmp_files)
        self.info.setText("Detected %d faces." % n_faces)

        self.faces = Table.from_list(domain, faces_list)
        comb = Table.concatenate([self.data, self.faces])
        self.send("Data", comb)

    def set_data(self, data):
        self.data = data
        self.faces = None
        if not self.data:
            self.info.setText("No data.")
            self.send("Data", None)
            return
        atts = [a for a in data.domain.metas if a.attributes.get("type") == "image"]
        self.img_attr = atts[0] if atts else None
        if not self.img_attr:
            self.info.setText("No image attribute.")
        else:
            self.info.setText("Image attribute: %s" % str(self.img_attr))
        if self.auto_run:
            self.commit()
