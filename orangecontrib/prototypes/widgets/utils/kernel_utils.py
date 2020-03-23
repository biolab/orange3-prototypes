import io
import sys
import pickle
from typing import List, Sequence

if sys.version_info < (3, 8):
    import pickle5 as pickle


def dump(obj, ) -> List[memoryview]:
    file = io.BytesIO()
    buffers = []  # type: List[pickle.PickleBuffer]
    pickler = pickle.Pickler(
        file, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=buffers.append
    )
    pickler.dump(obj)
    return [file.getbuffer(), *(memoryview(pb) for pb in buffers)]


def load(buffers: Sequence[bytes]) -> object:
    head = buffers[0]
    buffers = buffers[1:]
    unpickler = pickle.Unpickler(io.BytesIO(head), buffers=buffers)
    return unpickler.load()
