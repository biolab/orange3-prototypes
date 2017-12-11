from os.path import join as _path_join

from IPython.core.interactiveshell import InteractiveShell as _InteractiveShell


class IPythonStore:
    """
    A connector for getting (one-way) items stored with IPython/Jupyter
    %store magic.

    It wraps the underlying PickleStoreDB (_db) most thinly, stripping out
    the 'autorestore/' namespace added by %store magic.
    """
    _db = _InteractiveShell.instance().db  # IPython's PickleStore
    _NAMESPACE = 'autorestore/'  # IPython StoreMagic's "namespace"

    root = _path_join(str(_db.root), _NAMESPACE)  # The root directory of the store, used for watching

    def _trim(self, key: str, _ns=_NAMESPACE):
        # _ns = self._NAMESPACE
        return key[len(_ns):] if key.startswith(_ns) else key

    def keys(self):
        return (self._trim(key)
                for key in self._db.keys())

    def items(self):
        for key in self._db.keys():
            try:
                yield self._trim(key), self._db[key]
            except KeyError:  # Object unpickleable in this env; skip
                pass

    def get(self, key: str, default=None):
        return self._db.get(self._NAMESPACE + key, self._db.get(key))

    def __getitem__(self, key):
        return self._db[self._NAMESPACE + key]

    def __delitem__(self, key):
        del self._db[self._NAMESPACE + key]

    def __contains__(self, key):
        return (self._NAMESPACE + key) in self._db

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(list(self.keys()))


if __name__ == '__main__':
    print(list(IPythonStore().items()))
