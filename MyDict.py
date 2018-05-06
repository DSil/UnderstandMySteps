from collections.abc import MutableMapping

class MyDict(MutableMapping):

    def __init__(self, arg=None):
        self._map = {}
        if arg is not None:
            self.update(arg)

    def __getitem__(self, key):
        try:
            if isinstance(key, tuple):
                return self._map[frozenset(key)]
            return self._map[key]
        except KeyError:
            self.__setitem__(key, set())
            return self.__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self._map[frozenset(key)] = value
        else:
            self._map[key] = value

    def __delitem__(self, key):
        if isinstance(key, tuple):
            del self._map[frozenset(key)]
        else:
            del self.map[key]

    def __iter__(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)
    
    def __str__(self):
        return str(self._map)