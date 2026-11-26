"""Minimal replacement for the obsolete `attrdict` package (Python>=3.10 incompatible)."""
class AttrDict(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        for key, v in list(self.items()):
            if isinstance(v, dict) and not isinstance(v, AttrDict):
                self[key] = AttrDict(v)
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v
