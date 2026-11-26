"""Load the PyG TemporalData pickle without torch_geometric by stubbing modules,
then dump arrays to outputs/data.npz."""
import sys, types, pickle
import numpy as np
import torch


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.state = state


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        cls = type(name, (_Dummy,), {})
        setattr(self, name, cls)
        return cls


for m in ['torch_geometric', 'torch_geometric.data', 'torch_geometric.data.temporal',
          'torch_geometric.data.data', 'torch_geometric.data.storage',
          'torch_geometric.data.batch', 'torch_geometric.typing']:
    sys.modules[m] = _StubModule(m)

obj = torch.load('data/NF-UNSW-NB15-v2_3d.pt', weights_only=False)
st = obj.state if hasattr(obj, 'state') else obj
print(type(obj), list(st.keys()) if isinstance(st, dict) else st)
store = st['_store']
mp = store.state['_mapping'] if hasattr(store, 'state') else store['_mapping']
out = {}
for k, v in mp.items():
    a = v.numpy() if torch.is_tensor(v) else np.asarray(v)
    out[k] = a
    print(k, a.shape, a.dtype, a[:5] if a.ndim == 1 else a[0, :8])
np.savez_compressed('outputs/data.npz', **out)
