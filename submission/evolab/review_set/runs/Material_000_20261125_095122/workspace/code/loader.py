"""Safe loader for the custom-pickled PyG crystal-graph files.

The .pt files were pickled with a missing class `data_prepare.RealisticCrystalDataset`
and torch_geometric Data objects. We use a restricted Unpickler that only allows
tensor-rebuild globals and maps the unknown classes to inert stubs.
"""
import pickle, collections, types
import numpy as np
import torch

ELEMENTS = ['Fe','Co','Ni','Mn','Cr','V','Ti','Nd','Pr','Sm','Gd','Ho','Er','Yb',
            'O','F','Cl','Br','I','S','Se','Te','B','C','N','P','Si','H']
N_MAG = 14

class Stub:
    def __init__(self, *a, **k):
        self._args = a; self._kw = k; self._state = None
    def __setstate__(self, state):
        self._state = state
    def __call__(self, *a, **k):
        return Stub(*a, **k)

def _stub_factory(mod, name):
    return type(f"Stub_{mod.replace('.','_')}_{name}", (Stub,), {})

ALLOWED = {('torch._utils', '_rebuild_tensor_v2'), ('torch._utils', '_rebuild_tensor'),
           ('collections', 'OrderedDict'), ('torch._utils', '_rebuild_parameter')}

class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return getattr(__import__(module, fromlist=[name]), name)
        if module == 'torch' and name.endswith('Storage'):
            return getattr(torch, name)
        if module.startswith('data_prepare') or module.startswith('torch_geometric'):
            return _stub_factory(module, name)
        raise pickle.UnpicklingError(f'forbidden global {module}.{name}')

shim = types.ModuleType('shim')
shim.Unpickler = RestrictedUnpickler
shim.load = lambda f, **kw: RestrictedUnpickler(f, **kw).load()
shim.__name__ = 'pickle'

def _dig(obj):
    """Return the raw state of a stub (dict) recursively for Data objects."""
    return obj._state if isinstance(obj, Stub) else obj

def load_raw(path):
    return torch.load(path, pickle_module=shim, weights_only=False, map_location='cpu')

def load_graphs(path):
    raw = load_raw(path)
    st = _dig(raw)
    meta = {k: v for k, v in st.items() if k != 'data_list'} if isinstance(st, dict) else {}
    graphs = []
    for d in st['data_list']:
        s = _dig(d)
        store = _dig(s['_store']) if '_store' in s else s
        m = store['_mapping'] if '_mapping' in store else store
        g = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in m.items()}
        graphs.append(g)
    return graphs, meta

if __name__ == '__main__':
    for f in ['pretrain', 'finetune', 'candidate']:
        gs, meta = load_graphs(f'/workspace/data/{f}_data.pt')
        print(f, len(gs), {k: (v if not torch.is_tensor(v) else v.shape) for k, v in meta.items()})
        g = gs[0]; print({k: (tuple(v.shape), v.dtype) for k, v in g.items()})
        print(g['x'][:3], g['edge_index'][:, :6], g['edge_attr'][:6], g['y'])
