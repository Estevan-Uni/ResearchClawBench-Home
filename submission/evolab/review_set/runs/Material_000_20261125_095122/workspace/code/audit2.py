import sys; sys.path.insert(0, '/workspace/code')
import numpy as np, pandas as pd, collections
from common import *
D = load_cache()
for s in ['pretrain', 'finetune', 'candidate']:
    nonsym = []; multi = 0
    for i, g in enumerate(D[s]):
        ei = g['ei'].numpy(); pairs = [tuple(sorted(p)) for p in ei.T]
        c = collections.Counter(pairs)
        if max(c.values(), default=0) > 2: multi += 1; nonsym.append(i)
    y = np.array([g['y'] for g in D[s]])
    print(s, 'graphs with multi-edges (same pair >1 bond):', multi, 'pos rate among them', y[nonsym].mean() if nonsym else None, 'overall', y.mean())
g = D['pretrain'][[i for i, g in enumerate(D['pretrain']) if max(collections.Counter(tuple(sorted(p)) for p in g['ei'].numpy().T).values(), default=0) > 2][0]]
print(g['ei'], g['ea'])
for s in ['pretrain', 'finetune', 'candidate']:
    d = pd.read_csv(f'outputs/descriptors_{s}.csv')
    print(s, d.groupby('n_atoms').y.agg(['mean', 'size']).round(3).T.to_string())
