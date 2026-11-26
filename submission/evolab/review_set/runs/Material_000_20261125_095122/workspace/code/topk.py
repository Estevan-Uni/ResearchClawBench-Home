import sys; sys.path.insert(0, '/workspace/code')
import numpy as np, pandas as pd
from common import *
rows = []
for tag in ['real', 'control']:
    z = np.load(f'outputs/scores_{tag}.npz'); y = z['cd_y']
    for m in ['random', 'gbdt', 'wl_lr', 'scratch', 'pre_frozen', 'pre_ft']:
        for k in (10, 20, 50, 100):
            h = hits_at(y, z[f'cand_{m}'], k); p, lo, hi = wilson(h, k)
            rows.append(dict(task=tag, method=m, k=k, hits=h, precision=p, lo=lo, hi=hi, base=y.mean()))
T = pd.DataFrame(rows); T.to_csv('outputs/topk_precision.csv', index=False)
print(T.pivot_table(index=['task', 'method'], columns='k', values='hits').to_string())
