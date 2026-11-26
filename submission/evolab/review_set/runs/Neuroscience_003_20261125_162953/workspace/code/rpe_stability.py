"""Label-free subsample stability of top-30 panels (80% cells, 3 subsamples)."""
import sys, json; sys.path.insert(0, '/workspace/code')
import numpy as np, pandas as pd
from core import *
X, obs, var = load_rpe(); cfgj = json.load(open('/workspace/outputs/config_frozen.json'))
R = dict(np.load('/workspace/outputs/rpe_rankings.npz'))
rng = np.random.default_rng(1); rows = []
for s in range(3):
    idx = rng.choice(len(X), int(.8 * len(X)), replace=False)
    for name in ['DELVE-faithful', '+dedup', '+whiten']:
        r = delve(X[idx], cfgj['variants'][name])['ranking'][:30]
        rows.append(dict(sub=s, method=name, jaccard=len(set(r) & set(R[name][:30])) / len(set(r) | set(R[name][:30]))))
    for name, sc in baseline_scores(X[idx], 0, 30).items():
        if name in ('random',): continue
        r = np.argsort(-sc)[:30]
        rows.append(dict(sub=s, method=name, jaccard=len(set(r) & set(R[name][:30])) / len(set(r) | set(R[name][:30]))))
    print(s, flush=True)
df = pd.DataFrame(rows); df.to_csv('/workspace/outputs/rpe_stability.csv', index=False)
print(df.groupby('method').jaccard.agg(['mean', 'std']).round(3)); print('done')
