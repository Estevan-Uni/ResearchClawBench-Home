"""Label-free feature selection on RPE (no access to age/phase/batch except the clearly-labelled oracle)."""
import sys, json, time, os
sys.path.insert(0, '/workspace/code')
import numpy as np, pandas as pd
from core import *
X, obs, var = load_rpe()
cfgj = json.load(open('/workspace/outputs/config_frozen.json'))
rank = {}; seeds = {}
for name, c in cfgj['variants'].items():
    t0 = time.time(); r = delve(X, c)
    rank[name] = r['ranking']; seeds[name] = r['info']['seeds']
    print(name, f'{time.time()-t0:.0f}s', 'n_seeds', len(r['info']['seeds']), r['info'].get('noise_kept'), flush=True)
    if name == 'DELVE-faithful':
        pd.DataFrame({'feature': var, 'dyn_score': r['info']['dyn'], 'laplacian_seedgraph': r['info']['lap'],
                      'seed_consistency': r['info']['sel_counts']}).to_csv('/workspace/outputs/rpe_delve_faithful_scores.csv', index=False)
sc = baseline_scores(X, seed=0, n_select=30, oracle=None)
for m, v in sc.items(): rank[m] = np.argsort(-v)
# supervised oracle (reference only; uses age)
rank['oracle_rho_age'] = np.argsort(-np.abs([spearmanr(X[:, j], obs['age'])[0] for j in range(X.shape[1])]))
np.savez('/workspace/outputs/rpe_rankings.npz', **{k: v for k, v in rank.items()})
np.savez('/workspace/outputs/rpe_seeds.npz', **seeds)
df = pd.DataFrame({m: var[r[:50]] for m, r in rank.items()}); df.index.name = 'rank'
df.to_csv('/workspace/outputs/rpe_top50_features.csv')
for name, s in seeds.items():
    print(name, 'seeds:', list(var[s])[:60])
print('done')
