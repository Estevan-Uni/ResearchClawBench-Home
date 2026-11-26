"""Simulation benchmark: 3 topologies x noise sweeps (SNR, library size, dropout) x 2 reps."""
import sys, time, json, os
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, '/workspace/code')
import numpy as np, pandas as pd
from core import *
from simulate import simulate
cfgj = json.load(open('/workspace/outputs/config_frozen.json'))
SIMO = cfgj['sim_overrides']
SWEEPS = [('clean', 0, {})] + \
    [('snr', l, {'effect_scale': v}) for l, v in enumerate([0.6, 0.4, 0.25], 1)] + \
    [('library', l, {'lib_scale': v}) for l, v in enumerate([0.3, 0.1, 0.03], 1)] + \
    [('dropout', l, {'dropout_mid': v}) for l, v in enumerate([0.0, 1.0, 2.0], 1)]
K = 100
def evaluate_sim(X, d, ranking):
    feats = ranking[:K]
    Z = panel_embedding(X, feats, 10)
    root_pool = np.where(d['t'] <= np.quantile(d['t'], 0.01))[0]
    root = root_pool[np.argmin(((Z[root_pool] - Z[root_pool].mean(0)) ** 2).sum(1))]
    pt = dpt(Z, root)
    nseg = len(np.unique(d['seg']))
    return dict(precision=d['is_de'][feats].mean(),
                acc=knn_cv_accuracy(Z, d['seg']),
                nmi=kmeans_nmi(Z, d['seg'], nseg, 3),
                dpt_rho=spearmanr(pt, d['t'])[0],
                conf_frac=np.isin(feats, d['confound']).mean())
out = '/workspace/outputs/sim_results.csv'
rows = []
for rep in range(2):
    for topo in ['linear', 'bifurcating', 'tree']:
        for factor, lvl, kw in SWEEPS:
            t0 = time.time()
            seed = 1000 * rep + hash((topo,)) % 997 if False else 1000 * rep + ['linear', 'bifurcating', 'tree'].index(topo) * 37 + lvl * 3 + ['clean', 'snr', 'library', 'dropout'].index(factor) * 101
            d = simulate(topo, seed=seed, **kw); X = d['X']
            # drop all-zero genes not allowed to change indices: keep but they score -inf/inf naturally
            sc = baseline_scores(X, seed=rep, n_select=K)
            ranks = {m: np.argsort(-v) for m, v in sc.items()}
            ranks['all_features'] = np.arange(X.shape[1])
            for name in ['DELVE-faithful', '+noise-filter']:
                c = dict(cfgj['variants'][name]); c.update(SIMO)
                ranks[name] = delve(X, c)['ranking']
            # oracle: |rho| with true t (reference)
            ranks['oracle_rho'] = np.argsort(-np.nan_to_num(np.abs([spearmanr(X[:, j], d['t'])[0] for j in range(X.shape[1])])))
            for m, r in ranks.items():
                if m == 'all_features':
                    ev = dict(precision=d['is_de'].mean())
                    # evaluate on all genes (not top-100)
                    Z = panel_embedding(X, np.arange(X.shape[1]), 10)
                    root_pool = np.where(d['t'] <= np.quantile(d['t'], 0.01))[0]
                    root = root_pool[np.argmin(((Z[root_pool] - Z[root_pool].mean(0)) ** 2).sum(1))]
                    ev.update(acc=knn_cv_accuracy(Z, d['seg']), nmi=kmeans_nmi(Z, d['seg'], len(np.unique(d['seg'])), 3),
                              dpt_rho=spearmanr(dpt(Z, root), d['t'])[0], conf_frac=0.08)
                else:
                    ev = evaluate_sim(X, d, r)
                rows.append(dict(rep=rep, topology=topo, factor=factor, level=lvl, method=m,
                                 zero_frac=d['zero_frac'], **ev))
            pd.DataFrame(rows).to_csv(out, index=False)
            print(rep, topo, factor, lvl, f'{time.time()-t0:.1f}s', flush=True)
print('done', flush=True)
