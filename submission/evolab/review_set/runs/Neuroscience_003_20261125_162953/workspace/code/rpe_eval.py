"""evaluate(): the ONLY place labels are used (plus DPT root orientation)."""
import sys, json; sys.path.insert(0, '/workspace/code')
import numpy as np, pandas as pd
from core import *
X, obs, var = load_rpe()
R = dict(np.load('/workspace/outputs/rpe_rankings.npz'))
cfgj = json.load(open('/workspace/outputs/config_frozen.json'))
KS = cfgj['eval']['k_grid']
cyc = obs['phase'] != 'G0'

def root_of(Z, mask):
    a = obs['age'].copy(); a[~mask] = np.inf
    pool = np.where(mask & (a <= np.quantile(obs['age'][mask], 0.02)))[0]
    return pool[np.argmin(((Z[pool] - Z[pool].mean(0)) ** 2).sum(1))]

def evaluate(feats, return_pt=False):
    Z = panel_embedding(X, feats, 10)
    r = root_of(Z, cyc)
    pt_all = dpt(Z, r)
    Zc = Z[cyc]; rc = root_of(Z, cyc); rc_idx = np.where(np.where(cyc)[0] == rc)[0][0]
    pt_cyc = dpt(Zc, rc_idx)
    res = dict(acc=knn_cv_accuracy(Z, obs['phase']),
               nmi=kmeans_nmi(Z, obs['phase'], 4, 5),
               dpt_rho_all=spearmanr(pt_all, obs['age'])[0],
               dpt_rho_all_graph_cyc_cells=spearmanr(pt_all[cyc], obs['age'][cyc])[0],
               dpt_rho_cyc=spearmanr(pt_cyc, obs['age'][cyc])[0],
               batch_bal_acc=knn_cv_accuracy(Z, obs['batch'], balanced=True),
               batch_entropy=mixing_entropy(Z, obs['batch']))
    if return_pt: return res, pt_all, pt_cyc
    return res

if __name__ == '__main__':
    rows = []
    ref = evaluate(np.arange(X.shape[1])); rows.append(dict(method='all_features', k=241, **ref))
    for m, rk in R.items():
        for k in KS:
            rows.append(dict(method=m, k=k, **evaluate(rk[:k])))
        print(m, rows[-2], flush=True)
    df = pd.DataFrame(rows); df.to_csv('/workspace/outputs/rpe_metrics.csv', index=False)
    # bootstrap at k=30 for DPT rho (all cells and cycling) + paired diff vs best baseline
    pts = {}
    for m, rk in list(R.items()) + [('all_features', np.arange(X.shape[1]))]:
        feats = rk[:30] if m != 'all_features' else rk
        _, pa, pc = evaluate(feats, True); pts[m] = (pa, pc)
    rng = np.random.default_rng(0); B = 500
    n = len(X); ic = np.where(cyc)[0]
    boot = {m: ([], []) for m in pts}
    for b in range(B):
        i = rng.integers(0, n, n); j = rng.integers(0, len(ic), len(ic))
        for m, (pa, pc) in pts.items():
            boot[m][0].append(spearmanr(pa[i], obs['age'][i])[0]); boot[m][1].append(spearmanr(pc[j], obs['age'][ic][j])[0])
    boot_df = pd.DataFrame({f'{m}|{s}': v for m, (a, c) in boot.items() for s, v in (('all', a), ('cyc', c))})
    boot_df.to_csv('/workspace/outputs/rpe_bootstrap_k30.csv', index=False)
    print('done')
