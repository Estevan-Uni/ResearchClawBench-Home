"""Label-free diagnostics of seed sets (redundancy, compartment, global-intensity)."""
import sys; sys.path.insert(0, '/workspace/code')
import numpy as np, pandas as pd
from core import *
X, obs, var = load_rpe(); Xz = zscore(X)
med, lab = dedup_medoids(Xz, 0.9); gsize = pd.Series(lab).map(pd.Series(lab).value_counts()).values
S = dict(np.load('/workspace/outputs/rpe_seeds.npz')); R = dict(np.load('/workspace/outputs/rpe_rankings.npz'))
gint = Xz.mean(1)
rows = []
for name in ['DELVE-faithful', '+noise-filter', '+dedup', '+whiten']:
    s = S[name]; comp = pd.Series([v.split('_')[-1] for v in var[s]]).value_counts(normalize=True)
    Zs = pca(zscore(X[:, s]), 1)[0][:, 0]
    top = R[name][:30]
    rows.append(dict(variant=name, n_seeds=len(s), frac_seeds_in_redundant_group=float((gsize[s] > 1).mean()),
                     n_distinct_groups_in_seeds=len(np.unique(lab[s])), frac_nuc_seeds=float(comp.get('nuc', 0)),
                     abs_r_seedPC1_global_intensity=float(abs(np.corrcoef(Zs, gint)[0, 1])),
                     n_distinct_groups_top30=len(np.unique(lab[top])), frac_nuc_top30=float(np.mean([v.endswith('nuc') for v in var[top]]))))
df = pd.DataFrame(rows); df.to_csv('/workspace/outputs/rpe_seed_diagnostics.csv', index=False); print(df.round(3).to_string())
