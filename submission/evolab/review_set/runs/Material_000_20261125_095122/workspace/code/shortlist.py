"""Ranked top-50 shortlists (hidden labels appended for evaluation only)."""
import sys; sys.path.insert(0, '/workspace/code')
import numpy as np, pandas as pd
from common import *
D = load_cache()
def formula(g):
    c = np.bincount(g['x'].numpy(), minlength=28); return ''.join(f"{ELEMENTS[i]}{'' if c[i]==1 else c[i]}" for i in np.argsort(-c) if c[i] > 0 and i < 14) + \
        ''.join(f"{ELEMENTS[i]}{'' if c[i]==1 else c[i]}" for i in range(14, 28) if c[i] > 0)
for tag, idxf in [('real', 'idx_real_cand.npy'), ('control', 'idx_control_cand.npy')]:
    z = np.load(f'outputs/scores_{tag}.npz'); idx = np.load('outputs/' + idxf); cal = np.load(f'outputs/calibrated_{tag}_pre_ft.npy')
    s = z['cand_pre_ft']; o = np.argsort(-s, kind='stable')[:50]
    rows = []
    for r, i in enumerate(o):
        g = D['candidate'][idx[i]]
        rows.append(dict(rank=r + 1, candidate_index=int(idx[i]), formula=formula(g), magnetic_species=','.join(sorted({ELEMENTS[a] for a in g['x'].tolist() if a < 14})),
                         n_atoms=len(g['x']), p_raw=round(float(s[i]), 4), p_platt=round(float(cal[i]), 4),
                         gbdt_rank=int((z['cand_gbdt'] > z['cand_gbdt'][i]).sum() + 1), hidden_label_eval_only=int(z['cd_y'][i])))
    T = pd.DataFrame(rows); T.to_csv(f'outputs/top50_{tag}_pre_ft.csv', index=False); print(tag, T.hidden_label_eval_only.sum()); print(T.head(12).to_string())
