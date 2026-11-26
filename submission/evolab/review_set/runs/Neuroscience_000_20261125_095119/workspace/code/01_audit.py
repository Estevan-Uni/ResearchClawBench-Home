"""Data audit: shapes, alignment, index-like columns, label structure, ethogram."""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *
X, T = load_together(); R = load_reference()
res = {'features_shape': X.shape, 'targets_shape': T.shape, 'reference_shape': R.shape,
       'nan_features': int(X.isna().sum().sum()), 'nan_targets': int(T.isna().sum().sum()),
       'index_aligned': bool((X.index == T.index).all())}
shared = [c for c in X.columns if c in T.columns]
res['max_abs_diff_shared'] = float((X[shared] - T[shared]).abs().max().max())
res['Feature_1_equals_index'] = bool((X['Feature_1'].values == np.arange(len(X))).all())
res['Feature_2_equals_3000_minus_index'] = bool((X['Feature_2'].values == 3000 - np.arange(len(X))).all())
mono = [c for c in X.columns if X[c].is_monotonic_increasing or X[c].is_monotonic_decreasing]
res['monotone_columns'] = [c for c in mono if X[c].nunique() == len(X)]
pcols = [c for c in POSE_COLS if c.endswith('_p')]
res['likelihood_gt1_count'] = int((X[pcols] > 1).sum().sum())
res['likelihood_lt0.5_count'] = int((X[pcols] < 0.5).sum().sum())
res['ref_zero_coords_frac'] = float((R[[c for c in POSE_COLS if not c.endswith('_p')]] == 0).mean().mean())
res['ref_zero_tailbase1_frames'] = int((R['Tail_base_1_x'] == 0).sum())
res['ref_prob_equals_0.9_label'] = {b: bool(np.allclose(R[f'Probability_{b}'], 0.9 * R[b])) for b in BEHAVIOURS}
lab = {}
for name, D in [('Together_1', T), ('Reference', R)]:
    for b in BEHAVIOURS:
        y = D[b].values; bt = bouts(y); yc = close_gaps(y)
        gaps = sum(1 for (s0, e0), (s1, e1) in zip(bt[:-1], bt[1:]) if s1 - e0 - 1 == 1)
        lens = [e - s + 1 for s, e in bt]
        lab[f'{name}_{b}'] = dict(n_pos=int(y.sum()), prevalence=float(y.mean()), n_bouts=len(bt),
            median_bout_len=float(np.median(lens)) if lens else 0, one_frame_gaps=gaps,
            n_episodes_after_gap_closing=len(bouts(yc)), n_pos_after_gap_closing=int(yc.sum()),
            episodes=[(int(s), int(e)) for s, e in bouts(yc)])
res['labels'] = lab
res['cooccurrence_Together_1'] = int((T.Attack & T.Sniffing).sum())
json.dump(res, open(f'{OUT}/audit.json', 'w'), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))

# Ethogram figure
fig, ax = plt.subplots(3, 1, figsize=(11, 5.2), sharex=True, gridspec_kw={'height_ratios': [1, 1, 2]})
cols = {'Attack': '#2a78d6', 'Sniffing': '#eb6834'}
for i, b in enumerate(BEHAVIOURS):
    ax[i].fill_between(T.index, 0, T[b], step='mid', color=cols[b], lw=0)
    ax[i].set_yticks([]); ax[i].set_ylabel(b, rotation=0, ha='right', va='center')
    for s, e in bouts(close_gaps(T[b].values)):
        ax[i].axvspan(s, e, ymin=0.0, ymax=1, color=cols[b], alpha=0.15, lw=0)
    ax[i].set_title(f'{b}: {lab["Together_1_"+b]["n_pos"]} frames, {lab["Together_1_"+b]["n_bouts"]} annotated bouts '
                    f'({lab["Together_1_"+b]["one_frame_gaps"]} one-frame gaps) → {lab["Together_1_"+b]["n_episodes_after_gap_closing"]} episodes', fontsize=9, loc='left')
cd = np.hypot(T.Center_1_x - T.Center_2_x, T.Center_1_y - T.Center_2_y)
ax[2].plot(T.index, cd, color='k', lw=0.8); ax[2].set_ylabel('Centroid\ndistance (px)')
for b in BEHAVIOURS:
    for k, (s0, e0) in enumerate(bouts(close_gaps(T[b].values))):
        ax[2].axvspan(s0, e0, color=cols[b], alpha=0.15, lw=0, label=f'{b} episode' if k == 0 else None)
ax[2].legend(loc='upper left', fontsize=8, frameon=False)
ax[2].set_xlabel('Frame (Together_1, assumed 30 fps)')
inset = ax[0].inset_axes([0.80, 0.08, 0.18, 0.84]); z = slice(430, 470)
inset.step(T.index[z], T.Attack.values[z], where='mid', color=cols['Attack']); inset.set_title('zoom frames 430–470: 5-on/1-off comb', fontsize=6); inset.tick_params(labelsize=5); inset.set_yticks([0, 1])
plt.tight_layout(); plt.savefig(f'{IMG}/fig01_ethogram.png', dpi=150); plt.close()
