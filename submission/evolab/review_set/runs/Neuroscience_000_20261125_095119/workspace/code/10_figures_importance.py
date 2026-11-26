"""Figures: permutation top-15 (F1 and AP drop), grouped permutation by category, MDI top-15 (mean +- SD)."""
import numpy as np, pandas as pd
from common import *; from style import *
P = pd.read_csv(f'{OUT}/permutation_importance_all.csv'); G = pd.read_csv(f'{OUT}/grouped_permutation_importance.csv'); D = pd.read_csv(f'{OUT}/mdi_importance_all.csv')
def hbar(ax, d, col, sd, title, xlabel):
    y = np.arange(len(d)); ax.barh(y, d[col], xerr=d[sd], color=[CAT_COL.get(c, '#8a8984') for c in d.category], capsize=2, edgecolor='#fcfcfb')
    ax.set_yticks(y, d.feature, fontsize=7); ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.8); ax.set_title(title, fontsize=9); ax.set_xlabel(xlabel, fontsize=8)
from matplotlib.patches import Patch
leg = [Patch(color=CAT_COL[c], label=c) for c in ['distance', 'movement', 'shape', 'angle', 'tracking', 'position', 'index_control']]
for fs, tag in [('engineered', 'fig07'), ('provided50', 'fig08a'), ('reference_simba517', 'fig08b')]:
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for i, b in enumerate(BEHAVIOURS):
        s = P[(P.featureset == fs) & (P.behaviour == b)]
        hbar(axs[i, 0], s.sort_values('F1_drop_mean', ascending=False).head(15), 'F1_drop_mean', 'F1_drop_sd', f'{b}: top-15 permutation importance (F1 drop)', 'mean F1 drop (5 repeats) ± SD')
        hbar(axs[i, 1], s.sort_values('AP_drop_mean', ascending=False).head(15), 'AP_drop_mean', 'AP_drop_sd', f'{b}: top-15 permutation importance (AP drop)', 'mean AP drop (5 repeats) ± SD')
    fig.legend(handles=leg, ncol=7, loc='upper center', fontsize=8); fig.suptitle(f'Feature set: {fs} — held-out test frames, seed-0 split, 300-tree RF (entropy, sqrt)', y=0.965, fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig(f'{IMG}/{tag}_permutation_top15_{fs}.png', dpi=140); plt.close()
# grouped permutation by category
fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, fs in zip(axs, ['engineered', 'provided50', 'reference_simba517']):
    g = G[(G.featureset == fs) & (G.group_type == 'category')]; cats = [c for c in CAT_ORDER if c in g.group.unique()]
    for k, b in enumerate(BEHAVIOURS):
        d = g[g.behaviour == b].set_index('group').reindex(cats); x = np.arange(len(cats)) + (k - 0.5) * 0.36
        ax.bar(x, d.AP_drop_mean, 0.34, yerr=d.AP_drop_sd, color=BEH_COL[b], capsize=2, label=f'{b} (base AP {d.base_AP.iloc[0]:.2f})')
    ax.set_xticks(range(len(cats)), [f'{c}\n({int(n)})' for c, n in zip(cats, g[g.behaviour == "Attack"].set_index("group").reindex(cats).n_features)], fontsize=7)
    ax.axhline(0, color=INK2, lw=0.8); ax.set_title(f'{fs}: grouped permutation (AP drop, 20 reps)', fontsize=9); ax.legend(fontsize=7)
axs[0].set_ylabel('AP drop when whole category permuted'); plt.tight_layout(); plt.savefig(f'{IMG}/fig09_grouped_permutation.png', dpi=150); plt.close()
# MDI
fig, axs = plt.subplots(2, 2, figsize=(13, 9))
for i, b in enumerate(BEHAVIOURS):
    for j, fs in enumerate(['engineered', 'provided50']):
        d = D[(D.featureset == fs) & (D.behaviour == b)].sort_values('MDI_mean', ascending=False).head(15)
        hbar(axs[i, j], d, 'MDI_mean', 'MDI_sd', f'{b} · {fs}: MDI top-15 (mean ± SD, 5 seeds)', 'mean decrease in impurity (entropy)')
fig.legend(handles=leg, ncol=7, loc='upper center', fontsize=8); plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.savefig(f'{IMG}/fig15_mdi_top15.png', dpi=140); plt.close()
print('ok')
