"""Figures: PR curves (Together_1 random split, reference CV), confusion matrices, metric summary,
feature-validation scatter, validity diagnostics."""
import json, numpy as np, pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
from common import *; from style import *
P = pd.read_csv(f'{OUT}/together1_test_predictions_engineered.csv')
M = pd.read_csv(f'{OUT}/together1_random_split_metrics.csv')
# ---- Fig: PR curves Together_1 (engineered, 5 seeds) original + gap-closed eval ----
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, b in zip(axs, BEHAVIOURS):
    d = P[P.behaviour == b]; aps, apc = [], []
    for s, g in d.groupby('seed'):
        pr, rc, _ = precision_recall_curve(g.y, g.p); ax.step(rc, pr, where='post', color=BEH_COL[b], lw=1.2, alpha=0.6, label='original labels' if s == 0 else None)
        pr2, rc2, _ = precision_recall_curve(g.y_gapclosed, g.p); ax.step(rc2, pr2, where='post', color=INK2, lw=1.0, alpha=0.6, ls='--', label='1-frame gaps closed (sensitivity)' if s == 0 else None)
        aps.append(average_precision_score(g.y, g.p)); apc.append(average_precision_score(g.y_gapclosed, g.p))
    prev = d.y.mean(); ax.axhline(prev, color=INK2, lw=1, ls=':'); ax.text(0.02, prev + 0.015, f'prevalence {prev:.3f}', color=INK2, fontsize=8)
    ax.set_title(f'{b} – held-out frames (5 random 80/20 splits)\nAP = {np.mean(aps):.3f} ± {np.std(aps):.3f}  |  gap-closed AP = {np.mean(apc):.3f}')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_ylim(0, 1.03); ax.set_xlim(0, 1.01); ax.legend(loc='lower left')
plt.tight_layout(); plt.savefig(f'{IMG}/fig02_pr_curves_together1.png', dpi=150); plt.close()

# ---- Fig: confusion matrices (seed 0, thr 0.5 and OOB thr), counts + row % ----
fig, axs = plt.subplots(2, 2, figsize=(8.4, 7.4))
for i, b in enumerate(BEHAVIOURS):
    g = P[(P.behaviour == b) & (P.seed == 0)]
    for j, (lab, thr) in enumerate([('0.5', 0.5), ('OOB-max-F1', g.thr_oob.iloc[0])]):
        cm = confusion_matrix(g.y, g.p >= thr); rp = cm / cm.sum(1, keepdims=True); ax = axs[i, j]
        ax.imshow(rp, cmap='Blues', vmin=0, vmax=1); ax.grid(False)
        for r in range(2):
            for c in range(2): ax.text(c, r, f'{cm[r, c]}\n({100*rp[r, c]:.1f}%)', ha='center', va='center', color='white' if rp[r, c] > 0.6 else INK, fontsize=10)
        ax.set_xticks([0, 1], [f'pred not-{b}', f'pred {b}']); ax.set_yticks([0, 1], [f'true not-{b}', f'true {b}'])
        ax.set_title(f'{b}, threshold {lab} = {thr:.2f} (seed 0, n={len(g)})')
plt.tight_layout(); plt.savefig(f'{IMG}/fig03_confusion_matrices.png', dpi=150); plt.close()

# ---- Fig: metric summary across feature sets ----
S = M[(M.labels == 'original') & (M.threshold_rule == 'oob_maxF1')]
fs_order = ['engineered', 'pose48', 'provided50', 'index_only']
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, met in zip(axs, ['AP', 'ROC_AUC', 'F1']):
    for k, b in enumerate(BEHAVIOURS):
        v = [S[(S.featureset == f) & (S.behaviour == b)][met] for f in fs_order]
        x = np.arange(4) + (k - 0.5) * 0.36
        ax.bar(x, [a.mean() for a in v], 0.34, yerr=[a.std() for a in v], color=BEH_COL[b], label=b, capsize=2, edgecolor='#fcfcfb', lw=1)
    ax.set_xticks(range(4), ['engineered\n(498)', 'pose\n(48)', 'provided\n(50, incl. index)', 'index only\n(Feature_1/2)'], fontsize=8)
    ax.set_title(f'{met} – random 80/20 split'); ax.set_ylim(0, 1.05)
axs[0].legend(loc='lower left'); plt.tight_layout(); plt.savefig(f'{IMG}/fig04_metrics_by_featureset.png', dpi=150); plt.close()

# ---- Fig: reference recording PR curves (stratified random CV vs blocked) ----
RP = pd.read_csv(f'{OUT}/reference_cv_predictions.csv')
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, b in zip(axs, BEHAVIOURS):
    for fs, ls in [('reference_simba517', '-'), ('reference_engineered_recomputed', '--')]:
        for sch, col in [('stratified_random_5fold', BEH_COL[b]), ('blocked_5fold', INK2)]:
            g = RP[(RP.behaviour == b) & (RP.featureset == fs) & (RP.scheme == sch) & (RP.seed == 0)]
            pr, rc, _ = precision_recall_curve(g.y, g.p); ap = average_precision_score(g.y, g.p)
            ax.step(rc, pr, where='post', color=col, ls=ls, lw=1.4, label=f'{"SimBA-517" if "517" in fs else "recomputed-498"}, {sch.split("_")[0]} CV: AP={ap:.3f}')
    prev = g.y.mean(); ax.axhline(prev, color=INK2, lw=1, ls=':'); ax.text(0.6, prev + 0.02, f'prevalence {prev:.3f}', color=INK2, fontsize=8)
    ax.set_title(f'{b} – reference recording (300 frames, {len(bouts(g.y))} bout(s))'); ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_ylim(0, 1.03); ax.legend(fontsize=7, loc='center left')
plt.tight_layout(); plt.savefig(f'{IMG}/fig05_pr_curves_reference.png', dpi=150); plt.close()

# ---- Fig: feature validation (recomputed vs SimBA reference) ----
V = pd.read_csv(f'{OUT}/feature_validation_vs_reference.csv'); R = load_reference(); FR = pd.read_csv(f'{OUT}/features_engineered_reference.csv', index_col=0)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, c in zip(axs[:2], ['Centroid_distance', 'Movement_mouse_1_nose']):
    ax.scatter(FR[c], R[c], s=10, color=CAT_COL['distance' if 'distance' in c else 'movement'], alpha=0.7)
    r = V.set_index('feature').loc[c, 'pearson']; ax.set_title(f'{c}\nPearson r = {r:.4f}'); ax.set_xlabel('re-implemented (this work)'); ax.set_ylabel('SimBA reference table')
vv = V[V.recomputed]; cats = [c for c in CAT_ORDER if c in vv.category.unique()]
data = [vv[vv.category == c].pearson.abs().dropna() for c in cats]
bp = axs[2].boxplot(data, tick_labels=[f'{c}\n(n={len(d)})' for c, d in zip(cats, data)], patch_artist=True, widths=0.5)
for patch, c in zip(bp['boxes'], cats): patch.set_facecolor(CAT_COL[c]); patch.set_alpha(0.7)
axs[2].set_title('|r| recomputed vs reference, per category'); axs[2].set_ylim(-0.02, 1.05); axs[2].tick_params(axis='x', labelsize=7)
plt.tight_layout(); plt.savefig(f'{IMG}/fig06_feature_validation.png', dpi=150); plt.close()
print('ok')
