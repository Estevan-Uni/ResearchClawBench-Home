"""Figures for SHAP category/role shares, cross-condition comparisons, top-15 diffs, episode profiles."""
import numpy as np, pandas as pd
from common import *; from style import *
SH = pd.read_csv(f'{OUT}/shap_category_role_shares.csv'); DF = pd.read_csv(f'{OUT}/shap_share_differences.csv')
TOP = pd.read_csv(f'{OUT}/shap_mean_abs_all.csv'); EP = pd.read_csv(f'{OUT}/shap_episode_profiles.csv'); TW = pd.read_csv(f'{OUT}/shap_role_balance_twin_features.csv')
conds = [('Together_1', 'Attack'), ('Together_1', 'Sniffing'), ('Reference', 'Attack'), ('Reference', 'Sniffing')]
cats = ['distance', 'movement', 'shape', 'angle', 'tracking']; roles = ['M1', 'M2', 'inter']
# ---- Fig: category and role shares (grouped bars with block-bootstrap CIs) + enrichment ----
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={'width_ratios': [5, 5, 3]})
w = 0.2; ccol = ['#2a78d6', '#eb6834', '#9fc3ee', '#f5b597']
for k, (rec, b) in enumerate(conds):
    for ax, grp, keys in [(axs[0], 'category', cats), (axs[1], 'enrich', cats)]:
        d = SH[(SH.recording == rec) & (SH.behaviour == b) & (SH.grouping == 'category')].set_index('group').reindex(keys)
        x = np.arange(len(keys)) + (k - 1.5) * w
        if grp == 'category':
            ax.bar(x, d.share, w * 0.92, yerr=[d.share - d.lo, d.hi - d.share], color=ccol[k], capsize=2, label=f'{b} · {rec}')
        else:
            ax.bar(x, d.enrichment, w * 0.92, color=ccol[k], label=f'{b} · {rec}')
    d = SH[(SH.recording == rec) & (SH.behaviour == b) & (SH.grouping == 'role')].set_index('group').reindex(roles)
    x = np.arange(3) + (k - 1.5) * w; axs[2].bar(x, d.share, w * 0.92, yerr=[d.share - d.lo, d.hi - d.share], color=ccol[k], capsize=2)
axs[0].set_xticks(range(5), cats); axs[0].set_ylabel('share of total mean |SHAP|'); axs[0].set_title('A. Attribution share by feature category (95% block-bootstrap CI)')
axs[1].set_xticks(range(5), cats); axs[1].axhline(1, color=INK2, lw=1, ls=':'); axs[1].set_ylabel('share / fraction of features'); axs[1].set_title('B. Per-feature enrichment (1 = proportional to feature count)')
axs[2].set_xticks(range(3), ['M1\n(resident?)', 'M2\n(intruder?)', 'inter /\nboth']); axs[2].set_title('C. Share by animal role')
axs[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{IMG}/fig10_shap_category_role_shares.png', dpi=150); plt.close()

# ---- Fig: differences with CIs (category, role) for the three comparisons + twin-feature balance ----
comps = ['Attack_minus_Sniffing@Together_1', 'Together_1_minus_Reference@Attack', 'Together_1_minus_Reference@Sniffing']
labels = {'Attack_minus_Sniffing@Together_1': 'Attack − Sniffing (Together_1)', 'Together_1_minus_Reference@Attack': 'Together_1 − Reference (Attack)',
          'Together_1_minus_Reference@Sniffing': 'Together_1 − Reference (Sniffing)'}
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for ax, gname, keys in [(axs[0], 'category', cats), (axs[1], 'role', roles)]:
    for k, c in enumerate(comps):
        d = DF[(DF.comparison == c) & (DF.grouping == gname)].set_index('group').reindex(keys)
        y = np.arange(len(keys)) + (k - 1) * 0.26
        ax.errorbar(d['diff'], y, xerr=[d['diff'] - d.lo, d.hi - d['diff']], fmt='o', ms=6, color=['#6250d6', '#2a78d6', '#eb6834'][k], capsize=2, label=labels[c])
    ax.axvline(0, color=INK2, lw=1); ax.set_yticks(range(len(keys)), keys); ax.invert_yaxis(); ax.set_xlabel('difference in attribution share (95% CI)')
axs[0].set_title('A. Category-share differences'); axs[1].set_title('B. Role-share differences'); axs[0].legend(fontsize=7, loc='lower left')
t = TW.set_index(['recording', 'behaviour'])
for k, (rec, b) in enumerate(conds):
    r = t.loc[(rec, b)]; axs[2].errorbar(r.M1_fraction, k, xerr=[[r.M1_fraction - r.lo], [r.hi - r.M1_fraction]], fmt='o', ms=7, color=ccol[k], capsize=3)
axs[2].axvline(0.5, color=INK2, lw=1, ls=':'); axs[2].set_yticks(range(4), [f'{b} · {rec}' for rec, b in conds]); axs[2].invert_yaxis(); axs[2].set_xlim(0.3, 0.8)
axs[2].set_xlabel('M1 fraction of |SHAP| over 148 M1/M2 twin features'); axs[2].set_title('C. Resident (M1) vs intruder (M2), matched features')
plt.tight_layout(); plt.savefig(f'{IMG}/fig11_shap_condition_differences.png', dpi=150); plt.close()

# ---- Fig: top-15 feature diffs ----
def paired(ax, A, B, la, lb, ca, cb, title):
    a = A.set_index('feature').mean_abs_shap; b = B.set_index('feature').mean_abs_shap
    a, b = a / a.sum(), b / b.sum()  # normalised so the two models are comparable
    feats = list(dict.fromkeys(list(a.sort_values(ascending=False).index[:10]) + list(b.sort_values(ascending=False).index[:10])))[:18]
    feats = sorted(feats, key=lambda f: -(a[f] + b[f])); y = np.arange(len(feats))
    ax.barh(y - 0.2, a[feats], 0.38, color=ca, label=la); ax.barh(y + 0.2, b[feats], 0.38, color=cb, label=lb)
    ax.set_yticks(y, [f'{f} [{A.set_index("feature").category[f][:4]}]' for f in feats], fontsize=7); ax.invert_yaxis(); ax.set_xlabel('normalised mean |SHAP|'); ax.set_title(title); ax.legend(fontsize=8, loc='lower right')
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
g = lambda r, b: TOP[(TOP.recording == r) & (TOP.behaviour == b)]
paired(axs[0], g('Together_1', 'Attack'), g('Together_1', 'Sniffing'), 'Attack', 'Sniffing', BEH_COL['Attack'], BEH_COL['Sniffing'], 'A. Attack vs Sniffing classifier (Together_1): union of top-10 features')
paired(axs[1], g('Together_1', 'Attack'), g('Reference', 'Attack'), 'Together_1', 'Reference', REC_COL['Together_1'], '#9fc3ee', 'B. Attack classifier: Together_1 vs reference recording')
plt.tight_layout(); plt.savefig(f'{IMG}/fig12_shap_top_feature_diffs.png', dpi=150); plt.close()

# ---- Fig: per-episode category profiles ----
fig, ax = plt.subplots(figsize=(9, 3.6)); left = np.zeros(len(EP))
lbl = [f'{r.behaviour} ep{r.episode} ({r.start}–{r.end}, n={r.n_frames})' for r in EP.itertuples()]
for c in cats:
    ax.barh(range(len(EP)), EP[f'cat_{c}'], left=left, color=CAT_COL[c], label=c, edgecolor='#fcfcfb', lw=1.5); left += EP[f'cat_{c}'].values
ax.set_yticks(range(len(EP)), lbl, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel('share of mean |SHAP| within episode'); ax.legend(ncol=5, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.18))
ax.grid(axis='y', visible=False); plt.tight_layout(); plt.savefig(f'{IMG}/fig13_shap_episode_profiles.png', dpi=150); plt.close()
print('ok')
