"""TreeSHAP explanations, feature-category x animal-role attribution shares with bootstrap CIs,
and cross-condition comparisons: (1) Attack vs Sniffing (Together_1), (2) Together_1 vs reference recording
(identical recomputed feature definitions), (3) per-episode profiles within Together_1."""
import time, numpy as np, pandas as pd, shap, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from common import *; from features import tag_feature
X, T = load_together(); R = load_reference()
FT = pd.read_csv(f'{OUT}/features_engineered_together1.csv', index_col=0)
FR = pd.read_csv(f'{OUT}/features_engineered_reference.csv', index_col=0)[FT.columns]
cats = pd.DataFrame([tag_feature(c) for c in FT.columns], index=FT.columns, columns=['category', 'role'])
N_TREES_SHAP = 200; rng = np.random.default_rng(SEED)

def fit_explain(F, y, explain_idx, seed=0, train_idx=None):
    m = RandomForestClassifier(N_TREES_SHAP, criterion='entropy', max_features='sqrt', n_jobs=2, random_state=seed)
    m.fit(F.values[train_idx], y[train_idx])
    ex = shap.TreeExplainer(m); sv = ex.shap_values(F.values[explain_idx], check_additivity=False)
    sv = sv[:, :, 1] if np.ndim(sv) == 3 else (sv[1] if isinstance(sv, list) else sv)
    base = ex.expected_value[1] if np.ndim(ex.expected_value) else ex.expected_value
    return m, sv, base

BLOCK = 30  # frames (~1 s at the assumed 30 fps); block bootstrap respects temporal autocorrelation

def shares(sv, groups, rows=None):
    A = np.abs(sv if rows is None else sv[rows]).mean(0)
    s = pd.Series(A, index=FT.columns).groupby(groups).sum(); return s / s.sum()

def block_resample(frames):
    """Resample rows by contiguous 30-frame blocks (moving-block style) -> row indices into `frames`."""
    blk = np.asarray(frames) // BLOCK; ub = np.unique(blk); pick = rng.choice(ub, len(ub), replace=True)
    return np.concatenate([np.where(blk == u)[0] for u in pick])

def boot_shares(sv, groups, frames, n=200):
    B = pd.concat([shares(sv, groups, block_resample(frames)) for _ in range(n)], axis=1)
    nf = groups.value_counts(); base = shares(sv, groups)
    return pd.DataFrame(dict(share=base, lo=B.quantile(0.025, axis=1), hi=B.quantile(0.975, axis=1), n_features=nf.reindex(base.index),
                             feature_share=(nf / nf.sum()).reindex(base.index), enrichment=base / (nf / nf.sum()).reindex(base.index)))

t0 = time.time(); SV = {}; meta = []
g_cat = cats.category; g_role = cats.role; g_cr = cats.category + '|' + cats.role
for b in BEHAVIOURS:
    y = T[b].values; tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=0)
    # explain ALL Together_1 frames with a model trained on the 80% split; held-out rows flagged for test-only summaries
    m, sv, base = fit_explain(FT, y, np.arange(len(y)), 0, tr)
    SV[('Together_1', b)] = dict(sv=sv, test=np.isin(np.arange(len(y)), te), X=FT.values, y=y, base=base)
    print('T1', b, f'{time.time()-t0:.0f}s', flush=True)
    yR = R[b].values; trR, teR = train_test_split(np.arange(len(yR)), test_size=0.2, stratify=yR, random_state=0)
    m, svR, baseR = fit_explain(FR, yR, np.arange(len(yR)), 0, trR)
    SV[('Reference', b)] = dict(sv=svR, test=np.isin(np.arange(len(yR)), teR), X=FR.values, y=yR, base=baseR)
    print('Ref', b, f'{time.time()-t0:.0f}s', flush=True)
for (rec, b), d in SV.items():  # evaluation rows: held-out frames for Together_1; all 300 frames for the reference recording
    d['eval'] = d['test'] if rec == 'Together_1' else np.ones(len(d['y']), bool)
# save SHAP values (held-out Together_1 rows, + all reference rows)
for (rec, b), d in SV.items():
    df = pd.DataFrame(d['sv'], columns=FT.columns); df.insert(0, 'frame', np.arange(len(df))); df.insert(1, 'is_test', d['test']); df.insert(2, 'y', d['y'])
    df.to_csv(f'{OUT}/shap_values_{rec}_{b}.csv', index=False)
pd.concat([pd.read_csv(f'{OUT}/shap_values_Together_1_{b}.csv').query('is_test').assign(behaviour=b) for b in BEHAVIOURS]).to_csv(f'{OUT}/shap_values.csv', index=False)

# ---------- Top-15 tables, beeswarm and bar plots (Together_1 held-out rows) ----------
top = []
for (rec, b), d in SV.items():
    rows = d['test'] if rec == 'Together_1' else np.ones(len(d['y']), bool)
    ma = np.abs(d['sv'][rows]).mean(0); sg = np.array([np.corrcoef(d['X'][rows, j], d['sv'][rows, j])[0, 1] if d['X'][rows, j].std() > 0 else np.nan for j in range(len(ma))])
    s = pd.DataFrame(dict(recording=rec, behaviour=b, feature=FT.columns, mean_abs_shap=ma, corr_value_vs_shap=sg, category=cats.category.values, role=cats.role.values))
    top.append(s.sort_values('mean_abs_shap', ascending=False))
TOP = pd.concat(top); TOP.to_csv(f'{OUT}/shap_mean_abs_all.csv', index=False)
TOP.groupby(['recording', 'behaviour']).head(15).to_csv(f'{OUT}/shap_top15.csv', index=False)
for b in BEHAVIOURS:
    d = SV[('Together_1', b)]; rows = d['test']
    Xdf = pd.DataFrame(d['X'][rows], columns=FT.columns)
    for kind in ['dot', 'bar']:
        plt.figure(); shap.summary_plot(d['sv'][rows], Xdf, max_display=15, plot_type=kind, show=False, plot_size=(8, 6))
        plt.title(f'TreeSHAP – {b} classifier (Together_1 held-out frames, n={rows.sum()})', fontsize=10); plt.tight_layout()
        plt.savefig(f'{IMG}/fig_shap_{"beeswarm" if kind=="dot" else "bar"}_{b}.png', dpi=140, bbox_inches='tight'); plt.close()

# ---------- Category x role shares with bootstrap CIs ----------
sh_rows = []
for (rec, b), d in SV.items():
    rows = np.where(d['test'])[0] if rec == 'Together_1' else np.arange(len(d['y']))
    for gname, grp in [('category', g_cat), ('role', g_role), ('category_x_role', g_cr)]:
        bs = boot_shares(d['sv'][rows], grp, rows)
        for k, r in bs.iterrows(): sh_rows.append(dict(recording=rec, behaviour=b, grouping=gname, group=k, **r.to_dict()))
    # positive-frame-only shares (what drives positive calls)
    pos = rows[d['y'][rows] == 1]
    bs = boot_shares(d['sv'][pos], g_cat, pos)
    for k, r in bs.iterrows(): sh_rows.append(dict(recording=rec, behaviour=b, grouping='category_positive_frames', group=k, **r.to_dict()))
SH = pd.DataFrame(sh_rows); SH.to_csv(f'{OUT}/shap_category_role_shares.csv', index=False)
# difference CIs via paired bootstrap for the two comparisons
diff_rows = []
def boot_diff(svA, svB, grp, fA, fB, n=200):
    obs = shares(svA, grp) - shares(svB, grp)
    B = pd.concat([shares(svA, grp, block_resample(fA)) - shares(svB, grp, block_resample(fB)) for _ in range(n)], axis=1)
    return pd.DataFrame(dict(diff=obs, lo=B.quantile(0.025, axis=1), hi=B.quantile(0.975, axis=1)))
cmp = {'Attack_minus_Sniffing@Together_1': (SV[('Together_1', 'Attack')], SV[('Together_1', 'Sniffing')]),
       'Together_1_minus_Reference@Attack': (SV[('Together_1', 'Attack')], SV[('Reference', 'Attack')]),
       'Together_1_minus_Reference@Sniffing': (SV[('Together_1', 'Sniffing')], SV[('Reference', 'Sniffing')])}
for name, (A, B) in cmp.items():
    sa, sb = A['sv'][A['eval']], B['sv'][B['eval']]; fa, fb = np.where(A['eval'])[0], np.where(B['eval'])[0]
    for gname, grp in [('category', g_cat), ('role', g_role), ('category_x_role', g_cr)]:
        for k, r in boot_diff(sa, sb, grp, fa, fb).iterrows():
            diff_rows.append(dict(comparison=name, grouping=gname, group=k, **r.to_dict(), ci_excludes_zero=bool(r.lo > 0 or r.hi < 0)))
DF = pd.DataFrame(diff_rows); DF.to_csv(f'{OUT}/shap_share_differences.csv', index=False)
print(SH[SH.grouping.isin(['category', 'role'])].round(3).to_string()); print(DF[DF.grouping != 'category_x_role'].round(3).to_string())

# ---------- Role balance on symmetric twin features (M1 vs M2 versions of the same quantity) ----------
def twin(n):
    for a, b in [('Mouse_1', 'Mouse_2'), ('Mouse1', 'Mouse2'), ('mouse_1', 'mouse_2'), ('M1', 'M2'), ('_1_', '_2_')]:
        if a in n: return n.replace(a, b)
    return None
pairs = [(c, twin(c)) for c in FT.columns if cats.loc[c, 'role'] == 'M1' and twin(c) in FT.columns and cats.loc[twin(c), 'role'] == 'M2']
tw_rows = []
for (rec, b), d in SV.items():
    A = pd.Series(np.abs(d['sv'][d['eval']]).mean(0), index=FT.columns)
    m1 = sum(A[p] for p, q in pairs); m2 = sum(A[q] for p, q in pairs); fr = np.where(d['eval'])[0]
    bs = []
    for _ in range(200):
        ii = block_resample(fr); Ab = pd.Series(np.abs(d['sv'][d['eval']][ii]).mean(0), index=FT.columns)
        bs.append(sum(Ab[p] for p, q in pairs) / (sum(Ab[p] for p, q in pairs) + sum(Ab[q] for p, q in pairs)))
    tw_rows.append(dict(recording=rec, behaviour=b, n_twin_pairs=len(pairs), M1_fraction=m1 / (m1 + m2), lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975)))
TW = pd.DataFrame(tw_rows); TW.to_csv(f'{OUT}/shap_role_balance_twin_features.csv', index=False); print(TW.round(3))
pd.DataFrame(pairs, columns=['M1_feature', 'M2_feature']).to_csv(f'{OUT}/twin_feature_pairs.csv', index=False)

# ---------- Per-episode profiles (Together_1, all frames of each gap-closed episode; descriptive, partly in-sample) ----------
ep_rows = []
for b in BEHAVIOURS:
    d = SV[('Together_1', b)]
    for k, (s, e) in enumerate(bouts(close_gaps(T[b].values))):
        rows = np.arange(s, e + 1); sh = shares(d['sv'], g_cat, rows); rl = shares(d['sv'], g_role, rows)
        ep_rows.append(dict(behaviour=b, episode=k + 1, start=s, end=e, n_frames=len(rows), frac_test=d['test'][rows].mean(),
                            **{f'cat_{i}': v for i, v in sh.items()}, **{f'role_{i}': v for i, v in rl.items()}))
EP = pd.DataFrame(ep_rows); EP.to_csv(f'{OUT}/shap_episode_profiles.csv', index=False); print(EP.round(3).to_string())
# top-15 feature diff Attack vs Sniffing, Together_1 vs Reference
print('total', time.time() - t0)
