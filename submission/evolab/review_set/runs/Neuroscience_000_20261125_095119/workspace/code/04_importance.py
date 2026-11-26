"""Feature importance: permutation (F1 and AP drop, held-out test frames), grouped permutation by category / role,
and MDI (mean +- SD over seeds). Feature sets: engineered (Together_1), provided 50-column table (Together_1),
reference recording's own 517 SimBA features."""
import time, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, average_precision_score
from common import *; from features import tag_feature
X, T = load_together(); R = load_reference()
FT = pd.read_csv(f'{OUT}/features_engineered_together1.csv', index_col=0)
ref_feats = [c for c in R.columns if c not in POSE_COLS and not c.startswith('Probability_') and c not in BEHAVIOURS]
SETS = {'engineered': (FT, T), 'provided50': (X, T), 'reference_simba517': (R[ref_feats], R)}
rf = lambda n, s: RandomForestClassifier(n, criterion='entropy', max_features='sqrt', n_jobs=2, random_state=s)
N_PERM_TREES = 300  # forest used for permutation importance (same hyper-parameters; reduced size for runtime)
perm_rows, grp_rows, mdi_rows = [], [], []
t0 = time.time()
for fs, (F, Y) in SETS.items():
    cats = pd.DataFrame([tag_feature(c) for c in F.columns], index=F.columns, columns=['category', 'role'])
    for b in BEHAVIOURS:
        y = Y[b].values
        # MDI over seeds
        mdis = []
        for seed in range(5):
            tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
            m = rf(500, seed).fit(F.values[tr], y[tr]); mdis.append(m.feature_importances_)
            if seed == 0:
                tr0, te0 = tr, te; m0 = rf(N_PERM_TREES, 0).fit(F.values[tr], y[tr]); m0.set_params(n_jobs=1)
        mdis = np.array(mdis)
        for j, c in enumerate(F.columns):
            mdi_rows.append(dict(featureset=fs, behaviour=b, feature=c, category=cats.loc[c, 'category'], role=cats.loc[c, 'role'],
                                 MDI_mean=mdis[:, j].mean(), MDI_sd=mdis[:, j].std()))
        # permutation importance on held-out frames of seed-0 split (one model, 5 repeats, F1 and AP computed from the same permutations)
        Xte, yte = F.iloc[te0], y[te0]
        pi = permutation_importance(m0, Xte.values, yte, scoring={'f1': 'f1', 'AP': 'average_precision'}, n_repeats=5, random_state=0, n_jobs=2)
        for j, c in enumerate(F.columns):
            perm_rows.append(dict(featureset=fs, behaviour=b, feature=c, category=cats.loc[c, 'category'], role=cats.loc[c, 'role'],
                                  F1_drop_mean=pi['f1'].importances_mean[j], F1_drop_sd=pi['f1'].importances_std[j],
                                  AP_drop_mean=pi['AP'].importances_mean[j], AP_drop_sd=pi['AP'].importances_std[j]))
        # grouped permutation (joint row-permutation of all columns in a group) — robust to redundant rolling-window variants
        p0 = m0.predict_proba(Xte.values)[:, 1]; base = dict(F1=f1_score(yte, p0 >= 0.5), AP=average_precision_score(yte, p0))
        rng = np.random.default_rng(0)
        groups = {('category', k): v.index for k, v in cats.groupby('category')}
        groups.update({('role', k): v.index for k, v in cats.groupby('role')})
        groups.update({('category_x_role', f'{k[0]}|{k[1]}'): v.index for k, v in cats.groupby(['category', 'role'])})
        for (gtype, g), cols in groups.items():
            ci = [F.columns.get_loc(c) for c in cols]; d1, d2 = [], []
            for r in range(20):
                Xp = Xte.values.copy(); perm = rng.permutation(len(Xp)); Xp[:, ci] = Xp[perm][:, ci]
                pp = m0.predict_proba(Xp)[:, 1]
                d1.append(base['F1'] - f1_score(yte, pp >= 0.5, zero_division=0)); d2.append(base['AP'] - average_precision_score(yte, pp))
            grp_rows.append(dict(featureset=fs, behaviour=b, group_type=gtype, group=g, n_features=len(cols), F1_drop_mean=np.mean(d1),
                                 F1_drop_sd=np.std(d1), AP_drop_mean=np.mean(d2), AP_drop_sd=np.std(d2), base_F1=base['F1'], base_AP=base['AP']))
        print(fs, b, f'{time.time()-t0:.0f}s', flush=True)
        pd.DataFrame(perm_rows).to_csv(f'{OUT}/permutation_importance_all.csv', index=False)
P = pd.DataFrame(perm_rows); P.to_csv(f'{OUT}/permutation_importance_all.csv', index=False)
G = pd.DataFrame(grp_rows); G.to_csv(f'{OUT}/grouped_permutation_importance.csv', index=False)
D = pd.DataFrame(mdi_rows); D.to_csv(f'{OUT}/mdi_importance_all.csv', index=False)
for fs in SETS:
    for b in BEHAVIOURS:
        s = P[(P.featureset == fs) & (P.behaviour == b)]
        s.sort_values('F1_drop_mean', ascending=False).head(15).to_csv(f'{OUT}/perm_top15_F1_{fs}_{b}.csv', index=False)
        s.sort_values('AP_drop_mean', ascending=False).head(15).to_csv(f'{OUT}/perm_top15_AP_{fs}_{b}.csv', index=False)
        d = D[(D.featureset == fs) & (D.behaviour == b)].sort_values('MDI_mean', ascending=False).head(15)
        d.to_csv(f'{OUT}/mdi_top15_{fs}_{b}.csv', index=False)
        print('\n==', fs, b, '\nperm F1 top:\n', s.sort_values('F1_drop_mean', ascending=False).head(8)[['feature', 'F1_drop_mean', 'AP_drop_mean']].round(4).to_string(),
              '\nMDI top:\n', d.head(5)[['feature', 'MDI_mean', 'MDI_sd']].round(4).to_string())
print(G[G.group_type == 'category'].round(4).to_string())
