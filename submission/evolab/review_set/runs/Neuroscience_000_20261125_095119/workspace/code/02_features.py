"""Compute SimBA-family features for Together_1 and the reference recording; validate against reference SimBA columns."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from common import *; from features import simba_features, tag_feature
X, T = load_together(); R = load_reference()
FT = simba_features(X[POSE_COLS]); FR = simba_features(R[POSE_COLS])
FT.to_csv(f'{OUT}/features_engineered_together1.csv'); FR.to_csv(f'{OUT}/features_engineered_reference.csv')
ref_feats = [c for c in R.columns if c not in POSE_COLS and not c.startswith('Probability_') and c not in BEHAVIOURS]
print('n engineered', FT.shape, 'n reference SimBA features', len(ref_feats))
rows = []
for c in ref_feats:
    cat, role = tag_feature(c)
    if c in FR.columns:
        a, b = FR[c].values, R[c].values
        if np.std(a) > 0 and np.std(b) > 0:
            pr, sr = pearsonr(a, b)[0], spearmanr(a, b)[0]
        else: pr = sr = np.nan
        ratio = np.median(b[np.abs(a) > 1e-9] / a[np.abs(a) > 1e-9]) if (np.abs(a) > 1e-9).any() else np.nan
        rows.append(dict(feature=c, category=cat, role=role, recomputed=True, pearson=pr, spearman=sr, median_ratio_ref_over_ours=ratio))
    else: rows.append(dict(feature=c, category=cat, role=role, recomputed=False))
V = pd.DataFrame(rows); V.to_csv(f'{OUT}/feature_validation_vs_reference.csv', index=False)
print(V.groupby('category')[['pearson', 'spearman']].median()); print('not recomputed:', V[~V.recomputed].feature.tolist())
print(V[V.recomputed].sort_values('pearson').head(25).to_string())
# category tags for every feature used anywhere
allf = sorted(set(FT.columns) | set(ref_feats) | set(X.columns))
tags = pd.DataFrame([dict(feature=f, category=tag_feature(f)[0], role=tag_feature(f)[1], in_engineered=f in FT.columns, in_reference=f in ref_feats, in_provided_table=f in X.columns) for f in allf])
tags.to_csv(f'{OUT}/feature_categories.csv', index=False)
print(tags.groupby(['category', 'role']).size())
