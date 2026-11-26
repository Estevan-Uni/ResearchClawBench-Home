"""Compact validity checks: index-only control (in 03), blocked temporal CV with purge, leakage curve vs block length,
circular-shift label null, determinism hash."""
import hashlib, json, time, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from common import *
X, T = load_together(); FT = pd.read_csv(f'{OUT}/features_engineered_together1.csv', index_col=0)
rf = lambda n, s: RandomForestClassifier(n, criterion='entropy', max_features='sqrt', n_jobs=2, random_state=s)
n = len(FT); t0 = time.time(); res = {}

def blocked_cv(F, y, k=5, purge=15, ntree=200, seed=0):
    p = np.zeros(len(y)); folds = []
    for i, bk in enumerate(np.array_split(np.arange(len(y)), k)):
        tr = np.setdiff1d(np.arange(len(y)), np.arange(max(bk[0] - purge, 0), min(bk[-1] + purge + 1, len(y))))
        m = rf(ntree, seed).fit(F[tr], y[tr]); p[bk] = m.predict_proba(F[bk])[:, 1]
        folds.append(dict(fold=i, start=int(bk[0]), end=int(bk[-1]), n_pos_test=int(y[bk].sum()), n_pos_train=int(y[tr].sum()),
                          AUC=roc_auc_score(y[bk], p[bk]) if 0 < y[bk].sum() < len(bk) else np.nan,
                          AP=average_precision_score(y[bk], p[bk]) if y[bk].sum() > 0 else np.nan, prevalence=y[bk].mean()))
    return p, folds

rows = []
for fs, F in {'engineered': FT.values, 'provided50': X.values, 'index_only': X[['Feature_1', 'Feature_2']].values}.items():
    for b in BEHAVIOURS:
        y = T[b].values; p, folds = blocked_cv(F, y)
        for f in folds: rows.append(dict(featureset=fs, behaviour=b, **f))
        rows.append(dict(featureset=fs, behaviour=b, fold='pooled', AUC=roc_auc_score(y, p), AP=average_precision_score(y, p), prevalence=y.mean(),
                         F1=f1_score(y, p >= 0.5)))
        print('blocked', fs, b, f'{time.time()-t0:.0f}s', flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/blocked_cv_metrics.csv', index=False)

# leakage curve: random assignment of contiguous blocks of length L to test (20%), no purge
lrows = []
for b in BEHAVIOURS:
    y = T[b].values
    for L in [1, 6, 30, 120, 348]:
        for seed in range(3):
            blk = np.arange(n) // L; ub = np.unique(blk); r = np.random.default_rng(seed)
            # stratify-ish: resample until test fold contains positives
            for _ in range(200):
                tb = r.choice(ub, max(1, int(round(0.2 * len(ub)))), replace=False); te = np.isin(blk, tb)
                if 0 < y[te].sum() < te.sum(): break
            m = rf(100, seed).fit(FT.values[~te], y[~te]); p = m.predict_proba(FT.values[te])[:, 1]
            lrows.append(dict(behaviour=b, block_len=L, seed=seed, AP=average_precision_score(y[te], p), AUC=roc_auc_score(y[te], p), prevalence_test=y[te].mean()))
    print('leak', b, f'{time.time()-t0:.0f}s', flush=True)
pd.DataFrame(lrows).to_csv(f'{OUT}/leakage_curve.csv', index=False)

# circular-shift null: labels rotated by random offsets (>=150 frames) keep autocorrelation but break feature-label link
nrows = []
for b in BEHAVIOURS:
    y = T[b].values; r = np.random.default_rng(1)
    shifts = [0] + list(r.integers(150, n - 150, 15))
    for sh in shifts:
        ys = np.roll(y, sh)
        tr, te = train_test_split(np.arange(n), test_size=0.2, stratify=ys, random_state=0)
        p = rf(100, 0).fit(FT.values[tr], ys[tr]).predict_proba(FT.values[te])[:, 1]
        pb, _ = blocked_cv(FT.values, ys, ntree=100)
        nrows.append(dict(behaviour=b, shift=int(sh), is_real=sh == 0, AP_random_split=average_precision_score(ys[te], p),
                          AP_blocked=average_precision_score(ys, pb), prevalence=ys.mean()))
    print('null', b, f'{time.time()-t0:.0f}s', flush=True)
pd.DataFrame(nrows).to_csv(f'{OUT}/circular_shift_null.csv', index=False)

# determinism
y = T.Attack.values; tr, te = train_test_split(np.arange(n), test_size=0.2, stratify=y, random_state=0)
h = [hashlib.sha256(rf(200, 0).fit(FT.values[tr], y[tr]).predict_proba(FT.values[te]).tobytes()).hexdigest() for _ in range(2)]
h3 = hashlib.sha256(rf(200, 1).fit(FT.values[tr], y[tr]).predict_proba(FT.values[te]).tobytes()).hexdigest()
json.dump(dict(same_seed_hashes=h, identical=h[0] == h[1], other_seed_hash=h3), open(f'{OUT}/determinism.json', 'w'), indent=1)
print('done', time.time() - t0)
