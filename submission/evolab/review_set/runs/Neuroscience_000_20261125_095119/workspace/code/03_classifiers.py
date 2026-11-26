"""Headline SimBA-protocol classifiers (RF, entropy, sqrt, 2000 trees; random stratified 80/20 split, 5 seeds),
plus control feature sets, gap-closed AP sensitivity, reference-recording CV and cross-recording transfer."""
import json, sys, time, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score, precision_score, recall_score,
                             precision_recall_curve, confusion_matrix, classification_report)
from common import *
N_TREES = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEEDS = [0, 1, 2, 3, 4]
X, T = load_together(); R = load_reference()
FT = pd.read_csv(f'{OUT}/features_engineered_together1.csv', index_col=0)
FR = pd.read_csv(f'{OUT}/features_engineered_reference.csv', index_col=0)
FEATSETS = {'engineered': FT, 'pose48': X[POSE_COLS], 'provided50': X, 'index_only': X[['Feature_1', 'Feature_2']]}

def rf(n=N_TREES, seed=0, oob=True):
    return RandomForestClassifier(n_estimators=n, criterion='entropy', max_features='sqrt', bootstrap=True,
                                  oob_score=oob, n_jobs=2, random_state=seed)

def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p); f = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return float(th[np.argmax(f[:-1])])

def metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    return dict(AP=average_precision_score(y, p), ROC_AUC=roc_auc_score(y, p), F1=f1_score(y, yh, zero_division=0),
                precision=precision_score(y, yh, zero_division=0), recall=recall_score(y, yh, zero_division=0), threshold=thr)

rows, preds, reports, cms = [], [], {}, {}
t0 = time.time()
for fs, F in FEATSETS.items():
    for b in BEHAVIOURS:
        y = T[b].values; yc = close_gaps(y)
        for seed in (SEEDS if fs == 'engineered' else SEEDS[:3]):
            idx = np.arange(len(y)); tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
            ntree = N_TREES if fs == 'engineered' else 500
            m = rf(ntree, seed).fit(F.values[tr], y[tr])
            oob = m.oob_decision_function_[:, 1]; thr_oob = best_f1_threshold(y[tr], np.nan_to_num(oob, nan=0.0))
            p = m.predict_proba(F.values[te])[:, 1]
            for thr_name, thr in [('0.5', 0.5), ('oob_maxF1', thr_oob)]:
                rows.append(dict(featureset=fs, behaviour=b, seed=seed, n_trees=ntree, threshold_rule=thr_name, labels='original',
                                 prevalence_test=y[te].mean(), **metrics(y[te], p, thr)))
                # annotation-artifact sensitivity: same predictions, test labels with 1-frame gaps closed
                rows.append(dict(featureset=fs, behaviour=b, seed=seed, n_trees=ntree, threshold_rule=thr_name, labels='gap_closed_eval',
                                 prevalence_test=yc[te].mean(), **metrics(yc[te], p, thr)))
            if fs == 'engineered':
                preds.append(pd.DataFrame(dict(behaviour=b, seed=seed, frame=te, y=y[te], y_gapclosed=yc[te], p=p, thr_oob=thr_oob)))
                yh = (p >= 0.5).astype(int)
                reports[f'{b}_seed{seed}_thr0.5'] = classification_report(y[te], yh, target_names=[f'not_{b}', b], output_dict=True, zero_division=0)
                reports[f'{b}_seed{seed}_thrOOB'] = classification_report(y[te], (p >= thr_oob).astype(int), target_names=[f'not_{b}', b], output_dict=True, zero_division=0)
                cms[f'{b}_seed{seed}_thr0.5'] = confusion_matrix(y[te], yh).tolist()
                cms[f'{b}_seed{seed}_thrOOB'] = confusion_matrix(y[te], (p >= thr_oob).astype(int)).tolist()
                gap_frames = set(np.where((yc == 1) & (y == 0))[0])
                fp = te[(yh == 1) & (y[te] == 0)]
                reports[f'{b}_seed{seed}_FP_on_gap_frames'] = dict(n_FP=int(len(fp)), n_FP_on_gap=int(sum(f in gap_frames for f in fp)))
            print(fs, b, seed, f'{time.time()-t0:.0f}s', flush=True)
    # trained on gap-closed labels (engineered only, 3 seeds)
for b in BEHAVIOURS:
    yc = close_gaps(T[b].values)
    for seed in SEEDS[:3]:
        tr, te = train_test_split(np.arange(len(yc)), test_size=0.2, stratify=yc, random_state=seed)
        m = rf(500, seed).fit(FT.values[tr], yc[tr]); p = m.predict_proba(FT.values[te])[:, 1]
        thr = best_f1_threshold(yc[tr], np.nan_to_num(m.oob_decision_function_[:, 1]))
        rows.append(dict(featureset='engineered', behaviour=b, seed=seed, n_trees=500, threshold_rule='oob_maxF1', labels='gap_closed_train_and_eval',
                         prevalence_test=yc[te].mean(), **metrics(yc[te], p, thr)))
M = pd.DataFrame(rows); M.to_csv(f'{OUT}/together1_random_split_metrics.csv', index=False)
pd.concat(preds).to_csv(f'{OUT}/together1_test_predictions_engineered.csv', index=False)
json.dump(reports, open(f'{OUT}/classification_reports.json', 'w'), indent=1); json.dump(cms, open(f'{OUT}/confusion_matrices.json', 'w'), indent=1)
S = M.groupby(['featureset', 'behaviour', 'threshold_rule', 'labels'])[['AP', 'ROC_AUC', 'F1', 'precision', 'recall', 'threshold', 'prevalence_test']].agg(['mean', 'std'])
S.to_csv(f'{OUT}/together1_random_split_summary.csv'); print(S.round(3).to_string())

# ---- Reference recording ("second condition"): its own 517 SimBA features and our recomputed features ----
ref_feats = [c for c in R.columns if c not in POSE_COLS and not c.startswith('Probability_') and c not in BEHAVIOURS]
RSETS = {'reference_simba517': R[ref_feats], 'reference_engineered_recomputed': FR}
rrows, rpred = [], []
for fs, F in RSETS.items():
    for b in BEHAVIOURS:
        y = R[b].values
        for scheme in ['stratified_random_5fold', 'blocked_5fold']:
            for seed in SEEDS[:3]:
                if scheme == 'stratified_random_5fold':
                    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(F, y))
                else:
                    if seed > 0: continue
                    blocks = np.array_split(np.arange(len(y)), 5)
                    folds = [(np.setdiff1d(np.arange(len(y)), np.arange(max(bk[0] - 15, 0), min(bk[-1] + 16, len(y)))), bk) for bk in blocks]
                p = np.full(len(y), np.nan)
                for tr, te in folds:
                    if y[tr].sum() == 0: p[te] = 0.0; continue
                    m = rf(500, seed, oob=False).fit(F.values[tr], y[tr]); p[te] = m.predict_proba(F.values[te])[:, 1]
                rrows.append(dict(featureset=fs, behaviour=b, scheme=scheme, seed=seed, prevalence=y.mean(), n_bouts=len(bouts(y)), **metrics(y, p, 0.5)))
                rpred.append(pd.DataFrame(dict(featureset=fs, behaviour=b, scheme=scheme, seed=seed, frame=np.arange(len(y)), y=y, p=p)))
RM = pd.DataFrame(rrows); RM.to_csv(f'{OUT}/reference_cv_metrics.csv', index=False); pd.concat(rpred).to_csv(f'{OUT}/reference_cv_predictions.csv', index=False)
print(RM.groupby(['featureset', 'behaviour', 'scheme'])[['AP', 'ROC_AUC', 'F1', 'prevalence']].mean().round(3))

# ---- Cross-recording transfer on identical feature definitions ----
common_cols = [c for c in FT.columns if c in FR.columns]
trows = []
for b in BEHAVIOURS:
    for name, (Ftr, ytr, Fte, yte) in {'Together1_to_Reference': (FT[common_cols], T[b].values, FR[common_cols], R[b].values),
                                       'Reference_to_Together1': (FR[common_cols], R[b].values, FT[common_cols], T[b].values)}.items():
        m = rf(500, 0, oob=False).fit(Ftr.values, ytr); p = m.predict_proba(Fte.values)[:, 1]
        trows.append(dict(direction=name, behaviour=b, prevalence_target=yte.mean(), **metrics(yte, p, 0.5),
                          n_predicted_positive=int((p >= 0.5).sum()), max_p=float(p.max())))
TR = pd.DataFrame(trows); TR.to_csv(f'{OUT}/cross_recording_transfer.csv', index=False); print(TR.round(3))
print('total', time.time() - t0)
