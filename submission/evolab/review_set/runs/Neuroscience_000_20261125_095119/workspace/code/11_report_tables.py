"""Emit markdown tables (from outputs/) used verbatim in the report."""
import pandas as pd, numpy as np, json
from common import *
M = pd.read_csv(f'{OUT}/together1_random_split_metrics.csv'); out = []
f = lambda s: f'{s.mean():.3f} ± {s.std(ddof=0):.3f}' if len(s) > 1 else f'{s.mean():.3f}'
out.append('### T1: Together_1 random 80/20 split\n\n| Feature set | Behaviour | Threshold | Labels | n seeds | AP | ROC-AUC | F1 | Precision | Recall |\n|---|---|---|---|---|---|---|---|---|---|')
for (fs, b, th, lab), g in M.groupby(['featureset', 'behaviour', 'threshold_rule', 'labels'], sort=False):
    out.append(f'| {fs} | {b} | {th} | {lab} | {len(g)} | {f(g.AP)} | {f(g.ROC_AUC)} | {f(g.F1)} | {f(g.precision)} | {f(g.recall)} |')
R = pd.read_csv(f'{OUT}/reference_cv_metrics.csv')
out.append('\n### T2: Reference recording CV\n\n| Feature set | Behaviour | Scheme | AP | ROC-AUC | F1@0.5 | prevalence | bouts |\n|---|---|---|---|---|---|---|---|')
for (fs, b, sc), g in R.groupby(['featureset', 'behaviour', 'scheme']):
    out.append(f'| {fs} | {b} | {sc} | {f(g.AP)} | {f(g.ROC_AUC)} | {f(g.F1)} | {g.prevalence.iloc[0]:.3f} | {g.n_bouts.iloc[0]} |')
TR = pd.read_csv(f'{OUT}/cross_recording_transfer.csv')
out.append('\n### T3: Transfer\n\n| Direction | Behaviour | target prevalence | AP | ROC-AUC | F1@0.5 | predicted positives | max p |\n|---|---|---|---|---|---|---|---|')
for r in TR.itertuples(): out.append(f'| {r.direction} | {r.behaviour} | {r.prevalence_target:.3f} | {r.AP:.3f} | {r.ROC_AUC:.3f} | {r.F1:.3f} | {r.n_predicted_positive} | {r.max_p:.3f} |')
B = pd.read_csv(f'{OUT}/blocked_cv_metrics.csv'); B = B[B.fold == 'pooled']
out.append('\n### T4: Blocked CV pooled\n\n| Feature set | Behaviour | AP | ROC-AUC | F1@0.5 | prevalence |\n|---|---|---|---|---|---|')
for r in B.itertuples(): out.append(f'| {r.featureset} | {r.behaviour} | {r.AP:.3f} | {r.AUC:.3f} | {r.F1:.3f} | {r.prevalence:.3f} |')
open(f'{OUT}/report_tables.md', 'w').write('\n'.join(out)); print('\n'.join(out))
