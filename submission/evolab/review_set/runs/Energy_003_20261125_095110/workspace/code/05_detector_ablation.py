"""Detector ablation: threshold k, reference mode, rules on/off, and a global z-score / IQR baseline (5 seeds, 5 % + 2 %)."""
import numpy as np, pandas as pd
from common import *
from cleaning import *
y = load_energy("BN001")["E"].values; rows = []
for seed in range(5):
    yc, miss, out, _ = corrupt(y, seed, 0.05, 0.02)
    for mode in ["daily", "weekly"]:
        for k in [3, 3.5, 4, 5, 6]:
            rows.append(dict(seed=seed, detector=f"Hampel-{mode} + rules", k=k, **prf(detect(yc, k, mode), out)))
    rows.append(dict(seed=seed, detector="Hampel-daily, no rules", k=4, **prf(detect(yc, 4, "daily", rules=False), out)))
    z = (yc - np.nanmean(yc)) / np.nanstd(yc)
    for k in [2, 3]:
        rows.append(dict(seed=seed, detector="global z-score", k=k, **prf(np.nan_to_num(np.abs(z)) > k, out)))
    q1, q3 = np.nanpercentile(yc, [25, 75]); iqr = q3 - q1
    rows.append(dict(seed=seed, detector="global IQR (1.5)", k=1.5, **prf(np.nan_to_num((yc < q1 - 1.5 * iqr) | (yc > q3 + 1.5 * iqr)), out)))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/detector_ablation_raw.csv", index=False)
A = R.groupby(["detector", "k"])[["precision", "recall", "F1"]].agg(["mean", "std"]).round(3)
A.to_csv(f"{OUT}/detector_ablation_summary.csv"); print(A.to_string())
