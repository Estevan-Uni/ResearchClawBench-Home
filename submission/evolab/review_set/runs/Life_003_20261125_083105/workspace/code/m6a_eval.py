"""m6A detection: Uncalled4 vs Nanopolish alignments fed to m6Anet.
PR/ROC with paired bootstrap, DeLong, TP gain at matched precision (headline), threshold sweeps,
site_id-decile robustness, random half-split stability, score distributions."""
import json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

rng = np.random.default_rng(SEED)
lab = pd.read_csv(f"{DATA}/m6a_labels.csv")
pu = pd.read_csv(f"{DATA}/m6a_predictions_uncalled4.csv")
pn = pd.read_csv(f"{DATA}/m6a_predictions_nanopolish.csv")
df = lab.merge(pu.rename(columns={"probability": "U4"}), on="site_id").merge(
    pn.rename(columns={"probability": "NP"}), on="site_id")
y = df.label.values.astype(int); su = df.U4.values; sn = df.NP.values
N, P = len(y), int(y.sum())
R = {"n_sites": N, "n_pos": P, "prevalence": P / N}
PGRID = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
B = 2000


def tp_at_precision(yy, ss, grid):
    """Largest TP achievable at any score threshold (ties kept together) with precision >= p."""
    o = np.argsort(-ss, kind="stable")
    ys, s_sorted = yy[o], ss[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    last = np.r_[s_sorted[1:] != s_sorted[:-1], True]  # threshold boundaries
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp)
    out = []
    for p in grid:
        ok = prec >= p - 1e-12
        out.append(int(tp[ok].max()) if ok.any() else 0)
    return np.array(out)


def thresh_stats(yy, ss, t):
    c = ss >= t
    tp = int((c & (yy == 1)).sum()); fp = int((c & (yy == 0)).sum())
    return tp, fp, tp / max(tp + fp, 1), tp / max(yy.sum(), 1)


# point metrics
for k, s in [("U4", su), ("NP", sn)]:
    R[f"AUPRC_{k}"] = float(average_precision_score(y, s))
    R[f"AUROC_{k}"] = float(roc_auc_score(y, s))
tpu = tp_at_precision(y, su, PGRID); tpn = tp_at_precision(y, sn, PGRID)


# DeLong
def delong(y, a, b):
    pos, neg = y == 1, y == 0
    def comps(s):
        X, Y = s[pos], s[neg]
        # V10: for each positive, mean over negatives of psi
        allv = np.r_[X, Y]
        r_all = stats.rankdata(allv); r_x = stats.rankdata(X); r_y = stats.rankdata(Y)
        m, n = len(X), len(Y)
        auc = (r_all[:m].sum() - m * (m + 1) / 2) / (m * n)
        v10 = (r_all[:m] - r_x) / n
        v01 = 1 - (r_all[m:] - r_y) / m
        return auc, v10, v01
    a1, v10a, v01a = comps(a); a2, v10b, v01b = comps(b)
    s10 = np.cov(np.vstack([v10a, v10b])); s01 = np.cov(np.vstack([v01a, v01b]))
    m, n = pos.sum(), neg.sum()
    S = s10 / m + s01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var)
    return float(a1), float(a2), float(z), float(2 * stats.norm.sf(abs(z)))


R["delong"] = dict(zip(["AUROC_U4", "AUROC_NP", "z", "p"], delong(y, su, sn)))

# paired bootstrap
boot = dict(apu=[], apn=[], aru=[], arn=[], tpu=[], tpn=[], gain09=[])
for b in range(B):
    idx = rng.integers(0, N, N)
    yy, a, c = y[idx], su[idx], sn[idx]
    boot["apu"].append(average_precision_score(yy, a)); boot["apn"].append(average_precision_score(yy, c))
    boot["aru"].append(roc_auc_score(yy, a)); boot["arn"].append(roc_auc_score(yy, c))
    boot["tpu"].append(tp_at_precision(yy, a, PGRID)); boot["tpn"].append(tp_at_precision(yy, c, PGRID))
    t1 = thresh_stats(yy, a, 0.9)[0]; t2 = thresh_stats(yy, c, 0.9)[0]
    boot["gain09"].append(100 * (t1 / max(t2, 1) - 1))
for k in boot: boot[k] = np.array(boot[k])
ci = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
R["AUPRC_U4_CI"] = ci(boot["apu"]); R["AUPRC_NP_CI"] = ci(boot["apn"])
R["AUROC_U4_CI"] = ci(boot["aru"]); R["AUROC_NP_CI"] = ci(boot["arn"])
R["dAUPRC"] = R["AUPRC_U4"] - R["AUPRC_NP"]; R["dAUPRC_CI"] = ci(boot["apu"] - boot["apn"])
R["dAUROC"] = R["AUROC_U4"] - R["AUROC_NP"]; R["dAUROC_CI"] = ci(boot["aru"] - boot["arn"])
R["dAUPRC_boot_p_le0"] = float(np.mean(boot["apu"] - boot["apn"] <= 0))
R["relative_AUPRC_gain_pct"] = 100 * (R["AUPRC_U4"] / R["AUPRC_NP"] - 1)

# matched precision table
rows = []
for i, p in enumerate(PGRID):
    bu, bn = boot["tpu"][:, i], boot["tpn"][:, i]
    gain = np.where(bn > 0, 100 * (bu / np.maximum(bn, 1) - 1), np.nan)
    rows.append(dict(precision=p, TP_U4=int(tpu[i]), TP_U4_CI=ci(bu), TP_NP=int(tpn[i]), TP_NP_CI=ci(bn),
                     recall_U4=tpu[i] / P, recall_NP=tpn[i] / P, extra_TP=int(tpu[i] - tpn[i]),
                     extra_TP_CI=ci(bu - bn),
                     gain_pct=float(100 * (tpu[i] / tpn[i] - 1)) if tpn[i] > 0 else None,
                     gain_pct_CI=ci(gain[~np.isnan(gain)]) if np.isfinite(gain).sum() > 100 else None,
                     frac_boot_NP_zero=float(np.mean(bn == 0))))
mp = pd.DataFrame(rows); mp.to_csv(f"{OUT}/m6a_matched_precision.csv", index=False)
R["matched_precision"] = rows

# fixed-threshold comparisons
thr = []
for t in [0.5, 0.7, 0.8, 0.9, 0.95]:
    a = thresh_stats(y, su, t); c = thresh_stats(y, sn, t)
    thr.append(dict(threshold=t, TP_U4=a[0], FP_U4=a[1], prec_U4=a[2], rec_U4=a[3],
                    TP_NP=c[0], FP_NP=c[1], prec_NP=c[2], rec_NP=c[3],
                    gain_pct=100 * (a[0] / c[0] - 1) if c[0] else None))
R["fixed_threshold"] = thr
R["gain09_CI"] = ci(boot["gain09"])
pd.DataFrame(thr).to_csv(f"{OUT}/m6a_fixed_threshold.csv", index=False)

# McNemar at 0.5 (correctness of call)
cu = (su >= 0.5) == (y == 1); cn = (sn >= 0.5) == (y == 1)
b01 = int((cu & ~cn).sum()); b10 = int((~cu & cn).sum())
R["mcnemar_0.5"] = dict(U4_right_NP_wrong=b01, NP_right_U4_wrong=b10,
                        p=float(stats.binomtest(b01, b01 + b10).pvalue))

# precision@K
R["precision_at_K"] = []
for Kk in [100, 250, 500, 1024]:
    R["precision_at_K"].append(dict(K=Kk, U4=float(y[np.argsort(-su, kind="stable")[:Kk]].mean()),
                                    NP=float(y[np.argsort(-sn, kind="stable")[:Kk]].mean())))

# score distribution / point masses / agreement
R["point_masses"] = {f"{k}_{v}_label{l}": int(((s == v) & (y == l)).sum())
                     for k, s in [("U4", su), ("NP", sn)] for v in [0.0, 1.0] for l in [0, 1]}
R["spearman_U4_NP"] = float(stats.spearmanr(su, sn)[0])
R["spearman_U4_NP_pos"] = float(stats.spearmanr(su[y == 1], sn[y == 1])[0])
R["spearman_U4_NP_neg"] = float(stats.spearmanr(su[y == 0], sn[y == 0])[0])
R["KS_U4"] = float(stats.ks_2samp(su[y == 1], su[y == 0]).statistic)
R["KS_NP"] = float(stats.ks_2samp(sn[y == 1], sn[y == 0]).statistic)
R["median_score"] = {f"{k}_label{l}": float(np.median(s[y == l])) for k, s in [("U4", su), ("NP", sn)] for l in [0, 1]}
# sensitivity: remove point masses (exact 0/1) and recompute AUPRC
m = (su > 0) & (su < 1) & (sn > 0) & (sn < 1)
R["no_pointmass"] = dict(n=int(m.sum()), n_pos=int(y[m].sum()),
                         AUPRC_U4=float(average_precision_score(y[m], su[m])),
                         AUPRC_NP=float(average_precision_score(y[m], sn[m])),
                         AUROC_U4=float(roc_auc_score(y[m], su[m])), AUROC_NP=float(roc_auc_score(y[m], sn[m])))
# rank-only version (ties broken randomly) to show ties do not drive result
jit = rng.uniform(0, 1e-9, N)
R["AUPRC_tiebreak"] = dict(U4=float(average_precision_score(y, su + jit)), NP=float(average_precision_score(y, sn + jit)))

# site_id-decile robustness
dec = pd.qcut(df.site_id, 10, labels=False)
drows = []
for d in range(10):
    mm = dec.values == d
    drows.append(dict(decile=d + 1, n=int(mm.sum()), n_pos=int(y[mm].sum()),
                      AUPRC_U4=float(average_precision_score(y[mm], su[mm])),
                      AUPRC_NP=float(average_precision_score(y[mm], sn[mm])),
                      AUROC_U4=float(roc_auc_score(y[mm], su[mm])), AUROC_NP=float(roc_auc_score(y[mm], sn[mm]))))
drows = pd.DataFrame(drows); drows.to_csv(f"{OUT}/m6a_siteid_deciles.csv", index=False)
R["decile_dAUPRC_min_max"] = [float((drows.AUPRC_U4 - drows.AUPRC_NP).min()), float((drows.AUPRC_U4 - drows.AUPRC_NP).max())]
R["decile_U4_wins"] = int((drows.AUPRC_U4 > drows.AUPRC_NP).sum())
R["decile_posrate_chi2_p"] = float(stats.chi2_contingency(np.c_[drows.n_pos, drows.n - drows.n_pos])[1])
# label autocorrelation along site_id (does order carry structure?)
R["label_lag1_autocorr"] = float(np.corrcoef(y[:-1], y[1:])[0, 1])
R["spearman_siteid_U4"] = float(stats.spearmanr(df.site_id, su)[0])
R["spearman_siteid_NP"] = float(stats.spearmanr(df.site_id, sn)[0])

# random half-split stability of TP gain
hs = []
for r_ in range(500):
    perm = rng.permutation(N); h = perm[: N // 2]
    a = tp_at_precision(y[h], su[h], PGRID); c = tp_at_precision(y[h], sn[h], PGRID)
    hs.append([100 * (a[i] / c[i] - 1) if c[i] > 0 else np.nan for i in range(len(PGRID))]
              + [average_precision_score(y[h], su[h]) - average_precision_score(y[h], sn[h])])
hs = np.array(hs)
R["halfsplit"] = [dict(precision=p, median_gain=float(np.nanmedian(hs[:, i])), q025=float(np.nanquantile(hs[:, i], .025)),
                       q975=float(np.nanquantile(hs[:, i], .975)), frac_positive=float(np.nanmean(hs[:, i] > 0)),
                       frac_NP_zero=float(np.mean(np.isnan(hs[:, i]))))
                  for i, p in enumerate(PGRID)]
R["halfsplit_dAUPRC"] = dict(median=float(np.median(hs[:, -1])), min=float(hs[:, -1].min()), frac_pos=float(np.mean(hs[:, -1] > 0)))
json.dump(R, open(f"{OUT}/m6a_results.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in R.items() if k not in ["matched_precision", "halfsplit"]}, indent=1, default=float))
print(mp.to_string())
print(pd.DataFrame(R["halfsplit"]).to_string())

# ---------------- figures ----------------
CU, CN = TOOL_COLORS["Uncalled4"], TOOL_COLORS["Nanopolish"]
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
ax = axes[0]
for s, c, k in [(su, CU, "U4"), (sn, CN, "NP")]:
    pr, rc, _ = precision_recall_curve(y, s)
    ax.plot(rc, pr, color=c, lw=2, label=f"{'Uncalled4' if k == 'U4' else 'Nanopolish'}  AUPRC={R['AUPRC_' + k]:.3f} "
                                          f"[{R['AUPRC_' + k + '_CI'][0]:.3f}–{R['AUPRC_' + k + '_CI'][1]:.3f}]")
ax.axhline(P / N, color="k", ls=":", lw=1, label=f"prevalence = {P / N:.3f}")
for p in [0.8, 0.9]:
    ax.axhline(p, color="#bbbbbb", lw=0.7)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision–recall (m6Anet site level)")
ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.set_ylim(0, 1.02)
ax = axes[1]
for s, c, k in [(su, CU, "U4"), (sn, CN, "NP")]:
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, color=c, lw=2, label=f"{'Uncalled4' if k == 'U4' else 'Nanopolish'}  AUROC={R['AUROC_' + k]:.3f}")
ax.plot([0, 1], [0, 1], "k:", lw=1)
ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
ax.set_title(f"ROC (DeLong p = {R['delong']['p']:.1e})"); ax.legend(frameon=False, fontsize=8, loc="lower right")
ax = axes[2]
ax.hist(boot["apu"] - boot["apn"], bins=50, color=CU, alpha=0.7, label="ΔAUPRC")
ax.hist(boot["aru"] - boot["arn"], bins=50, color="#e07b39", alpha=0.7, label="ΔAUROC")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("Uncalled4 − Nanopolish"); ax.set_ylabel("bootstrap replicates")
ax.set_title(f"Paired bootstrap (B={B})"); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig(f"{IMG}/fig7_m6a_pr_roc.png", dpi=150); plt.close()

# headline TP gain figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
pp = np.array(PGRID)
for k, col, lab_ in [("tpu", CU, "Uncalled4"), ("tpn", CN, "Nanopolish")]:
    med = tpu if k == "tpu" else tpn
    lo = np.quantile(boot[k], .025, axis=0); hi = np.quantile(boot[k], .975, axis=0)
    ax.errorbar(pp, med, yerr=[med - lo, hi - med], fmt="o-", color=col, capsize=3, label=lab_)
ax.set_xlabel("Precision floor"); ax.set_ylabel(f"True-positive m6A sites (of {P})")
ax.set_title("Maximum TP sites at matched precision (95% bootstrap CI)"); ax.legend(frameon=False)
ax = axes[1]
g = mp.gain_pct.values.astype(float)
glo = [r["gain_pct_CI"][0] if r["gain_pct_CI"] else np.nan for r in rows]
ghi = [r["gain_pct_CI"][1] if r["gain_pct_CI"] else np.nan for r in rows]
ok = np.isfinite(g)
ax.errorbar(pp[ok], g[ok], yerr=[g[ok] - np.array(glo)[ok], np.array(ghi)[ok] - g[ok]], fmt="s-", color=CU, capsize=3,
            label="full data (bootstrap 95% CI)")
hsm = [h["median_gain"] for h in R["halfsplit"]]; hlo = [h["q025"] for h in R["halfsplit"]]; hhi = [h["q975"] for h in R["halfsplit"]]
ax.fill_between(pp, hlo, hhi, color="#e07b39", alpha=0.2, label="random half-splits (2.5–97.5%)")
ax.plot(pp, hsm, "--", color="#e07b39")
ax.axhline(0, color="k", lw=1)
ax.set_ylim(0, None)
ax.set_xlabel("Precision floor"); ax.set_ylabel("% more TP sites with Uncalled4")
ax.set_title("True-positive gain of Uncalled4 over Nanopolish"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{IMG}/fig8_m6a_tp_gain.png", dpi=150); plt.close()

# distributions + threshold sweep + deciles
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
ax = axes[0]
bins = np.linspace(0, 1, 41)
for s, c, k in [(su, CU, "Uncalled4"), (sn, CN, "Nanopolish")]:
    ax.hist(s[y == 1], bins=bins, histtype="step", lw=2, color=c, label=f"{k}, m6A (label 1)")
    ax.hist(s[y == 0], bins=bins, histtype="step", lw=1.2, ls="--", color=c, label=f"{k}, unmodified")
ax.set_yscale("log"); ax.set_xlabel("m6Anet site probability"); ax.set_ylabel("sites [log]")
ax.set_title("Score distributions by ground-truth label"); ax.legend(frameon=False, fontsize=7.5)
ax = axes[1]
ts = np.linspace(0.01, 0.99, 99)
for s, c, k in [(su, CU, "Uncalled4"), (sn, CN, "Nanopolish")]:
    st = np.array([thresh_stats(y, s, t) for t in ts])
    ax.plot(ts, st[:, 2], color=c, lw=2, label=f"{k} precision")
    ax.plot(ts, st[:, 3], color=c, lw=2, ls="--", label=f"{k} recall")
ax.axvline(0.9, color="k", lw=0.7, ls=":")
ax.set_xlabel("probability threshold"); ax.set_ylabel("value"); ax.set_title("Threshold sweep")
ax.legend(frameon=False, fontsize=8)
ax = axes[2]
xd = drows.decile.values
ax.plot(xd, drows.AUPRC_U4, "o-", color=CU, label="Uncalled4")
ax.plot(xd, drows.AUPRC_NP, "o-", color=CN, label="Nanopolish")
ax.plot(xd, drows.n_pos / drows.n, ":", color="k", label="positive rate")
ax.set_xlabel("site_id decile"); ax.set_ylabel("AUPRC"); ax.set_ylim(0, 1)
ax.set_title("Robustness across site_id deciles"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{IMG}/fig9_m6a_distributions_robustness.png", dpi=150); plt.close()

# scatter agreement
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(sn[y == 0], su[y == 0], s=4, alpha=0.3, color="#999999", label="unmodified")
ax.scatter(sn[y == 1], su[y == 1], s=6, alpha=0.5, color="#d62839", label="m6A")
ax.plot([0, 1], [0, 1], "k:", lw=1)
ax.set_xlabel("Nanopolish-based probability"); ax.set_ylabel("Uncalled4-based probability")
ax.set_title(f"Per-site agreement (Spearman ρ={R['spearman_U4_NP']:.2f})"); ax.legend(frameon=False, markerscale=3)
plt.tight_layout(); plt.savefig(f"{IMG}/figS2_m6a_scatter.png", dpi=150); plt.close()
