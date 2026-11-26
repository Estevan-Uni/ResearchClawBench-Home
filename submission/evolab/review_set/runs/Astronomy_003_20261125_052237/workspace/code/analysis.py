"""Statistical analysis of SXS-style catalog waveform-difference distributions.

Reads data/fig6_data.csv, fig7_data.csv, fig8_data.csv and writes outputs/summary.json.
Run from the workspace root:  python code/analysis.py
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats, optimize

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261125)
NBOOT = 5000


def boot(x, fn, n=NBOOT):
    idx = rng.integers(0, len(x), size=(n, len(x)))
    vals = np.array([fn(x[i]) for i in idx])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], float(vals.std(ddof=1))


def describe(x):
    lx = np.log10(x)
    med_ci, med_se = boot(x, np.median)
    p95_ci, _ = boot(x, lambda v: np.percentile(v, 95))
    d = dict(
        n=int(len(x)), n_nonpositive=int((x <= 0).sum()), n_nan=int(np.isnan(x).sum()),
        min=float(x.min()), max=float(x.max()), median=float(np.median(x)),
        median_ci95=med_ci, median_se=med_se, mean=float(x.mean()),
        p5=float(np.percentile(x, 5)), p16=float(np.percentile(x, 16)),
        p84=float(np.percentile(x, 84)), p95=float(np.percentile(x, 95)), p95_ci95=p95_ci,
        p99=float(np.percentile(x, 99)),
        log10_mean=float(lx.mean()), log10_std=float(lx.std(ddof=1)),
        log10_skew=float(stats.skew(lx)), log10_kurtosis_excess=float(stats.kurtosis(lx)),
    )
    for name, thr in [("1em4", 1e-4), ("1em3", 1e-3), ("1em2", 1e-2)]:
        f = (x < thr).mean()
        ci, _ = boot(x, lambda v, t=thr: (v < t).mean())
        d[f"frac_lt_{name}"] = float(f)
        d[f"frac_lt_{name}_ci95"] = ci
    d["frac_1em4_to_1em2"] = float(((x >= 1e-4) & (x < 1e-2)).mean())
    return d


def family_fits(x):
    """Compare distribution families for log10(E) (and gamma / Weibull directly on E)."""
    lx = np.log10(x)
    res = {}
    cands = {"normal(log10E)": stats.norm, "student_t(log10E)": stats.t,
             "skewnorm(log10E)": stats.skewnorm, "logistic(log10E)": stats.logistic,
             "gumbel_r(log10E)": stats.gumbel_r}
    for name, dist in cands.items():
        p = dist.fit(lx)
        ll = dist.logpdf(lx, *p).sum()
        # convert log10-space likelihood to E-space for comparability with gamma fit
        ll_E = ll - np.sum(np.log(x * np.log(10)))
        res[name] = dict(params=[float(v) for v in p], k=len(p), loglike_E=float(ll_E),
                         AIC=float(2 * len(p) - 2 * ll_E),
                         KS_p=float(stats.kstest(lx, dist.cdf, args=p).pvalue))
    for name, dist in {"gamma(E)": stats.gamma, "weibull(E)": stats.weibull_min}.items():
        p = dist.fit(x, floc=0)
        ll = dist.logpdf(x, *p).sum()
        res[name] = dict(params=[float(v) for v in p], k=2, loglike_E=float(ll),
                         AIC=float(2 * 2 - 2 * ll),
                         KS_p=float(stats.kstest(x, dist.cdf, args=p).pvalue))
    best = min(res, key=lambda k: res[k]["AIC"])
    for k in res:
        res[k]["dAIC"] = res[k]["AIC"] - res[best]["AIC"]
    ad = stats.anderson(lx, dist="norm")
    return dict(families=res, best_by_AIC=best,
                shapiro_p=float(stats.shapiro(lx).pvalue),
                anderson_stat=float(ad.statistic),
                anderson_crit_5pct=float(ad.critical_values[2]),
                dagostino_p=float(stats.normaltest(lx).pvalue))


def max_of_n_check(x):
    """Is the observed maximum consistent with the max of N draws from the fitted log-normal?"""
    lx = np.log10(x)
    mu, sd = lx.mean(), lx.std(ddof=1)
    sims = rng.normal(mu, sd, size=(20000, len(x))).max(axis=1)
    obs = lx.max()
    return dict(obs_log10_max=float(obs), sim_log10_max_median=float(np.median(sims)),
                sim_log10_max_95=[float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))],
                p_value_ge=float((sims >= obs).mean()),
                expected_max_E=float(10 ** np.median(sims)),
                prob_any_ge_0p5=float(1 - stats.norm.cdf(np.log10(0.5), mu, sd) ** len(x)),
                obs_log10_min=float(lx.min()),
                sim_log10_min_95=[float(np.percentile(-rng.normal(-mu, sd, size=(5000, len(x))).max(axis=1), q)) for q in (2.5, 97.5)])


summary = {}

# ---------------- Fig 6: resolution error ----------------
f6 = pd.read_csv(os.path.join(DATA, "fig6_data.csv"))
E6 = f6["waveform_difference"].to_numpy(float)
summary["fig6"] = describe(E6)
summary["fig6"]["fits"] = family_fits(E6)
summary["fig6"]["max_of_n"] = max_of_n_check(E6)
# critical SNR (flat-noise approximation rho_crit = 1/sqrt(2E))
summary["fig6"]["rho_crit_median"] = float(1 / np.sqrt(2 * np.median(E6)))
summary["fig6"]["rho_crit_p95_sim"] = float(1 / np.sqrt(2 * np.percentile(E6, 95)))
summary["fig6"]["frac_rho_crit_gt_30"] = float((1 / np.sqrt(2 * E6) > 30).mean())
summary["fig6"]["frac_rho_crit_gt_100"] = float((1 / np.sqrt(2 * E6) > 100).mean())

# ---------------- Fig 7: per-ell error ----------------
f7 = pd.read_csv(os.path.join(DATA, "fig7_data.csv"))
ells = np.array([int(c[3:]) for c in f7.columns])
per_ell = {}
for c, l in zip(f7.columns, ells):
    x = f7[c].to_numpy(float)
    d = describe(x)
    d["fits"] = {"shapiro_p_log10": float(stats.shapiro(np.log10(x)).pvalue)}
    per_ell[int(l)] = d
med = np.array([per_ell[l]["median"] for l in ells])
se_log = np.array([per_ell[l]["median_se"] / per_ell[l]["median"] / np.log(10) for l in ells])
lmed = np.log10(med)
w = 1 / se_log ** 2


def wfit(X, y, w):
    A = np.vstack([np.ones_like(X), X]).T
    W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A)
    beta = cov @ A.T @ W @ y
    chi2 = float(np.sum(w * (y - A @ beta) ** 2))
    return beta, np.sqrt(np.diag(cov)), chi2


b_pl, e_pl, chi_pl = wfit(np.log10(ells.astype(float)), lmed, w)
b_ex, e_ex, chi_ex = wfit(ells.astype(float), lmed, w)
dof = len(ells) - 2
sig = np.array([per_ell[l]["log10_std"] for l in ells])
b_sig, e_sig, _ = wfit(ells.astype(float), sig, np.ones_like(sig))
mw = {}
for a, b in zip(ells[:-1], ells[1:]):
    p = stats.mannwhitneyu(f7[f"ell{b}"], f7[f"ell{a}"], alternative="greater").pvalue
    mw[f"{b}>{a}"] = float(p)
lev = stats.levene(*[np.log10(f7[c]) for c in f7.columns]).pvalue
spear7 = f7.apply(np.log10).corr(method="spearman").to_numpy()
offdiag = spear7[~np.eye(len(ells), dtype=bool)]
summary["fig7"] = dict(
    per_ell=per_ell,
    medians=med.tolist(), ells=ells.tolist(),
    median_ratio_l8_l2=float(med[-1] / med[0]),
    band16_84_width_dex=[float(np.log10(per_ell[l]["p84"] / per_ell[l]["p16"])) for l in ells],
    powerlaw=dict(log10A=float(b_pl[0]), slope=float(b_pl[1]), slope_err=float(e_pl[1]),
                  chi2=chi_pl, dof=dof, p=float(stats.chi2.sf(chi_pl, dof))),
    exponential=dict(log10A=float(b_ex[0]), dlog10_per_ell=float(b_ex[1]), err=float(e_ex[1]),
                     factor_per_ell=float(10 ** b_ex[1]),
                     chi2=chi_ex, dof=dof, p=float(stats.chi2.sf(chi_ex, dof))),
    sigma_log_vs_ell=dict(intercept=float(b_sig[0]), slope_dex_per_ell=float(b_sig[1])),
    mannwhitney_successive=mw, levene_p_log10=float(lev),
    spearman_offdiag_max_abs=float(np.abs(offdiag).max()),
    spearman_offdiag_mean=float(offdiag.mean()),
)

# Illustrative full-waveform combination: E_tot = sum_l w_l E_l, w_l = mode power fraction.
# Assumed power fractions for a comparable-mass, non-precessing binary (illustrative only):
# l=2 dominates (~95%+); each higher l down by ~ an order of magnitude-ish.
power = {2: 0.955, 3: 0.03, 4: 0.01, 5: 0.003, 6: 0.0012, 7: 0.0005, 8: 0.0003}
pw = np.array([power[l] for l in ells]); pw = pw / pw.sum()
Ecomb = (f7.to_numpy() * pw).sum(axis=1)
contrib = (np.median(f7.to_numpy(), axis=0) * pw)
summary["fig7"]["combination"] = dict(
    assumed_power_fractions=dict(zip([int(l) for l in ells], pw.tolist())),
    median_combined=float(np.median(Ecomb)), log10_std_combined=float(np.log10(Ecomb).std(ddof=1)),
    fractional_contrib_of_median=dict(zip([int(l) for l in ells], (contrib / contrib.sum()).tolist())),
    ell2_share=float(contrib[0] / contrib.sum()),
    fig6_log10_std=summary["fig6"]["log10_std"],
)
# truncation: drop modes l > lmax, error introduced ~ power fraction of the dropped modes
summary["fig7"]["truncation_power_dropped"] = {int(L): float(pw[ells > L].sum()) for L in ells}

# ---------------- Fig 8: extrapolation order ----------------
f8 = pd.read_csv(os.path.join(DATA, "fig8_data.csv"))
E3 = f8["N2vsN3"].to_numpy(float)
E4 = f8["N2vsN4"].to_numpy(float)
summary["fig8"] = dict(N2vsN3=describe(E3), N2vsN4=describe(E4))
summary["fig8"]["N2vsN3"]["fits"] = family_fits(E3)
summary["fig8"]["N2vsN4"]["fits"] = family_fits(E4)
m6 = np.median(E6)
# bootstrap the ratio of medians
r3, r4 = [], []
for _ in range(NBOOT):
    a = np.median(rng.choice(E6, len(E6)))
    r3.append(a / np.median(rng.choice(E3, len(E3))))
    r4.append(a / np.median(rng.choice(E4, len(E4))))
summary["fig8"]["ratio_res_over_N3"] = float(m6 / np.median(E3))
summary["fig8"]["ratio_res_over_N3_ci95"] = [float(np.percentile(r3, 2.5)), float(np.percentile(r3, 97.5))]
summary["fig8"]["ratio_res_over_N4"] = float(m6 / np.median(E4))
summary["fig8"]["ratio_res_over_N4_ci95"] = [float(np.percentile(r4, 2.5)), float(np.percentile(r4, 97.5))]
summary["fig8"]["ratio_N4_over_N3"] = float(np.median(E4) / np.median(E3))
summary["fig8"]["mannwhitney_N4_gt_N3_p"] = float(stats.mannwhitneyu(E4, E3, alternative="greater").pvalue)
summary["fig8"]["wilcoxon_paired_p"] = float(stats.wilcoxon(E4, E3, alternative="greater").pvalue)
summary["fig8"]["frac_rows_N4_gt_N3"] = float((E4 > E3).mean())
summary["fig8"]["spearman_N3_N4"] = float(stats.spearmanr(E3, E4).correlation)
summary["fig8"]["frac_N3_gt_res_median"] = float((E3 > m6).mean())
summary["fig8"]["frac_N4_gt_res_median"] = float((E4 > m6).mean())
# probability a random extrapolation error exceeds a random resolution error (population level)
summary["fig8"]["P_N3_gt_res"] = float((rng.choice(E3, 200000) > rng.choice(E6, 200000)).mean())
summary["fig8"]["P_N4_gt_res"] = float((rng.choice(E4, 200000) > rng.choice(E6, 200000)).mean())
# KS test: are N2vsN3, N2vsN4 merely shifted copies (same log-width)?
summary["fig8"]["log10_std_ratio_N4_N3"] = float(np.log10(E4).std() / np.log10(E3).std())

# ---------------- combined error budget ----------------
n = 200000
s6 = rng.choice(E6, n); s3 = rng.choice(E3, n)
inc = s6 + s3                              # incoherent: mismatches add
coh = (np.sqrt(s6) + np.sqrt(s3)) ** 2     # coherent: residual norms add linearly
summary["budget"] = dict(
    median_res_only=float(np.median(s6)), median_incoherent=float(np.median(inc)),
    median_coherent=float(np.median(coh)),
    frac_increase_incoherent=float(np.median(inc) / np.median(s6) - 1),
    frac_increase_coherent=float(np.median(coh) / np.median(s6) - 1),
)

# ---------------- cross-file structure ----------------
summary["structure"] = dict(
    spearman_fig7_offdiag_max_abs=summary["fig7"]["spearman_offdiag_max_abs"],
    spearman_fig6_vs_fig7_ell2=float(stats.spearmanr(E6, f7["ell2"]).correlation),
    spearman_fig8=summary["fig8"]["spearman_N3_N4"],
    fig6_vs_ell2_KS_p=float(stats.ks_2samp(E6, f7["ell2"]).pvalue),
    fig6_log10_std=summary["fig6"]["log10_std"],
    ell2_log10_std=per_ell[2]["log10_std"],
)

with open(os.path.join(OUT, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)

# compact table outputs
rows = []
for l in ells:
    d = per_ell[l]
    rows.append(dict(ell=l, median=d["median"], ci_lo=d["median_ci95"][0], ci_hi=d["median_ci95"][1],
                     p16=d["p16"], p84=d["p84"], p5=d["p5"], p95=d["p95"], log10_std=d["log10_std"],
                     shapiro_p=d["fits"]["shapiro_p_log10"]))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "fig7_per_ell_stats.csv"), index=False)
fam = []
for lab, d in [("fig6", summary["fig6"]), ("N2vsN3", summary["fig8"]["N2vsN3"]), ("N2vsN4", summary["fig8"]["N2vsN4"])]:
    for k, v in d["fits"]["families"].items():
        fam.append(dict(dataset=lab, family=k, AIC=v["AIC"], dAIC=v["dAIC"], KS_p=v["KS_p"]))
pd.DataFrame(fam).to_csv(os.path.join(OUT, "family_fits.csv"), index=False)
print(json.dumps({k: summary[k] for k in ["budget", "structure"]}, indent=1))
print("fig6 median", summary["fig6"]["median"], summary["fig6"]["median_ci95"], "max", summary["fig6"]["max"])
print("fig7 medians", med, "ratio", med[-1] / med[0])
print("powerlaw", summary["fig7"]["powerlaw"]); print("exp", summary["fig7"]["exponential"])
print("fig8 medians", np.median(E3), np.median(E4), summary["fig8"]["ratio_res_over_N3"], summary["fig8"]["ratio_res_over_N4"])
