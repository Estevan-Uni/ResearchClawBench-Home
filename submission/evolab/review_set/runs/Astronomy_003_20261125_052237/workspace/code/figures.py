"""Figures for the SXS-style error-distribution study. Run after analysis.py:
    python code/figures.py
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
IMG = os.path.join(ROOT, "report", "images")
os.makedirs(IMG, exist_ok=True)
S = json.load(open(os.path.join(ROOT, "outputs", "summary.json")))
rng = np.random.default_rng(7)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 160, "figure.dpi": 100})
BLUE, ORANGE, GREEN, GRAY, RED = "#2a6fb0", "#e07b28", "#3a9a5b", "#7f7f7f", "#c23b3b"

E6 = pd.read_csv(os.path.join(DATA, "fig6_data.csv"))["waveform_difference"].to_numpy()
f7 = pd.read_csv(os.path.join(DATA, "fig7_data.csv"))
f8 = pd.read_csv(os.path.join(DATA, "fig8_data.csv"))
E3, E4 = f8["N2vsN3"].to_numpy(), f8["N2vsN4"].to_numpy()
ells = np.array(S["fig7"]["ells"])


def fmt(v):
    e = int(np.floor(np.log10(v)))
    m = v / 10 ** e
    return rf"{m:.1f}\times10^{{{e}}}"


def lognormal_counts(x, bins):
    """Expected counts per (log-spaced) bin under the fitted log-normal."""
    lx = np.log10(x)
    mu, sd = lx.mean(), lx.std(ddof=1)
    c = stats.norm.cdf(np.log10(bins), mu, sd)
    return len(x) * np.diff(c)


# ---------------------------------------------------------------- Fig 6
bins = np.logspace(-6, 0, 61)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axes[0]
ax.axvspan(1e-4, 1e-2, color=GREEN, alpha=0.08, label=r"$10^{-4}\leq\mathcal{E}<10^{-2}$" +
           f" ({100*S['fig6']['frac_1em4_to_1em2']:.1f}%)")
ax.hist(E6, bins=bins, color=BLUE, alpha=0.85, edgecolor="white", lw=0.4, label=f"SXS sims (N={len(E6)})")
exp = lognormal_counts(E6, bins)
ctr = np.sqrt(bins[1:] * bins[:-1])
ax.plot(ctr, exp, color="k", lw=1.3, ls="--", label="log-normal fit")
m = S["fig6"]["median"]
ax.axvline(m, color=RED, lw=1.8, label=rf"median $={fmt(m)}$")
ax.text(m * 1.25, 160, rf"median $\mathcal{{E}}={fmt(m)}$", color=RED, fontsize=10)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-6, 1); ax.set_ylim(0.7, 400)
ax.set_xlabel(r"waveform difference $\mathcal{E}$ (two highest resolutions)")
ax.set_ylabel("number of simulations")
ax.set_title("(a) log-spaced bins, logarithmic axes")
ax.legend(fontsize=9, loc="upper left", frameon=False)

ax = axes[1]
lbins = np.linspace(0, E6.max() * 1.02, 60)
ax.hist(E6, bins=lbins, color=BLUE, alpha=0.85, edgecolor="white", lw=0.4)
lx = np.log10(E6); mu, sd = lx.mean(), lx.std(ddof=1)
c = stats.norm.cdf(np.log10(np.clip(lbins, 1e-12, None)), mu, sd)
ax.plot(0.5 * (lbins[1:] + lbins[:-1]), len(E6) * np.diff(c), "k--", lw=1.3, label="log-normal fit")
ax.axvline(m, color=RED, lw=1.8, label=rf"median $={fmt(m)}$")
ax.axvline(S["fig6"]["p95"], color=ORANGE, lw=1.4, ls=":", label=rf"95th pct $={fmt(S['fig6']['p95'])}$")
ax.set_yscale("log"); ax.set_ylim(0.7, 2000)
ax.set_xlabel(r"waveform difference $\mathcal{E}$ (linear)")
ax.set_ylabel("number of simulations")
ax.set_title("(b) linear bins, logarithmic count axis")
ax.legend(fontsize=9, frameon=False)
fig.suptitle("Reproduction of catalog Fig. 6: resolution error of the SXS simulations", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "fig6_resolution_histogram.png"), bbox_inches="tight")
plt.close(fig)

# ECDF version of Fig 6 with thresholds
fig, ax = plt.subplots(figsize=(6.4, 4.4))
xs = np.sort(E6)
ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), color=BLUE, where="post", label="empirical CDF")
gx = np.logspace(-6, 0, 400)
ax.plot(gx, stats.norm.cdf(np.log10(gx), mu, sd), "k--", lw=1, label="log-normal fit")
for thr, key in [(1e-4, "frac_lt_1em4"), (1e-3, "frac_lt_1em3"), (1e-2, "frac_lt_1em2")]:
    f = S["fig6"][key]
    ax.axvline(thr, color=GRAY, lw=0.8, ls=":")
    ax.text(thr * 1.15, f - 0.07 if thr > 1e-4 else f + 0.03, f"{100*f:.1f}%", fontsize=9)
ax.set_xscale("log"); ax.set_xlim(1e-6, 1)
ax.set_xlabel(r"$\mathcal{E}$"); ax.set_ylabel(r"fraction of simulations with difference $<\mathcal{E}$")
ax.set_title("Cumulative resolution-error distribution")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig6_resolution_cdf.png")); plt.close(fig)

# ---------------------------------------------------------------- Fig 7
cmap = plt.get_cmap("viridis")
cols = [cmap(i / (len(ells) - 1) * 0.9) for i in range(len(ells))]
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
ax = axes[0]
b7 = np.logspace(-6, 0, 61)
for l, cl in zip(ells, cols):
    x = f7[f"ell{l}"].to_numpy()
    ax.hist(x, bins=b7, histtype="step", lw=1.6, color=cl,
            label=rf"$\ell={l}$: median ${fmt(np.median(x))}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-5, 0.2); ax.set_ylim(0.7, 5000)
ax.set_xlabel(r"waveform difference $\mathcal{E}_\ell$ (mode $\ell$ only)")
ax.set_ylabel("number of simulations")
ax.set_title(r"(a) per-$\ell$ resolution-error histograms")
ax.legend(fontsize=8.5, frameon=False, loc="upper right")

ax = axes[1]
P = {l: S["fig7"]["per_ell"][str(l)] for l in ells}
med = np.array([P[l]["median"] for l in ells])
p5 = np.array([P[l]["p5"] for l in ells]); p95 = np.array([P[l]["p95"] for l in ells])
p16 = np.array([P[l]["p16"] for l in ells]); p84 = np.array([P[l]["p84"] for l in ells])
ci = np.array([P[l]["median_ci95"] for l in ells])
ax.fill_between(ells, p5, p95, color=BLUE, alpha=0.12, label="5th–95th percentile")
ax.fill_between(ells, p16, p84, color=BLUE, alpha=0.30, label="16th–84th percentile")
ax.errorbar(ells, med, yerr=[med - ci[:, 0], ci[:, 1] - med], fmt="o", color=BLUE, ms=6,
            capsize=3, label="median (95% bootstrap CI)")
pl = S["fig7"]["powerlaw"]; ex = S["fig7"]["exponential"]
lg = np.linspace(2, 8, 100)
ax.plot(lg, 10 ** (pl["log10A"] + pl["slope"] * np.log10(lg)), color=RED, lw=1.5,
        label=rf"power law $\propto\ell^{{{pl['slope']:.2f}\pm{pl['slope_err']:.2f}}}$ ($\chi^2/\nu={pl['chi2']:.1f}/{pl['dof']}$)")
ax.plot(lg, 10 ** (ex["log10A"] + ex["dlog10_per_ell"] * lg), color=GRAY, lw=1.2, ls="--",
        label=rf"exponential $\propto{ex['factor_per_ell']:.2f}^{{\ell}}$ ($\chi^2/\nu={ex['chi2']:.0f}/{ex['dof']}$)")
for l, mv, w in zip(ells, med, S["fig7"]["band16_84_width_dex"]):
    ax.text(l, p95[l - 2] * 1.25, f"{w:.2f} dex", ha="center", fontsize=8, color=BLUE)
ax.annotate(rf"median ratio $\ell{{=}}8/\ell{{=}}2 = {S['fig7']['median_ratio_l8_l2']:.1f}$",
            xy=(0.03, 0.04), xycoords="axes fraction", fontsize=10)
ax.set_yscale("log"); ax.set_xticks(ells)
ax.set_ylim(3e-5, 5e-2)
ax.set_xlabel(r"spherical-harmonic index $\ell$"); ax.set_ylabel(r"$\mathcal{E}_\ell$")
ax.set_title(r"(b) median and percentile bands vs $\ell$ (labels: 16–84% width)")
ax.legend(fontsize=8.3, frameon=False, loc="upper left")
fig.suptitle(r"Reproduction of catalog Fig. 7: resolution error decomposed by mode $\ell$", y=1.01)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig7_per_ell_errors.png"), bbox_inches="tight"); plt.close(fig)

# Fig 7 supplementary: violins + sigma growth + assumed-power contributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
ax = axes[0]
parts = ax.violinplot([np.log10(f7[f"ell{l}"]) for l in ells], positions=ells, showmedians=True, widths=0.8)
for pc, cl in zip(parts["bodies"], cols):
    pc.set_facecolor(cl); pc.set_alpha(0.6)
ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$\log_{10}\mathcal{E}_\ell$"); ax.set_title("(a) per-mode distributions")
ax = axes[1]
sig = np.array([P[l]["log10_std"] for l in ells])
sl = S["fig7"]["sigma_log_vs_ell"]
ax.plot(ells, sig, "o", color=BLUE, label=r"$\sigma(\log_{10}\mathcal{E}_\ell)$")
ax.plot(ells, sl["intercept"] + sl["slope_dex_per_ell"] * ells, color=RED,
        label=f"linear: +{sl['slope_dex_per_ell']:.3f} dex per $\\ell$")
ax.axhline(S["fig6"]["log10_std"], color=GRAY, ls=":", label="full-waveform (Fig. 6) width")
ax.set_xlabel(r"$\ell$"); ax.set_ylabel("log-scatter [dex]"); ax.set_title("(b) scatter grows with $\\ell$")
ax.legend(frameon=False, fontsize=9)
ax = axes[2]
comb = S["fig7"]["combination"]
pw = np.array([comb["assumed_power_fractions"][str(l)] for l in ells])
fc = np.array([comb["fractional_contrib_of_median"][str(l)] for l in ells])
wdt = 0.38
ax.bar(ells - wdt / 2, pw, wdt, color=GRAY, label="assumed power fraction $w_\\ell$")
ax.bar(ells + wdt / 2, fc, wdt, color=BLUE, label=r"share of $\sum_\ell w_\ell\,\tilde{\mathcal{E}}_\ell$")
ax.set_yscale("log"); ax.set_xlabel(r"$\ell$"); ax.set_title("(c) illustrative power-weighted budget")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig7_supplementary.png")); plt.close(fig)

# ---------------------------------------------------------------- Fig 8
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
b8 = np.logspace(-8, -1, 71)
m3, m4 = np.median(E3), np.median(E4)
ax.hist(E6, bins=b8, histtype="stepfilled", color=GRAY, alpha=0.18,
        label=rf"resolution (Fig. 6), median ${fmt(m)}$")
ax.hist(E3, bins=b8, histtype="step", lw=1.8, color=BLUE, label=rf"$N{{=}}2$ vs $N{{=}}3$, median ${fmt(m3)}$")
ax.hist(E4, bins=b8, histtype="step", lw=1.8, color=ORANGE, label=rf"$N{{=}}2$ vs $N{{=}}4$, median ${fmt(m4)}$")
ax.axvline(m3, color=BLUE, ls="--", lw=1); ax.axvline(m4, color=ORANGE, ls="--", lw=1)
ax.axvline(m, color="k", lw=1.6, ls="-.", label="resolution-error median (reference)")
ax.annotate("", xy=(m3, 260), xytext=(m, 260), arrowprops=dict(arrowstyle="<->", color=BLUE))
ax.text(np.sqrt(m * m3), 300, f"×{S['fig8']['ratio_res_over_N3']:.0f}", color=BLUE, ha="center")
ax.annotate("", xy=(m4, 150), xytext=(m, 150), arrowprops=dict(arrowstyle="<->", color=ORANGE))
ax.text(np.sqrt(m * m4), 170, f"×{S['fig8']['ratio_res_over_N4']:.0f}", color=ORANGE, ha="center")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-8, 1e-1); ax.set_ylim(0.7, 800)
ax.set_xlabel(r"waveform difference $\mathcal{E}$"); ax.set_ylabel("number of simulations")
ax.set_title("(a) extrapolation-order differences vs resolution error")
ax.legend(fontsize=8.8, frameon=False, loc="upper left")
ax = axes[1]
for x, cl, lab in [(E6, GRAY, "resolution"), (E3, BLUE, "N2 vs N3"), (E4, ORANGE, "N2 vs N4")]:
    xs = np.sort(x); ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color=cl, lw=1.6, label=lab)
ax.axvline(m, color="k", ls="-.", lw=1)
ax.set_xscale("log"); ax.set_xlim(1e-8, 1e-1)
ax.set_xlabel(r"$\mathcal{E}$"); ax.set_ylabel("cumulative fraction of simulations")
ax.set_title("(b) cumulative distributions")
ax.legend(frameon=False, fontsize=9)
fig.suptitle("Reproduction of catalog Fig. 8: extrapolation-order waveform differences", y=1.01)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig8_extrapolation_histogram.png"), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Validation: Q-Q
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
for ax, x, lab, key in [(axes[0], E6, "resolution (Fig. 6)", S["fig6"]),
                        (axes[1], E3, "N2 vs N3", S["fig8"]["N2vsN3"]),
                        (axes[2], E4, "N2 vs N4", S["fig8"]["N2vsN4"])]:
    lx = np.sort(np.log10(x)); n = len(lx)
    mu, sd = lx.mean(), lx.std(ddof=1)
    q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    sims = np.sort(rng.normal(0, 1, size=(2000, n)), axis=1)
    lo, hi = np.percentile(sims, [2.5, 97.5], axis=0)
    ax.fill_between(q, mu + sd * lo, mu + sd * hi, color=GRAY, alpha=0.3, label="95% envelope (log-normal)")
    ax.plot(q, lx, ".", ms=3, color=BLUE, label="data")
    ax.plot(q, mu + sd * q, "k-", lw=0.8)
    f = key["fits"]
    ax.set_title(f"{lab}\nShapiro p={f['shapiro_p']:.2f}, AD={f['anderson_stat']:.2f} (5% crit {f['anderson_crit_5pct']:.2f})", fontsize=10)
    ax.set_xlabel("standard-normal quantile"); ax.set_ylabel(r"$\log_{10}\mathcal{E}$")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(IMG, "validation_qq.png")); plt.close(fig)

# Validation: AIC comparison
fam = pd.read_csv(os.path.join(ROOT, "outputs", "family_fits.csv"))
fig, ax = plt.subplots(figsize=(9, 4))
dsets = ["fig6", "N2vsN3", "N2vsN4"]; fams = list(fam.family.unique())
wb = 0.12
for i, fm in enumerate(fams):
    vals = [fam[(fam.dataset == d) & (fam.family == fm)].dAIC.values[0] for d in dsets]
    ax.bar(np.arange(3) + (i - len(fams) / 2) * wb, np.array(vals) + 0.5, wb, label=fm)
ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels(["resolution", "N2 vs N3", "N2 vs N4"])
ax.axhline(2.5, color="k", ls=":", lw=0.8); ax.text(2.45, 2.7, "ΔAIC=2", fontsize=8)
ax.set_ylabel(r"$\Delta$AIC + 0.5 (relative to best)"); ax.set_title("Distribution-family comparison")
ax.legend(fontsize=8, ncol=2, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "validation_aic.png")); plt.close(fig)

# Structure diagnostics
allcols = pd.concat([pd.Series(E6, name="fig6"), f7.iloc[:len(E6)]], axis=1)
corr = np.log10(allcols).corr(method="spearman")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax, label="Spearman ρ"); ax.set_title("(a) row-wise rank correlations")
ax = axes[1]
w = np.array([comb["assumed_power_fractions"][str(l)] for l in ells])
Ec = (f7.to_numpy() * w).sum(axis=1)
bb = np.logspace(-6, -1, 51)
ax.hist(E6, bins=bb, histtype="step", color=BLUE, lw=1.6, label=f"Fig. 6 data (σ={S['fig6']['log10_std']:.2f} dex)")
ax.hist(Ec, bins=bb, histtype="step", color=ORANGE, lw=1.6,
        label=f"power-weighted Fig. 7 modes (σ={comb['log10_std_combined']:.2f} dex)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.7, 1000)
ax.set_xlabel(r"$\mathcal{E}$"); ax.set_ylabel("number of simulations")
ax.set_title("(b) Fig. 6 vs mode combination"); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "structure_diagnostics.png")); plt.close(fig)

# Overview: all error sources on one axis
fig, ax = plt.subplots(figsize=(8, 4.2))
data = [("resolution (Fig. 6)", E6, "k")] + [(rf"$\ell={l}$", f7[f"ell{l}"].to_numpy(), c) for l, c in zip(ells, cols)] + \
       [("N2 vs N3", E3, BLUE), ("N2 vs N4", E4, ORANGE)]
for i, (lab, x, c) in enumerate(data):
    p = np.percentile(x, [5, 16, 50, 84, 95])
    ax.plot([p[0], p[4]], [i, i], color=c, lw=1)
    ax.plot([p[1], p[3]], [i, i], color=c, lw=5, alpha=0.6, solid_capstyle="butt")
    ax.plot(p[2], i, "o", color=c, mec="white")
ax.set_yticks(range(len(data))); ax.set_yticklabels([d[0] for d in data]); ax.invert_yaxis()
ax.set_xscale("log"); ax.set_xlabel(r"waveform difference $\mathcal{E}$")
ax.axvline(m, color=RED, ls=":", lw=1)
ax.set_title("Data overview: median, 16–84% (thick) and 5–95% (thin) ranges")
fig.tight_layout(); fig.savefig(os.path.join(IMG, "data_overview.png")); plt.close(fig)
print("figures written:", sorted(os.listdir(IMG)))
