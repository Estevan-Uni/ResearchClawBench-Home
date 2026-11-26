"""Mechanism test (simulation, NOT real-data stratification): site-level m6A-like detection vs read
coverage for true alignment, basecaller-guided banded DTW and move-table-only alignment.
RNA001 5-mer model; a modified A shifts the current of the 5 k-mers that contain it."""
import json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simlib import *

rng = np.random.default_rng(SEED + 1)
M = Model("RNA001 (5-mer)")
K = M.K; c = K // 2
NSITE = 120          # per class
MAXCOV = 40
COVS = [5, 10, 20, 40]
STOICH = 0.5         # fraction of modified reads at modified sites
DELTA = 0.35         # current shift at centre position (model SD units)
WPOS = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
FLANK = 15
NOISE = 3.0
JIT = 1.5
recs = []
for site in range(2 * NSITE):
    modsite = site < NSITE
    seq = random_seq(rng, 2 * FLANK + K)
    seq[FLANK + c] = 0  # central A (in k-mer index FLANK the A sits at the centre position)
    codes = kmer_codes(seq, K); L = len(codes)
    # k-mers containing the A at seq pos FLANK+c: k-mer j contains it at position FLANK+c-j
    shift = np.zeros(L)
    for p in range(K):
        j = FLANK + c - p
        if 0 <= j < L: shift[j] = DELTA * WPOS[p]
    mu, sd = M.mu[codes], M.sd[codes] * NOISE
    for r in range(MAXCOV):
        is_mod = modsite and (rng.random() < STOICH)
        rd = simulate_read(rng, M, codes, noise_infl=NOISE, mod_shift=shift if is_mod else None)
        mvb = move_table(rng, rd, jitter_strides=JIT)
        mv_lab = labels_from_bounds(mvb, rd["T"])
        # normalisation: assume a long-read estimate is available (true values with 1% error)
        s = rd["scale"] * (1 + rng.normal(0, 0.01)); b = rd["shift"] + rng.normal(0, 0.05) * rd["scale"]
        z = (rd["raw"] - b) / s
        labs = {"true": rd["lab"], "move table": mv_lab}
        lab_bc, _ = dtw(z, mu, sd, band_center=mv_lab, W=5)
        if lab_bc is None:
            lab_bc, _ = dtw(z, mu, sd)
        labs["bcDTW W=5"] = lab_bc
        for a, lab in labs.items():
            em, n = event_means(z, lab, L)
            # per-read feature: weighted deviation over the 5 affected k-mers (like m6Anet's 3-kmer window)
            js = [FLANK + c - p for p in range(K)]
            dev = np.array([em[j] - M.mu[codes[j]] if n[j] > 0 else 0.0 for j in js])
            score = float(np.sum(dev * WPOS) / np.sum(WPOS ** 2))
            recs.append(dict(site=site, mod_site=int(modsite), read=r, read_mod=int(is_mod), aligner=a,
                             score=score, dwell_centre=int(n[FLANK])))
    if site % 40 == 0: print("site", site, flush=True)
df = pd.DataFrame(recs); df.to_csv(f"{OUT}/mod_sim_reads.csv", index=False)

res = {"per_read_AUROC": {}, "site": []}
for a, g in df.groupby("aligner"):
    gg = g[(g.mod_site == 0) | (g.read_mod == 1)]
    res["per_read_AUROC"][a] = float(roc_auc_score(gg.read_mod, gg.score))
B = 500
for a, g in df.groupby("aligner"):
    for cov in COVS:
        site_score = g[g.read < cov].groupby("site").agg(score=("score", "mean"), y=("mod_site", "first"))
        au = roc_auc_score(site_score.y, site_score.score); ap = average_precision_score(site_score.y, site_score.score)
        bs = []
        yv, sv = site_score.y.values, site_score.score.values
        for _ in range(B):
            i = rng.integers(0, len(yv), len(yv))
            if yv[i].min() == yv[i].max(): continue
            bs.append(roc_auc_score(yv[i], sv[i]))
        res["site"].append(dict(aligner=a, coverage=cov, AUROC=float(au), AUPRC=float(ap),
                                AUROC_CI=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]))
st = pd.DataFrame(res["site"]); st.to_csv(f"{OUT}/mod_sim_site_by_coverage.csv", index=False)
# sites detected at matched specificity: TPR at FPR=5%
tpr = []
for a, g in df.groupby("aligner"):
    for cov in COVS:
        ss = g[g.read < cov].groupby("site").agg(score=("score", "mean"), y=("mod_site", "first"))
        thr = np.quantile(ss.score[ss.y == 0], 0.95)
        tpr.append(dict(aligner=a, coverage=cov, TPR_at_FPR05=float(np.mean(ss.score[ss.y == 1] > thr))))
res["tpr_fpr05"] = tpr
json.dump(res, open(f"{OUT}/mod_sim_results.json", "w"), indent=1)
print(json.dumps(res["per_read_AUROC"], indent=1)); print(st.to_string()); print(pd.DataFrame(tpr).to_string())

cols = {"true": "k", "move table": "#e07b39", "bcDTW W=5": "#1f5aa6"}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
ax = axes[0]
for a in cols:
    g = df[(df.aligner == a) & ((df.mod_site == 0) | (df.read_mod == 1))]
    ax.hist(g.score[g.read_mod == 0], bins=60, histtype="step", color=cols[a], ls="--", density=True)
    ax.hist(g.score[g.read_mod == 1], bins=60, histtype="step", color=cols[a], lw=2, density=True,
            label=f"{a} (per-read AUROC {res['per_read_AUROC'][a]:.3f})")
ax.set_xlabel("per-read current deviation score"); ax.set_ylabel("density")
ax.set_title("Per-read scores (solid: modified reads; dashed: unmodified)"); ax.legend(frameon=False, fontsize=8)
for j, met in enumerate(["AUROC", "TPR_at_FPR05"]):
    ax = axes[j + 1]
    T_ = st if met == "AUROC" else pd.DataFrame(tpr)
    for a in cols:
        g = T_[T_.aligner == a].sort_values("coverage")
        if met == "AUROC":
            lo = [x[0] for x in g.AUROC_CI]; hi = [x[1] for x in g.AUROC_CI]
            ax.fill_between(g.coverage, lo, hi, color=cols[a], alpha=0.15)
        ax.plot(g.coverage, g[met], "o-", color=cols[a], label=a)
    ax.set_xscale("log", base=2); ax.set_xticks(COVS); ax.set_xticklabels(COVS)
    ax.set_xlabel("reads per site (coverage bin)")
    ax.set_ylabel("site-level AUROC" if met == "AUROC" else "sensitivity at 5% FPR")
    ax.set_title("Site-level detection vs coverage (SIMULATION)"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{IMG}/fig11_sim_mod_coverage.png", dpi=150); plt.close()
