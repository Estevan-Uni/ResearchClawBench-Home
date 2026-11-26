"""Simulation: alignment quality (model MAD proxy, accuracy, boundary error) vs DP cost for
move-table-only, basecaller-guided banded DTW (W=2,5,10) and full DTW."""
import json, time, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simlib import *

rng = np.random.default_rng(SEED)
NREADS = 12; LK = 250
CHEMS = ["DNA r9.4.1 (6-mer)", "DNA r10.4.1 (9-mer)"]
JITTERS = [0.5, 1.0, 2.0]
rows = []
for chem in CHEMS:
    M = Model(chem)
    for jit in JITTERS:
        for r in range(NREADS):
            seq = random_seq(rng, LK + M.K - 1)
            codes = kmer_codes(seq, M.K)
            mu, sd = M.mu[codes], M.sd[codes] * 2.0  # aligner uses inflated noise model consistent w/ sim
            rd = simulate_read(rng, M, codes, noise_infl=2.0)
            mvb = move_table(rng, rd, jitter_strides=jit)
            mv_lab = labels_from_bounds(mvb, rd["T"])
            out = {"true": (rd["lab"], (rd["scale"], rd["shift"]), 0)}
            s, b = norm_from_assignment(rd["raw"], mv_lab, mu)
            out["move table"] = (mv_lab, (s, b), 0)
            tt = {}
            for W in [2, 5, 10]:
                t0 = time.time(); out[f"bcDTW W={W}"] = bcdtw(rd["raw"], mu, sd, mvb, W); tt[f"bcDTW W={W}"] = time.time() - t0
            t0 = time.time(); out["full DTW"] = full_dtw(rd["raw"], mu, sd, mv_lab); tt["full DTW"] = time.time() - t0
            for k, (lab, (s, b), cells) in out.items():
                z = (rd["raw"] - b) / s
                em, n = event_means(z, lab, LK)
                ok = n > 0
                mad = np.nanmedian(np.abs(em[ok] - M.mu[codes][ok]))
                first = bounds_from_labels(lab, LK)
                tb = rd["bounds"][:-1]
                be = np.abs(first[first >= 0] - tb[first >= 0])
                rows.append(dict(chem=chem, jitter=jit, read=r, aligner=k, model_MAD=mad,
                                 sample_acc=float(np.mean(lab == rd["lab"])),
                                 boundary_err=float(np.median(be)), boundary_err_mean=float(np.mean(be)),
                                 frac_kmer_zero=float(np.mean(~ok)), cells=cells,
                                 cells_per_sample=cells / rd["T"], T=rd["T"],
                                 scale_err=abs(s / rd["scale"] - 1), wall=tt.get(k, 0.0)))
        print(chem, jit, flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/signal_sim_reads.csv", index=False)
order = ["true", "move table", "bcDTW W=2", "bcDTW W=5", "bcDTW W=10", "full DTW"]
summ = df.groupby(["chem", "jitter", "aligner"]).agg(model_MAD=("model_MAD", "median"),
                                                     sample_acc=("sample_acc", "mean"),
                                                     boundary_err=("boundary_err_mean", "mean"),
                                                     cells_per_sample=("cells_per_sample", "mean"),
                                                     scale_err=("scale_err", "mean")).reset_index()
full = summ[summ.aligner == "full DTW"].set_index(["chem", "jitter"]).cells_per_sample
summ["cells_rel_full"] = [r.cells_per_sample / full.loc[(r.chem, r.jitter)] for r in summ.itertuples()]
summ.to_csv(f"{OUT}/signal_sim_summary.csv", index=False)
print(summ.round(4).to_string())

fig, axes = plt.subplots(len(CHEMS), 3, figsize=(16, 8.5))
cols = {"true": "k", "move table": "#e07b39", "bcDTW W=2": "#9ecae1", "bcDTW W=5": "#4292c6",
        "bcDTW W=10": "#08519c", "full DTW": "#7a7a7a"}
for i, chem in enumerate(CHEMS):
    S = summ[summ.chem == chem]
    for j, (met, lab) in enumerate([("model_MAD", "median |event mean − model mean| (model SD units)"),
                                    ("sample_acc", "fraction of samples on correct k-mer")]):
        ax = axes[i, j]
        x = np.arange(len(JITTERS)); w = 0.13
        for k, a in enumerate(order):
            v = S.set_index("aligner").loc[a].sort_values("jitter")[met].values
            ax.bar(x + (k - 2.5) * w, v, w, color=cols[a], label=a)
        ax.set_xticks(x); ax.set_xticklabels([f"jitter {j_} strides" for j_ in JITTERS])
        ax.set_ylabel(lab, fontsize=8.5); ax.set_title(f"{chem}: {'model MAD' if j == 0 else 'assignment accuracy'}")
        if met == "sample_acc": ax.set_ylim(0.4, 1.0)
        if i == 0 and j == 0: ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax = axes[i, 2]
    for a in order[2:]:
        Sa = S[S.aligner == a]
        ax.scatter(Sa.cells_rel_full * 100, Sa.model_MAD, color=cols[a], s=60, label=a)
    Sm = S[S.aligner == "move table"]; St = S[S.aligner == "true"]
    for jj, (mm, tt_) in enumerate(zip(Sm.model_MAD, St.model_MAD)):
        ax.axhline(mm, color=cols["move table"], lw=0.8, ls="--")
        ax.axhline(tt_, color="k", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("DP cells, % of full DTW [log]"); ax.set_ylabel("model MAD")
    ax.set_title("Quality vs cost (dashed: move table; dotted: truth)")
    if i == 0: ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{IMG}/fig10_sim_alignment_quality.png", dpi=150); plt.close()
