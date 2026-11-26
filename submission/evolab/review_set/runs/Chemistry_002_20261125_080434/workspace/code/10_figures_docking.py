"""Step 10: docking figures (score funnels per scenario, success summary, template dependence)."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = f"{ROOT}/report/images"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
# ordinal CAPRI classes: one-hue blue ramp + neutral grey for incorrect
QCOL = {"incorrect": "#c9c8c3", "acceptable": "#86b6ef", "medium": "#2a78d6", "high": "#104281"}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "legend.frameon": False})

mods = pd.read_csv(f"{ROOT}/outputs/docking_models.csv")
summ = pd.read_csv(f"{ROOT}/outputs/docking_summary.csv")
cl = pd.read_csv(f"{ROOT}/outputs/docking_clusters.csv")
ORDER = [r for r in ["unbound_ideal", "unbound_hotspot", "unbound_noisy", "unbound_onesided", "unbound_abinitio",
                     "bound_hotspot", "bound_abinitio"] if r in set(mods.run)]
TITLE = {"unbound_ideal": "Unbound · ideal interface AIRs", "unbound_hotspot": "Unbound · SKEMPI hot-spot AIRs",
         "unbound_noisy": "Unbound · noisy AIRs (50% false)", "unbound_onesided": "Unbound · one-sided AIRs",
         "unbound_abinitio": "Unbound · ab initio (CM + surface)", "bound_hotspot": "Bound · SKEMPI hot-spot AIRs",
         "bound_abinitio": "Bound · ab initio (CM + surface)"}

# ---------------- Fig 2: funnels ----------------
n = len(ORDER)
fig, axs = plt.subplots(2, n, figsize=(3.0 * n, 6.4), squeeze=False)
for j, run in enumerate(ORDER):
    for i, stage in enumerate(["rigid", "refined"]):
        ax = axs[i, j]
        t = mods[(mods.run == run) & (mods.stage == stage)]
        for q in ["incorrect", "acceptable", "medium", "high"]:
            s = t[t.capri == q]
            ax.scatter(s.irmsd, s.score, s=22 if stage == "rigid" else 40, color=QCOL[q], edgecolor=SURF, lw=0.8, zorder=3)
        top = t.sort_values("score").iloc[0]
        ax.scatter(top.irmsd, top.score, s=110, facecolor="none", edgecolor=INK, lw=1.4, zorder=4)
        ax.axvline(4, color=MUTED, lw=0.8, ls="--")
        ax.set_xlim(0, max(16, t.irmsd.max() + 1))
        if i == 0:
            ax.set_title(TITLE[run], fontsize=9)
        if j == 0:
            ax.set_ylabel(("Rigid-body (it0)" if stage == "rigid" else "Refined (flexref+emref)") + "\nHADDOCK score")
        if i == 1:
            ax.set_xlabel("i-RMSD to 1BRS (Å)")
handles = [Line2D([], [], marker="o", ls="", color=QCOL[q], ms=7, label=q) for q in QCOL] + \
          [Line2D([], [], marker="o", ls="", mfc="none", mec=INK, ms=9, label="top-ranked model")]
fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(f"{IMG}/fig2_docking_funnels.png", dpi=160)
plt.close(fig)

# ---------------- Fig 5: success summary ----------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1.3, 1]))
SHORT = {"unbound_ideal": "U · ideal", "unbound_hotspot": "U · hot-spot", "unbound_noisy": "U · noisy",
         "unbound_onesided": "U · one-sided", "unbound_abinitio": "U · ab initio", "bound_hotspot": "B · hot-spot",
         "bound_abinitio": "B · ab initio"}
labels = [SHORT[r] for r in ORDER]
x = np.arange(n)
ax = axs[0]
w = 0.38
for k, (stage, col) in enumerate((("rigid", MUTED), ("refined", BLUE))):
    s = summ[summ.stage == stage].set_index("run").loc[ORDER]
    ax.bar(x + (k - 0.5) * w, s.top1_dockq, width=w * 0.92, color=col, label=f"top-1, {stage}")
    ax.scatter(x + (k - 0.5) * w, s.best_dockq, marker="_", s=250, color=INK, lw=2, zorder=4)
for yv, lab in ((0.23, "acceptable"), (0.49, "medium"), (0.80, "high")):
    ax.axhline(yv, color=MUTED, lw=0.8, ls=":")
    ax.text(-0.6, yv + 0.01, lab, ha="left", fontsize=7.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, rotation=25, ha="right")
ax.set_ylabel("DockQ"); ax.set_ylim(0, 1)
ax.set_title("Top-ranked model (bars) vs best model (—)")
ax.legend(fontsize=8, loc="upper left")
ax = axs[1]
s = summ[summ.stage == "refined"].set_index("run").loc[ORDER]
s0 = summ[summ.stage == "rigid"].set_index("run").loc[ORDER]
ax.bar(x - w / 2, 100 * s0.frac_acceptable, width=w * 0.92, color=MUTED, label="rigid-body (all models)")
ax.bar(x + w / 2, 100 * s.frac_acceptable, width=w * 0.92, color=BLUE, label="refined (top-20 / top-10)")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, rotation=25, ha="right")
ax.set_ylabel("% models ≥ acceptable (DockQ ≥ 0.23)")
ax.set_title("Fraction of acceptable-or-better models"); ax.legend(fontsize=8)
ax = axs[2]
# best cluster rank with acceptable quality
cl["ok"] = cl.dockq >= 0.23
rows = []
for r in ORDER:
    c = cl[cl.run == r].sort_values("cluster_rank")
    good = c[c.ok]
    rows.append((r, int(good.cluster_rank.min()) if len(good) else np.nan, len(c)))
rk = pd.DataFrame(rows, columns=["run", "first_ok_cluster", "n_clusters"])
for i, row in rk.iterrows():
    if np.isnan(row.first_ok_cluster):
        ax.text(i, 0.3, "none", ha="center", color=INK2, fontsize=8, rotation=90)
    else:
        ax.bar(i, row.first_ok_cluster, color=AQUA, width=0.6)
        ax.text(i, row.first_ok_cluster + 0.05, f"rank {int(row.first_ok_cluster)} of {row.n_clusters}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=8.5, rotation=25, ha="right")
ax.set_ylabel("Rank of first acceptable FCC cluster")
ax.set_yticks([0, 1, 2, 3]); ax.set_ylim(0, 3)
ax.set_title("Cluster-based ranking (lower = better)")
fig.tight_layout()
fig.savefig(f"{IMG}/fig5_docking_success.png", dpi=160)
plt.close(fig)
print("docking figures written")
