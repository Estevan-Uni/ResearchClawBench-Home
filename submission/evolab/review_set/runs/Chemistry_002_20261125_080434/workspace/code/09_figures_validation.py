"""Step 9: figures for data overview and SKEMPI validation (static PNGs for the report)."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = f"{ROOT}/report/images"
V = f"{ROOT}/outputs/validation"
os.makedirs(IMG, exist_ok=True)

# reference palette (dataviz skill, light mode)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 150})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{IMG}/{name}.png", dpi=170)
    plt.close(fig)


# ---------------- Fig 1: data overview ----------------
iface = pd.read_csv(f"{ROOT}/outputs/interface_residues.csv")
sk = pd.read_csv(f"{ROOT}/outputs/skempi_1brs.csv")
single = sk[sk.n_mut == 1].copy()
single["key"] = single.mutations.str[1] + single.mutations.str[2:-1]
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw=dict(width_ratios=[2.3, 1, 1]))
ax = axs[0]
iface["key"] = iface.chain + iface.resid.astype(str)
ii = iface[(iface.bsa > 5) | iface.key.isin(single.key)].sort_values(["chain", "resid"]).reset_index(drop=True)
ii["key"] = ii.chain + ii.resid.astype(str)
cols = [BLUE if c == "A" else ORANGE for c in ii.chain]
ax.bar(range(len(ii)), ii.bsa, color=cols, width=0.75)
ax.set_xticks(range(len(ii)))
ax.set_xticklabels([f"{a}{r}" for a, r in zip(ii.aa, ii.resid)], rotation=90, fontsize=7)
ax.set_ylabel("Buried SASA (Å$^2$)")
ax.set_title("Interface residues of barnase (A) – barstar (D), 1BRS")
mx = single.groupby("key").ddG.max()
ax2 = ax.twinx() if False else None
for i, k in enumerate(ii.key):
    if k in mx.index:
        ax.text(i, ii.bsa[i] + 3, f"{mx[k]:.1f}", ha="center", fontsize=6.5, color=INK2, rotation=90)
ax.text(0.99, 0.97, "numbers: max. SKEMPI ΔΔG of a\nsingle substitution (kcal/mol)", transform=ax.transAxes,
        ha="right", va="top", fontsize=8, color=INK2)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="barnase"), Patch(color=ORANGE, label="barstar")], loc="upper left")
ax = axs[1]
b = np.arange(-1.5, 11.5, 1.0)
h1 = np.histogram(sk[sk.n_mut == 1].ddG, bins=b)[0]; h2 = np.histogram(sk[sk.n_mut == 2].ddG, bins=b)[0]
ctr = (b[:-1] + b[1:]) / 2
ax.bar(ctr - 0.21, h1, width=0.4, color=BLUE, label=f"single (n={int((sk.n_mut==1).sum())})")
ax.bar(ctr + 0.21, h2, width=0.4, color=ORANGE, label=f"double (n={int((sk.n_mut==2).sum())})")
ax.set_xlabel("ΔΔG$_{bind}$ (kcal/mol)"); ax.set_ylabel("SKEMPI entries"); ax.legend()
ax.set_title("SKEMPI 2.0 entries for 1BRS")
ax = axs[2]
rep = sk.groupby("mutations").ddG.agg(list)
pr = np.array([(v[0], v[1]) for v in rep if len(v) >= 2])
ax.plot([-1, 9], [-1, 9], color=MUTED, lw=1, ls="--")
ax.scatter(pr[:, 0], pr[:, 1], s=40, color=BLUE, edgecolor=SURF, lw=1.5, zorder=3)
r = stats.pearsonr(pr[:, 0], pr[:, 1])[0]
ax.set_xlabel("ΔΔG measurement 1"); ax.set_ylabel("ΔΔG measurement 2")
ax.set_title("Experimental reproducibility"); ax.text(0.05, 0.92, f"r = {r:.2f} (n={len(pr)})", transform=ax.transAxes)
save(fig, "fig1_data_overview")

# ---------------- Fig 3: predicted vs experimental (crystal) ----------------
m = pd.read_csv(f"{V}/xtal_merged.csv")
cr = pd.read_csv(f"{V}/correlations.csv")
fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
ax = axs[0]
for (lab, sub, c, mk) in (("single", m[m.n_mut == 1], BLUE, "o"), ("double", m[m.n_mut == 2], ORANGE, "s")):
    ax.scatter(sub.d_score, sub.ddG, s=38, color=c, marker=mk, edgecolor=SURF, lw=1.2, label=lab, zorder=3)
sl, ic, r, p, _ = stats.linregress(m.d_score, m.ddG)
xx = np.linspace(m.d_score.min(), m.d_score.max(), 10)
ax.plot(xx, ic + sl * xx, color=INK2, lw=1.5)
for _, row in m[m.n_mut == 1].iterrows():
    if abs(row.ddG - (ic + sl * row.d_score)) > 2.3 or row.ddG > 6:
        ax.annotate(row.mutations, (row.d_score, row.ddG), fontsize=7, color=INK2, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("ΔHADDOCK score (mutant − WT, a.u.)"); ax.set_ylabel("Experimental ΔΔG (kcal/mol)")
c = cr[(cr.template == "xtal") & (cr.subset == "all") & (cr.predictor == "HADDOCK score")].iloc[0]
ax.set_title("HADDOCK3 scoring vs SKEMPI (crystal)")
ax.text(0.03, 0.95, f"Pearson r = {c.pearson:.2f} [{c.pearson_lo:.2f}, {c.pearson_hi:.2f}]\nSpearman ρ = {c.spearman:.2f}, n = {int(c.n)}",
        transform=ax.transAxes, va="top", fontsize=8.5)
ax.legend(loc="lower right")
# component bars
ax = axs[1]
preds = ["HADDOCK score", "elec", "vdW", "desolv", "-dBSA", "re-weighted (LOO-CV)", "residue BSA", "residue contacts"]
sub = cr[(cr.template == "xtal") & (cr.subset == "all")].set_index("predictor").loc[preds]
y = np.arange(len(preds))[::-1]
colr = [BLUE] * 5 + [AQUA] + [MUTED] * 2
ax.barh(y, sub.pearson, color=colr, height=0.6)
ax.errorbar(sub.pearson, y, xerr=[sub.pearson - sub.pearson_lo, sub.pearson_hi - sub.pearson], fmt="none", ecolor=INK2, lw=1, capsize=2)
ax.set_yticks(y); ax.set_yticklabels(["ΔHADDOCK score", "ΔE$_{elec}$", "ΔE$_{vdW}$", "ΔE$_{desolv}$", "−ΔBSA",
                                       "re-weighted terms\n(LOO-CV)", "residue BSA\n(baseline)", "# contacts\n(baseline)"], fontsize=8.5)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlabel("Pearson r with exp. ΔΔG (95% bootstrap CI)"); ax.set_title("Which terms carry the signal? (n=75)")
# subsets
ax = axs[2]
subsets = ["all", "single", "single_ala", "double"]
names = ["all (75)", "single (30)", "single Ala (14)", "double (45)"]
w = 0.26
for k, (pname, colc) in enumerate((("HADDOCK score", BLUE), ("elec", AQUA), ("residue contacts", MUTED))):
    vals = [cr[(cr.template == "xtal") & (cr.subset == s) & (cr.predictor == pname)].spearman.iloc[0] for s in subsets]
    ax.bar(np.arange(4) + (k - 1) * w, vals, width=w * 0.9, color=colc, label={"HADDOCK score": "ΔHADDOCK score", "elec": "ΔE$_{elec}$", "residue contacts": "# contacts"}[pname])
ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylabel("Spearman ρ"); ax.set_title("Performance by mutation subset"); ax.legend(fontsize=8)
ax.axhline(0, color=MUTED, lw=0.8)
save(fig, "fig3_skempi_correlation")

# ---------------- Fig 4: alascan module + hotspot ROC + coupling ----------------
al = pd.read_csv(f"{V}/alascan_xtal.csv")
alm = pd.read_csv(f"{V}/alascan_vs_skempi.csv")
fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[1.8, 1, 1]))
ax = axs[0]
al = al.sort_values(["chain", "res"]).reset_index(drop=True)
ax.bar(range(len(al)), -al.delta_score, color=[BLUE if c == "A" else ORANGE for c in al.chain], width=0.75)
ax.set_xticks(range(len(al))); ax.set_xticklabels(al.mutations.str[0] + al.res.astype(str), rotation=90, fontsize=7)
ax.set_ylabel("−Δscore = score$_{mut}$ − score$_{wt}$ (a.u.)")
ax.set_title("HADDOCK3 alascan module on 1BRS (positive = destabilising)")
expd = dict(zip(alm.mutations, alm.ddG))
for i, mu in enumerate(al.mutations):
    if mu in expd:
        ax.text(i, max(-al.delta_score[i], 0) + 0.8, f"{expd[mu]:.1f}", ha="center", va="bottom", fontsize=7,
                rotation=90, color=INK, fontweight="bold")
ax.set_ylim(top=30)
ax.text(0.99, 0.97, "bold numbers: SKEMPI ΔΔG of the Ala mutant (kcal/mol)", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
ax.legend(handles=[Patch(color=BLUE, label="barnase"), Patch(color=ORANGE, label="barstar")], loc="upper left")
ax = axs[1]
s = m[m.n_mut == 1]
lab = (s.ddG >= 2).values
for pname, col, colc in (("ΔHADDOCK score", "d_score", BLUE), ("ΔE$_{elec}$", "d_elec", AQUA), ("# contacts", "ncont_res", MUTED)):
    sc = s[col].values
    th = np.sort(np.unique(sc))[::-1]
    tpr = [0] + [((sc >= t) & lab).sum() / lab.sum() for t in th]
    fpr = [0] + [((sc >= t) & ~lab).sum() / (~lab).sum() for t in th]
    auc = np.trapezoid(tpr, fpr)
    ax.plot(fpr, tpr, color=colc, lw=2, label=f"{pname} (AUC {auc:.2f})")
ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
ax.set_title(f"Hot-spot detection (ΔΔG ≥ 2; {lab.sum()}/{len(lab)})"); ax.legend(fontsize=8, loc="lower right")
ax = axs[2]
cyc = pd.read_csv(f"{V}/double_mutant_cycles.csv")
ax.scatter(cyc.dist, cyc.coupling_exp, s=36, color=BLUE, edgecolor=SURF, lw=1.2, label="experiment (kcal/mol)", zorder=3)
k = np.std(cyc.coupling_exp) / np.std(cyc.coupling_pred)
ax.scatter(cyc.dist, cyc.coupling_pred * k, s=36, color=ORANGE, marker="s", edgecolor=SURF, lw=1.2,
           label=f"HADDOCK (scaled ×{k:.2f})", zorder=3)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel("Min. distance between mutated residues (Å)"); ax.set_ylabel("Coupling ΔΔG$_{int}$")
rho_e = stats.spearmanr(cyc.dist, cyc.coupling_exp)[0]; rho_p = stats.spearmanr(cyc.dist, cyc.coupling_pred)[0]
ax.set_title("Double-mutant cycles (n=%d)" % len(cyc))
ax.text(0.97, 0.05, f"ρ(dist) exp = {rho_e:.2f}\nρ(dist) HADDOCK = {rho_p:.2f}", transform=ax.transAxes, ha="right", fontsize=8.5)
ax.legend(fontsize=8, loc="upper right")
save(fig, "fig4_alascan_hotspots_coupling")
print("validation figures written")
