"""Step 11: can experimental mutagenesis data (SKEMPI singles) discriminate docking models?
For each refined docking model, single-mutant HADDOCK scores are correlated with experimental ddG.
The per-model Spearman rho is compared to the model quality (DockQ, i-RMSD) and to the HADDOCK score.
Output: outputs/validation/model_discrimination.csv, report/images/fig6_template_dependence.png
"""
import os, glob
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = f"{ROOT}/report/images"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False})

sk = pd.read_csv(f"{ROOT}/outputs/skempi_1brs.csv")
exp = sk[sk.n_mut == 1].groupby("mutations").ddG.mean()
mods = pd.read_csv(f"{ROOT}/outputs/docking_models.csv")
mods = mods[mods.stage == "refined"].copy()
mods["name"] = mods.run.str.replace("unbound_", "") + "_" + mods.model.str.extract(r"(emref_\d+)")[0]

rows = []
files = sorted(glob.glob(f"{ROOT}/outputs/mutscan/models/*_single.csv"))
extra = {"xtal": f"{ROOT}/outputs/mutscan/xtal_scores.csv"}
for f in files + list(extra.values()):
    name = os.path.basename(f).replace("_single.csv", "") if f not in extra.values() else "xtal"
    d = pd.read_csv(f).set_index("label")
    d = d.loc[[i for i in d.index if i in exp.index]]
    rho = stats.spearmanr(d.d_score, exp.loc[d.index])[0]
    r = stats.pearsonr(d.d_score, exp.loc[d.index])[0]
    rho_e = stats.spearmanr(d.d_elec, exp.loc[d.index])[0]
    row = dict(model=name, n=len(d), spearman=rho, pearson=r, spearman_elec=rho_e)
    if name == "xtal":
        row.update(dockq=1.0, irmsd=0.0, score=np.nan)
    else:
        m = mods[mods.name == name].iloc[0]
        row.update(dockq=m.dockq, irmsd=m.irmsd, fnat=m.fnat, score=m.score, capri=m.capri)
    rows.append(row)
df = pd.DataFrame(rows).sort_values("dockq")
df.to_csv(f"{ROOT}/outputs/validation/model_discrimination.csv", index=False)
dm = df[df.model != "xtal"]
print(df.round(3).to_string())
r1 = stats.spearmanr(dm.dockq, dm.spearman)
r2 = stats.spearmanr(dm.dockq, -dm.score)
print("Spearman(DockQ, mutational agreement) = %.2f p=%.3g ; Spearman(DockQ, -HADDOCK score) = %.2f p=%.3g" % (r1[0], r1[1], r2[0], r2[1]))
open(f"{ROOT}/outputs/validation/model_discrimination_stats.txt", "w").write(
    f"n_models={len(dm)}\nspearman(DockQ, rho_mut)={r1[0]:.3f} p={r1[1]:.3g}\nspearman(DockQ, -score)={r2[0]:.3f} p={r2[1]:.3g}\n")

# ------------ ensemble averaging (as done per cluster by the alascan module) ------------
ens_rows = []
preds = {}
for f in files:
    name = os.path.basename(f).replace("_single.csv", "")
    d = pd.read_csv(f).set_index("label")
    preds[name] = d.loc[[i for i in d.index if i in exp.index], "d_score"]
P = pd.DataFrame(preds)
q = df.set_index("model").dockq
groups = {"correct (DockQ>=0.23)": [m for m in P.columns if q[m] >= 0.23],
          "incorrect (DockQ<0.23)": [m for m in P.columns if q[m] < 0.23],
          "all 13 models": list(P.columns)}
for g, ms in groups.items():
    avg = P[ms].mean(1)
    rho, p = stats.spearmanr(avg, exp.loc[avg.index])
    r = stats.pearsonr(avg, exp.loc[avg.index])[0]
    ens_rows.append(dict(ensemble=g, n_models=len(ms), spearman=rho, p=p, pearson=r))
# per-model noise: mean pairwise rank correlation of predictions between correct models
cm = groups["correct (DockQ>=0.23)"]
pw = [stats.spearmanr(P[a], P[b])[0] for i, a in enumerate(cm) for b in cm[i + 1:]]
ens = pd.DataFrame(ens_rows)
ens.to_csv(f"{ROOT}/outputs/validation/ensemble_average.csv", index=False)
print(ens.round(3))
print("mean pairwise Spearman between predictions of correct models: %.2f (range %.2f-%.2f)" % (np.mean(pw), min(pw), max(pw)))
with open(f"{ROOT}/outputs/validation/model_discrimination_stats.txt", "a") as fh:
    fh.write(ens.round(3).to_string() + f"\nmean pairwise spearman (correct models) = {np.mean(pw):.3f}\n")

# ------------ figure 6 ------------
cr = pd.read_csv(f"{ROOT}/outputs/validation/correlations.csv")
fig, axs = plt.subplots(1, 4, figsize=(18, 4.5))
ax = axs[0]
tmpl = ["xtal", "hotspot_top1_acceptable", "hotspot_decoy_incorrect"]
tl = ["crystal\n(1BRS)", "docked model\nacceptable", "docked decoy\nincorrect"]
w = 0.38
for k, (sub, col, lab) in enumerate((("all", BLUE, "all mutants (75)"), ("single", ORANGE, "single mutants (30)"))):
    v = [cr[(cr.template == t) & (cr.subset == sub) & (cr.predictor == "HADDOCK score")].iloc[0] for t in tmpl]
    ax.bar(np.arange(3) + (k - 0.5) * w, [x.pearson for x in v], width=w * 0.92, color=col, label=lab)
    ax.errorbar(np.arange(3) + (k - 0.5) * w, [x.pearson for x in v],
                yerr=[[x.pearson - x.pearson_lo for x in v], [x.pearson_hi - x.pearson for x in v]], fmt="none", ecolor=INK2, lw=1, capsize=2)
ax.set_xticks(range(3)); ax.set_xticklabels(tl, fontsize=9)
ax.set_ylabel("Pearson r (ΔHADDOCK vs exp. ΔΔG)"); ax.axhline(0, color=MUTED, lw=0.8)
ax.set_title("Template dependence of ΔΔG prediction"); ax.legend(fontsize=8)
ax = axs[1]
col = [BLUE if "ideal" in m else (ORANGE if "hotspot" in m else INK) for m in df.model]
ax.scatter(df.dockq, df.spearman, s=55, c=col, edgecolor=SURF, lw=1.2, zorder=3)
for _, row in df.iterrows():
    if row.model == "xtal":
        ax.annotate("crystal", (row.dockq, row.spearman), xytext=(-20, -16), textcoords="offset points", fontsize=8, color=INK2)
ax.set_xlabel("DockQ of template model"); ax.set_ylabel("Spearman ρ, single mutants (n=30)")
ax.set_title("Agreement with mutagenesis vs model quality")
ax.text(0.03, 0.95, f"ρ(DockQ, agreement) = {r1[0]:.2f} (p={r1[1]:.2g}, n={len(dm)})", transform=ax.transAxes, fontsize=8.5, va="top")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="ideal-AIR run"),
                   Line2D([], [], marker="o", ls="", color=ORANGE, label="hot-spot-AIR run"),
                   Line2D([], [], marker="o", ls="", color=INK, label="crystal")], fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.62))
ax.axvline(0.23, color=MUTED, ls=":", lw=0.8)
ax = axs[2]
ax.scatter(dm.dockq, dm.score, s=55, c=[c for c, m in zip(col, df.model) if m != "xtal"], edgecolor=SURF, lw=1.2, zorder=3)
ax.set_xlabel("DockQ of template model"); ax.set_ylabel("HADDOCK score of model (emref)")
ax.set_title("HADDOCK score vs quality (same models)")
ax.text(0.40, 0.95, f"ρ(DockQ, −score) = {r2[0]:.2f} (p={r2[1]:.2g})", transform=ax.transAxes, fontsize=8.5, va="top")
ax.axvline(0.23, color=MUTED, ls=":", lw=0.8)
ax = axs[3]
labs = ["crystal\n(single)", "best single\nmodel", "median single\nmodel", "ensemble\ncorrect (7)", "ensemble\nincorrect (6)", "ensemble\nall (13)"]
vals = [df[df.model == "xtal"].spearman.iloc[0], dm.spearman.max(), dm.spearman.median()] + list(ens.spearman)
cols = [INK, MUTED, MUTED, BLUE, ORANGE, AQUA]
ax.bar(range(6), vals, color=cols, width=0.65)
for i, v in enumerate(vals):
    ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(range(6)); ax.set_xticklabels([l.replace("\n", " ") for l in labs], fontsize=8, rotation=35, ha="right")
ax.set_ylabel("Spearman ρ, single mutants (n=30)")
ax.set_title("Ensemble-averaged ΔΔG predictions")
fig.tight_layout()
fig.savefig(f"{IMG}/fig6_template_dependence.png", dpi=170)
print("fig6 written")

