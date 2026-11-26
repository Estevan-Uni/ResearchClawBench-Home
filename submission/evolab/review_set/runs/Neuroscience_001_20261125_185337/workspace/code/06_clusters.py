"""Signed cluster analysis: provided UMAP/GMM clusters (responses to naturalistic stimuli)
of T4c, T4a, T5c, Mi4, Mi9 coupled to signed preferred directions, polar tuning, and
the ON/OFF polarity of the T4 inputs Mi4 / Mi9 (contingency tests)."""
import sys, os, pickle, json
sys.path.insert(0, "/workspace/code")
import numpy as np, pandas as pd
import flyvis  # needed for unpickling
import fastsim as fs, metrics as M
from scipy.stats import fisher_exact, chi2_contingency, mannwhitneyu
from flyvis.utils import groundtruth_utils as gt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/workspace/outputs/clusters"; os.makedirs(OUT, exist_ok=True)
IMG = "/workspace/report/images"
types = fs.central_type_order(); ti = {t: i for i, t in enumerate(types)}
losses = np.load("/workspace/outputs/validation_losses.npy")
E = [np.load(f"/workspace/outputs/ensemble/model_{i:03d}.npz") for i in range(50)]
edf = pd.read_csv("/workspace/outputs/ensemble/edge_arg_df.csv")
FRI = np.stack([e["fri"] for e in E]); DSI = np.stack([e["dsi"] for e in E]); PD = np.stack([e["pd"] for e in E])
PEAK = np.stack([e["peak"] for e in E])  # (50, samples, types)
inten = list(E[0]["intensity"]); ON, OFF = inten.index(1), inten.index(0)
angles = np.arange(0, 360, 30)


def tuning(i, ct, intensity):
    out = []
    for a in angles:
        sel = (edf.angle.values == a) & (edf.intensity.values == intensity)
        out.append(PEAK[i, sel, ti[ct]].mean())
    return np.array(out)


def load_cl(ct):
    e = pickle.load(open(f"{fs.DATA_DIR}/umap_and_clustering/{ct}.pickle", "rb"))
    lab = np.asarray(e.labels)
    # sort cluster ids by mean task error (0 = best), as flyvis' get_cluster_to_indices
    order = sorted(np.unique(lab), key=lambda c: losses[lab == c].mean())
    rel = np.array([order.index(c) for c in lab])
    return e.embedding.embedding, rel


def pd_class(theta, dsi, pref, thr=0.2):
    d = np.rad2deg(M.ang_dist(theta, np.deg2rad(pref)))
    if dsi < thr:
        return "non-DS"
    if d <= 45:
        return "correct"
    if d >= 135:
        return "inverted"
    return "oblique"


res = {}
rows = []
for ct, inten_ct, pref in [("T4c", 1, 90), ("T4a", 1, 180), ("T5c", 0, 90), ("T4b", 1, 0), ("T5a", 0, 180)]:
    emb, lab = load_cl(ct)
    k = ON if inten_ct == 1 else OFF
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        th = PD[idx, k, ti[ct]]; ds = DSI[idx, k, ti[ct]]
        cm = np.angle(np.mean(ds * np.exp(1j * th)))
        cls = [pd_class(a, b, pref) for a, b in zip(th, ds)]
        rows.append(dict(cell_type=ct, cluster=int(c), n=len(idx), mean_loss=losses[idx].mean(),
                         circ_mean_pd_deg=float(np.rad2deg(cm) % 360), mean_dsi=float(ds.mean()),
                         n_correct=cls.count("correct"), n_inverted=cls.count("inverted"),
                         n_nonDS=cls.count("non-DS"), n_oblique=cls.count("oblique"),
                         Mi4_ON_frac=float(np.mean(FRI[idx, ti["Mi4"]] > 0)), Mi9_ON_frac=float(np.mean(FRI[idx, ti["Mi9"]] > 0)),
                         Tm9_OFF_frac=float(np.mean(FRI[idx, ti["Tm9"]] < 0)), CT1Lo1_OFF_frac=float(np.mean(FRI[idx, ti["CT1(Lo1)"]] < 0)),
                         models=",".join(map(str, idx))))
cl = pd.DataFrame(rows); cl.to_csv(f"{OUT}/cluster_summary.csv", index=False)
print(cl.drop(columns="models").to_string())

# ---- contingency: T4c PD class x (Mi4, Mi9) polarity, all 50 models ------------
cls_T4c = np.array([pd_class(PD[i, ON, ti["T4c"]], DSI[i, ON, ti["T4c"]], 90) for i in range(50)])
mi4 = np.where(FRI[:, ti["Mi4"]] > 0, "Mi4 ON", "Mi4 OFF"); mi9 = np.where(FRI[:, ti["Mi9"]] > 0, "Mi9 ON", "Mi9 OFF")
combo = np.char.add(np.char.add(mi4, " / "), mi9)
tab = pd.crosstab(pd.Series(combo, name="Mi4/Mi9 polarity"), pd.Series(cls_T4c, name="T4c PD class"))
tab.to_csv(f"{OUT}/contingency_T4c_Mi4Mi9.csv")
print(tab)
stats = {}
chi2, p, dof, _ = chi2_contingency(tab.values)
stats["chi2_full"] = dict(chi2=float(chi2), p=float(p), dof=int(dof))
# 2x2: Mi9 polarity (OFF = correct) vs T4c correct vs not
for inp, lab_ok in [("Mi9", "Mi9 OFF"), ("Mi4", "Mi4 ON")]:
    arr = mi9 if inp == "Mi9" else mi4
    t22 = np.array([[np.sum((arr == lab_ok) & (cls_T4c == "correct")), np.sum((arr == lab_ok) & (cls_T4c != "correct"))],
                    [np.sum((arr != lab_ok) & (cls_T4c == "correct")), np.sum((arr != lab_ok) & (cls_T4c != "correct"))]])
    o, pp = fisher_exact(t22)
    stats[f"fisher_{inp}_correctpol_vs_T4c_correct"] = dict(table=t22.tolist(), odds=float(o), p=float(pp))
both = (mi4 == "Mi4 ON") & (mi9 == "Mi9 OFF")
t22 = np.array([[np.sum(both & (cls_T4c == "correct")), np.sum(both & (cls_T4c != "correct"))],
                [np.sum(~both & (cls_T4c == "correct")), np.sum(~both & (cls_T4c != "correct"))]])
o, pp = fisher_exact(t22)
stats["fisher_both_correct_vs_T4c_correct"] = dict(table=t22.tolist(), odds=float(o), p=float(pp))
# among DS models: inverted vs correct by Mi4/Mi9 configuration
ds_mask = np.isin(cls_T4c, ["correct", "inverted"])
stats["n_T4c_classes"] = {k: int(v) for k, v in zip(*np.unique(cls_T4c, return_counts=True))}
# same for T5c with Tm9 / CT1(Lo1)
cls_T5c = np.array([pd_class(PD[i, OFF, ti["T5c"]], DSI[i, OFF, ti["T5c"]], 90) for i in range(50)])
tab5 = pd.crosstab(pd.Series(np.char.add(np.where(FRI[:, ti["Tm9"]] < 0, "Tm9 OFF", "Tm9 ON"), np.where(FRI[:, ti["CT1(Lo1)"]] < 0, " / CT1(Lo1) OFF", " / CT1(Lo1) ON")), name="T5c inputs"),
                   pd.Series(cls_T5c, name="T5c PD class"))
tab5.to_csv(f"{OUT}/contingency_T5c_Tm9CT1.csv"); print(tab5)
stats["n_T5c_classes"] = {k: int(v) for k, v in zip(*np.unique(cls_T5c, return_counts=True))}
# loss difference between T4c-correct and others
a, b = losses[cls_T4c == "correct"], losses[cls_T4c != "correct"]
stats["loss_T4c_correct_vs_other"] = dict(median_correct=float(np.median(a)), median_other=float(np.median(b)),
                                          p_mwu=float(mannwhitneyu(a, b).pvalue) if len(b) else None)
json.dump(stats, open(f"{OUT}/cluster_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
pd.DataFrame(dict(model=range(50), loss=losses, T4c_class=cls_T4c, T5c_class=cls_T5c, Mi4_fri=FRI[:, ti["Mi4"]],
                  Mi9_fri=FRI[:, ti["Mi9"]], T4c_pd_deg=np.rad2deg(PD[:, ON, ti["T4c"]]) % 360, T4c_dsi=DSI[:, ON, ti["T4c"]],
                  T5c_pd_deg=np.rad2deg(PD[:, OFF, ti["T5c"]]) % 360, T5c_dsi=DSI[:, OFF, ti["T5c"]])).to_csv(f"{OUT}/per_model_T4c.csv", index=False)

# ---------------------------------------------------------------- figure ------
cmap = plt.cm.Set1
fig = plt.figure(figsize=(17, 10))
for r, (ct, inten_ct, pref) in enumerate([("T4c", 1, 90), ("T5c", 0, 90)]):
    emb, lab = load_cl(ct)
    ax = fig.add_subplot(3, 6, r * 6 + 1)
    for c in np.unique(lab):
        ax.scatter(*emb[lab == c].T, s=18, color=cmap(c), label=f"C{c}")
    ax.scatter(*emb[0], s=90, facecolor="none", edgecolor="k", lw=1.5, label="best")
    ax.set_title(f"{ct} UMAP clusters\n(sorted by task loss)", fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=6)
    for c in np.unique(lab)[:4]:
        idx = np.where(lab == c)[0]
        ax = fig.add_subplot(3, 6, r * 6 + 2 + c, projection="polar")
        for i in idx:
            tc = tuning(i, ct, inten_ct); tc = tc / (tc.max() + 1e-9)
            ax.plot(np.deg2rad(np.r_[angles, 0]), np.r_[tc, tc[0]], color=cmap(c), alpha=.25, lw=.8)
        g = np.array(gt.tuning_curves[ct]); g = g / g.max()
        ax.plot(np.deg2rad(np.r_[angles, 0]), np.r_[g, g[0]], "k--", lw=1.2)
        row = cl[(cl.cell_type == ct) & (cl.cluster == c)].iloc[0]
        ax.set_title(f"{ct} C{c} (n={row.n}, loss {row.mean_loss:.2f})\ncirc. PD {row.circ_mean_pd_deg:.0f}°, DSI {row.mean_dsi:.2f}", fontsize=8)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270])); ax.set_xticklabels(["→", "↑", "←", "↓"]); ax.set_yticks([])
ax = fig.add_subplot(3, 3, 7)
colors = {"correct": "C2", "inverted": "C3", "non-DS": "gray", "oblique": "C1"}
for k, c in colors.items():
    m = cls_T4c == k
    ax.scatter(FRI[m, ti["Mi4"]], FRI[m, ti["Mi9"]], color=c, s=25, label=f"T4c {k} (n={m.sum()})")
ax.scatter(FRI[0, ti["Mi4"]], FRI[0, ti["Mi9"]], s=110, facecolor="none", edgecolor="k", lw=1.5, label="best model")
ax.axhline(0, c="k", lw=.5); ax.axvline(0, c="k", lw=.5)
ax.set_xlabel("Mi4 FRI (known ON > 0)"); ax.set_ylabel("Mi9 FRI (known OFF < 0)"); ax.legend(fontsize=7)
ax.set_title("T4c direction tuning vs. polarity of its inputs Mi4/Mi9 (50 models)", fontsize=9)
ax = fig.add_subplot(3, 3, 8)
tabn = tab.reindex(columns=[c for c in ["correct", "inverted", "oblique", "non-DS"] if c in tab.columns])
im = ax.imshow(tabn.values, cmap="Greys"); ax.set_xticks(range(tabn.shape[1])); ax.set_xticklabels(tabn.columns, fontsize=8)
ax.set_yticks(range(tabn.shape[0])); ax.set_yticklabels(tabn.index, fontsize=8)
for (a, b), v in np.ndenumerate(tabn.values):
    ax.text(b, a, v, ha="center", va="center", color="r", fontsize=11)
ax.set_title(f"contingency (χ² p={stats['chi2_full']['p']:.2g}; Fisher Mi9-OFF p={stats['fisher_Mi9_correctpol_vs_T4c_correct']['p']:.2g})", fontsize=8)
ax = fig.add_subplot(3, 3, 9)
for k, c in colors.items():
    m = cls_T4c == k
    ax.scatter(losses[m], DSI[m, ON, ti["T4c"]], color=c, s=20)
ax.set_xlabel("validation loss"); ax.set_ylabel("T4c DSI (ON)"); ax.set_title("T4c DSI vs task loss (colour = PD class)", fontsize=9)
plt.tight_layout(); plt.savefig(f"{IMG}/fig_clusters_signed.png", dpi=140); plt.close()
