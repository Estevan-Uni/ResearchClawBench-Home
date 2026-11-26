"""Centrepiece: BN001 electricity fault injection -> two-stage clean -> MAE/RMSE -> daily K-means(k=3)
cluster recovery (true vs corrupted vs cleaned)."""
import json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
from cleaning import *

SEED, MISS, OUTR, K_DET, K_CL = 0, 0.05, 0.02, 4.0, 3
D = load_all(); idx = D["BN001"].index
y = D["BN001"]["E"].values
sib = np.c_[[D[b]["E"].values for b in BUILDINGS if b != "BN001"]].T
yc, miss, out, otype = corrupt(y, SEED, MISS, OUTR)
pd.DataFrame({"time": idx, "truth": y, "corrupted": yc, "missing": miss, "outlier": out, "outlier_type": otype}).to_csv(
    f"{OUT}/bn001_corruption_seed{SEED}.csv", index=False)
bad = miss | out

# ---------------- Stage 1
flag = detect(yc, k=K_DET, mode="daily", half=7)
det = prf(flag, out); print("detection", det)
det_by_type = {t: dict(n=int((otype == t).sum()), detected=int((flag & (otype == t)).sum())) for t in ["spike", "zero", "scale", "stuck"]}
y1 = yc.copy(); y1[flag] = np.nan  # stage-1 output: flagged points removed
oracle = yc.copy(); oracle[out] = np.nan

# ---------------- Stage 2 variants
methods = {
    "B0 corrupted + linear (no detection)": linear_fill(yc),
    "B1 detect + linear": linear_fill(y1),
    "P  detect + seasonal-profile anchored (proposed, univariate)": profile_fill(y1),
    "P' detect + cluster-centroid anchored (univariate)": cluster_fill(y1, K_CL, SEED),
    "X  detect + sibling ridge (cross-sectional)": ridge_fill(y1, sib),
    "H  detect + hierarchy identity (CN01 - sum others)": hierarchy_fill(y1, D["CN01"]["E"].values, sib.sum(1)),
    "O  oracle mask + seasonal-profile anchored": profile_fill(oracle),
}
rows = []
for name, yh in methods.items():
    for scope, m in [("corrupted points", bad), ("missing points", miss), ("outlier points", out), ("all hours", None)]:
        rows.append(dict(method=name, scope=scope, **err(y, yh, m)))
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/cleaning_metrics_seed{SEED}.csv", index=False)
print(res.pivot(index="method", columns="scope", values="MAE").round(3).to_string())
print(res.pivot(index="method", columns="scope", values="RMSE").round(3).to_string())

# by gap length (missing runs) for main methods
gl = []
for a, b in runs(np.isnan(y1)):
    L = b - a
    for name in list(methods)[:6]:
        e = methods[name][a:b] - y[a:b]
        gl.append(dict(method=name, L=L, abs_err_sum=np.abs(e).sum(), sq_sum=(e ** 2).sum()))
gl = pd.DataFrame(gl); gl["bin"] = pd.cut(gl.L, [0, 1, 2, 6, 12, 24, 1000], labels=["1", "2", "3-6", "7-12", "13-24", ">24"])
glb = gl.groupby(["method", "bin"], observed=True).apply(lambda g: pd.Series(dict(MAE=g.abs_err_sum.sum() / g.L.sum(), n_hours=g.L.sum())), include_groups=False).reset_index()
glb.to_csv(f"{OUT}/cleaning_by_gaplen_seed{SEED}.csv", index=False)

# ---------------- clustering of daily load curves
Yt = y.reshape(-1, 24)
cand = {"corrupted (linear-filled)": methods["B0 corrupted + linear (no detection)"],
        "cleaned (proposed P)": methods["P  detect + seasonal-profile anchored (proposed, univariate)"],
        "cleaned (cluster-aware P')": methods["P' detect + cluster-centroid anchored (univariate)"],
        "detect + linear (B1)": methods["B1 detect + linear"]}
cl = {}
for name, yh in cand.items():
    cl[name] = cluster_compare(Yt, yh.reshape(-1, 24), K_CL, SEED)
cl_tab = pd.DataFrame([dict(series=n, centroid_RMSE=v["cent_RMSE"], centroid_MAE=v["cent_MAE"], ARI=v["ARI"], label_agreement=v["agree"])
                       for n, v in cl.items()]); cl_tab.to_csv(f"{OUT}/cluster_recovery_seed{SEED}.csv", index=False)
print(cl_tab.round(4).to_string())
Ct = cl["cleaned (proposed P)"]["C_true"]; lt = cl["cleaned (proposed P)"]["l_true"]
np.savetxt(f"{OUT}/kmeans_true_centroids.csv", Ct, delimiter=",", fmt="%.4f")
dow = idx[::24].day_of_week
ct = pd.crosstab(pd.Series(lt, name="cluster(true)"), pd.Series(np.array(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])[dow], name="weekday"))
ct = ct[["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]; ct.to_csv(f"{OUT}/cluster_vs_weekday.csv"); print(ct)
ctm = pd.crosstab(pd.Series(lt, name="cluster(true)"), pd.Series(idx[::24].month, name="month")); ctm.to_csv(f"{OUT}/cluster_vs_month.csv")

# choice of k: inertia + silhouette (numpy) for k=2..6 on true daily curves
def silhouette(X, lab):
    Dm = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)); s = []
    for i in range(len(X)):
        same = lab == lab[i]; a = Dm[i, same].sum() / max(same.sum() - 1, 1)
        b = min(Dm[i, lab == j].mean() for j in set(lab) if j != lab[i]); s.append((b - a) / max(a, b))
    return float(np.mean(s))
ks = []
for k in range(2, 7):
    C, l = kmeans(Yt, k, SEED); ks.append(dict(k=k, inertia=float(((Yt - C[l]) ** 2).sum()), silhouette=silhouette(Yt, l)))
pd.DataFrame(ks).to_csv(f"{OUT}/kmeans_k_selection.csv", index=False); print(pd.DataFrame(ks))

json.dump(dict(seed=SEED, miss_rate=MISS, out_rate=OUTR, n_missing=int(miss.sum()), n_outlier=int(out.sum()),
               n_flagged=int(flag.sum()), detection=det, detection_by_type=det_by_type,
               n_missing_runs=len(runs(miss)), longest_missing_run=int(max(b - a for a, b in runs(miss))),
               n_gap_hours_after_stage1=int(np.isnan(y1).sum())),
          open(f"{OUT}/cleaning_summary_seed{SEED}.json", "w"), indent=2)

# ---------------- figures
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
P = methods["P  detect + seasonal-profile anchored (proposed, univariate)"]
# Fig 9: time-window overlay. choose a window containing a long missing block and outliers
best, bw = -1, 0
for s in range(0, 8760 - 336, 24):
    sc = miss[s:s + 336].sum() + 3 * out[s:s + 336].sum()
    if sc > best and any((b - a) >= 12 and s <= a and b <= s + 336 for a, b in runs(miss)): best, bw = sc, s
sl = slice(bw, bw + 336); t = idx[sl]
fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.2]))
ax[0].plot(t, y[sl], color="0.2", lw=1.6, label="ground truth")
ycp = np.where(np.isnan(yc), np.nan, yc)
ax[0].plot(t, ycp[sl], color="tab:orange", lw=0.9, alpha=0.9, label="corrupted (observed)")
ax[0].plot(t, P[sl], "--", color="tab:blue", lw=1.3, label="cleaned (two-stage, P)")
fl = flag[sl]; ax[0].scatter(t[fl], yc[sl][fl], marker="x", color="red", s=40, zorder=5, label="flagged by stage 1")
for a, b in runs(miss[sl]): ax[0].axvspan(t[a], t[min(b, len(t) - 1)], color="0.85", alpha=0.6, lw=0)
ax[0].axvspan(t[0], t[0], color="0.85", label="missing (injected)")
ax[0].set_ylabel("BN001 electricity [kW]"); ax[0].legend(ncol=3, fontsize=8, frameon=False, loc="upper left")
ax[0].set_title(f"BN001 electricity: corruption and two-stage cleaning (seed {SEED}, {int(MISS*100)} % missing + {int(OUTR*100)} % outliers)")
ax[0].set_ylim(-2, max(np.nanmax(yc[sl]), y[sl].max()) * 1.15)
ax[1].plot(t, (P - y)[sl], color="tab:blue", lw=0.9, label="cleaned − truth")
ax[1].plot(t, (methods["B0 corrupted + linear (no detection)"] - y)[sl], color="tab:orange", lw=0.7, alpha=0.8, label="corrupted+linear − truth")
ax[1].axhline(0, color="k", lw=0.5); ax[1].set_ylabel("error [kW]"); ax[1].legend(fontsize=8, frameon=False, ncol=2)
import matplotlib.dates as mdates
ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=2)); ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.tight_layout(); fig.savefig(f"{IMG}/fig09_cleaning_window.png", dpi=150); plt.close()

# Fig 10: K-means centroid comparison
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
Cc = cl["corrupted (linear-filled)"]["C_other"]; Cp = cl["cleaned (proposed P)"]["C_other"]
order = np.argsort(Ct.mean(1))
for j, ax in enumerate(axs):
    c = order[j]; n = int((lt == c).sum())
    ax.plot(range(24), Ct[c], "-o", color="k", ms=3, lw=2, label="true")
    ax.plot(range(24), Cp[c], "--s", color="tab:blue", ms=3, lw=1.5, label="cleaned (P)")
    ax.plot(range(24), Cc[c], ":^", color="tab:orange", ms=3, lw=1.5, label="corrupted (lin.-filled)")
    ax.set_title(f"Cluster {j+1} (true: {n} days)"); ax.set_xlabel("hour of day")
axs[0].set_ylabel("BN001 electricity [kW]"); axs[0].legend(frameon=False, fontsize=8)
fig.suptitle(f"K-means (k=3) daily load-profile centroids — centroid RMSE: cleaned {cl['cleaned (proposed P)']['cent_RMSE']:.3f} kW, "
             f"corrupted {cl['corrupted (linear-filled)']['cent_RMSE']:.3f} kW; ARI: cleaned {cl['cleaned (proposed P)']['ARI']:.3f}, corrupted {cl['corrupted (linear-filled)']['ARI']:.3f}", fontsize=10)
fig.tight_layout(); fig.savefig(f"{IMG}/fig10_kmeans_centroids.png", dpi=150); plt.close()

# Fig 11: metric bars + gap length
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
sub = res[res.scope == "corrupted points"].set_index("method")[["MAE", "RMSE"]]
sub.index = [m.split("  ")[0].split(" ")[0] for m in sub.index]
sub.plot(kind="bar", ax=ax[0], color=["tab:blue", "tab:gray"]); ax[0].set_yscale("log"); ax[0].set_ylabel("error on corrupted hours [kW] (log)")
ax[0].set_title("Reconstruction error on the 613 corrupted hours"); ax[0].tick_params(axis="x", rotation=0)
for i, (a, b) in enumerate(sub.values): ax[0].text(i - 0.2, a * 1.1, f"{a:.2f}", fontsize=7, ha="center")
for name, c in zip(list(methods)[1:5], ["tab:gray", "tab:blue", "tab:green", "tab:purple"]):
    g = glb[glb.method == name]; ax[1].plot(g.bin.astype(str), g.MAE, "-o", color=c, label=name.split("  ")[0].split(" (")[0])
ax[1].set_xlabel("gap length after stage 1 [h]"); ax[1].set_ylabel("MAE [kW]"); ax[1].legend(fontsize=7, frameon=False)
ax[1].set_title("Reconstruction MAE by gap length")
fig.tight_layout(); fig.savefig(f"{IMG}/fig11_cleaning_metrics.png", dpi=150); plt.close()

# Fig 12: daily-profile heat maps true/corrupted/cleaned + cluster labels
fig, ax = plt.subplots(1, 4, figsize=(14, 5), gridspec_kw=dict(width_ratios=[4, 4, 4, 0.6]))
vmax = max(y.max(), 80)
for a_, (nm, Y_) in zip(ax[:3], [("truth", Yt), ("corrupted (NaN = white)", yc.reshape(-1, 24)), ("cleaned (P)", P.reshape(-1, 24))]):
    im = a_.imshow(Y_, aspect="auto", cmap="magma", vmin=0, vmax=vmax); a_.set_title(nm); a_.set_xlabel("hour")
ax[0].set_ylabel("day of year 2014"); plt.colorbar(im, ax=ax[2], label="kW")
rank = np.empty(K_CL, int); rank[order] = np.arange(K_CL)
ax[3].imshow(rank[lt][:, None], aspect="auto", cmap="Set2", interpolation="nearest"); ax[3].set_xticks([]); ax[3].set_title("true\ncluster", fontsize=8); ax[3].set_yticks([])
fig.tight_layout(); fig.savefig(f"{IMG}/fig12_daily_matrix.png", dpi=150); plt.close()
print("done")
