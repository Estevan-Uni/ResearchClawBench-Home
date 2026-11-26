"""Sensitivity of the BN001 two-stage cleaning to seed and missing rate; generalisation to all
buildings and to H/C loads (single seed)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
from cleaning import *

D = load_all(); y = D["BN001"]["E"].values
sib = np.c_[[D[b]["E"].values for b in BUILDINGS if b != "BN001"]].T
rows = []
for rate in [0.02, 0.05, 0.10]:
    for seed in range(5):
        yc, miss, out, _ = corrupt(y, seed, rate, 0.02); bad = miss | out
        for mode in ["daily", "weekly"]:
            f = detect(yc, 4.0, mode); d = prf(f, out)
            rows.append(dict(rate=rate, seed=seed, kind="detect", method=f"Hampel-{mode}", F1=d["F1"], precision=d["precision"], recall=d["recall"]))
        f = detect(yc, 4.0, "daily"); y1 = yc.copy(); y1[f] = np.nan
        M = {"B0 corrupted+linear": linear_fill(yc), "B1 detect+linear": linear_fill(y1), "P detect+profile": profile_fill(y1),
             "P' detect+cluster": cluster_fill(y1, 3, 0), "X detect+ridge": ridge_fill(y1, sib)}
        for name, yh in M.items():
            e = err(y, yh, bad); e_all = err(y, yh)
            r = dict(rate=rate, seed=seed, kind="clean", method=name, MAE=e["MAE"], RMSE=e["RMSE"], MAPE=e["MAPE"], RMSE_all=e_all["RMSE"])
            if name in ("B0 corrupted+linear", "P detect+profile", "P' detect+cluster"):
                c = cluster_compare(y.reshape(-1, 24), yh.reshape(-1, 24), 3, 0); r.update(cent_RMSE=c["cent_RMSE"], ARI=c["ARI"])
            rows.append(r)
S = pd.DataFrame(rows); S.to_csv(f"{OUT}/sensitivity_raw.csv", index=False)
agg = S.groupby(["kind", "method", "rate"]).agg(["mean", "std"]).drop(columns="seed", level=0)
agg = agg.dropna(axis=1, how="all"); agg.to_csv(f"{OUT}/sensitivity_summary.csv"); print(agg.round(3).to_string())

# generalisation: all buildings x {E, H, C}, seed 0, 5 % missing + 2 % outliers
g = []
for b in BUILDINGS:
    for v in ["E", "H", "C"]:
        yy = D[b][v].values; sb = np.c_[[D[o][v].values for o in BUILDINGS if o != b]].T
        yc, miss, out, _ = corrupt(yy, 0, 0.05, 0.02); bad = miss | out
        f = detect(yc, 4.0, "daily"); d = prf(f, out); y1 = yc.copy(); y1[f] = np.nan
        for name, yh in {"B0": linear_fill(yc), "P": profile_fill(y1), "X": ridge_fill(y1, sb)}.items():
            e = err(yy, yh, bad)
            g.append(dict(building=b, var=v, method=name, F1=d["F1"], MAE=e["MAE"], RMSE=e["RMSE"], NRMSE=e["RMSE"] / yy.std()))
G = pd.DataFrame(g); G.to_csv(f"{OUT}/generalisation_buildings.csv", index=False)
print(G.groupby(["var", "method"])[["F1", "MAE", "RMSE", "NRMSE"]].agg(["mean", "std"]).round(3).to_string())

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
C = S[S.kind == "clean"]
for m, c in zip(["B0 corrupted+linear", "B1 detect+linear", "P detect+profile", "P' detect+cluster", "X detect+ridge"],
                ["tab:orange", "tab:gray", "tab:blue", "tab:green", "tab:purple"]):
    q = C[C.method == m].groupby("rate").RMSE.agg(["mean", "std"])
    ax[0].errorbar(q.index * 100, q["mean"], q["std"], marker="o", capsize=3, color=c, label=m)
ax[0].set_yscale("log"); ax[0].set_xlabel("missing rate [%] (+2 % outliers)"); ax[0].set_ylabel("RMSE on corrupted hours [kW]")
ax[0].legend(fontsize=7, frameon=False); ax[0].set_title("Reconstruction RMSE (5 seeds, mean ± sd)")
for m, c in zip(["B0 corrupted+linear", "P detect+profile", "P' detect+cluster"], ["tab:orange", "tab:blue", "tab:green"]):
    q = C[C.method == m].groupby("rate").ARI.agg(["mean", "std"])
    ax[1].errorbar(q.index * 100, q["mean"], q["std"], marker="o", capsize=3, color=c, label=m)
ax[1].set_ylim(0, 1.02); ax[1].set_xlabel("missing rate [%]"); ax[1].set_ylabel("ARI vs true day clusters"); ax[1].legend(fontsize=7, frameon=False)
ax[1].set_title("K-means (k=3) label recovery")
piv = G[G.method != "B0"].pivot_table(index="building", columns=["var", "method"], values="NRMSE")
piv.plot(ax=ax[2], marker="o", lw=1); ax[2].set_ylabel("NRMSE on corrupted hours (σ units)"); ax[2].set_title("All buildings × E/H/C (seed 0)")
ax[2].legend(fontsize=6, ncol=2, frameon=False); ax[2].tick_params(axis="x", rotation=45)
fig.tight_layout(); fig.savefig(f"{IMG}/fig13_sensitivity.png", dpi=150); plt.close()
print("done")
