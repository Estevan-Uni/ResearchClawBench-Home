"""Data overview + hierarchical aggregation consistency + GHG identity check."""
import json, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

D = load_all(); W = load_weather()
V = ["E", "H", "C", "PV", "GHG"]
UNITS = {"E": "kW", "H": "mmBTU", "C": "ton", "PV": "kW", "GHG": "ton CO2e"}

# ---------- overview ----------
rows = []
for n, df in D.items():
    for v in V:
        s = df[v]
        rows.append(dict(node=n, var=v, mean=s.mean(), std=s.std(), min=s.min(), max=s.max(),
                         n_nan=int(s.isna().sum()), n_neg=int((s < 0).sum()), n_zero=int((s == 0).sum())))
ov = pd.DataFrame(rows); ov.to_csv(f"{OUT}/overview_stats.csv", index=False)
wst = W.describe().T; wst["n_nan"] = W.isna().sum(); wst.to_csv(f"{OUT}/weather_stats.csv")
dup = {n: int(df.index.duplicated().sum()) for n, df in D.items()}
print("records energy:", sum(len(d) for d in D.values()), "weather:", len(W), "dups", sum(dup.values()))

# ---------- hierarchy ----------
sumBN = sum(D[b][V] for b in BUILDINGS)
res = []
for lvl, a, b in [("CN01 - sum(BN001..BN010)", D["CN01"][V], sumBN), ("Total - CN01", D["Total"][V], D["CN01"][V])]:
    r = a - b
    for v in V:
        denom = a[v].abs()
        rel = (r[v].abs()[denom > 0] / denom[denom > 0]).mean() * 100
        res.append(dict(level=lvl, var=v, max_abs=float(r[v].abs().max()), MAE=float(r[v].abs().mean()),
                        RMSE=float(np.sqrt((r[v] ** 2).mean())), mean_rel_err_pct=float(rel),
                        parent_mean=float(a[v].mean())))
hr = pd.DataFrame(res); hr.to_csv(f"{OUT}/hierarchy_consistency.csv", index=False); print(hr.to_string())
h_cn = sha256(f"{DATA}/CN01_energy.csv"); h_tot = sha256(f"{DATA}/Total_energy.csv")
# also check residual with float32-level noise: summation order
r_order = (D["CN01"]["E"] - sum(D[b]["E"] for b in BUILDINGS[::-1])).abs().max()

# ---------- GHG identity ----------
ghg = []
for n, df in D.items():
    X = np.c_[df[["E", "H", "C", "PV"]].values, np.ones(len(df))]
    beta, *_ = np.linalg.lstsq(X, df["GHG"].values, rcond=None)
    pred = X @ beta; r2 = 1 - ((df.GHG - pred) ** 2).sum() / ((df.GHG - df.GHG.mean()) ** 2).sum()
    fixed = 0.5 * df.E + 0.3 * df.H + 0.2 * df.C - 0.5 * df.PV
    ghg.append(dict(node=n, bE=beta[0], bH=beta[1], bC=beta[2], bPV=beta[3], intercept=beta[4], R2=r2,
                    max_abs_dev_fixed_formula=float((df.GHG - fixed).abs().max())))
gg = pd.DataFrame(ghg); gg.to_csv(f"{OUT}/ghg_identity.csv", index=False); print(gg.round(6).to_string())

# building shares
share = pd.DataFrame({b: D[b][V].mean() / D["CN01"][V].mean() for b in BUILDINGS}).T
share.to_csv(f"{OUT}/building_shares.csv")

json.dump(dict(n_files=13, records_energy=int(sum(len(d) for d in D.values())), records_weather=len(W),
               duplicates=dup, sha_CN01=h_cn, sha_Total=h_tot, identical=h_cn == h_tot,
               max_abs_reverse_order_E=float(r_order),
               n_nan_total=int(ov.n_nan.sum()), n_neg_total=int(ov.n_neg.sum())),
          open(f"{OUT}/hierarchy_summary.json", "w"), indent=2)

# ---------- figures ----------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
cols = plt.cm.tab10(np.arange(10))

# Fig 1: overview — Total annual series (daily mean) for 5 energy vars + weather
fig, ax = plt.subplots(6, 1, figsize=(11, 11), sharex=True)
dm = D["Total"][V].resample("D").mean()
for i, v in enumerate(V):
    ax[i].plot(D["Total"].index, D["Total"][v], lw=0.3, color="0.75", label="hourly")
    ax[i].plot(dm.index + pd.Timedelta(hours=12), dm[v], lw=1.4, color=cols[i], label="daily mean")
    ax[i].set_ylabel(f"{v}\n[{UNITS[v]}]")
ax[0].legend(loc="upper right", ncol=2, frameon=False)
ax[5].plot(W.index, W.Temp, lw=0.3, color="0.75"); ax[5].plot(W.Temp.resample("D").mean(), color="firebrick", lw=1.4)
ax[5].set_ylabel("Temp\n[°F]")
ax[0].set_title("HEEW Mini-Dataset 2014 — Total node energy variables and outdoor temperature")
fig.tight_layout(); fig.savefig(f"{IMG}/fig01_overview_total.png", dpi=150); plt.close()

# Fig 2: building mean levels + heatmap of shares
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
mm = pd.DataFrame({b: D[b][V].mean() for b in BUILDINGS}).T
mm[["E", "H", "C", "PV"]].plot(kind="bar", ax=ax[0], width=0.8)
ax[0].set_ylabel("annual mean (native units)"); ax[0].set_title("Annual mean per building")
im = ax[1].imshow(share.values.T * 100, cmap="viridis", aspect="auto")
ax[1].set_xticks(range(10)); ax[1].set_xticklabels(BUILDINGS, rotation=45); ax[1].set_yticks(range(5)); ax[1].set_yticklabels(V)
for i in range(10):
    for j in range(5):
        ax[1].text(i, j, f"{share.values[i, j]*100:.1f}", ha="center", va="center", fontsize=7, color="w")
plt.colorbar(im, ax=ax[1], label="share of CN01 annual mean [%]"); ax[1].set_title("Building share of community total")
fig.tight_layout(); fig.savefig(f"{IMG}/fig02_building_levels_shares.png", dpi=150); plt.close()

# Fig 3: single representative day (Wed 15 Jan 2014) — CN01 vs sum BN for E with stacked contributions
day = "2014-01-15"
sl = slice(f"{day} 00:00", f"{day} 23:00")
fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
hrs = np.arange(24); bottom = np.zeros(24)
for i, b in enumerate(BUILDINGS):
    y = D[b]["E"][sl].values
    ax[0].bar(hrs, y, bottom=bottom, color=cols[i], width=0.85, label=b, alpha=0.85); bottom += y
ax[0].plot(hrs, D["CN01"]["E"][sl].values, "k-o", ms=5, lw=1.5, label="CN01 (reported)")
ax[0].plot(hrs, sumBN["E"][sl].values, "w--x", ms=6, lw=1.0, label="Σ BN001–BN010", mec="w")
ax[0].plot(hrs, D["Total"]["E"][sl].values, "s", mfc="none", mec="red", ms=9, label="Total (reported)")
ax[0].set_ylabel("Electricity [kW]"); ax[0].legend(ncol=4, fontsize=8, loc="upper left", frameon=False)
ax[0].set_title(f"Hierarchical additivity on a representative day ({day}, Wednesday)")
ax[0].set_ylim(0, bottom.max() * 1.25)
ax[1].plot(hrs, (D["CN01"]["E"] - sumBN["E"])[sl].values, "o-", label="CN01 − Σ BN")
ax[1].plot(hrs, (D["Total"]["E"] - D["CN01"]["E"])[sl].values, "s--", mfc="none", label="Total − CN01")
ax[1].set_ylim(-1e-9, 1e-9); ax[1].set_ylabel("residual [kW]"); ax[1].set_xlabel("hour of day"); ax[1].legend(fontsize=8, frameon=False)
ax[1].set_xticks(hrs)
fig.tight_layout(); fig.savefig(f"{IMG}/fig03_hierarchy_single_day.png", dpi=150); plt.close()

# Fig 4: full-year residuals for all vars + scatter
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for i, v in enumerate(V):
    ax[0].plot(D["CN01"].index, (D["CN01"][v] - sumBN[v]).values + i * 0, lw=0.8, label=v)
ax[0].set_ylim(-1e-9, 1e-9); ax[0].set_title("CN01 − Σ BN residual, all 8,760 h (all variables)")
ax[0].set_ylabel("residual (native units)"); ax[0].legend(ncol=5, fontsize=8, frameon=False)
for i, v in enumerate(V):
    a = D["CN01"][v] / D["CN01"][v].max(); b = sumBN[v] / D["CN01"][v].max()
    ax[1].scatter(b, a, s=1, alpha=0.3, color=cols[i], label=v)
ax[1].plot([0, 1], [0, 1], "k--", lw=0.8); ax[1].set_xlabel("Σ BN (normalised by max)"); ax[1].set_ylabel("CN01 (normalised)")
ax[1].set_title("Parent vs. summed children (8,760 h × 5 vars)"); ax[1].legend(markerscale=8, fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f"{IMG}/fig04_hierarchy_residual_year.png", dpi=150); plt.close()

# Fig 5: typical profiles - mean diurnal & weekly at Total + PV hour pattern
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
T = D["Total"]
for i, v in enumerate(["E", "H", "C", "PV"]):
    p = T[v].groupby(T.index.hour).mean(); ax[0].plot(p.index, p / p.max(), "-o", ms=3, label=v, color=cols[i])
ax[0].set_xlabel("hour"); ax[0].set_ylabel("normalised mean"); ax[0].set_title("Mean diurnal profile (Total)"); ax[0].legend(frameon=False)
for i, v in enumerate(["E", "H", "C", "PV"]):
    p = T[v].groupby(T.index.day_of_week).mean(); ax[1].plot(p.index, p / p.max(), "-o", ms=3, label=v, color=cols[i])
ax[1].set_xticks(range(7)); ax[1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]); ax[1].set_title("Mean weekly profile (Total)")
for i, v in enumerate(["E", "H", "C", "PV"]):
    p = T[v].groupby(T.index.month).mean(); ax[2].plot(p.index, p / p.max(), "-o", ms=3, label=v, color=cols[i])
p = W.Temp.groupby(W.index.month).mean(); ax[2].plot(p.index, p / p.max(), "k--", label="Temp")
ax[2].set_xlabel("month"); ax[2].set_title("Monthly mean (Total) vs temperature"); ax[2].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{IMG}/fig05_profiles.png", dpi=150); plt.close()
print("done")
