"""Energy–weather Pearson correlation (raw hourly) + robustness analyses + plausibility checks."""
import json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
from common import *

T = load_energy("Total"); W = load_weather()
V = ["E", "H", "C", "PV", "GHG"]; WV = list(W.columns)
X = pd.concat([T[V], W], axis=1)
C_raw = X.corr(method="pearson"); C_raw.to_csv(f"{OUT}/corr_raw_hourly.csv")
C_spear = X.corr(method="spearman"); C_spear.to_csv(f"{OUT}/corr_spearman_hourly.csv")

# robustness: remove mean diurnal profile; remove hour×weekday profile; daily means; remove hour×weekday + monthly mean
def deseason(df, keys):
    return df - df.groupby(keys).transform("mean")
h = X.index.hour; dw = X.index.day_of_week; mo = X.index.month
X_d = deseason(X, [h])
X_hw = deseason(X, [h, dw])
X_hwm = deseason(X_hw, [mo])
X_day = X.resample("D").mean()
X_day_dw = deseason(X_day, [X_day.index.day_of_week])
var_ok = lambda d: d.loc[:, d.std() > 1e-9]
variants = {"raw hourly": X, "minus diurnal": X_d, "minus hour×weekday": X_hw,
            "minus hour×weekday & month": X_hwm, "daily means": X_day, "daily minus weekday": X_day_dw}
rows = []
for name, d in variants.items():
    c = var_ok(d).corr()
    for v in V[:4]:
        for w in WV:
            rows.append(dict(variant=name, energy=v, weather=w,
                             r=c.loc[v, w] if (v in c and w in c) else np.nan))
rob = pd.DataFrame(rows); rob.to_csv(f"{OUT}/corr_robustness.csv", index=False)
piv = rob.pivot_table(index=["energy", "weather"], columns="variant", values="r")[list(variants)]
print(piv.round(3).to_string())

# variance decomposition: share of variance explained by hour×weekday profile
r2 = {c: 1 - X_hw[c].var() / X[c].var() for c in X.columns}
r2_hwm = {c: 1 - X_hwm[c].var() / X[c].var() for c in X.columns}
pd.DataFrame({"R2_hour_x_weekday": r2, "R2_hour_x_weekday_plus_month": r2_hwm}).to_csv(f"{OUT}/calendar_variance_share.csv")

# plausibility checks
mon_T = W.Temp.groupby(W.index.month).mean()
phoenix_normal = pd.Series([56.1, 59.7, 64.8, 72.0, 81.0, 90.8, 94.2, 93.0, 87.1, 75.5, 63.4, 55.1], index=range(1, 13))  # NOAA 1991-2020 normals, approx. °F
Tc = (W.Temp - 32) / 1.8; Tdc = (W.DewPt - 32) / 1.8
rh_magnus = 100 * np.exp(17.625 * Tdc / (243.04 + Tdc)) / np.exp(17.625 * Tc / (243.04 + Tc))
pv_hours = sorted(set(T.index.hour[T.PV > 0]))
pv_on_by_month = T.PV.gt(0).groupby([T.index.month, T.index.hour]).mean().unstack()
dew_gt_T = float((W.DewPt > W.Temp).mean())
gust_lt_wind = float((W.Gust < W.Wind).mean())
weekly_acf = {c: float(X_day[c].autocorr(7)) for c in ["Temp", "RH", "Wind", "E"]}
lag1_acf = {c: float(X_day[c].autocorr(1)) for c in ["Temp", "RH", "Wind", "E"]}
precip_hours = int((W.Precip > 0).sum())
plaus = dict(monthly_temp_F=mon_T.round(2).to_dict(), phoenix_normals_F=phoenix_normal.to_dict(),
             corr_RH_magnus_vs_reported=float(np.corrcoef(rh_magnus, W.RH)[0, 1]),
             MAE_RH_magnus_vs_reported=float((rh_magnus - W.RH).abs().mean()),
             frac_dewpoint_above_temp=dew_gt_T, frac_gust_below_wind=gust_lt_wind,
             pv_nonzero_hours=[int(x) for x in pv_hours], precip_hours=precip_hours,
             daily_acf_lag7=weekly_acf, daily_acf_lag1=lag1_acf,
             temp_min=float(W.Temp.min()), temp_max=float(W.Temp.max()))
json.dump(plaus, open(f"{OUT}/weather_plausibility.json", "w"), indent=2)
print(json.dumps(plaus, indent=1))

key = {f"{a}-{b}": float(C_raw.loc[a, b]) for a, b in
       [("E", "Temp"), ("H", "Temp"), ("C", "Temp"), ("PV", "Temp"), ("PV", "RH"), ("E", "RH"), ("C", "RH"),
        ("E", "Wind"), ("PV", "Wind"), ("E", "Press"), ("PV", "Press"), ("H", "C"), ("E", "C"), ("E", "PV"), ("Temp", "RH"), ("Wind", "Gust"),
        ("E", "Precip"), ("PV", "Precip"), ("E", "DewPt"), ("H", "RH")]}
json.dump(key, open(f"{OUT}/corr_key_values.json", "w"), indent=2); print(key)

# ---------- figures ----------
labels = ["Electricity", "Heat", "Cooling", "PV", "GHG", "Temperature", "Dew point", "Humidity", "Wind speed", "Wind gust", "Pressure", "Precipitation"]
fig, ax = plt.subplots(figsize=(10, 8.5))
sns.heatmap(C_raw.values, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, ax=ax,
            xticklabels=labels, yticklabels=labels, cbar_kws=dict(label="Pearson r", shrink=0.8), annot_kws=dict(size=8))
ax.set_title("Pearson correlation — Total-node energy variables and weather (2014, hourly, n = 8,760)")
plt.xticks(rotation=45, ha="right"); fig.tight_layout(); fig.savefig(f"{IMG}/fig06_corr_raw_heatmap.png", dpi=150); plt.close()

# robustness figure: bar chart of r for E,H,C,PV vs Temp, RH across variants
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, w in zip(axs, ["Temp", "RH", "Wind"]):
    sub = rob[rob.weather == w].pivot(index="energy", columns="variant", values="r")[list(variants)].loc[["E", "H", "C", "PV"]]
    sub.plot(kind="bar", ax=ax, width=0.85, legend=(w == "Temp"), colormap="viridis")
    ax.axhline(0, color="k", lw=0.6); ax.set_title(f"r(energy, {w})"); ax.set_xlabel(""); ax.set_ylim(-1, 1)
axs[0].legend(fontsize=7, frameon=False, loc="lower left"); axs[0].set_ylabel("Pearson r")
fig.suptitle("Robustness of energy–weather correlations to calendar-profile removal and temporal aggregation")
fig.tight_layout(); fig.savefig(f"{IMG}/fig07_corr_robustness.png", dpi=150); plt.close()

# plausibility figure
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].plot(mon_T.index, mon_T.values, "o-", label="HEEW Mini Total_weather")
ax[0].plot(phoenix_normal.index, phoenix_normal.values, "s--", label="Phoenix normals (approx.)")
ax[0].set_xlabel("month"); ax[0].set_ylabel("°F"); ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("Monthly mean temperature")
ax[1].scatter(rh_magnus, W.RH, s=1, alpha=0.3); ax[1].plot([0, 100], [0, 100], "k--", lw=0.8)
ax[1].set_xlabel("RH from (T, Td) via Magnus [%]"); ax[1].set_ylabel("reported RH [%]"); ax[1].set_title("Psychrometric consistency")
im = ax[2].imshow(pv_on_by_month.values, aspect="auto", cmap="Greys", extent=[-0.5, 23.5, 12.5, 0.5])
ax[2].set_xlabel("hour"); ax[2].set_ylabel("month"); ax[2].set_title("Fraction of hours with PV > 0")
plt.colorbar(im, ax=ax[2]); fig.tight_layout(); fig.savefig(f"{IMG}/fig08_weather_plausibility.png", dpi=150); plt.close()
print("done")
