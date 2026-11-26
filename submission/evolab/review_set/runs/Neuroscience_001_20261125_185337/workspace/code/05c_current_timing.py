"""Half-rise latency of each input current onto central T4c/T5c, PD (90 deg) vs ND (270 deg),
10 best models. Latency = first time the current change reaches 50% of its max |change| from baseline."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
rows = []
for ct in ["T4c", "T5c"]:
    z = np.load(f"/workspace/outputs/mechanism/currents_{ct}.npz")
    C, st, ang, tm = z["C"], list(z["src_types"]), list(z["angles"]), z["time"]
    for j, s in enumerate(st):
        for m in range(C.shape[0]):
            lat = {}
            for a in [90, 270]:
                c = C[m, ang.index(a), :, j]
                base = c[np.searchsorted(tm, 0.0)]
                d = np.abs(c - base); win = (tm >= 0) & (tm <= 1.0)
                if d[win].max() < 1e-3:
                    lat[a] = np.nan; continue
                k = np.where(win & (d >= 0.5 * d[win].max()))[0][0]
                lat[a] = tm[k]
            rows.append(dict(target=ct, source=s, model=m, lat_PD=lat[90], lat_ND=lat[270]))
df = pd.DataFrame(rows); df["ND_minus_PD_ms"] = 1000 * (df.lat_ND - df.lat_PD)
out = df.groupby(["target", "source"]).agg(n=("ND_minus_PD_ms", lambda x: int(np.isfinite(x).sum())),
                                          lat_PD_ms=("lat_PD", lambda x: 1000 * np.nanmedian(x)),
                                          lat_ND_ms=("lat_ND", lambda x: 1000 * np.nanmedian(x)),
                                          median_ND_minus_PD_ms=("ND_minus_PD_ms", "median")).reset_index()
out["p_wilcoxon"] = [float(wilcoxon(g.ND_minus_PD_ms.dropna()).pvalue) if (g.ND_minus_PD_ms.dropna() != 0).sum() > 4 else np.nan
                     for _, g in df.groupby(["target", "source"])]
out.to_csv("/workspace/outputs/mechanism/current_latencies.csv", index=False)
print(out.round(3).to_string())
