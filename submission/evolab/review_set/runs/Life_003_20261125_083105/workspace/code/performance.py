"""Benchmark reproduction (Table 1): time and file size per chemistry x tool, ratios, geometric means."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

perf = pd.read_csv(f"{DATA}/performance_summary.csv")
chems = list(dict.fromkeys(perf.Chemistry)); tools = ["Uncalled4", "f5c", "Nanopolish", "Tombo"]
T = perf.pivot(index="Chemistry", columns="Tool", values="Time_min").loc[chems, tools]
S = perf.pivot(index="Chemistry", columns="Tool", values="FileSize_MB").loc[chems, tools]
rt = T.div(T["Uncalled4"], axis=0); rs = S.div(S["Uncalled4"], axis=0)
tab = []
for c in chems:
    for t in tools:
        tab.append(dict(Chemistry=c, Tool=t, Time_min=T.loc[c, t], FileSize_MB=S.loc[c, t],
                        time_ratio_vs_U4=rt.loc[c, t], size_ratio_vs_U4=rs.loc[c, t],
                        U4_fastest=bool(T.loc[c].min() == T.loc[c, "Uncalled4"]),
                        U4_smallest=bool(S.loc[c].min() == S.loc[c, "Uncalled4"])))
tab = pd.DataFrame(tab); tab.to_csv(f"{OUT}/performance_table.csv", index=False)
summary = {}
for c in chems:
    comp = [t for t in tools[1:] if not np.isnan(T.loc[c, t])]
    summary[c] = dict(competitors=comp,
                      speedup_range=[float(rt.loc[c, comp].min()), float(rt.loc[c, comp].max())],
                      size_ratio_range=[float(rs.loc[c, comp].min()), float(rs.loc[c, comp].max())],
                      speedup_geomean=float(np.exp(np.log(rt.loc[c, comp]).mean())),
                      size_geomean=float(np.exp(np.log(rs.loc[c, comp]).mean())))
allr = rt[tools[1:]].values.ravel(); alls = rs[tools[1:]].values.ravel()
summary["overall"] = dict(speedup_range=[float(np.nanmin(allr)), float(np.nanmax(allr))],
                          speedup_geomean=float(np.exp(np.nanmean(np.log(allr)))),
                          size_range=[float(np.nanmin(alls)), float(np.nanmax(alls))],
                          size_geomean=float(np.exp(np.nanmean(np.log(alls)))),
                          f5c_speedup=[float(x) for x in rt["f5c"]], f5c_size=[float(x) for x in rs["f5c"]],
                          U4_fastest_all=bool(tab.U4_fastest.all()), U4_smallest_all=bool(tab.U4_smallest.all()),
                          # size per minute? throughput proxy: same input per chemistry, so time ratio = throughput ratio
                          )
# Size vs Tombo (Tombo writes into FAST5 – smallest competitor)
json.dump(summary, open(f"{OUT}/performance_summary.json", "w"), indent=1)
print(tab.round(2).to_string()); print(json.dumps(summary, indent=1))

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
x = np.arange(len(chems)); w = 0.2
for row, (M, R, lab) in enumerate([(T, rt, "Alignment time (min)"), (S, rs, "Output file size (MB)")]):
    ax = axes[row, 0]
    for i, t in enumerate(tools):
        v = M[t].values
        ax.bar(x + (i - 1.5) * w, np.nan_to_num(v, nan=0), w, color=TOOL_COLORS[t], label=t)
        for j, vv in enumerate(v):
            if np.isnan(vv):
                ax.text(x[j] + (i - 1.5) * w, 1.2, "n/s", ha="center", va="bottom", fontsize=8, rotation=90)
            else:
                ax.text(x[j] + (i - 1.5) * w, vv * 1.08, f"{vv:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_yscale("log"); ax.set_ylim(1, M.max().max() * 6)
    ax.set_xticks(x); ax.set_xticklabels(chems); ax.set_ylabel(lab + " [log]")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper right")
    ax = axes[row, 1]
    for i, t in enumerate(tools[1:]):
        v = R[t].values
        ax.bar(x + (i - 1) * 0.26, np.nan_to_num(v, nan=0), 0.26, color=TOOL_COLORS[t], label=t)
        for j, vv in enumerate(v):
            ax.text(x[j] + (i - 1) * 0.26, (vv * 1.08) if not np.isnan(vv) else 1.05,
                    "n/s" if np.isnan(vv) else f"{vv:.1f}×", ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.axhline(1, color="k", lw=1, ls="--")
    ax.set_yscale("log"); ax.set_ylim(0.8, R.max().max() * 4)
    ax.set_xticks(x); ax.set_xticklabels(chems)
    ax.set_ylabel(("Time" if row == 0 else "Size") + " relative to Uncalled4 [log]")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
axes[0, 0].set_title("Absolute alignment time"); axes[0, 1].set_title("Slow-down of competitors vs Uncalled4")
axes[1, 0].set_title("Absolute output size"); axes[1, 1].set_title("Size inflation of competitors vs Uncalled4")
plt.tight_layout(); plt.savefig(f"{IMG}/fig6_benchmark.png", dpi=150); plt.close()
