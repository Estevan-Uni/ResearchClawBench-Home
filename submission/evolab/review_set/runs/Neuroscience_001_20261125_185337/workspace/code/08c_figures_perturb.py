"""Figures/tables for ablations (04), mechanism (05), flow decoding (07)."""
import sys, os, json
sys.path.insert(0, "/workspace/code")
import numpy as np, pandas as pd
import fastsim as fs, metrics as M
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = "/workspace/report/images"; OUT = "/workspace/outputs"
T = fs.topo(); types = fs.central_type_order(); ti = {t: i for i, t in enumerate(types)}
res = {}

# ===================================================================== ablations
ab = pd.read_csv(f"{OUT}/ablations/ablation_results.csv")
CONDS = ["intact", "untrained_init", "permuted_params", "shuffled_sources", "polarity_only_counts",
         "no_spatial_all", "no_spatial_T4T5", "mirrored_T4T5", "tied_58types", "tied_EI_classes"]
LAB = {"intact": "intact\ntrained", "untrained_init": "connectome +\nuntrained init", "permuted_params": "trained params\npermuted",
       "shuffled_sources": "shuffled\nconnectome", "polarity_only_counts": "polarity only\n(uniform counts)",
       "no_spatial_all": "no spatial\nstructure (all)", "no_spatial_T4T5": "no spatial\n(onto T4/T5)",
       "mirrored_T4T5": "mirrored filters\n(onto T4/T5)", "tied_58types": "T4/T5 subtypes\ntied (59 groups)",
       "tied_EI_classes": "E/I/mixed\nclasses tied"}
CONDS = [c for c in CONDS if c in ab.condition.unique()]
ab["unstable"] = ab.unstable.fillna(False).astype(bool)
_mcols = [c for c in ab.columns if c not in ("condition", "model", "unstable")]
ab.loc[ab.unstable, _mcols] = np.nan  # numerically exploding runs (|V|>1e4) are failures, excluded from medians
NUNST = ab.groupby("condition").unstable.sum().to_dict()
for c in CONDS:
    if NUNST.get(c, 0):
        LAB[c] = LAB[c] + f"\n[{NUNST[c]}/10 unstable]"
mets = [("fri_acc_all", "FRI polarity accuracy\n(32 known types)", (0, 1)), ("dsi_corr_known", "corr(DSI, known DS)", (-1, 1)),
        ("dsi_T4_on", "T4 DSI (ON)", (0, 1)), ("dsi_T5_off", "T5 DSI (OFF)", (0, 1)),
        ("pd_err_T4_deg", "T4 PD error (deg)", (0, 180)), ("pd_err_T5_deg", "T5 PD error (deg)", (0, 180)),
        ("flow_cos", "decoded-flow cosine\n(translating textures)", (-1, 1))]
tab = ab.groupby("condition")[[m[0] for m in mets] + ["unstable"]].agg(["median", "mean", "count"])
tab.to_csv(f"{OUT}/ablation_summary_table.csv")
# paired Wilcoxon vs intact (same models)
pv = {}
for c in CONDS[1:]:
    for m, _, _ in mets:
        a = ab[ab.condition == "intact"].set_index("model")[m]; b = ab[ab.condition == c].set_index("model")[m]
        j = a.index.intersection(b.index); x, y = a[j].values, b[j].values; ok = np.isfinite(x) & np.isfinite(y)
        try:
            pv[(c, m)] = float(wilcoxon(x[ok], y[ok]).pvalue) if ok.sum() > 4 and np.any(x[ok] != y[ok]) else np.nan
        except ValueError:
            pv[(c, m)] = np.nan
pd.Series(pv).to_csv(f"{OUT}/ablation_wilcoxon_vs_intact.csv")
fig, axs = plt.subplots(1, len(mets), figsize=(24, 5.2), sharey=True)
for ax, (m, lab, lim) in zip(axs, mets):
    for k, c in enumerate(CONDS):
        v = ab[ab.condition == c][m].values
        ax.scatter(v, k + np.random.default_rng(k).uniform(-.15, .15, len(v)), s=12, color="C0" if c == "intact" else "C1", alpha=.7)
        if np.isfinite(v).any():
            ax.plot([np.nanmedian(v)] * 2, [k - .3, k + .3], "k", lw=2)
        if c != "intact" and np.isfinite(pv.get((c, m), np.nan)) and pv[(c, m)] < 0.01:
            ax.text(lim[1], k, "**", va="center", ha="right", fontsize=9, color="r")
    ax.set_xlim(*lim); ax.set_title(lab, fontsize=9); ax.grid(axis="x", alpha=.3)
    if m == "flow_cos":
        ax.axvline(0, color="gray", ls=":")
axs[0].set_yticks(range(len(CONDS))); axs[0].set_yticklabels([LAB[c] for c in CONDS], fontsize=8); axs[0].invert_yaxis()
plt.suptitle("Inference-time ablations / coarse-graining of the 10 best trained DMNs (dots = models or random draws, bar = median; ** p<0.01 paired Wilcoxon vs intact). No retraining.", fontsize=10)
plt.tight_layout(); plt.savefig(f"{IMG}/fig_ablation_summary.png", dpi=140); plt.close()
res["ablation_medians"] = {c: {m: float(ab[ab.condition == c][m].median()) for m, _, _ in mets} for c in CONDS}
res["ablation_n_unstable"] = {c: int(ab[ab.condition == c].unstable.fillna(False).astype(bool).sum()) for c in CONDS}

# ===================================================================== silencing
si = pd.read_csv(f"{OUT}/mechanism/silencing.csv")
inp = json.load(open(f"{OUT}/mechanism/silenced_inputs.json"))
intact = si[si.condition == "intact"].set_index("model")
rows = []
for grp, ins, it in [("T4", inp["T4_inputs"], "on"), ("T5", inp["T5_inputs"], "off")]:
    subs = M.T4 if grp == "T4" else M.T5
    for s in ins:
        d = si[si.condition == f"{grp}-{s}"].set_index("model")
        for i in d.index:
            for t in subs:
                base = intact.loc[i, f"dsi_{it}_{t}"]
                if base < 0.2:  # only intact DS (model, subtype)
                    continue
                rows.append(dict(group=grp, silenced=s, model=i, subtype=t, dsi_intact=base,
                                 dsi_sil=d.loc[i, f"dsi_{it}_{t}"], d_dsi=d.loc[i, f"dsi_{it}_{t}"] - base,
                                 gain=d.loc[i, f"peakmax_{t}"] / (intact.loc[i, f"peakmax_{t}"] + 1e-9),
                                 pd_shift_deg=np.rad2deg(M.ang_dist(d.loc[i, f"pd_{it}_{t}"], intact.loc[i, f"pd_{it}_{t}"]))))
sil = pd.DataFrame(rows); sil.to_csv(f"{OUT}/silencing_effects_long.csv", index=False)
ss = sil.groupby(["group", "silenced"]).agg(n=("d_dsi", "size"), median_d_dsi=("d_dsi", "median"),
                                            median_gain=("gain", "median"), median_pd_shift=("pd_shift_deg", "median"),
                                            frac_dsi_drop_gt_0p1=("d_dsi", lambda x: float(np.mean(x < -0.1)))).reset_index()
ss.to_csv(f"{OUT}/silencing_summary.csv", index=False); print(ss)
fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))
for ax, grp in zip(axs[:2], ["T4", "T5"]):
    d = sil[sil.group == grp]; order = inp["T4_inputs"] if grp == "T4" else inp["T5_inputs"]
    data = [d[d.silenced == s].d_dsi.values for s in order]
    ax.boxplot(data, showfliers=False);
    for k, v in enumerate(data):
        ax.scatter(k + 1 + np.random.default_rng(k).uniform(-.15, .15, len(v)), v, s=5, alpha=.5)
    ax2 = ax.twinx(); ax2.plot(range(1, len(order) + 1), [max(np.median(d[d.silenced == s].gain), 0.01) for s in order], "rD-", ms=5)
    ax2.set_ylabel("median peak-response gain (silenced/intact)", color="r"); ax2.set_yscale("log")
    ax.axhline(0, color="k", lw=.5); ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order, rotation=60)
    ax.set_ylabel(f"ΔDSI of {grp} subtypes"); ax.set_title(f"Silencing single input types onto all {grp} subtypes\n(10 best models; DS-in-intact (model, subtype) pairs; red: median gain, clipped at 0.01)", fontsize=9)
# TmY parallel pathway
ax = axs[2]
tmy = ["TmY3", "TmY4", "TmY5a", "TmY13", "TmY18", "TmY14", "TmY15", "T4c", "T5c", "T2", "T3"]
x = np.arange(len(tmy))
nt = si[si.condition == "noT4T5out"].set_index("model")
mx = lambda df, t: np.maximum(df[f"dsi_on_{t}"], df[f"dsi_off_{t}"])
a = [mx(intact, t).values for t in tmy]; b = [mx(nt, t).reindex(intact.index).values for t in tmy]
ax.boxplot(a, positions=x - .18, widths=.3, showfliers=False, patch_artist=True, boxprops=dict(facecolor="C0", alpha=.5))
ax.boxplot(b, positions=x + .18, widths=.3, showfliers=False, patch_artist=True, boxprops=dict(facecolor="C1", alpha=.5))
ax.set_xticks(x); ax.set_xticklabels(tmy, rotation=60); ax.set_ylabel("DSI (max over ON/OFF)")
ax.set_title("DSI intact (blue) vs. all T4/T5 output synapses silenced (orange)", fontsize=9)
plt.tight_layout(); plt.savefig(f"{IMG}/fig_silencing_tmy.png", dpi=140); plt.close()
tm = []
for t in tmy:
    va, vb = mx(intact, t), mx(nt, t).reindex(intact.index)
    tm.append(dict(cell_type=t, median_dsi_intact=float(va.median()), median_dsi_noT4T5=float(vb.median()),
                   p_wilcoxon=float(wilcoxon(va, vb).pvalue) if np.any(va != vb) else np.nan))
pd.DataFrame(tm).to_csv(f"{OUT}/tmy_dsi_noT4T5.csv", index=False); res["tmy_noT4T5"] = tm

# ===================================================================== currents
fig, axs = plt.subplots(2, 3, figsize=(17, 8))
for r, ct in enumerate(["T4c", "T5c"]):
    z = np.load(f"{OUT}/mechanism/currents_{ct}.npz")
    C, V, st, angs, tm_ = z["C"], z["V"], list(z["src_types"]), z["angles"], z["time"]
    top = np.argsort(-np.abs(C).mean(axis=(0, 1, 2)))[:7]
    lags = []
    for col, a in enumerate([90, 270]):
        ax = axs[r, col]; k = list(angs).index(a)
        for j in top:
            mmed = np.median(C[:, k, :, j], 0)
            ax.plot(tm_, mmed, label=st[j])
            ax.fill_between(tm_, np.percentile(C[:, k, :, j], 25, 0), np.percentile(C[:, k, :, j], 75, 0), alpha=.15)
        ax.set_xlim(0, 1.2); ax.axhline(0, color="k", lw=.5)
        ax.set_title(f"{ct}: input currents, {'ON' if ct=='T4c' else 'OFF'} edge moving {a}° ({'PD' if a==90 else 'ND'}), n=10 models (median, IQR)", fontsize=8)
        ax.set_xlabel("time (s)")
    axs[r, 0].legend(fontsize=7, ncol=2); axs[r, 0].set_ylabel("synaptic current (a.u.)")
    ax = axs[r, 2]
    for a, col in [(90, "C2"), (270, "C3")]:
        k = list(angs).index(a)
        ax.plot(tm_, np.median(V[:, k], 0), color=col, label=f"{a}°")
    ax.set_xlim(0, 1.2); ax.legend(); ax.set_title(f"{ct} voltage (median of 10 models)", fontsize=8)
    # peak-time lags of the top inputs (PD), relative to the strongest excitatory input
    k = list(angs).index(90)
    pk = {st[j]: float(np.median([tm_[np.argmax(np.abs(C[m, k, :, j]))] for m in range(C.shape[0])])) for j in top}
    res[f"current_peak_times_PD_{ct}"] = pk
    tot_pd = {st[j]: float(np.median(np.abs(C[:, list(angs).index(90), :, j]).sum(-1))) for j in top}
    tot_nd = {st[j]: float(np.median(np.abs(C[:, list(angs).index(270), :, j]).sum(-1))) for j in top}
    res[f"current_abs_integral_PD_{ct}"] = tot_pd; res[f"current_abs_integral_ND_{ct}"] = tot_nd
plt.tight_layout(); plt.savefig(f"{IMG}/fig_currents.png", dpi=140); plt.close()

# ===================================================================== flow
fl = pd.read_csv(f"{OUT}/flow/flow_results.csv")
fs_ = fl.groupby("condition")[["cos", "hex_cos", "ang_err", "rot_hex_cos", "exp_hex_cos", "gain"]].median()
fs_.to_csv(f"{OUT}/flow_summary.csv"); print(fs_)
res["flow_median"] = fs_.to_dict()
order = [c for c in ["intact", "noT4", "noT5", "noT4T5", "noMi1Tm3", "noTm9CT1Lo1"] if c in fl.condition.unique()]
fig = plt.figure(figsize=(19, 9))
z = np.load(f"{OUT}/flow/dense_intact_000.npz"); st_ = np.load(f"{OUT}/flow/stimuli.npz")
x, y = np.sqrt(3) * (T.u_in + T.v_in / 2), 1.5 * T.v_in
# receptor positions of BoxEye are (y,x) pixels; decoded flow indexed by the same hexal order
import textures as tx
px, py = tx.RC[:, 1], -tx.RC[:, 0]
ii = np.arange(0, 721, 3)
for k, (dd, g, title) in enumerate([(z["d"][2], np.tile(st_["gt"][2][:, None], (1, 721)), "translation ↑"),
                                    (z["d"][0], np.tile(st_["gt"][0][:, None], (1, 721)), "translation →"),
                                    (z["d2"][0], st_["gt2"][0], "rotation (ccw)"), (z["d2"][1], st_["gt2"][1], "expansion")]):
    ax = fig.add_subplot(2, 4, k + 1)
    gn = g / (np.abs(g).max() + 1e-9); dn = dd / (np.linalg.norm(dd, axis=0).max() + 1e-9)
    ax.quiver(px[ii], py[ii], gn[0, ii], gn[1, ii], color="gray", alpha=.6, scale=12, width=.004)
    ax.quiver(px[ii], py[ii], dn[0, ii], dn[1, ii], color="C3", scale=12, width=.004)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"best model decoded flow (red) vs GT (gray)\n{title}", fontsize=9)
ax = fig.add_subplot(2, 2, 3)
for k, c in enumerate(order):
    v = fl[fl.condition == c].cos.values
    ax.scatter(k + np.random.default_rng(k).uniform(-.15, .15, len(v)), v, s=8, alpha=.6)
    ax.plot([k - .3, k + .3], [np.median(v)] * 2, "k", lw=2)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order); ax.axhline(0, color="gray", ls=":")
ax.set_ylabel("cosine(mean decoded flow, true direction)"); ax.set_title("Translation decoding, 50 models × 8 directions (mean per model)", fontsize=9)
ax = fig.add_subplot(2, 2, 4)
for k, c in enumerate(order):
    for off, m, col in [(-.15, "rot_hex_cos", "C4"), (.15, "exp_hex_cos", "C5")]:
        v = fl[fl.condition == c][m].values
        ax.scatter(k + off + np.random.default_rng(k).uniform(-.07, .07, len(v)), v, s=6, alpha=.6, color=col, label=m if k == 0 else None)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order); ax.axhline(0, color="gray", ls=":"); ax.legend(fontsize=7)
ax.set_ylabel("mean per-column cosine with GT flow"); ax.set_title("Rotation / expansion fields, 50 models", fontsize=9)
plt.tight_layout(); plt.savefig(f"{IMG}/fig_flow_decoding.png", dpi=140); plt.close()

# ===================================================================== hex maps
sn = np.load(f"{OUT}/flow/snapshot_000_up.npz")
act = sn["act"][2]  # frame 30
show = ["R1", "L1", "L2", "L3", "L4", "L5", "Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9", "CT1(M10)", "CT1(Lo1)",
        "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "TmY3", "TmY4", "TmY5a", "TmY13", "TmY14", "TmY15", "TmY18", "T2", "T3", "C3", "Lawf2", "Am"]
fig, axs = plt.subplots(4, 9, figsize=(20, 9.6))
for ax, t in zip(axs.ravel(), show):
    idx = T.layer_index[t]
    v = act[idx] - np.median(act[idx]); xx, yy = np.sqrt(3) * (T.u[idx] + T.v[idx] / 2), 1.5 * T.v[idx]
    lim = np.abs(v).max() + 1e-9
    ax.scatter(xx, yy, c=v, cmap="seismic", vmin=-lim, vmax=lim, marker="h", s=10)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(f"{t} (median {np.median(act[idx]):.2f}, ±{lim:.2f})", fontsize=8)
ax = axs.ravel()[-1] if len(show) < 36 else None
plt.suptitle("Best model: simulated membrane voltage of every neuron of 36 cell types (hex lattice, frame 30 of an upward-translating texture, dt=20 ms); colour = deviation from the cell type's spatial median (red more depolarised), scaled per type", fontsize=10)
plt.tight_layout(); plt.savefig(f"{IMG}/fig_hex_voltage_maps.png", dpi=130); plt.close()
json.dump(res, open(f"{OUT}/08c_results.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float)[:4000])
