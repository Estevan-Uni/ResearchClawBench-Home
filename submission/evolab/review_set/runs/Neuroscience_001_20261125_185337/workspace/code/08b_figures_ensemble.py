"""Ensemble-level prediction figures + tables (intact trained models, 6 speeds, dt=1/200)."""
import sys, os, json
sys.path.insert(0, "/workspace/code")
import numpy as np, pandas as pd
import fastsim as fs, metrics as M
from flyvis.utils import groundtruth_utils as gt
from scipy.stats import spearmanr, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = "/workspace/report/images"; OUT = "/workspace/outputs"
types = fs.central_type_order(); ti = {t: i for i, t in enumerate(types)}
losses = np.load(f"{OUT}/validation_losses.npy")
E = [np.load(f"{OUT}/ensemble/model_{i:03d}.npz") for i in range(50)]
edf = pd.read_csv(f"{OUT}/ensemble/edge_arg_df.csv"); fdf = pd.read_csv(f"{OUT}/ensemble/flash_arg_df.csv")
FRI = np.stack([e["fri"] for e in E]); DSI = np.stack([e["dsi"] for e in E]); PD = np.stack([e["pd"] for e in E])
TC = np.stack([e["tuning_corr"] for e in E]); PEAK = np.stack([e["peak"] for e in E])
inten = list(E[0]["intensity"]); ON, OFF = inten.index(1), inten.index(0)
rows = []
for i in range(50):
    s = M.summarise(FRI[i], DSI[i], PD[i], TC[i], inten); s["model"] = i; s["loss"] = losses[i]; rows.append(s)
summ = pd.DataFrame(rows); summ.to_csv(f"{OUT}/ensemble_per_model_summary.csv", index=False)
res = {}
res["best_model_fri_correct_all"] = f"{summ.fri_n_correct_all[0]}/{summ.fri_n_known_all[0]}"
res["best_model_fri_acc_noPR"] = summ.fri_acc_noPR[0]
res["median_fri_acc_all"] = summ.fri_acc_all.median(); res["median_fri_acc_noPR"] = summ.fri_acc_noPR.median()
res["iqr_fri_acc_all"] = summ.fri_acc_all.quantile([.25, .75]).tolist()
# ensemble-median FRI sign per known type
med = np.median(FRI, 0)
res["ensemble_median_fri_correct_all"] = int(sum(np.sign(med[ti[t]]) == p for t, p in M.KNOWN_POL.items()))
res["n_known_all"] = len(M.KNOWN_POL); res["n_known_noPR"] = len(M.KNOWN_POL_NOPR)
res["ensemble_median_fri_correct_noPR"] = int(sum(np.sign(med[ti[t]]) == p for t, p in M.KNOWN_POL_NOPR.items()))
# per type fraction of models correct
pt = pd.DataFrame([dict(cell_type=t, known=p, frac_models_correct=float(np.mean(np.sign(FRI[:, ti[t]]) == p)),
                        median_fri=float(med[ti[t]]), best_model_fri=float(FRI[0, ti[t]])) for t, p in M.KNOWN_POL.items()])
pt.to_csv(f"{OUT}/fri_per_type.csv", index=False)
for k in ["dsi_corr_known", "dsi_T4_on", "dsi_T5_off", "pd_err_T4_deg", "pd_err_T5_deg", "tuning_corr_T4", "tuning_corr_T5", "fri_corr_known", "dsi_nonDS_max"]:
    res[f"median_{k}"] = float(summ[k].median()); res[f"best_{k}"] = float(summ[k][0])
    r, p = spearmanr(summ.loss, summ[k]); res[f"spearman_loss_vs_{k}"] = dict(rho=float(r), p=float(p))
r, p = spearmanr(summ.loss, summ.fri_acc_all); res["spearman_loss_vs_fri_acc_all"] = dict(rho=float(r), p=float(p))
# T5 vs T4 DSI and paired test across models
res["wilcoxon_T4on_vs_T5off"] = float(wilcoxon(summ.dsi_T4_on, summ.dsi_T5_off).pvalue)
# DSI per type (max over intensities)
dsi_t = pd.DataFrame([dict(cell_type=t, median_dsi_on=float(np.median(DSI[:, ON, ti[t]])), median_dsi_off=float(np.median(DSI[:, OFF, ti[t]])),
                           best_dsi_on=float(DSI[0, ON, ti[t]]), best_dsi_off=float(DSI[0, OFF, ti[t]])) for t in types])
dsi_t.to_csv(f"{OUT}/dsi_per_type.csv", index=False)
res["types_median_dsi_gt_0.3"] = dsi_t[(dsi_t[["median_dsi_on", "median_dsi_off"]].max(1) > 0.3)].cell_type.tolist()
json.dump(res, open(f"{OUT}/08b_ensemble_results.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))

# --- Figure: FRI swarm --------------------------------------------------------
known = list(M.KNOWN_POL.keys())
order = known + [t for t in types if t not in known]
fig, axs = plt.subplots(2, 1, figsize=(18, 8.5))
ax = axs[0]
for k, t in enumerate(order):
    v = FRI[:, ti[t]]
    xx = k + np.random.default_rng(k).uniform(-.25, .25, 50)
    p = gt.polarity.get(t, 0)
    col = "C3" if p == 1 else ("C0" if p == -1 else "gray")
    ax.scatter(xx, v, s=6, color=col, alpha=.5)
    ax.scatter(k, FRI[0, ti[t]], marker="*", s=70, color="k", zorder=3)
    if p != 0:
        ax.text(k, 1.12, "ON" if p == 1 else "OFF", color=col, ha="center", fontsize=6, rotation=90)
ax.axhline(0, color="k", lw=.5); ax.axvline(len(known) - .5, color="k", ls=":")
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=7); ax.set_ylim(-1.05, 1.3)
ax.set_ylabel("flash response index (FRI)")
ax.set_title(f"Predicted contrast preference. Left of dotted line: {len(known)} types with known polarity (red ON, blue OFF). "
             f"★ best model: {res['best_model_fri_correct_all']} correct; ensemble median: {res['ensemble_median_fri_correct_all']}/{len(known)}; "
             f"per-model median accuracy {100*res['median_fri_acc_all']:.0f}%", fontsize=9)
ax = axs[1]
order2 = gt.motion_tuning + gt.no_motion_tuning + [t for t in types if t not in gt.known_dsi_types]
for k, t in enumerate(order2):
    for inten_k, col, off in [(ON, "C3", -.17), (OFF, "C0", .17)]:
        v = DSI[:, inten_k, ti[t]]
        ax.scatter(k + off + np.random.default_rng(k).uniform(-.1, .1, 50), v, s=5, color=col, alpha=.45)
        ax.scatter(k + off, DSI[0, inten_k, ti[t]], marker="*", s=45, color="k", zorder=3)
ax.axvline(7.5, color="k", ls=":"); ax.axvline(7.5 + len(gt.no_motion_tuning), color="k", ls=":")
ax.text(3.5, 0.95, "known DS", ha="center"); ax.text(8 + len(gt.no_motion_tuning) / 2, .95, "known non-DS", ha="center")
ax.set_xticks(range(len(order2))); ax.set_xticklabels(order2, rotation=90, fontsize=7)
ax.set_ylabel("direction selectivity index"); ax.set_ylim(0, 1)
ax.set_title(f"Predicted direction selectivity (red: ON edges, blue: OFF edges; ★ best model). Median corr(DSI, known DS) = {res['median_dsi_corr_known']:.2f}", fontsize=9)
plt.tight_layout(); plt.savefig(f"{IMG}/fig_fri_dsi_swarm.png", dpi=140); plt.close()

# --- Figure: T4/T5 polar tuning vs measured -------------------------------------
angles = np.arange(0, 360, 30)
fig, axs = plt.subplots(2, 4, figsize=(14, 7.4), subplot_kw=dict(projection="polar"))
for r, (grp, it) in enumerate([(M.T4, 1), (M.T5, 0)]):
    for c, t in enumerate(grp):
        ax = axs[r, c]
        tcs = []
        for i in range(50):
            tc = np.array([PEAK[i, (edf.angle.values == a) & (edf.intensity.values == it), ti[t]].mean() for a in angles])
            tc = tc / (tc.max() + 1e-9); tcs.append(tc)
            ax.plot(np.deg2rad(np.r_[angles, 0]), np.r_[tc, tc[0]], color="gray", alpha=.15, lw=.7)
        tcs = np.array(tcs)
        ax.plot(np.deg2rad(np.r_[angles, 0]), np.r_[tcs[0], tcs[0, 0]], color="C3" if it else "C0", lw=2, label="best model")
        mm = np.median(tcs, 0)
        ax.plot(np.deg2rad(np.r_[angles, 0]), np.r_[mm, mm[0]], color="k", lw=1.2, label="ensemble median")
        g = np.array(gt.tuning_curves[t]); g = g / g.max()
        ax.plot(np.deg2rad(np.r_[angles, 0]), np.r_[g, g[0]], "--", color="C2", lw=1.5, label="measured (Maisak 2013)")
        k = ON if it else OFF
        ax.set_title(f"{t} ({'ON' if it else 'OFF'} edges)\nPD err median {np.rad2deg(np.median(M.ang_dist(PD[:, k, ti[t]], M.PD_TRUE[t]))):.0f}°, "
                     f"r_tuning med {np.median(TC[:, k, r*4+c]):.2f}", fontsize=8)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270])); ax.set_xticklabels(["→", "↑", "←", "↓"]); ax.set_yticks([])
axs[0, 0].legend(fontsize=6, loc="lower left", bbox_to_anchor=(-.3, -.25))
plt.suptitle("Normalised peak responses of central T4/T5 cells to moving edges (12 directions, 6 speeds averaged), 50 models (gray)", y=0.995)
plt.tight_layout(); plt.subplots_adjust(top=0.88, hspace=0.45); plt.savefig(f"{IMG}/fig_T4T5_tuning.png", dpi=140); plt.close()

# --- Figure: traces of best model ---------------------------------------------
e0 = E[0]; tr = e0["traces19"].astype(float); ttypes = list(e0["traces19_types"]); ang = e0["traces19_angle"]; inn = e0["traces19_intensity"]
tt = np.arange(tr.shape[1]) * 2 / 200 - 1.0
fl = e0["flash"].astype(float); tf = np.arange(fl.shape[1]) / 200 - 1.0
fig, axs = plt.subplots(2, 6, figsize=(18, 5.6))
for k, t in enumerate(["L1", "L2", "Mi1", "Tm3", "Mi9", "Tm9"]):
    ax = axs[0, k]; j = types.index(t)
    for s in range(len(fdf)):
        col = "C3" if fdf.intensity[s] == 1 else "C0"
        if fdf.radius[s] == 6:
            ax.plot(tf, fl[s, :, j], color=col, label=f"{'ON' if fdf.intensity[s]==1 else 'OFF'} flash")
    ax.set_title(f"{t} (known {'ON' if gt.polarity[t]==1 else 'OFF'}), FRI={FRI[0, j]:.2f}", fontsize=9); ax.set_xlabel("time (s)")
axs[0, 0].legend(fontsize=7); axs[0, 0].set_ylabel("voltage (a.u.)")
for k, t in enumerate(["T4c", "T4a", "T5c", "T5a", "TmY3", "TmY18"]):
    ax = axs[1, k]; j = ttypes.index(t)
    pref = gt.preferred_directions.get(t, None)
    it = 0 if t.startswith("T5") else 1
    for a, col in [(90, "C2"), (270, "C1"), (0, "C4"), (180, "C5")]:
        s = np.where((ang == a) & (inn == it))[0][0]
        ax.plot(tt, tr[s, :, j], color=col, label=f"{a}°")
    ax.set_title(f"{t}, {'ON' if it else 'OFF'} edges 19°/s (known PD {pref}°)" if pref is not None else f"{t}, ON edges 19°/s", fontsize=9)
    ax.set_xlim(-0.2, 0.5); ax.set_xlabel("time (s)")
axs[1, 0].legend(fontsize=7); axs[1, 0].set_ylabel("voltage (a.u.)")
plt.suptitle("Best model (000): central-column voltage responses to full-field flashes (top) and moving edges at 19°/s (bottom; simulated until the edge has crossed the central column)")
plt.tight_layout(); plt.savefig(f"{IMG}/fig_traces_best.png", dpi=140); plt.close()

# --- Figure: metrics vs loss ----------------------------------------------------
fig, axs = plt.subplots(1, 5, figsize=(19, 3.6))
for ax, k, lab in zip(axs, ["fri_acc_all", "dsi_corr_known", "dsi_T4_on", "dsi_T5_off", "pd_err_T4_deg"],
                      ["FRI accuracy (32 known)", "corr(DSI, known DS)", "mean T4 DSI (ON)", "mean T5 DSI (OFF)", "T4 PD error (deg)"]):
    ax.scatter(summ.loss, summ[k], s=14); ax.scatter(summ.loss[0], summ[k][0], marker="*", s=120, color="k")
    rr = res[f"spearman_loss_vs_{k}"] if f"spearman_loss_vs_{k}" in res else res["spearman_loss_vs_fri_acc_all"]
    ax.set_title(f"{lab}\nSpearman ρ={rr['rho']:.2f}, p={rr['p']:.2g}", fontsize=9); ax.set_xlabel("validation loss")
plt.tight_layout(); plt.savefig(f"{IMG}/fig_metrics_vs_loss.png", dpi=140); plt.close()

# --- T5 DSI diagnostic: dependence on DSI definition --------------------------
def dsi_variant(i, t, it, mode):
    vec, norm_s = 0, []
    for a in angles:
        pass
    sums = {}
    for x in [0, 1]:
        r = []
        for a in angles:
            sel = (edf.angle.values == a) & (edf.intensity.values == x)
            p = PEAK[i, sel, ti[t]]  # per speed
            if mode == "baseline":
                p = np.clip(p - max(E[i]["v_rest"][ti[t]], 0), 0, None)
            r.append(p)
        sums[x] = np.array(r)  # (angles, speeds)
    R = sums[it]
    vs = np.abs((R * np.exp(1j * np.deg2rad(angles))[:, None]).sum(0))
    if mode == "own_intensity":
        nrm = R.sum(0)
    else:
        nrm = np.maximum(sums[0].sum(0), sums[1].sum(0))
    return float(np.mean(vs / (nrm + 1e-15)))


diag = []
for i in range(50):
    for t in M.T4 + M.T5:
        it = 1 if t.startswith("T4") else 0
        diag.append(dict(model=i, cell_type=t, native=DSI[i, ON if it else OFF, ti[t]],
                         own_intensity_norm=dsi_variant(i, t, it, "own_intensity"),
                         baseline_subtracted=dsi_variant(i, t, it, "baseline"),
                         v_rest=float(E[i]["v_rest"][ti[t]]),
                         peak_pref_contrast=float(np.nanmax(PEAK[i][edf.intensity.values == it][:, ti[t]])),
                         peak_other_contrast=float(np.nanmax(PEAK[i][edf.intensity.values != it][:, ti[t]]))))
diag = pd.DataFrame(diag); diag.to_csv(f"{OUT}/t4t5_dsi_definition_diagnostic.csv", index=False)
g = diag.assign(grp=diag.cell_type.str[:2]).groupby("grp")[["native", "own_intensity_norm", "baseline_subtracted", "v_rest", "peak_pref_contrast", "peak_other_contrast"]].median()
g.to_csv(f"{OUT}/t4t5_dsi_definition_summary.csv"); print(g)
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axs, ["native", "own_intensity_norm", "baseline_subtracted"]):
    for k, t in enumerate(M.T4 + M.T5):
        v = diag[diag.cell_type == t][col]
        ax.scatter(k + np.random.default_rng(k).uniform(-.2, .2, 50), v, s=6, color="C3" if t.startswith("T4") else "C0", alpha=.6)
        ax.plot([k - .3, k + .3], [v.median()] * 2, "k")
    ax.set_xticks(range(8)); ax.set_xticklabels(M.T4 + M.T5); ax.set_ylim(0, 1); ax.set_title(f"DSI: {col}", fontsize=9)
axs[0].set_ylabel("DSI (T4: ON edges, T5: OFF edges)")
plt.tight_layout(); plt.savefig(f"{IMG}/fig_T5_dsi_diagnostic.png", dpi=140); plt.close()

# --- Structure -> function: asymmetric connectome input predicts DS ------------
from scipy.stats import mannwhitneyu as _mwu
asym = [t for t, v in gt.asymmetric_input.items() if v == 1]
symm = gt.symmetric_inputs
dmax = dsi_t.set_index("cell_type")[["median_dsi_on", "median_dsi_off"]].max(1)
a_, s_ = dmax.reindex(asym).dropna(), dmax.reindex(symm).dropna()
res2 = dict(asym_types=asym, symm_types=symm, median_dsi_asym=float(a_.median()), median_dsi_symm=float(s_.median()),
            p_mwu=float(_mwu(a_, s_, alternative="greater").pvalue), n_asym=len(a_), n_symm=len(s_))
json.dump(res2, open(f"{OUT}/08b_asymmetry_vs_dsi.json", "w"), indent=1); print(res2)
fig, ax = plt.subplots(figsize=(6, 4))
for k, (lab, v) in enumerate([("asymmetric input", a_), ("symmetric input", s_)]):
    ax.scatter(k + np.random.default_rng(k).uniform(-.12, .12, len(v)), v.values, s=20)
    for t, val in v.items():
        ax.text(k + .15, val, t, fontsize=6)
ax.set_xticks([0, 1]); ax.set_xticklabels(["spatially asymmetric\nconnectome input", "symmetric input"])
ax.set_ylabel("ensemble-median DSI (max ON/OFF)"); ax.set_title(f"Connectome asymmetry predicts DS (MWU p={res2['p_mwu']:.2g})", fontsize=9)
plt.tight_layout(); plt.savefig(f"{IMG}/fig_asymmetry_dsi.png", dpi=140); plt.close()
