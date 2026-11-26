"""Inference-time ablation / coarse-graining baselines (no retraining) on the 10 best
models (or 10 random draws). For each condition: FRI polarity accuracy, DSI vs known,
T4/T5 DSI, PD error, tuning correlation, flow decoding on synthetic textures.
Output: outputs/ablations/ablation_results.csv (+ per-eval npz)."""
import sys, os, time, json
sys.path.insert(0, "/workspace/code")
import numpy as np, pandas as pd
import fastsim as fs, stimuli as S, metrics as M, ablations as A, textures as tx

OUT = "/workspace/outputs/ablations"; os.makedirs(OUT, exist_ok=True)
SPEEDS = (4.8, 13, 25)
DT = 1 / 200
MODELS = list(range(10))
T4T5 = M.T4 + M.T5
img = tx.pink_noise(seed=1)
MOV, GT = tx.translation_movies(img)

CONDS = ["intact", "untrained_init", "permuted_params", "polarity_only_counts", "no_spatial_all",
         "no_spatial_T4T5", "mirrored_T4T5", "shuffled_sources", "tied_58types", "tied_EI_classes"]
if len(sys.argv) > 1:
    CONDS = sys.argv[1].split(",")


def build(cond, i):
    P = fs.model_params(i)
    if cond == "intact":
        return fs.FastSim(P)
    if cond == "untrained_init":
        return fs.FastSim(A.init_params(P, seed=100 + i))
    if cond == "permuted_params":
        return fs.FastSim(A.permuted_params(P, seed=200 + i))
    if cond == "polarity_only_counts":
        return fs.FastSim(A.polarity_only_counts(P))
    if cond == "tied_58types":
        return fs.FastSim(A.tie_params(P, A.groups_58()))
    if cond == "tied_EI_classes":
        return fs.FastSim(A.tie_params(P, A.groups_EI()))
    if cond == "no_spatial_all":
        tab = A.collapse_spatial(P)
    elif cond == "no_spatial_T4T5":
        tab = A.collapse_spatial(P, targets=T4T5)
    elif cond == "mirrored_T4T5":
        tab = A.mirror_spatial(P, targets=T4T5)
    elif cond == "shuffled_sources":
        tab = A.shuffle_sources(P, seed=300 + i)
    else:
        raise ValueError(cond)
    return fs.FastSim(P, custom_edges=A.edges_from_filters(tab))


csv = f"{OUT}/ablation_results.csv"
rows = pd.read_csv(csv).to_dict("records") if os.path.exists(csv) else []
done = {(r["condition"], r["model"]) for r in rows}
for cond in CONDS:
    for i in MODELS:
        if (cond, i) in done:
            continue
        t0 = time.time()
        sim = build(cond, i)
        Vg = sim.steady_state(DT, 1.0)
        row = dict(condition=cond, model=i)
        if not np.all(np.isfinite(Vg.numpy())) or np.abs(Vg.numpy()).max() > 1e4:
            row["unstable"] = True
            rows.append(row); pd.DataFrame(rows).to_csv(csv, index=False)
            print(cond, i, "unstable", flush=True); continue
        fr, fdf, fcfg = S.simulate_flashes(sim, DT, V_grey=Vg)
        fri = S.fri(fr, fdf, fcfg)
        er, edf, ecfg = S.simulate_edges(sim, SPEEDS, DT, V_grey=Vg)
        m = S.edge_metrics(er, edf, ecfg)
        row.update(M.summarise(fri, m["dsi"], m["pd"], m["tuning_corr"], m["intensity"]))
        row["unstable"] = bool(np.nanmax(np.abs(er)) > 1e4)
        try:
            d = tx.decode(sim, MOV, i)
            fsc = tx.flow_scores(d, GT)
            row.update(flow_cos=float(np.mean(fsc["cos"])), flow_hex_cos=float(np.mean(fsc["hex_cos"])),
                       flow_ang_err=float(np.mean(fsc["ang_err"])), flow_gain=float(np.mean(fsc["gain"])))
        except Exception as e:
            print("flow failed", e)
        np.savez_compressed(f"{OUT}/{cond}_{i:03d}.npz", fri=fri, dsi=m["dsi"], pd=m["pd"], peak=m["peak"],
                            tuning_corr=m["tuning_corr"], intensity=m["intensity"])
        rows.append(row)
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(cond, i, f"{time.time()-t0:.0f}s", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                                                  if k in ("fri_acc_all", "dsi_corr_known", "dsi_T4_on", "dsi_T5_off", "pd_err_T4_deg", "flow_cos")}, flush=True)
