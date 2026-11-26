"""Intact ensemble: flash + moving-edge responses for all 50 trained models with the
validated fast simulator; native flyvis metrics (FRI, DSI, PD, tuning correlation).
Saves outputs/ensemble/model_XXX.npz (+ summary arrays)."""
import sys, os, time
sys.path.insert(0, "/workspace/code")
import numpy as np
import fastsim as fs
import stimuli as S

OUT = "/workspace/outputs/ensemble"; os.makedirs(OUT, exist_ok=True)
types = fs.central_type_order()
KEEP = [t for t in types if t.startswith(("T4", "T5", "TmY", "Mi", "Tm", "CT1", "L", "T2", "T3", "C3"))]
keep_idx = np.array([types.index(t) for t in KEEP])
models = range(50) if len(sys.argv) < 2 else [int(a) for a in sys.argv[1:]]
dt = 1 / 200
for i in models:
    f = f"{OUT}/model_{i:03d}.npz"
    if os.path.exists(f):
        continue
    t0 = time.time()
    P = fs.model_params(i)
    sim = fs.FastSim(P)
    Vg = sim.steady_state(dt, 1.0)
    fr, fdf, fcfg = S.simulate_flashes(sim, dt, V_grey=Vg)
    fri = S.fri(fr, fdf, fcfg)
    er, edf, ecfg = S.simulate_edges(sim, S.SPEEDS_ALL, dt, V_grey=Vg)
    m = S.edge_metrics(er, edf, ecfg)
    edf.to_csv(f"{OUT}/edge_arg_df.csv", index=False); fdf.to_csv(f"{OUT}/flash_arg_df.csv", index=False)
    # store traces for speed 19 (subset of types, every 2nd frame) for plotting
    sel = np.where(edf.speed.values == 19)[0]
    tr = er[sel][:, ::2][:, :, keep_idx].astype(np.float16)
    np.savez_compressed(f, fri=fri, dsi=m["dsi"], pd=m["pd"], peak=m["peak"], tuning_corr=m["tuning_corr"],
                        intensity=m["intensity"], flash=fr.astype(np.float16), traces19=tr,
                        traces19_types=np.array(KEEP), traces19_angle=edf.angle.values[sel],
                        traces19_intensity=edf.intensity.values[sel], v_rest=Vg[fs.topo().central_cells_index, 0].numpy())
    print(f"model {i:03d} done in {time.time()-t0:.1f}s", flush=True)
