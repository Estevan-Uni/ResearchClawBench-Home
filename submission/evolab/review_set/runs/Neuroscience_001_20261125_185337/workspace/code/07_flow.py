"""Optic-flow decoding on synthetic textures for all 50 models, intact and with
inference-time silencing of motion-pathway cells. Saves dense flow fields and
whole-network voltage snapshots of the best model for figures."""
import sys, os, time
sys.path.insert(0, "/workspace/code")
import numpy as np, pandas as pd, torch
import fastsim as fs, textures as tx, metrics as M

OUT = "/workspace/outputs/flow"; os.makedirs(OUT, exist_ok=True)
T = fs.topo()
img = tx.pink_noise(seed=2)
MOV, GT = tx.translation_movies(img, speed_px=4.0)
MOV2, GT2 = tx.rotation_expansion_movies(img)
np.savez_compressed(f"{OUT}/stimuli.npz", mov=MOV, gt=GT, mov2=MOV2, gt2=GT2)


def mask_src(types):
    m = np.ones(T.n_edges); m[np.isin(T.edge_src_type, types)] = 0; return m


CONDS = {"intact": None, "noT4": M.T4, "noT5": M.T5, "noT4T5": M.T4 + M.T5, "noMi1Tm3": ["Mi1", "Tm3"],
         "noTm9CT1Lo1": ["Tm9", "CT1(Lo1)"]}
csv = f"{OUT}/flow_results.csv"
rows = pd.read_csv(csv).to_dict("records") if os.path.exists(csv) else []
done = {(r["condition"], r["model"]) for r in rows}
for i in range(50):
    P = fs.model_params(i)
    for c, src in CONDS.items():
        if (c, i) in done:
            continue
        t0 = time.time()
        sim = fs.FastSim(P, edge_mask=None if src is None else mask_src(src))
        d = tx.decode(sim, MOV, i)
        sc = tx.flow_scores(d, GT)
        d2 = tx.decode(sim, MOV2, i)
        per = d2.mean(1)
        rc = []
        for k in range(2):
            num = (per[k] * GT2[k]).sum(0); den = np.linalg.norm(per[k], axis=0) * np.linalg.norm(GT2[k], axis=0)
            ok = np.linalg.norm(GT2[k], axis=0) > 0
            rc.append(float(np.mean(num[ok] / (den[ok] + 1e-12))))
        row = dict(condition=c, model=i, cos=float(sc["cos"].mean()), hex_cos=float(sc["hex_cos"].mean()),
                   ang_err=float(sc["ang_err"].mean()), gain=float(sc["gain"].mean()),
                   rot_hex_cos=rc[0], exp_hex_cos=rc[1])
        rows.append(row); pd.DataFrame(rows).to_csv(csv, index=False)
        if i == 0:
            np.savez_compressed(f"{OUT}/dense_{c}_000.npz", d=d.mean(1), d2=d2.mean(1))
        print(i, c, f"{time.time()-t0:.0f}s", {k: round(v, 3) for k, v in row.items() if isinstance(v, float)}, flush=True)

# whole-network voltage snapshot for the best model during upward translation
P = fs.model_params(0); sim = fs.FastSim(P)
dt = 1 / 50
Vg = sim.steady_state(dt, 1.0)
first = np.repeat(MOV[2:3, :1], 25, 1)
_, V = sim.simulate(first, dt=dt, V0=Vg, record=np.array([0]), return_last=True)
act = sim.simulate(MOV[2:3], dt=dt, V0=V, record="all")
np.savez_compressed(f"{OUT}/snapshot_000_up.npz", act=act[0, [10, 20, 30, 39]].astype(np.float32),
                    frames=np.array([10, 20, 30, 39]), stim=MOV[2, [10, 20, 30, 39]])
print("snapshot done")
