"""PD rotation produced by point-mirroring the T4/T5 input filters (same model, intact vs mirrored)."""
import sys; sys.path.insert(0, "/workspace/code")
import numpy as np, json, metrics as M
rows = []
for i in range(10):
    a = np.load(f"/workspace/outputs/ablations/intact_{i:03d}.npz"); b = np.load(f"/workspace/outputs/ablations/mirrored_T4T5_{i:03d}.npz")
    on = list(a["intensity"]).index(1); off = list(a["intensity"]).index(0)
    for t in M.T4 + M.T5:
        k = on if t.startswith("T4") else off; j = M.TYPES.index(t)
        if a["dsi"][k, j] >= 0.2 and b["dsi"][k, j] >= 0.2:
            rows.append(float(np.rad2deg(M.ang_dist(a["pd"][k, j], b["pd"][k, j]))))
rows = np.array(rows)
res = dict(n_pairs=int(len(rows)), median_pd_rotation_deg=float(np.median(rows)), frac_gt_135=float(np.mean(rows > 135)))
print(res); json.dump(res, open("/workspace/outputs/ablations/mirror_pd_rotation.json", "w"), indent=1)
