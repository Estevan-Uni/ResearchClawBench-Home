"""Quick skill check of a checkpoint (or cascade) on 8 test inits (debug aid)."""
import sys, numpy as np, torch
from forecast import test_inputs, run_system
from gen import load
from evaluate import truth_and_clim, scores
torch.set_num_threads(1)
nm, Z, X, d, inits, ok, ii = test_inputs()
sel = np.where(ok)[0][::22][:8]
pos = {k: i for k, i in zip(np.where(ok)[0], ii)}
names = sys.argv[1].split(',')
ms = [(load(n), 60 // len(names)) for n in names]
out = run_system(ms, Z, d, np.array([pos[s] for s in sel]), nm, bs=8)
_, T, C, P = truth_and_clim()
e2, acc, act = scores(out, T[sel], C[sel])
for v, j in [('Z500', 4), ('T2M', 1), ('T850', 9), ('MSL', 0)]:
    print(v, 'RMSE', [round(float(np.sqrt(np.nanmean(e2[:, k, j]))), 2) for k in (0, 3, 11, 19, 39, 59)], 'ACC', [round(float(np.nanmean(acc[:, k, j])), 3) for k in (3, 19, 39, 59)])
