"""Wall-time of energy+force evaluation for SR and LES models vs system size (replicated random-charge boxes)."""
import time, json
import numpy as np, torch
from les import LESPotential, Frame
torch.set_num_threads(1)
d = np.load('../outputs/random_charges_labeled.npz')
p0, q0, L0 = d['pos'][0], d['q'][0], float(d['L'])
out = []
for rep in [1, 2, 3]:
    shifts = np.array([[i, j, k] for i in range(rep) for j in range(rep) for k in range(rep)]) * L0
    pos = (p0[None] + shifts[:, None]).reshape(-1, 3)
    sp = np.tile((q0 < 0).astype(int), len(shifts))
    fr = Frame(pos, sp, cell=L0 * rep)
    for lr in [0, 1]:
        m = LESPotential(2, rc=5.0, n_chan=8, lmax=1, use_lr=bool(lr), pair_prior=True, dl=2.0)
        m.eval()
        o = m(fr)  # warm-up (+ neighbour list)
        ts = []
        for _ in range(3):
            t = time.time(); o = m(fr); ts.append(time.time() - t)
        out.append(dict(N=len(pos), les=lr, t=float(np.median(ts))))
        print(out[-1], flush=True)
json.dump(out, open('../outputs/results/bench_scaling.json', 'w'), indent=1)
