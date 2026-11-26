import numpy as np, json, os, sys
from collections import deque
sys.path.insert(0, os.path.dirname(__file__))
from data_io import *

def comps(grid):
    H, W = grid.shape; lab = -np.ones((H, W), int); c = 0
    for r in range(H):
        for q in range(W):
            if grid[r, q] == 0 and lab[r, q] < 0:
                dq = deque([(r, q)]); lab[r, q] = c
                while dq:
                    a, b = dq.popleft()
                    for da, db in ((1,0),(-1,0),(0,1),(0,-1)):
                        x, y = a+da, b+db
                        if 0 <= x < H and 0 <= y < W and grid[x, y] == 0 and lab[x, y] < 0:
                            lab[x, y] = c; dq.append((x, y))
                c += 1
    return lab

rows = []
for fam, name, n, d in list_configs():
    fs = instance_files(d)
    dens, bad = [], 0
    for f in fs:
        g, s, t = load_instance(f)
        lab = comps(g)
        ok = (len(set(map(tuple, s))) == n and len(set(map(tuple, t))) == n
              and (g[s[:,0], s[:,1]] == 0).all() and (g[t[:,0], t[:,1]] == 0).all()
              and (lab[s[:,0], s[:,1]] == lab[t[:,0], t[:,1]]).all())
        bad += (not ok)
        dens.append(n / (g == 0).sum())
    g, s, t = load_instance(fs[0])
    rows.append(dict(family=fam, config=name, agents=n, H=g.shape[0], W=g.shape[1], n_inst=len(fs),
                     free_cells=int((g == 0).sum()), density_mean=float(np.mean(dens)),
                     density_min=float(np.min(dens)), density_max=float(np.max(dens)), invalid=bad))
    print(rows[-1], flush=True)
# compare maps_60 dataset
import glob
a = instance_files(os.path.join(DATA, 'maps_60_10_10_0.175'))
b = instance_files(os.path.join(DATA, 'random_small', 'maps_60_10_10_0.175'))
same = 0
for fa, fb in zip(a, b):
    A = load_instance(fa); B = load_instance(fb)
    same += all(np.array_equal(x, y) for x, y in zip(A, B))
print('maps_60 duplicate check: identical', same, 'of', len(a), len(b))
json.dump(dict(configs=rows, maps60_identical=same, maps60_n=len(a)), open(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'data_summary.json'), 'w'), indent=1)
