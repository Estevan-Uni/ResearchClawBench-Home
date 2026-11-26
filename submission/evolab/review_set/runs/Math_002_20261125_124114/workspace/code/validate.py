"""Independent exact solution validator (pure numpy, does not share code with the solvers)."""
import numpy as np

def validate(grid, starts, goals, paths):
    """paths: list of (L_i,2) int arrays (row,col). Agents rest at goal after their path ends.
    Returns dict(valid, vertex, swap, errors)."""
    N = len(paths); errs = []
    H, W = grid.shape
    L = max(len(p) for p in paths)
    X = np.zeros((L, N), np.int64)
    for a, p in enumerate(paths):
        p = np.asarray(p)
        if not (p[0] == starts[a]).all(): errs.append(f'start {a}')
        if not (p[-1] == goals[a]).all(): errs.append(f'goal {a}')
        if (p[:, 0] < 0).any() or (p[:, 0] >= H).any() or (p[:, 1] < 0).any() or (p[:, 1] >= W).any():
            errs.append(f'bounds {a}'); continue
        if grid[p[:, 0], p[:, 1]].any(): errs.append(f'obstacle {a}')
        if len(p) > 1:
            st = np.abs(np.diff(p, axis=0)).sum(1)
            if (st > 1).any(): errs.append(f'jump {a}')
        c = p[:, 0] * W + p[:, 1]
        X[:len(p), a] = c; X[len(p):, a] = c[-1]
    vert = 0; swap = 0
    for t in range(L):
        row = X[t]
        u, cnt = np.unique(row, return_counts=True)
        vert += int((cnt > 1).sum())
        if t + 1 < L:
            a_ = row; b_ = X[t + 1]
            mv = a_ != b_
            e = set(zip(a_[mv].tolist(), b_[mv].tolist()))
            for (x, y) in e:
                if (y, x) in e: swap += 1
    swap //= 2
    return dict(valid=(not errs and vert == 0 and swap == 0), vertex=vert, swap=swap, errors=errs[:5])
