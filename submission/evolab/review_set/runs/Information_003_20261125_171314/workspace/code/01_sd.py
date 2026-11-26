"""Statistical disentanglement (SD, Eqs. 3-7 of 3D-IDS) solved as an LP in h = w*F.

For each flow the min-max normalised features F are sorted ascending (order-preserving
constraint Eq. 4 is then meaningful), and we solve
    max  h_N - h_1 + sum_{i=2}^{N-1} (2h_i - h_{i-1} - h_{i+1})
    s.t. Wmin*F_i <= h_i <= Wmax*F_i,  sum h_i <= B,  h_i <= h_{i+1},
         2h_i <= h_{i-1} + h_{i+1}
with HiGHS. Only unique feature rows are solved. Output: outputs/sd_h.npy (same order as msg).
"""
import sys, time, numpy as np
from multiprocessing import Pool
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
sys.path.insert(0, '/workspace/code')
from common import OUT, save_json

WMIN, WMAX, B = 0.1, 10.0, 40.0
N = 40


def build_static():
    rows = []
    # order: h_i - h_{i+1} <= 0
    A = lil_matrix((1 + (N - 1) + (N - 2), N))
    A[0, :] = 1.0                       # sum h <= B
    r = 1
    for i in range(N - 1):
        A[r, i] = 1; A[r, i + 1] = -1; r += 1
    for i in range(1, N - 1):           # 2h_i - h_{i-1} - h_{i+1} <= 0
        A[r, i] = 2; A[r, i - 1] = -1; A[r, i + 1] = -1; r += 1
    b = np.zeros(A.shape[0]); b[0] = B
    c = np.zeros(N)                     # minimise -objective
    c[N - 1] -= 1; c[0] += 1
    for i in range(1, N - 1):
        c[i] -= 2; c[i - 1] += 1; c[i + 1] += 1
    return A.tocsr(), b, c


A_FULL, B_FULL, C = build_static()
A_REL, B_REL = A_FULL[:N], B_FULL[:N]   # relaxed: drop convexity constraints


def solve(F):
    order = np.argsort(F, kind='stable')
    Fs = F[order].astype(float)
    bounds = [(WMIN * f, WMAX * f) for f in Fs]
    res = linprog(C, A_ub=A_FULL, b_ub=B_FULL, bounds=bounds, method='highs')
    status = 0
    if res.status != 0:
        res = linprog(C, A_ub=A_REL, b_ub=B_REL, bounds=bounds, method='highs')
        status = 1
        if res.status != 0:
            return Fs[np.argsort(order)].astype(np.float32), 2
    h = np.empty(N, np.float32); h[order] = res.x
    return h, status


if __name__ == '__main__':
    d = np.load(f'{OUT}/data.npz')
    X = d['msg'].astype(np.float64)
    U, inv = np.unique(X, axis=0, return_inverse=True)
    print('unique rows', len(U), flush=True)
    t0 = time.time()
    with Pool(8) as p:
        res = p.map(solve, list(U), chunksize=200)
    H = np.stack([r[0] for r in res]); st = np.array([r[1] for r in res])
    print('time', time.time() - t0, 'status counts', np.bincount(st, minlength=3), flush=True)
    h = H[inv.ravel()].astype(np.float32)
    np.save(f'{OUT}/sd_h.npy', h)
    W = np.where(U > 0, H / np.maximum(U, 1e-12), np.nan)
    save_json({'n_unique': len(U), 'time_s': time.time() - t0, 'Wmin': WMIN, 'Wmax': WMAX, 'B': B,
               'feasible_full_frac': float((st == 0).mean()), 'relaxed_frac': float((st == 1).mean()),
               'fallback_frac': float((st == 2).mean()),
               'w_mean_by_feature': np.nanmean(W, 0).tolist(),
               'sum_h_mean': float(H.sum(1).mean())}, f'{OUT}/sd_stats.json')
