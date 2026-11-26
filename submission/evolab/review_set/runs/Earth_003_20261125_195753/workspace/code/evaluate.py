"""Verification against ERA5 (2020 test inits): latitude-weighted RMSE and ACC (anomalies w.r.t. the
WB2 ERA5 1990-2019 6-hourly climatology), per init x lead x variable, for all systems.
Writes outputs/metrics.npz."""
import os, numpy as np
from common import *

FC = f'{ROOT}/fc'; B = f'{ROOT}/bench'
w = WLAT[None, :, None]


def truth_and_clim():
    X, d = load_year_physical(2020)
    pos = {int(t): i for i, t in enumerate(d.astype(np.int64))}
    inits = np.load(f'{B}/inits.npy').astype('datetime64[h]')
    clim = np.load(f'{B}/clim.npy')                  # (hour4, doy366, 11, 32, 64)
    n = len(inits)
    T = np.full((n, 60, 11, 32, 64), np.nan, np.float32); C = np.zeros_like(T); P = np.full((n, 11, 32, 64), np.nan, np.float32)
    for a, t0 in enumerate(inits):
        i0 = int(t0.astype(np.int64))
        if i0 in pos: P[a] = X[pos[i0]][VER_IDX]
        for k in range(60):
            t = t0 + np.timedelta64(6 * (k + 1), 'h')
            if i0 + 6 * (k + 1) in pos: T[a, k] = X[pos[i0 + 6 * (k + 1)]][VER_IDX]
            doy = int((t - t.astype('datetime64[Y]')).astype('timedelta64[D]').astype(int))  # 0-based
            C[a, k] = clim[(t.astype(np.int64) % 24) // 6, doy]
    return inits, T, C, P


def scores(F, T, C):
    """F,T,C: (n,60,11,H,W) -> rmse (n,60,11) [per-init spatial RMSE], mse, acc."""
    e2 = ((F - T) ** 2 * w).mean((-2, -1))
    fa, oa = F - C, T - C
    fa = fa - (fa * w).mean((-2, -1), keepdims=True) * 0   # uncentred ACC (WB2 convention)
    acc = (fa * oa * w).mean((-2, -1)) / np.sqrt((fa ** 2 * w).mean((-2, -1)) * (oa ** 2 * w).mean((-2, -1)))
    act = np.sqrt((fa ** 2 * w).mean((-2, -1)))      # forecast anomaly amplitude (activity)
    return e2, acc, act


if __name__ == '__main__':
    inits, T, C, P = truth_and_clim()
    systems = {}
    for s in ['cascade', 'short_only', 'shortplus', 'pretrain_only', 'medium_only', 'long_only']:
        if os.path.exists(f'{FC}/{s}.npy'): systems[s] = np.load(f'{FC}/{s}.npy', mmap_mode='r')
    for s, f in [('ens_mean', 'ens_mean'), ('hres', 'hres'), ('fuxi_wb2', 'fuxi_wb2')]:
        if os.path.exists(f'{B}/{f}.npy'): systems[s] = np.load(f'{B}/{f}.npy', mmap_mode='r')
    out = {'inits': inits.astype(np.int64)}
    for s, F in list(systems.items()) + [('persistence', None), ('climatology', None)]:
        mse = np.full((len(inits), 60, 11), np.nan); acc = mse.copy(); act = mse.copy()
        for a in range(len(inits)):
            if s == 'persistence': f = np.broadcast_to(P[a][None], T[a].shape)
            elif s == 'climatology': f = C[a]
            else: f = np.asarray(F[a], dtype=np.float32)
            e2, ac, at = scores(f[None], T[a][None], C[a][None])
            mse[a], acc[a], act[a] = e2[0], ac[0], at[0]
        out[f'{s}_mse'] = mse; out[f'{s}_acc'] = acc; out[f'{s}_act'] = act
        print(s, 'Z500 RMSE d1/d5/d10/d15', [round(float(np.sqrt(np.nanmean(mse[:, k, 4]))), 1) for k in (3, 19, 39, 59)],
              'ACC', [round(float(np.nanmean(acc[:, k, 4])), 3) for k in (3, 19, 39, 59)], flush=True)
    _, acc0, act0 = scores(T, T, C)
    out['truth_act'] = act0
    np.savez(f'{ROOT}/metrics.npz', **out)
