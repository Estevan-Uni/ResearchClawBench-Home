"""Download ECMWF IFS ENS mean, HRES, and the published FuXi forecasts (WeatherBench2, 64x32) for the 2020 test inits.
Verification variables (11): MSL,T2M,U10,V10,Z500,T500,U500,V500,Z850,T850,TP6.
Saved: outputs/bench/{name}.npy  (n_init, n_lead, 11, 32 lat, 64 lon), leads 6h..; NaN where missing."""
import numpy as np, os, sys, time
from wb2io import Store
import os
NT = int(os.environ.get("NT", 12))
OUT = '/workspace/outputs/bench'; os.makedirs(OUT, exist_ok=True)
VARS = ['MSL', 'T2M', 'U10', 'V10', 'Z500', 'T500', 'U500', 'V500', 'Z850', 'T850', 'TP6']
MAP = {'MSL': ('mean_sea_level_pressure', None), 'T2M': ('2m_temperature', None), 'U10': ('10m_u_component_of_wind', None),
       'V10': ('10m_v_component_of_wind', None), 'Z500': ('geopotential', 500), 'T500': ('temperature', 500),
       'U500': ('u_component_of_wind', 500), 'V500': ('v_component_of_wind', 500), 'Z850': ('geopotential', 850),
       'T850': ('temperature', 850), 'TP6': ('total_precipitation_6hr', None)}
# 44 groups (every 8 days) of 4 consecutive inits (day d and d+1, 00/12 UTC): chunk-aligned in the WB2 stores
inits = np.array([np.datetime64('2020-01-01T00') + np.timedelta64(24 * d + h, 'h') for d0 in range(0, 345, 8) for d in (d0, d0 + 1) for h in (0, 12)])
np.save(f'{OUT}/inits.npy', inits.astype('datetime64[h]').astype(np.int64))

def run(name, path, t0, nlead_out=60, subset='all', leadsel=None, skip=()):
    if os.path.exists(f'{OUT}/{name}.npy'): return
    s = Store(path); tm = np.datetime64(t0) + s.coord('time').astype('timedelta64[h]')
    lt = s.coord('prediction_timedelta'); lev = list(s.coord('level'))
    sel = np.arange(len(inits))
    if subset == 'first2': sel = sel[(sel % 4) < 2]                 # day d only (00/12 UTC)
    if subset == 'alt_groups': sel = sel[(sel // 4) % 2 == 0]        # every other group
    ii_all = np.array([np.where(tm == d)[0][0] for d in inits])
    ii = ii_all[sel]
    cache = {}
    out = np.full((len(inits), nlead_out, len(VARS), 32, 64), np.nan, np.float32)
    leads = [(j, int(np.where(lt == 6 * (j + 1))[0][0])) for j in range(nlead_out) if 6 * (j + 1) in lt]
    if leadsel is not None: leads = [(j, li) for j, li in leads if (j + 1) in leadsel]
    t = time.time()
    for vi, v in sorted(enumerate(VARS), key=lambda t: MAP[t[1]][0]):
        var, L = MAP[v]
        if v in skip: continue
        za = s.zarray(var); ck = za['chunks']; nd = len(za['shape'])
        cks = sorted(set(i // ck[0] for i in ii))
        if ck[1] == 1:   # chunk per lead
            keys = [(c, li) + (0,) * (nd - 2) for c in cks for (_, li) in leads]
        else:
            keys = [(c, 0) + (0,) * (nd - 2) for c in cks]
        if var not in cache:
            cache = {var: dict(zip(keys, s.read_many(var, keys, NT)))}
        data = cache[var]
        for n, i in zip(sel, ii):
            c = i // ck[0]; r = i - c * ck[0]
            for j, li in leads:
                a = data[(c, li) + (0,) * (nd - 2)][r, 0] if ck[1] == 1 else data[(c, 0) + (0,) * (nd - 2)][r, li]
                if L is not None: a = a[lev.index(L)]
                out[n, j, vi] = a.T
        print(name, v, f'{time.time()-t:.0f}s', flush=True)
    np.save(f'{OUT}/{name}.npy', out)

if __name__ == '__main__':
    which = sys.argv[1]
    daily = [1, 2] + list(range(4, 61, 4))        # 6 h, 12 h, then every 24 h
    if which == 'ens': run('ens_mean', 'ifs_ens/2018-2022-64x32_equiangular_conservative_mean.zarr', '2018-01-01T00', leadsel=daily)
    if which == 'hres': run('hres', 'hres/2016-2022-0012-64x32_equiangular_conservative.zarr', '2016-01-01T00', subset='first2', leadsel=[1, 4, 8, 12, 20, 28, 40])
    if which == 'fuxi': run('fuxi_wb2', 'fuxi/2020-64x32_equiangular_conservative.zarr', '2020-01-01T00', subset='alt_groups', skip=('TP6',))
