"""Download WB2 ERA5 1990-2019 6-hourly climatology (64x32) for the 11 verification variables + static fields."""
import numpy as np, os
from wb2io import Store
from download_benchmarks import VARS, MAP
OUT = '/workspace/outputs/bench'
import sys
if sys.argv[1:] == ['static']:
    e = Store('era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr')
    st = {k: e.read_chunk(k, (0, 0)).T for k in ['land_sea_mask', 'geopotential_at_surface']}
    st['lat'] = e.coord('latitude'); st['lon'] = e.coord('longitude')
    np.savez(f'{OUT}/static.npz', **st); print('static ok'); sys.exit()
s = Store('era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr')
lev = list(s.coord('level')); print('hours', s.coord('hour'), 'doy', s.coord('dayofyear')[:3])
clim = np.zeros((4, 366, len(VARS), 32, 64), np.float32)
cache = {}
for vi, v in enumerate(VARS):
    var, L = MAP[v]
    if var not in cache:
        nd = len(s.zarray(var)['shape']); cache[var] = s.read_chunk(var, (0,) * nd)
    a = cache[var]
    if L is not None: a = a[:, :, lev.index(L)]
    clim[:, :, vi] = a.transpose(0, 1, 3, 2); print(v, a.shape, flush=True)
np.save(f'{OUT}/clim.npy', clim)
e = Store('era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr')
st = {k: e.read_chunk(k, (0, 0)).T for k in ['land_sea_mask', 'geopotential_at_surface']}
st['lat'] = e.coord('latitude'); st['lon'] = e.coord('longitude')
np.savez(f'{OUT}/static.npz', **st); print('static ok', st['lat'][:2])
