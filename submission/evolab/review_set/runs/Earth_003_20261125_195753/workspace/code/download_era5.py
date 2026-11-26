"""Download ERA5 (WeatherBench2, 64x32 equiangular conservative, 6-hourly) for the 70 FuXi channels.
Output per year: outputs/era5/{year}.npy  shape (T, 70, 32 lat, 64 lon), lat ascending.
Train years are stored as float16 normalised by stats from 2017; test/val years as float32 physical."""
import numpy as np, sys, os, time, json
from wb2io import Store
OUT = '/workspace/outputs/era5'; os.makedirs(OUT, exist_ok=True)
s = Store('era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr')
tm = s.coord('time')  # hours since 1959-01-01
UP = ['geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'relative_humidity']
SF = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure', 'total_precipitation_6hr']
base = np.datetime64('1959-01-01T00')
dates = base + tm.astype('timedelta64[h]')

def year_idx(y):
    return np.where((dates >= np.datetime64(f'{y}-01-01T00')) & (dates < np.datetime64(f'{y+1}-01-01T00')))[0]

def fetch(y):
    idx = year_idx(y); c0, c1 = idx[0] // 100, idx[-1] // 100
    arrs = []
    for v in UP:
        ch = s.read_many(v, [(c, 0, 0, 0) for c in range(c0, c1 + 1)], 2)
        a = np.concatenate(ch)[idx - c0 * 100]          # (T,13,lon,lat)
        arrs.append(a)
    for v in SF:
        ch = s.read_many(v, [(c, 0, 0) for c in range(c0, c1 + 1)], 2)
        a = np.concatenate(ch)[idx - c0 * 100][:, None]
        arrs.append(a)
    X = np.concatenate(arrs, 1).transpose(0, 1, 3, 2)   # (T,70,lat,lon)
    return X.astype(np.float32), dates[idx]

if __name__ == '__main__':
    years = [int(a) for a in sys.argv[1:]]
    for y in years:
        f = f'{OUT}/{y}.npy'
        if os.path.exists(f): continue
        t = time.time(); X, d = fetch(y)
        if y == 2017 and not os.path.exists(f'{OUT}/stats.npz'):
            mu = X.mean((0, 2, 3)); sd = X.std((0, 2, 3))
            np.savez(f'{OUT}/stats.npz', mean=mu, std=sd)
        if y in (2017,) or y >= 2018:
            np.save(f + '.tmp.npy', X); os.rename(f + '.tmp.npy', f)
        else:
            st = np.load(f'{OUT}/stats.npz')
            Xn = ((X - st['mean'][None, :, None, None]) / st['std'][None, :, None, None]).astype(np.float16)
            np.save(f + '.tmp.npy', Xn); os.rename(f + '.tmp.npy', f)
        np.save(f'{OUT}/{y}_dates.npy', d.astype('datetime64[h]').astype(np.int64))
        import shutil; shutil.rmtree('/workspace/outputs/cache/era5', ignore_errors=True)
        print(y, X.shape, f'{time.time()-t:.0f}s', np.isnan(X).sum(), flush=True)
