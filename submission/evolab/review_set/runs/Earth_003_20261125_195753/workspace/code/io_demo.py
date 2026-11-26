"""End-to-end I/O demonstration on the *provided* files (format check only).
Reads data/20231012-06_input_netcdf.nc (2 x 70 x 181 x 360, 1 deg), conservatively (area-weighted box)
averages to the 5.625 deg model grid, runs the trained cascade for 60 x 6 h steps, bilinearly
interpolates back to 1 deg and writes outputs/io_demo/{006..360}.nc in the same layout as data/006.nc.
NOTE: the provided files are white noise (see diagnose_data.py), so the forecast values are meaningless;
this only demonstrates that the pipeline consumes and produces the task's file format."""
import os, numpy as np, torch
from scipy.io import netcdf_file
from scipy.interpolate import RegularGridInterpolator
from common import *
from gen import load
from train import step

OUT = f'{ROOT}/io_demo'; os.makedirs(OUT, exist_ok=True)
f = netcdf_file('/workspace/data/20231012-06_input_netcdf.nc', 'r', mmap=False)
X = np.array(f.variables['data'].data, np.float64)            # (2,70,181,360)
names = [r.tobytes().decode().replace('\x00', '').strip() for r in f.variables['level'].data]
assert names == NAMES, (names[:5], NAMES[:5])
lat1 = np.array(f.variables['lat'].data, np.float64); lon1 = np.array(f.variables['lon'].data, np.float64)
X = np.where(np.isfinite(X), X, 0)


def to_coarse(A):
    out = np.zeros(A.shape[:-2] + (32, 64))
    wl = np.cos(np.deg2rad(lat1))
    for i, la in enumerate(LAT):
        mi = np.abs(lat1 - la) <= 2.8125
        for j, lo in enumerate(LON):
            dl = (lon1 - lo + 180) % 360 - 180; mj = np.abs(dl) <= 2.8125
            sub = A[..., mi, :][..., mj]; ww = wl[mi][:, None] * np.ones(mj.sum())[None]
            out[..., i, j] = (sub * ww).sum((-2, -1)) / ww.sum()
    return out


Xc = to_coarse(X).astype(np.float32)
nm = Normaliser(); Z = nm.norm(Xc)
t0 = np.datetime64('2023-10-12T06')
dstd = torch.from_numpy(nm.dsd).view(1, -1, 1, 1)
models = [(load('short'), 20), (load('medium'), 20), (load('long'), 20)]
a, c = torch.from_numpy(Z[0:1]), torch.from_numpy(Z[1:2])
k = 0
latc = np.concatenate([[-90], LAT, [90]]); lonc = np.concatenate([LON, [360]])
P1 = np.stack(np.meshgrid(lat1, lon1, indexing='ij'), -1)
with torch.no_grad():
    for m, n in models:
        for _ in range(n):
            fo = torch.from_numpy(forcing(np.array([t0 + np.timedelta64(6 * k, 'h')])))
            a, c = c, step(m, a, c, fo, dstd); k += 1
            phys = nm.denorm(c.numpy())[0]                                   # (70,32,64)
            ext = np.concatenate([phys[:, :1], phys, phys[:, -1:]], 1); ext = np.concatenate([ext, ext[:, :, :1]], 2)
            fine = np.stack([RegularGridInterpolator((latc, lonc), e)(P1) for e in ext]).astype(np.float32)
            if k in (1, 20, 40, 60) or True:
                g = netcdf_file(f'{OUT}/{6*k:03d}.nc', 'w')
                for dn, sz in [('time', 1), ('step', 1), ('level', 70), ('string5', 5), ('lat', 181), ('lon', 360)]: g.createDimension(dn, sz)
                v = g.createVariable('lon', '>f4', ('lon',)); v[:] = lon1
                v = g.createVariable('lat', '>f4', ('lat',)); v[:] = lat1
                v = g.createVariable('level', 'S1', ('level', 'string5'))
                v[:] = np.array([list(s.ljust(5, '\x00')) for s in NAMES], 'S1')
                v = g.createVariable('data', '>f4', ('time', 'step', 'level', 'lat', 'lon')); v[:] = fine[None, None]
                v.description = b'Cascade (Short/Medium/Long) forecast; model run at 5.625 deg, interpolated to 1 deg (I/O demo on noise input)'
                v = g.createVariable('time', '>i4', ('time',)); v[:] = [0]; v.units = b'days since 2023-10-12 06:00:00'
                v = g.createVariable('step', '>i4', ('step',)); v[:] = [6 * k]
                g.close()
chk = netcdf_file(f'{OUT}/360.nc', 'r', mmap=False)
print('wrote', len(os.listdir(OUT)), 'files; 360.nc data', chk.variables['data'].shape, 'finite', np.isfinite(chk.variables['data'].data).all())
ref = netcdf_file('/workspace/data/006.nc', 'r', mmap=False)
print('layout matches 006.nc:', {k: v.shape for k, v in ref.variables.items()} == {k: v.shape for k, v in netcdf_file(f"{OUT}/006.nc", 'r', mmap=False).variables.items()})
