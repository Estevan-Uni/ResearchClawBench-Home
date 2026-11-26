"""Diagnose the provided ERA5-format sample files (input + 006.nc)."""
import numpy as np, json
from scipy.io import netcdf_file
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

def load(p):
    f = netcdf_file(p, 'r', mmap=False)
    info = {k: (v.shape, v.dimensions) for k, v in f.variables.items()}
    return f, info

fi, ii = load('data/20231012-06_input_netcdf.nc'); fo, io = load('data/006.nc')
print(ii); print(io)
def getdata(f):
    for k, v in f.variables.items():
        if len(v.shape) >= 4: return k, np.array(v.data, dtype=np.float64)
ki, X = getdata(fi); ko, Y = getdata(fo)
names = None
for k, v in fi.variables.items():
    if v.data.dtype.kind in 'SU' or k in ('level', 'channel'):
        try: names = [r.tobytes().decode().replace('\x00', '').strip() for r in v.data]
        except Exception: names = [str(x) for x in v.data]
print('names', names[:8], '...', names[-6:] if names else None)
Y = Y.reshape(-1, *Y.shape[-3:])
print(ki, X.shape, ko, Y.shape)
N = 181 * 360
res = {}
rows = []
for c in range(X.shape[1]):
    a, b, o = X[0, c], X[1, c], Y[0, c]
    lx = np.corrcoef(b[:, :-1].ravel(), b[:, 1:].ravel())[0, 1]
    ly = np.corrcoef(b[:-1].ravel(), b[1:].ravel())[0, 1]
    lt = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    lo = np.corrcoef(b.ravel(), o.ravel())[0, 1]
    rows.append([b.mean(), b.std(), lx, ly, lt, lo])
rows = np.array(rows)
flat = X[1].reshape(70, -1); C = np.corrcoef(flat); off = np.abs(C[~np.eye(70, dtype=bool)])
iz = names.index('z500') if names and 'z500' in names else 7
z = X[1, iz].ravel(); ks_z = stats.kstest(z / z.std(), 'norm')
tp = X[1, -1].ravel(); ks_tp = stats.kstest(tp, 'halfnorm', args=(0, np.sqrt((tp**2).mean())))
rmse_pers = np.sqrt(((Y[0] - X[1])**2).mean()); rmse_mean = np.sqrt(((Y[0] - X[1].mean(axis=(1, 2), keepdims=True))**2).mean())
res = dict(shape_input=list(X.shape), shape_output=list(Y.shape), names=names,
           max_abs_lag1_lon=float(np.abs(rows[:, 2]).max()), max_abs_lag1_lat=float(np.abs(rows[:, 3]).max()),
           max_abs_corr_t_minus_6h=float(np.abs(rows[:, 4]).max()), max_abs_corr_input_output=float(np.abs(rows[:, 5]).max()),
           max_offdiag_crosschannel=float(off.max()), noise_level_1sigma=float(1 / np.sqrt(N)),
           channel_std_range=[float(rows[:, 1].min()), float(rows[:, 1].max())], channel_mean_range=[float(rows[:, 0].min()), float(rows[:, 0].max())],
           ks_z500_normal_p=float(ks_z.pvalue), ks_tp_halfnormal_p=float(ks_tp.pvalue), tp_min=float(tp.min()),
           rmse_006_persistence=float(rmse_pers), rmse_006_channelmean=float(rmse_mean))
print(json.dumps(res, indent=1))
json.dump(res, open('outputs/data_diagnosis.json', 'w'), indent=1)

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
lon = np.arange(360); lat = np.linspace(90, -90, 181)
im = ax[0, 0].imshow(X[1, iz], cmap='RdBu_r', extent=[0, 360, -90, 90]); ax[0, 0].set_title(f'Provided input: {names[iz] if names else iz} (t0)'); plt.colorbar(im, ax=ax[0, 0], shrink=.7)
im = ax[0, 1].imshow(Y[0, iz], cmap='RdBu_r', extent=[0, 360, -90, 90]); ax[0, 1].set_title('Provided 006.nc: same channel (t0+6h)'); plt.colorbar(im, ax=ax[0, 1], shrink=.7)
ax[0, 2].hist(z, 100, density=True, alpha=.7, label='z500 values'); xs = np.linspace(-40, 40, 200)
ax[0, 2].plot(xs, stats.norm.pdf(xs, 0, z.std()), 'k', label=f'N(0,{z.std():.1f}²), KS p={ks_z.pvalue:.2f}'); ax[0, 2].legend(fontsize=8); ax[0, 2].set_title('Value distribution')
th = 3 / np.sqrt(N)
for j, (lab, k) in enumerate([('lag-1 lon', 2), ('lag-1 lat', 3), ('corr(t-6h,t)', 4), ('corr(t, 006)', 5)]):
    ax[1, 0 if j < 2 else 1].plot(rows[:, k], '.', label=lab)
for a_ in ax[1, :2]:
    a_.axhspan(-th, th, color='grey', alpha=.3, label='±3/√N noise'); a_.set_ylim(-0.05, 0.05); a_.legend(fontsize=8); a_.set_xlabel('channel index')
ax[1, 0].set_title('Spatial autocorrelation per channel (real ERA5: >0.95)'); ax[1, 1].set_title('Temporal correlation per channel (real ERA5: >0.9)')
im = ax[1, 2].imshow(C, cmap='RdBu_r', vmin=-.05, vmax=.05); ax[1, 2].set_title(f'Cross-channel corr (max |off-diag|={off.max():.3f})'); plt.colorbar(im, ax=ax[1, 2], shrink=.7)
plt.tight_layout(); plt.savefig('report/images/fig01_data_diagnosis.png', dpi=110)
