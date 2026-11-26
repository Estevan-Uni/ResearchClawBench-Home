"""Data overview figure: ERA5 (WB2 64x32) fields used by the models + dataset split."""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *

X, d = load_year_physical(2020)
i = int(np.where(d == np.datetime64('2020-02-02T00'))[0][0])
panels = [('Z500', 1 / 9.80665, 'gpm', 'viridis'), ('T850', 1, 'K', 'RdYlBu_r'), ('U500', 1, 'm/s', 'PuOr_r'), ('R700', 100, '%', 'BrBG'),
          ('T2M', 1, 'K', 'RdYlBu_r'), ('MSL', 0.01, 'hPa', 'viridis'), ('U10', 1, 'm/s', 'PuOr_r'), ('TP', 1000, 'mm / 6 h', 'Blues')]
fig, axs = plt.subplots(2, 4, figsize=(19, 6.5))
for ax, (v, sc, u, cm) in zip(axs.flat, panels):
    a = X[i, NAMES.index(v)] * sc
    im = ax.pcolormesh(LON, LAT, a, cmap=cm, vmin=None if v != 'TP' else 0, vmax=None if v != 'TP' else np.percentile(a, 99))
    plt.colorbar(im, ax=ax, shrink=0.85, label=u); ax.set_title(f'{v}  (ERA5, {str(d[i])} UTC)')
    ax.set_xlabel('lon'); ax.set_ylabel('lat')
plt.suptitle('Input/target data: ERA5 on the WeatherBench2 5.625° grid (64×32), 70 channels = 5 upper-air vars × 13 levels + 5 surface vars', fontsize=11)
plt.tight_layout(); plt.savefig('/workspace/report/images/fig02_data_overview.png', dpi=100); plt.close()

years = sorted(int(f[:4]) for f in os.listdir(ERA) if f.endswith('.npy') and f[:4].isdigit() and '_' not in f)
nm = Normaliser()
fig, ax = plt.subplots(1, 2, figsize=(16, 3.8))
for y in years:
    c = 'tab:red' if y == 2020 else 'tab:blue'
    ax[0].barh(0, 1, left=y, color=c, edgecolor='k')
ax[0].set_yticks([]); ax[0].set_xlabel('year'); ax[0].set_title('Years used: blue = training (last 45 days of 2017 held out for validation), red = 2020 test')
ax[1].bar(range(70), nm.dsd, color=['C0'] * 13 + ['C1'] * 13 + ['C2'] * 13 + ['C3'] * 13 + ['C4'] * 13 + ['C5'] * 5)
ax[1].set_xticks([6, 19, 32, 45, 58, 67]); ax[1].set_xticklabels(['Z', 'T', 'U', 'V', 'R', 'sfc'])
ax[1].set_title('6-h tendency std / field std per channel (scales the residual output)')
plt.tight_layout(); plt.savefig('/workspace/report/images/fig03_dataset.png', dpi=100); plt.close()
print('years', years)
