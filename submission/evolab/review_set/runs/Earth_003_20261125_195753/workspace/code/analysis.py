"""Maps, spatial RMSE / difference maps, spectra, activity, precipitation ETS, cascade-vs-single gains."""
import os, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from evaluate import truth_and_clim
from plots import M, SYS, boot_ci, common_sel, avail, LEADS, VI, V8, IM, groups

FC = f'{ROOT}/fc'; B = f'{ROOT}/bench'
inits, T, C, P = truth_and_clim()
F = {s: np.load(f'{FC}/{s}.npy', mmap_mode='r') for s in ['cascade', 'short_only', 'shortplus'] if os.path.exists(f'{FC}/{s}.npy')}
F.update({s: np.load(f'{B}/{s}.npy', mmap_mode='r') for s in ['ens_mean', 'hres', 'fuxi_wb2'] if os.path.exists(f'{B}/{s}.npy')})
res = {}
w = WLAT[:, None]

# ---------- 1. Z500 maps, one case -----------------------------------------------------------
case = 16                                      # 2020-02-02T00 (covered by all systems)
lead_days = [1, 3, 5, 7, 10, 15]
cols = [('truth', 'ERA5 (truth)'), ('hres', 'ECMWF HRES'), ('ens_mean', 'ECMWF ENS mean'), ('cascade', 'Cascade (ours)'),
        ('short_only', 'Single model (ours)'), ('fuxi_wb2', 'FuXi (published)')]
cols = [c for c in cols if c[0] == 'truth' or c[0] in F]
fig, axs = plt.subplots(len(lead_days), len(cols), figsize=(3.1 * len(cols), 2.0 * len(lead_days)))
j = VI['Z500']
for r, ld in enumerate(lead_days):
    k = ld * 4 - 1
    tr = T[case, k, j] / 9.80665
    for c, (s, lab) in enumerate(cols):
        ax = axs[r, c]
        fld = tr if s == 'truth' else np.asarray(F[s][case, k, j]) / 9.80665
        if np.all(np.isnan(fld)):
            ax.text(0.5, 0.5, 'n/a', ha='center', va='center', transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.contourf(LON, LAT, fld, levels=np.arange(4800, 6001, 50), cmap='viridis', extend='both')
            ax.contour(LON, LAT, tr, levels=np.arange(4800, 6001, 200), colors='w', linewidths=0.5)
            if s != 'truth':
                rm = np.sqrt(np.nanmean((fld - tr) ** 2 * w))
                ax.set_title(f'{lab}\nRMSE {rm:.0f} m' if r == 0 else f'RMSE {rm:.0f} m', fontsize=8)
            else:
                ax.set_title(lab if r == 0 else '', fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        if c == 0: ax.set_ylabel(f'day {ld}', fontsize=10)
plt.suptitle(f'Z500 [gpm], init {str(inits[case])} UTC; white contours = ERA5 truth (200 gpm)', fontsize=11)
plt.tight_layout(); plt.savefig(f'{IM}/fig09_z500_maps.png', dpi=100); plt.close()

# ---------- 2. time-mean spatial RMSE maps + differences -------------------------------------
def rmse_map(s, k, j, sel):
    return np.sqrt(np.nanmean((np.asarray(F[s][sel, k, j]) - T[sel, k, j]) ** 2, 0))

fig, axs = plt.subplots(2, 5, figsize=(22, 7))
for r, ld in enumerate([5, 10]):
    k = ld * 4 - 1; j = VI['Z500']
    sel = np.where(common_sel('cascade', 'ens_mean', 'hres') if 'hres' in F else common_sel('cascade', 'ens_mean'))[0]
    mc, me = rmse_map('cascade', k, j, sel), rmse_map('ens_mean', k, j, sel)
    mh = rmse_map('hres', k, j, sel) if 'hres' in F else np.full_like(mc, np.nan)
    vmax = np.nanpercentile(np.stack([mc, me, mh]), 99)
    for c, (fld, lab) in enumerate([(mh, 'HRES'), (me, 'ENS mean'), (mc, 'Cascade')]):
        im = axs[r, c].pcolormesh(LON, LAT, fld, vmin=0, vmax=vmax, cmap='magma_r'); axs[r, c].set_title(f'Z500 RMSE day {ld}: {lab}')
        plt.colorbar(im, ax=axs[r, c], shrink=0.8)
    for c, (fld, lab) in enumerate([(me - mc, 'ENS mean − Cascade'), (mh - mc, 'HRES − Cascade')]):
        v = np.nanpercentile(np.abs(me - mc), 98)
        im = axs[r, 3 + c].pcolormesh(LON, LAT, fld, vmin=-v, vmax=v, cmap='RdBu'); axs[r, 3 + c].set_title(f'day {ld}: {lab}\n(blue = cascade better)')
        plt.colorbar(im, ax=axs[r, 3 + c], shrink=0.8)
    res[f'rmse_map_frac_cascade_better_than_em_d{ld}'] = float(np.average((me - mc) > 0, weights=np.broadcast_to(w, mc.shape)))
    res[f'rmse_map_frac_cascade_better_than_hres_d{ld}'] = float(np.average((mh - mc) > 0, weights=np.broadcast_to(w, mc.shape)))
    res['rmse_map_nsel'] = int(len(sel))
for ax in axs.flat: ax.set_xticks([]); ax.set_yticks([])
plt.suptitle(f'Time-mean per-gridpoint Z500 RMSE [m²/s²] over {len(sel)} paired 2020 inits', fontsize=12)
plt.tight_layout(); plt.savefig(f'{IM}/fig10_rmse_maps.png', dpi=95); plt.close()

# ---------- 3. activity and zonal spectra -----------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))
for v, ax in zip(['Z500', 'T2M', 'U500'], axs):
    j = VI[v]
    ta = np.nanmean(M['truth_act'][:, :, j], 0)
    for s in ['cascade', 'short_only', 'shortplus', 'ens_mean', 'hres', 'fuxi_wb2']:
        if s not in SYS: continue
        a = np.nanmean(M[f'{s}_act'][:, :, j], 0) / ta; lm = avail(s) & np.isfinite(a)
        ax.plot(LEADS[lm], a[lm], color=SYS[s][1], ls=SYS[s][2], marker='o' if lm.sum() < 30 else None, ms=3, label=SYS[s][0])
    ax.axhline(1, color='k', lw=0.8); [ax.axvline(x, color='k', lw=.5, alpha=.4) for x in (5, 10)]
    ax.set_title(f'{v}: activity ratio  std(forecast anomaly)/std(ERA5 anomaly)'); ax.set_xlabel('lead (days)'); ax.grid(alpha=.3)
axs[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{IM}/fig11_activity.png', dpi=105); plt.close()


def zspec(A):     # A: (..., 32, 64) -> zonal power spectrum averaged over 20-70 deg in both hemispheres
    m = (np.abs(LAT) >= 20) & (np.abs(LAT) <= 70)
    S = np.abs(np.fft.rfft(A[..., m, :], axis=-1)) ** 2
    return np.nanmean(S.reshape(-1, m.sum(), S.shape[-1]), (0, 1))

fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))
kk = np.arange(33)
spec_res = {}
for ax, ld in zip(axs, [1, 5, 15]):
    k = ld * 4 - 1; j = VI['Z500']
    sel = np.where(np.isfinite(F['cascade'][:, 0, 0, 0, 0]))[0]
    st = zspec(T[sel, k, j]); ax.loglog(kk[1:], st[1:], 'k', lw=2.5, label='ERA5')
    for s in ['cascade', 'short_only', 'ens_mean', 'hres', 'fuxi_wb2']:
        if s not in F: continue
        ss = np.where(np.isfinite(np.asarray(F[s][:, k, j, 0, 0])))[0]
        if len(ss) == 0: continue
        sp = zspec(np.asarray(F[s][ss, k, j])); ax.loglog(kk[1:], sp[1:], color=SYS[s][1], ls=SYS[s][2], label=SYS[s][0])
        spec_res[f'{s}_d{ld}_ratio_k10_20'] = float(np.mean(sp[10:21] / zspec(T[ss, k, j])[10:21]))
    ax.set_title(f'Z500 zonal power spectrum (20-70°), day {ld}'); ax.set_xlabel('zonal wavenumber'); ax.grid(alpha=.3, which='both')
axs[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{IM}/fig12_spectra.png', dpi=105); plt.close()
res['spectra'] = spec_res

# ---------- 4. precipitation (6 h accumulations) ETS ----------------------------------------
def ets(f, o, thr):
    f, o = f >= thr, o >= thr
    wgt = np.broadcast_to(WLAT[:, None], f.shape[-2:])
    a = (f & o) * wgt; b = (f & ~o) * wgt; c = (~f & o) * wgt; tot = wgt.sum() * f.shape[0]
    a, b, c = a.sum(), b.sum(), c.sum(); ar = (a + b) * (a + c) / tot
    return (a - ar) / (a + b + c - ar)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
j = VI['TP6']; ets_res = {}
for ax, thr in zip(axs, [0.5e-3, 2e-3]):
    for s in ['cascade', 'short_only', 'ens_mean', 'hres']:
        if s not in F: continue
        y = []
        for k in range(60):
            fk = np.asarray(F[s][:, k, j]); ok = np.isfinite(fk[:, 0, 0]) & np.isfinite(T[:, k, j, 0, 0])
            y.append(ets(fk[ok], T[ok, k, j], thr) if ok.sum() else np.nan)
        y = np.array(y); lm = np.isfinite(y)
        ax.plot(LEADS[lm], y[lm], color=SYS[s][1], ls=SYS[s][2], marker='o' if lm.sum() < 30 else None, ms=3, label=SYS[s][0])
        ets_res[f'{s}_{thr*1e3:g}mm'] = {f'd{d}': float(y[d * 4 - 1]) for d in (1, 3, 5, 10, 15) if np.isfinite(y[d * 4 - 1])}
    ax.set_title(f'6-h precipitation ETS, threshold {thr*1e3:g} mm'); ax.set_xlabel('lead (days)'); ax.grid(alpha=.3)
axs[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{IM}/fig13_precip_ets.png', dpi=105); plt.close()
res['ets'] = ets_res

# ---------- 5. cascade vs single model gain with block-bootstrap CI ----------------------------
fig, axs = plt.subplots(2, 4, figsize=(18, 7.5))
gain = {}
others = [o for o in ['short_only', 'shortplus'] if o in SYS]
for ax, v in zip(axs.flat, V8):
    j = VI[v]
    for o in others:
        m_, lo, hi = [], [], []
        for k in range(60):
            rc = np.sqrt(M['cascade_mse'][:, k, j]); ro = np.sqrt(M[f'{o}_mse'][:, k, j])
            d = (ro - rc) / np.nanmean(ro)
            a, b, c = boot_ci(d, nb=300, seed=k); m_.append(a); lo.append(b); hi.append(c)
        m_, lo, hi = map(np.array, (m_, lo, hi))
        ax.plot(LEADS, 100 * m_, color=SYS[o][1], label=f'vs {SYS[o][0]}'); ax.fill_between(LEADS, 100 * lo, 100 * hi, color=SYS[o][1], alpha=.2)
        gain[f'{v}_vs_{o}'] = {f'd{d}': [round(100 * m_[d * 4 - 1], 2), round(100 * lo[d * 4 - 1], 2), round(100 * hi[d * 4 - 1], 2)] for d in (5, 7, 10, 12, 15)}
    ax.axhline(0, color='k', lw=.8); [ax.axvline(x, color='k', lw=.5, alpha=.4) for x in (5, 10)]
    ax.set_title(v); ax.set_xlabel('lead (days)'); ax.set_ylabel('RMSE reduction by cascade [%]'); ax.grid(alpha=.3)
axs.flat[0].legend(fontsize=8)
plt.suptitle('Cascade RMSE reduction relative to single-model baselines (95% block-bootstrap CI over 44 init groups)')
plt.tight_layout(); plt.savefig(f'{IM}/fig14_cascade_gain.png', dpi=105); plt.close()
res['cascade_gain_pct'] = gain

# sanity checks
sc = {}
dd = np.nanmax(np.abs(np.asarray(F['cascade'][:, :20]) - np.asarray(F['short_only'][:, :20])), axis=(0, 1, 3, 4))
sc['cascade_minus_short_first20_maxabs_per_var'] = dict(zip(VER, [float(x) for x in dd]))   # float32 thread-count round-off only
sc['climatology_rmse_over_clim_std'] = None
pers_worse = {}
for s in ['cascade', 'short_only']:
    r = np.sqrt(np.nanmean(M[f'{s}_mse'], 0)); p = np.sqrt(np.nanmean(M['persistence_mse'], 0))
    pers_worse[s] = bool(np.all(r[4:, :10] < p[4:, :10]))
sc['ml_vs_persistence_climatology'] = pers_worse
res['sanity'] = sc
json.dump(res, open(f'{ROOT}/analysis_results.json', 'w'), indent=1)
print(json.dumps(res, indent=1)[:3000])
