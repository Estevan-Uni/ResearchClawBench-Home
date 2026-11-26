"""Generate all report figures (PNG) into report/images/."""
import json, h5py, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                     'ytick.color': '#52514e', 'axes.grid': False, 'figure.dpi': 130, 'savefig.bbox': 'tight'})
C1, C2, C3, C4, INK, INK2 = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#0b0b0b', '#52514e'
IMG = 'report/images/'; O = 'outputs/'
f = h5py.File('data/raw_trARPES_data.h5', 'r')
E = f['energy_axis'][:]; k = f['kx_axis'][:]
off = f['pump_off_spectrum'][:]
angles = [int(a) for a in f['polarization_angles'][:]]
on = {a: f[f'pump_on_angle_{a}'][:] for a in angles}
R = json.load(open(O + 'analysis_results.json')); P = json.load(open(O + 'polarization_results.json'))
F = pd.read_csv(O + 'floquet_fit_per_angle.csv').set_index('angle')
hw = F.loc[0, 'hw']; v = F.loc[0, 'v']
KK, EE = np.meshgrid(k, E)
ext = dict(shading='auto', rasterized=True)

def cone(ED, v, g, A):
    k0 = np.abs(EE - ED) / v
    return A * (g**2 / ((KK - k0)**2 + g**2) + g**2 / ((KK + k0)**2 + g**2))
def model(p):
    A0, g0, Ap1, Am1, g1, Ap2, Am2, hw_, v_, c = p
    m = cone(0, v_, g0, A0) + cone(hw_, v_, g1, Ap1) + cone(-hw_, v_, g1, Am1)
    m += cone(2 * hw_, v_, g1 * np.sqrt(1.5), Ap2) + cone(-2 * hw_, v_, g1 * np.sqrt(1.5), Am2)
    return m + c
NAMES = ['A0', 'g0', 'A+1', 'A-1', 'g1', 'A+2', 'A-2', 'hw', 'v', 'c']
p0 = F.loc[0, NAMES].values

def guides(ax, orders=(0, 1, -1), lw=0.7):
    kk = np.linspace(-0.3, 0.3, 50)
    for n in orders:
        for s in (1, -1):
            ax.plot(kk, n * hw + s * v * np.abs(kk), ls='--', lw=lw, color='w' if n == 0 else '#ffd27a', alpha=0.8)
    ax.set_xlim(k[0], k[-1]); ax.set_ylim(E[0], E[-1])

# ---------------- Fig 1: data overview
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for a_, img, t in zip(ax, [off, on[0]], ['(a) pump off', r'(b) pump on, $\theta_p=0^\circ$, $t=0$']):
    m = a_.pcolormesh(k, E, img, cmap='Blues', norm=LogNorm(1, 220), **ext); a_.set_title(t, loc='left')
plt.colorbar(m, ax=ax[:2], label='intensity (counts, log)', pad=0.01)
m = ax[2].pcolormesh(k, E, on[0] - off, cmap='Blues', vmin=0, vmax=17, **ext)
plt.colorbar(m, ax=ax[2], label=r'$I_{on}-I_{off}$', pad=0.01)
ax[2].set_title('(c) difference (on − off)', loc='left')
guides(ax[2], orders=(1, -1))
for a_ in ax:
    a_.set_xlabel(r'$k_x - K$ (Å$^{-1}$)')
    a_.axhline(hw, color=C2, lw=0.6, ls=':'); a_.axhline(-hw, color=C2, lw=0.6, ls=':')
ax[0].set_ylabel(r'$E - E_D$ (eV)')
ax[2].text(0.29, hw + 0.012, r'$+\hbar\omega$', color=C2, ha='right'); ax[2].text(0.29, -hw + 0.012, r'$-\hbar\omega$', color=C2, ha='right')
plt.savefig(IMG + 'fig1_data_overview.png'); plt.close()

# ---------------- Fig 2: global fit, residuals, EDC
mod = model(p0); res = on[0] - mod
fig = plt.figure(figsize=(13, 7.6)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9])
a = fig.add_subplot(gs[0, 0]); m = a.pcolormesh(k, E, on[0] - off, cmap='Blues', vmin=0, vmax=17, **ext)
a.set_title('(a) data: on − off (0°)', loc='left'); plt.colorbar(m, ax=a)
b = fig.add_subplot(gs[0, 1], sharey=a); m = b.pcolormesh(k, E, mod - off, cmap='Blues', vmin=0, vmax=17, **ext)
b.set_title(r'(b) Floquet model: $\sum_{n=\pm1,\pm2} A_n\,\mathcal{C}(E-n\hbar\omega)$', loc='left'); plt.colorbar(m, ax=b)
c = fig.add_subplot(gs[0, 2], sharey=a); m = c.pcolormesh(k, E, res, cmap='RdBu_r', vmin=-0.2, vmax=0.2, **ext)
c.set_title(rf'(c) residual, $\chi^2_\nu$={F.loc[0, "chi2_red"]:.3f}', loc='left'); plt.colorbar(m, ax=c, label='counts')
for x in (a, b, c): x.set_xlabel(r'$k_x - K$ (Å$^{-1}$)')
a.set_ylabel(r'$E - E_D$ (eV)')
d = fig.add_subplot(gs[1, :2])
edc = np.load(O + 'edc_k0_diff.npy'); j0 = np.argmin(abs(k))
d.plot(E, edc[:, 2], color=INK2, lw=1.5, label='pump off')
d.plot(E, edc[:, 3], color=C1, lw=2, label=r'pump on ($\theta_p=0^\circ$)')
d.plot(E, 10 * edc[:, 1], color=C2, lw=2, label=r'10 × (on − off)')
mm = model(p0)[:, j0 - 1:j0 + 2].mean(1) - off[:, j0 - 1:j0 + 2].mean(1)
d.plot(E, 10 * mm, color=INK, lw=1, ls='--', label='10 × model difference')
for n in (-1, 1): d.axvline(n * hw, color=C2, lw=0.7, ls=':')
d.set_xlabel(r'$E - E_D$ (eV)'); d.set_ylabel(r'intensity at $k_x=K$'); d.legend(frameon=False, fontsize=9)
d.set_title(r'(d) energy distribution curve through the Dirac point ($|k_x-K|<0.005$ Å$^{-1}$)', loc='left')
e = fig.add_subplot(gs[1, 2])
sc = pd.read_csv(O + 'hw_scan.csv')
e.plot(sc.hw * 1e3, sc.chi2 - sc.chi2.min(), color=C1, lw=2, marker='o', ms=4)
e.axvline(248, color=C2, ls=':', lw=1); e.set_yscale('symlog', linthresh=10); e.set_ylim(0, 5e6)
e.set_xlabel(r'trial $\hbar\omega$ (meV)'); e.set_ylabel(r'$\Delta\chi^2$')
e.set_title(rf'(e) photon-energy profile: $\hbar\omega$={hw*1e3:.2f}±{F.loc[0,"hw_err"]*1e3:.2f} meV', loc='left', fontsize=9)
e.text(252, 30, r'$hc/5\,\mu$m = 248 meV', color=C2, fontsize=8)
plt.tight_layout(); plt.savefig(IMG + 'fig2_floquet_fit.png'); plt.close()

# ---------------- Fig 3: dispersions
md = pd.read_csv(O + 'pumpoff_mdc_fits.csv'); rp = pd.read_csv(O + 'replica_mdc_fits.csv')
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
a = ax[0]
a.pcolormesh(k, E, on[0] - off, cmap='Greys', vmin=0, vmax=20, **ext)
s = md[md.E.abs() > 0.03]
a.plot(s.k0, s.E, '.', color=C1, ms=4, label='n=0 (pump-off MDC peaks)'); a.plot(-s.k0, s.E, '.', color=C1, ms=4)
for n, col in ((1, C2), (-1, C3)):
    q = rp[rp.order == n]
    a.plot(q.k_rep, q.E, '.', color=col, ms=4, label=f'n={n:+d} replica (difference MDC peaks)'); a.plot(-q.k_rep, q.E, '.', color=col, ms=4)
kk = np.linspace(-0.3, 0.3, 10)
for n in (0, 1, -1):
    for sg in (1, -1): a.plot(kk, n * hw + sg * v * np.abs(kk), color=INK2, lw=0.6, ls='--')
a.set_xlim(-0.12, 0.12); a.set_ylim(-0.5, 0.5)
a.set_xlabel(r'$k_x - K$ (Å$^{-1}$)'); a.set_ylabel(r'$E - E_D$ (eV)')
a.legend(frameon=True, fontsize=8, loc='lower left', facecolor='white', framealpha=0.9)
a.set_title('(a) extracted band positions; dashed: $E=n\\hbar\\omega\\pm\\hbar v_F|k|$', loc='left', fontsize=9)
b = ax[1]
disp = R['replica_dispersion']
labs = list(disp.keys()); sl = [abs(disp[l]['slope']) for l in labs]; se = [disp[l]['slope_err'] for l in labs]
xs = np.arange(len(labs) + 2)
vals = [R['pumpoff_mdc']['hv_upper'], R['pumpoff_mdc']['hv_lower']] + sl
errs = [0, 0] + se
names = ['n=0 upper', 'n=0 lower'] + [l.replace('_', ' ').replace('n+1', 'n=+1').replace('n-1', 'n=−1') for l in labs]
cols = [C1, C1, C2, C2, C3, C3]
for x_, v_, e_, c_ in zip(xs, vals, errs, cols): b.errorbar(x_, v_, yerr=e_, fmt='o', color=c_, ms=8, capsize=3)
fr = R['free_replica_fit_angle0']
b.axhline(fr['v1']['val'], color=C2, lw=1, ls=':'); b.text(5.4, fr['v1']['val'] + 0.004, f"global fit $v_{{\\pm1}}$={fr['v1']['val']:.4f}±{fr['v1']['err']:.4f}", ha='right', fontsize=8, color=INK2)
b.set_xticks(xs); b.set_xticklabels(names, rotation=25, fontsize=8)
b.set_ylabel(r'$|dE/dk|$ (eV Å)'); b.set_ylim(5.9, 6.1)
b.set_title('(b) group velocity of each branch', loc='left', fontsize=9)
plt.tight_layout(); plt.savefig(IMG + 'fig3_dispersion.png'); plt.close()

# ---------------- Fig 4: difference maps vs polarization (small multiples)
fig, ax = plt.subplots(1, 7, figsize=(16, 3.3), sharey=True)
for a_, ang in zip(ax, angles):
    m = a_.pcolormesh(k, E, on[ang] - off, cmap='Blues', vmin=0, vmax=17, **ext)
    a_.set_title(rf'$\theta_p$={ang}°', fontsize=10); a_.set_xlim(-0.15, 0.15); a_.set_xlabel(r'$k_x-K$')
ax[0].set_ylabel(r'$E-E_D$ (eV)'); plt.colorbar(m, ax=ax, pad=0.01, label='on − off')
plt.savefig(IMG + 'fig4_polarization_maps.png'); plt.close()

# ---------------- Fig 5: polarization dependence + model tests
th = np.array(P['theta_deg']); y = np.array(P['R1']); ye = np.array(P['R1_err'])
tt = np.linspace(0, 180, 361); tr = np.deg2rad(tt)
Mo = P['models']
def getp(prefix): return [v_ for k_, v_ in Mo.items() if k_.startswith(prefix)][0]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
a = ax[0]
a.errorbar(th, y, yerr=ye, fmt='o', color=INK, ms=7, zorder=5, label=r'$A_{\pm1}/A_0$ (global fit)')
p = getp('M4')['params']; a.plot(tt, p[0] + p[1] * np.cos(2 * tr)**2, color=C1, lw=2, label=rf"M4: $a+b\cos^2 2\theta$, $\chi^2$={getp('M4')['chi2']:.1f}")
p = getp('M3')['params']; a.plot(tt, p[0] + p[1] * np.cos(2 * tr) + p[2] * np.sin(2 * tr), color=C2, lw=2, label=rf"M3: $a+b\cos2\theta+c\sin2\theta$, $\chi^2$={getp('M3')['chi2']:.1e}")
p = getp('M1')['params']; a.plot(tt, p[0] * np.cos(tr)**2, color=C3, lw=2, label=rf"M1: $a\cos^2\theta$, $\chi^2$={getp('M1')['chi2']:.1e}")
import sys; sys.path.insert(0, 'code')
from scipy.special import jv
def bessel_ratios(p, t, nmax=6):
    aF, aV, r, phi = p
    alF = aF * np.cos(t); alV = aV * np.sqrt(np.cos(t)**2 + r * np.sin(t)**2)
    M = {n: sum(jv(n - m, alV) * jv(m, alF) * np.exp(1j * m * phi) for m in range(-nmax, nmax + 1)) for n in (0, 1, 2)}
    return np.abs(M[1])**2 / np.abs(M[0])**2, np.abs(M[2])**2 / np.abs(M[0])**2
pb = getp('B1')['params']; r1b, r2b = bessel_ratios(pb, tr)
a.plot(tt, r1b, color=C4, lw=1.6, ls='--', label=rf"B1: FB+Volkov Bessel, $\chi^2$={getp('B1')['chi2']:.1e}")
a.set_xlabel(r'pump polarization $\theta_p$ (deg)'); a.set_ylabel(r'first-order sideband weight $A_{\pm1}/A_0$')
a.set_xticks(range(0, 181, 30)); a.legend(frameon=False, fontsize=7.5, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.16)); a.set_ylim(0.03, 0.095)
a.set_title('(a) replica weight vs polarization', loc='left')
b = ax[1]
y2 = np.array(P['R2']); y2e = np.array(P['R2_err'])
sa = pd.read_csv(O + 'single_alpha_consistency.csv')
b.errorbar(th, y2 * 1e3, yerr=y2e * 1e3, fmt='o', color=INK, ms=7, label=r'observed $A_{\pm2}/A_0$')
b.plot(sa.angle, sa.R2_pred * 1e3, 's', color=C2, ms=8, label=r'Bessel prediction from $A_{\pm1}/A_0$ ($J_2^2/J_0^2$)')
b.plot(tt, r2b * 1e3, color=C4, ls='--', lw=1.5, label='B1 model prediction')
b.axhline(0, color=INK2, lw=0.6); b.set_ylim(-0.1, 2.0)
b.set_xlabel(r'$\theta_p$ (deg)'); b.set_ylabel(r'second-order weight $A_{\pm2}/A_0$ ($\times10^{-3}$)'); b.set_xticks(range(0, 181, 30))
b.legend(frameon=False, fontsize=8, loc='upper center'); b.set_title('(b) second-order sidebands: test of Bessel dressing', loc='left', fontsize=9)
c = ax[2]
C = pd.read_csv('data/polarization_dependence_data.csv')
bx = pd.read_csv(O + 'replica_box_integrals.csv')
c.plot(C.angle_degrees, C.intensity / C.intensity.mean(), 'o-', color=C3, ms=7, lw=1.5, label='provided CSV (main-band pixel)')
c.plot(bx.angle, bx['box_n+1'] / bx['box_n+1'].mean(), 's-', color=C1, ms=7, lw=1.5, label='box-integrated replica n=+1 (model-free)')
c.plot(th, y / y.mean(), 'd--', color=C2, ms=6, lw=1.2, label='global-fit $A_{\\pm1}/A_0$')
c.set_xlabel(r'$\theta_p$ (deg)'); c.set_ylabel('normalized to mean'); c.set_xticks(range(0, 181, 30))
c.legend(frameon=False, fontsize=8, loc='center right'); c.set_title(f"(c) contrast: CSV {P['csv_contrast']*100:.1f}% vs replica {P['contrast_R1']*100:.1f}%", loc='left', fontsize=9)
plt.tight_layout(); plt.savefig(IMG + 'fig5_polarization_dependence.png'); plt.close()

# ---------------- Fig 6: validation (provided products, noise, gap)
js = json.load(open('data/processed_band_data.json'))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
a = ax[0]
a.pcolormesh(k, E, on[0], cmap='Blues', norm=LogNorm(1, 220), **ext)
guides(a, orders=(1, -1))
for rb in js['replica_bands']:
    a.plot(rb['kx'], rb['energy'], 'x', color=C2, ms=10, mew=2.5)
a.plot(*js['dirac_point'], marker='*', color='#e34948', ms=16, clip_on=False, label='JSON "dirac_point"')
a.plot(C.target_kx[0], C.target_energy[0], 'o', mfc='none', mec=C3, ms=12, mew=2.5)
a.plot([], [], 'x', color=C2, mew=2.5, label='JSON "replica_bands"'); a.plot([], [], 'o', mfc='none', mec=C3, mew=2.5, label='CSV target pixel')
a.plot(0, 0, '+', color='w', ms=14, mew=2, label='refit Dirac point (0,0)')
bd = pd.DataFrame(js['band_dispersion'])
a.plot(bd.kx, bd.energy, '.', color='#eda100', ms=3, label='JSON "band_dispersion"')
a.legend(fontsize=7, loc='lower left', facecolor='white', framealpha=0.85)
a.set_xlabel(r'$k_x-K$ (Å$^{-1}$)'); a.set_ylabel(r'$E-E_D$ (eV)'); a.set_title('(a) provided processed products on raw map', loc='left', fontsize=9)
b = ax[1]
r0 = np.load(O + 'resid_angle0.npy').ravel(); sig = R['noise']['sigma_from_0_180']
b.hist(r0, bins=80, density=True, color=C1, alpha=0.85, label='fit residuals (0°)')
xx = np.linspace(-0.25, 0.25, 200); b.plot(xx, np.exp(-xx**2 / (2 * sig**2)) / np.sqrt(2 * np.pi) / sig, color=C2, lw=2,
                                           label=rf'Gaussian, $\sigma$={sig:.4f} (from 0°−180° maps)')
b.set_xlabel('residual (counts)'); b.set_ylabel('density'); b.legend(frameon=False, fontsize=8)
b.set_title('(b) residuals are consistent with pure noise', loc='left', fontsize=9)
c = ax[2]
gs_ = pd.read_csv(O + 'gap_scan.csv')
c.plot(gs_.Delta * 1e3, gs_.chi2 - gs_.chi2.min(), 'o-', color=C1, ms=4)
c.axhline(3.84, color=C2, ls=':'); c.text(30, 5, r'$\Delta\chi^2$=3.84 (95% CL)', color=C2, fontsize=8)
c.set_yscale('symlog', linthresh=10); c.set_ylim(0, None); c.set_xlabel(r'half-gap $\Delta$ at $n=0/\pm1$ crossings (meV)'); c.set_ylabel(r'$\Delta\chi^2$')
c.set_title(rf"(c) avoided-crossing test: $\Delta<{R['gap_test']['Delta_95CL_upper']*1e3:.1f}$ meV (95% CL)", loc='left', fontsize=9)
plt.tight_layout(); plt.savefig(IMG + 'fig6_validation.png'); plt.close()

# ---------------- Fig 7: MDC / EDC cuts with model components (didactic)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
i = np.argmin(abs(E - hw))
comp = {'n=0': cone(0, v, p0[1], p0[0]), 'n=+1': cone(hw, v, p0[4], p0[2]), 'n=−1': cone(-hw, v, p0[4], p0[3])}
a = ax[0]
a.plot(k, on[0][i], color=INK, lw=0, marker='o', ms=2.5, label=r'data, pump on (0°)')
for (nm, cc), col in zip(comp.items(), (C1, C2, C3)): a.plot(k, cc[i], color=col, lw=1.8, label=nm)
a.plot(k, model(p0)[i], color=C4, lw=1, ls='--', label='total model')
a.set_yscale('log'); a.set_ylim(0.3, 300); a.set_xlabel(r'$k_x-K$ (Å$^{-1}$)'); a.set_ylabel('intensity')
a.set_title(rf'(a) MDC at $E={E[i]:.3f}$ eV $\approx+\hbar\omega$', loc='left'); a.legend(frameon=False, fontsize=8)
b = ax[1]
jc = np.argmin(abs(k - hw / (2 * v)))
b.plot(E, on[0][:, jc], 'o', color=INK, ms=2.5, label='data, pump on (0°)')
for (nm, cc), col in zip(comp.items(), (C1, C2, C3)): b.plot(E, cc[:, jc], color=col, lw=1.8, label=nm)
b.plot(E, model(p0)[:, jc], color=C4, ls='--', lw=1, label='total model (no gap)')
b.axvline(hw / 2, color=INK2, ls=':', lw=0.8)
b.set_yscale('log'); b.set_ylim(0.3, 300); b.set_xlabel(r'$E-E_D$ (eV)'); b.set_ylabel('intensity')
b.set_title(rf'(b) EDC at crossing momentum $k_x-K={k[jc]:.4f}$ Å$^{{-1}}$', loc='left'); b.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(IMG + 'fig7_cuts_components.png'); plt.close()
print('done')
