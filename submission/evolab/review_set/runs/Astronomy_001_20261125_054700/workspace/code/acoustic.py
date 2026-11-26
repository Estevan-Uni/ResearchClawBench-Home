"""Acoustic tension in the (Omega_m, h r_d) plane.
DESI-only LCDM constraint reconstructed from the digitised Fig.6 BAO points relative to the
selected (Planck18) fiducial:  X_true(z) = X_fid(z) * (1 + Δ).  Flat LCDM BAO ratios depend only on
(Omega_m, h r_d). The analytic E(z) is used for both model and fiducial so that small
radiation/neutrino terms cancel in the ratio."""
import os, json, numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load_data, MODELS, NICE, COLORS, OUT, IMG

d = load_data()
dv = np.array(d['desi_dvrd_points']); fap = np.array(d['desi_fap_points'])
C = 2997.92458  # c/100 km/s in Mpc
zg = np.linspace(0, 2.5, 3001)

def bao(om, hrd, z):
    E = np.sqrt(om * (1 + zg)**3 + 1 - om)
    chi = np.concatenate([[0], cumulative_trapezoid(1 / E, zg)])
    DM = C * np.interp(z, zg, chi) / hrd
    DH = C / np.interp(z, zg, E) / hrd
    return (z * DM**2 * DH)**(1 / 3), DM / DH

fid = json.load(open(os.path.join(OUT, 'distance_fiducial_tests.json')))['Planck18 LCDM']
om_f, hrd_f = fid['fid_params']['omega_m'], fid['fid_hrd']
dvf, ff = bao(om_f, hrd_f, dv[:, 0]); _, fff = bao(om_f, hrd_f, fap[:, 0])
DV_obs, DV_err = dvf * (1 + dv[:, 1]), dvf * dv[:, 2]
F_obs, F_err = fff * (1 + fap[:, 1]), fff * fap[:, 2]

oms = np.linspace(0.22, 0.42, 241); hrds = np.linspace(90, 110, 241)
chi2 = np.zeros((len(oms), len(hrds)))
for i, om in enumerate(oms):
    for j, h in enumerate(hrds):
        a, _ = bao(om, h, dv[:, 0]); _, b = bao(om, h, fap[:, 0])
        chi2[i, j] = np.sum(((a - DV_obs) / DV_err)**2) + np.sum(((b - F_obs) / F_err)**2)
i0, j0 = np.unravel_index(chi2.argmin(), chi2.shape)
P = np.exp(-0.5 * (chi2 - chi2.min())); P /= P.sum()
om_mean = (P.sum(1) * oms).sum(); om_sd = np.sqrt((P.sum(1) * (oms - om_mean)**2).sum())
h_mean = (P.sum(0) * hrds).sum(); h_sd = np.sqrt((P.sum(0) * (hrds - h_mean)**2).sum())
print(f'DESI(digitised)-only LCDM: Om={om_mean:.4f}±{om_sd:.4f}, h rd={h_mean:.2f}±{h_sd:.2f}, chi2min={chi2.min():.2f} (dof={len(dv)+len(fap)-2})')

# distances from CMB+DESI model predictions to DESI-only constraint (Gaussian approx)
cov_desi = np.zeros((2, 2))
OM, HR = np.meshgrid(oms, hrds, indexing='ij')
cov_desi[0, 0] = (P * (OM - om_mean)**2).sum(); cov_desi[1, 1] = (P * (HR - h_mean)**2).sum()
cov_desi[0, 1] = cov_desi[1, 0] = (P * (OM - om_mean) * (HR - h_mean)).sum()
out = dict(desi_only=dict(om=om_mean, om_sd=om_sd, hrd=h_mean, hrd_sd=h_sd, chi2min=float(chi2.min()),
                          rho=cov_desi[0, 1] / np.sqrt(cov_desi[0, 0] * cov_desi[1, 1])))
fig, ax = plt.subplots(figsize=(6.6, 5.4))
lev = chi2.min() + np.array([2.30, 6.18])
ax.contourf(oms, hrds, chi2.T, levels=[chi2.min(), lev[0], lev[1]], colors=['#555555', '#BBBBBB'], alpha=0.7)
ax.contour(oms, hrds, chi2.T, levels=lev, colors='k', linewidths=0.8)
ax.plot([], [], 's', color='#888888', label='DESI DR2 BAO only ($\\Lambda$CDM, from digitised pts)')
for m in MODELS:
    s = np.load(os.path.join(OUT, f'samples_{m}.npy'))
    ax.scatter(s[:, 0], s[:, 1], s=5, color=COLORS[m], alpha=0.35)
    mu = s[:, :2].mean(0); cv = np.cov(s[:, :2].T)
    # 2D Gaussian tension between CMB+DESI model prediction and DESI-only LCDM constraint (in the LCDM-BAO plane)
    dlt = mu - np.array([om_mean, h_mean]); t2 = dlt @ np.linalg.inv(cv + cov_desi) @ dlt
    out[m] = dict(om=float(mu[0]), hrd=float(mu[1]), om_sd=float(np.sqrt(cv[0, 0])), hrd_sd=float(np.sqrt(cv[1, 1])),
                  chi2_2d=float(t2))
    ax.errorbar(*mu, xerr=np.sqrt(cv[0, 0]), yerr=np.sqrt(cv[1, 1]), fmt='o', color=COLORS[m], mec='k', ms=7,
                capsize=3, label=NICE[m] + ' CMB+DESI (mock-chain samples)')
ax.plot(om_f, hrd_f, '*', color='gold', mec='k', ms=14, label='Planck18 $\\Lambda$CDM (fiducial)')
ax.set_xlabel(r'$\Omega_\mathrm{m}$'); ax.set_ylabel(r'$h\,r_d$ [Mpc]')
ax.set_xlim(0.26, 0.41); ax.set_ylim(88, 106); ax.legend(fontsize=7.5, loc='upper right', frameon=True)
ax.set_title('BAO acoustic plane: CMB+DESI model predictions vs DESI-only', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'figure_acoustic_plane.png'), dpi=140)
json.dump(out, open(os.path.join(OUT, 'acoustic_plane.json'), 'w'), indent=1, default=float)
for k, v in out.items(): print(k, {a: round(float(b), 4) for a, b in v.items()})
