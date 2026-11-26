"""Distance comparison (Fig. 6 reproduction): Δμ, Δ(D_V/r_d), ΔF_AP for LCDM/EDE/w0wa at the
table means, relative to a fiducial chosen by testing candidates against the digitised points."""
import os, json, numpy as np
from scipy.optimize import brentq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load_data, MODELS, NICE, COLORS, OUT, IMG
from cosmo import background, distances

d = load_data()
dv = np.array(d['desi_dvrd_points']); fap = np.array(d['desi_fap_points']); sn = np.array(d['sne_mu_points'])
means = {m: {k: v[0] for k, v in d[key].items()} for m, key in MODELS.items()}

# ---------------- candidate fiducials ----------------
planck18 = dict(H0=67.36, ombh2=0.02237, omega_m=(0.1200 + 0.02237 + 0.000644) / 0.6736**2,
                ns=0.9649, ln10As=3.044, tau=0.0544)
# DESI DR2 BAO-only LCDM best fit (Omega_m=0.2975, h r_d=101.54 Mpc; literature values, EXTERNAL)
def desi_like():
    def f(H0):
        p = dict(H0=H0, ombh2=0.02218, omega_m=0.2975)
        res, der = background(p)
        return H0 / 100 * der['rdrag'] - 101.54
    H0 = brentq(f, 60, 80)
    return dict(H0=H0, ombh2=0.02218, omega_m=0.2975)
CANDS = {'Planck18 LCDM': (planck18, 'lcdm'),
         'Paper LCDM (CMB+DESI) means': (means['lcdm'], 'lcdm'),
         'DESI-DR2-like LCDM (ext.)': (desi_like(), 'lcdm')}

zgrid = np.linspace(0.01, 2.6, 400)
zs_all = np.unique(np.concatenate([zgrid, dv[:, 0], fap[:, 0], sn[:, 0]]))

def get(p, m):
    res, der = background(p, m)
    out = distances(res, der, zs_all)
    out['theta_star'] = der['thetastar']; out['H0'] = p['H0']
    return out
MOD = {m: get(means[m], m) for m in MODELS}

def at(D, key, z):
    return np.interp(z, zs_all, D[key])

def deltas(D, F, fap_mode='frac'):
    r = dict(dvrd=D['DVrd'] / F['DVrd'] - 1,
             fap=(D['FAP'] / F['FAP'] - 1) if fap_mode == 'frac' else (D['FAP'] - F['FAP']),
             mu=5 * np.log10(D['DL'] / F['DL']))
    return r

def chi2(model_vals_fn, pts):
    z, y, e = pts.T
    return float(np.sum(((y - model_vals_fn(z)) / e)**2))

def chi2_offset(model_fn, pts):
    z, y, e = pts.T
    r = y - model_fn(z); w = 1 / e**2
    off = np.sum(w * r) / np.sum(w)
    return float(np.sum(((r - off) / e)**2)), float(off)

results = {}
for cname, (cp, cm) in CANDS.items():
    F = get(cp, cm)
    results[cname] = {'fid_params': {k: float(v) for k, v in cp.items()}, 'fid_rd': float(F['rd']),
                      'fid_hrd': float(cp['H0'] / 100 * F['rd'])}
    for fap_mode in ['frac', 'abs']:
        tot = 0; per = {}
        for m in MODELS:
            dl = deltas(MOD[m], F, fap_mode)
            c_dv = chi2(lambda z: np.interp(z, zs_all, dl['dvrd']), dv)
            c_fap = chi2(lambda z: np.interp(z, zs_all, dl['fap']), fap)
            c_mu = chi2(lambda z: np.interp(z, zs_all, dl['mu']), sn)
            c_mu_off, off = chi2_offset(lambda z: np.interp(z, zs_all, dl['mu']), sn)
            per[m] = dict(chi2_dvrd=c_dv, chi2_fap=c_fap, chi2_mu=c_mu, chi2_mu_offmarg=c_mu_off,
                          mu_offset=off)
            tot += c_dv + c_fap + c_mu
        # data relative to fiducial itself (zero line)
        z0 = dict(chi2_dvrd=chi2(lambda z: 0 * z, dv), chi2_fap=chi2(lambda z: 0 * z, fap),
                  chi2_mu=chi2(lambda z: 0 * z, sn))
        results[cname][fap_mode] = dict(per_model=per, total_3models=tot, zero_line=z0)

for c, r in results.items():
    print(f"\n== {c}: h*rd_fid={r['fid_hrd']:.2f}  rd={r['fid_rd']:.2f}")
    for fm in ['frac', 'abs']:
        print(f"  F_AP {fm}: total chi2 (3 models, all panels) = {r[fm]['total_3models']:.2f}")
        for m, v in r[fm]['per_model'].items():
            print('    ', m, {k: round(x, 2) for k, x in v.items()})

# choose fiducial: minimal total chi2 (BAO+SN, 3 models) with the better F_AP definition
best = min(((c, fm) for c in results for fm in ['frac', 'abs']),
           key=lambda t: results[t[0]][t[1]]['total_3models'])
print('\nSelected fiducial:', best)
results['selected'] = {'fiducial': best[0], 'fap_mode': best[1]}
json.dump(results, open(os.path.join(OUT, 'distance_fiducial_tests.json'), 'w'), indent=1)

# ----------------------------- figure 6 -----------------------------
FID = get(*CANDS[best[0]]); fm = best[1]
fig, axs = plt.subplots(3, 1, figsize=(7.2, 10), sharex=True, gridspec_kw=dict(hspace=0.06))
ylab = [r'$\Delta\mu$ [mag]', r'$\Delta(D_V/r_d)$', r'$\Delta F_\mathrm{AP}$' + (' (fractional)' if fm == 'frac' else ' (absolute)')]
keys = ['mu', 'dvrd', 'fap']; pts = [sn, dv, fap]
zb = np.load(os.path.join(OUT, 'curves_z.npy'))
res_f, der_f = background(*CANDS[best[0]])
Fb = distances(res_f, der_f, zb); nb = len(zb)
def band(m, k):
    c = np.load(os.path.join(OUT, f'curves_{m}.npy'))
    if k == 'dvrd': y = c[:, :nb] / Fb['DVrd'] - 1
    elif k == 'fap': y = (c[:, nb:2*nb] / Fb['FAP'] - 1) if fm == 'frac' else c[:, nb:2*nb] - Fb['FAP']
    else: y = c[:, 2*nb:] - Fb['mu']
    return np.percentile(y, [16, 84], axis=0)
for ia, (ax, k, pt, yl) in enumerate(zip(axs, keys, pts, ylab)):
    for m in MODELS:
        lo, hi = band(m, k)
        ax.fill_between(zb, lo, hi, color=COLORS[m], alpha=0.15, lw=0)
        dl = deltas(MOD[m], FID, fm)
        ax.plot(zs_all, dl[k], color=COLORS[m], lw=2.2, label=NICE[m] + ' (CMB+DESI mean)')
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    if k == 'mu':
        ax.errorbar(pt[:, 0], pt[:, 1], pt[:, 2], fmt='s', color='k', mfc='w', ms=6, capsize=3,
                    label='Union3 SNe (digitised)')
    else:
        ax.errorbar(pt[:, 0], pt[:, 1], pt[:, 2], fmt='o', color='k', ms=6, capsize=3,
                    label='DESI DR2 BAO (digitised)')
    ax.set_ylabel(yl, fontsize=12)
    if k == 'mu': ax.set_ylim(-0.25, 0.2)
    if k == 'dvrd': ax.set_ylim(-0.045, 0.06)
    if k == 'fap': ax.set_ylim(-0.08, 0.07)
    if ia == 0: ax.legend(fontsize=8.5, frameon=False, loc='upper right', ncol=2)
    else: ax.legend(handles=[ax.get_legend_handles_labels()[0][-1]], fontsize=8.5, frameon=False, loc='upper left')
axs[0].set_xlim(0, 2.6)
axs[-1].set_xlabel('Redshift $z$', fontsize=12)
axs[0].set_title(f'Distances relative to fiducial: {best[0]} (lines: table means; bands: 68% from mock chains)', fontsize=10)
fig.savefig(os.path.join(IMG, 'figure6_distance.png'), dpi=140, bbox_inches='tight')
plt.close(fig)

# ---- supplementary: offset-marginalised Δμ, DM/rd and DH/rd separately ----
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
for m in MODELS:
    dl = deltas(MOD[m], FID, fm)
    off = results[best[0]][fm]['per_model'][m]['mu_offset']
    axs[0].plot(zs_all, dl['mu'] + off, color=COLORS[m], lw=2, label=NICE[m] + f' (offset {off:+.3f} mag)')
    axs[1].plot(zs_all, MOD[m]['DMrd'] / FID['DMrd'] - 1, color=COLORS[m], lw=2, label=NICE[m])
    axs[2].plot(zs_all, MOD[m]['DHrd'] / FID['DHrd'] - 1, color=COLORS[m], lw=2, label=NICE[m])
axs[0].errorbar(sn[:, 0], sn[:, 1], sn[:, 2], fmt='s', color='k', mfc='w', capsize=3, label='Union3 (digitised)')
axs[0].set_xlim(0, 0.8); axs[0].set_title(r'$\Delta\mu$ with best-fit offset per model (marginalised $M$)')
axs[1].set_title(r'$\Delta(D_M/r_d)$'); axs[2].set_title(r'$\Delta(D_H/r_d)$')
for a in axs: a.axhline(0, color='gray', ls=':'); a.set_xlabel('$z$'); a.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'figure6b_distance_supplement.png'), dpi=130)

# low-z sanity check for DV/rd: -> (h rd)_fid/(h rd)_model - 1
hrd_fid = CANDS[best[0]][0]['H0'] / 100 * FID['rd']
san = {}
for m in MODELS:
    hrd = means[m]['H0'] / 100 * MOD[m]['rd']
    san[m] = dict(h_rd=float(hrd), rd=float(MOD[m]['rd']), theta_star=float(MOD[m]['theta_star']),
                  lowz_pred=float(hrd_fid / hrd - 1),
                  lowz_camb=float(np.interp(0.01, zs_all, MOD[m]['DVrd'] / FID['DVrd'] - 1)),
                  dvrd_z_lt1_chi2=chi2(lambda z: np.interp(z, zs_all, MOD[m]['DVrd'] / FID['DVrd'] - 1), dv[dv[:, 0] < 1]))
    print(m, {k: round(v, 5) for k, v in san[m].items()})
json.dump(san, open(os.path.join(OUT, 'model_derived_quantities.json'), 'w'), indent=1)
# store curves
np.savetxt(os.path.join(OUT, 'distance_curves.txt'),
           np.column_stack([zs_all] + [deltas(MOD[m], FID, fm)[k] for m in MODELS for k in keys]),
           header='z ' + ' '.join(f'{m}_{k}' for m in MODELS for k in keys), fmt='%.6e')
