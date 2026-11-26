"""Main analysis pipeline: Floquet-Bloch replicas in graphene tr-ARPES data.

Steps
 1. Pump-off (equilibrium) Dirac cone: MDC fits + global 2D Lorentzian model.
 2. Pump-on spectra (7 polarization angles): global 2D Floquet-sideband model
    I_on = sum_n A_n * Cone(E - n*hw; v, g_n) + c, fitted with least squares,
    uncertainties from the Jacobian with the empirically measured noise level.
 3. Model-independent checks: EDC at k=0, replica dispersion via MDC fits,
    photon-energy scan, replica-velocity vs main-band velocity.
 4. Avoided-crossing (Floquet gap) test at the n=0 / n=+-1 crossings.
 5. Validation of the provided processed products (JSON / CSV).
All numbers are written to outputs/*.json / *.csv.
"""
import json, h5py, numpy as np, pandas as pd
from scipy.optimize import least_squares, curve_fit

D = 'data/'; O = 'outputs/'
f = h5py.File(D + 'raw_trARPES_data.h5', 'r')
E = f['energy_axis'][:]; k = f['kx_axis'][:]
angles = list(f['polarization_angles'][:])
off = f['pump_off_spectrum'][:]
on = {int(a): f[f'pump_on_angle_{int(a)}'][:] for a in angles}
HW_NOM = float(f.attrs['pump_energy_eV'])
KK, EE = np.meshgrid(k, E)
dE, dk = E[1] - E[0], k[1] - k[0]
out = {}

# ---------------------------------------------------------------- noise level
# 0, 90 and 180 deg maps are nominally equivalent -> differences give noise
pair = on[0] - on[180]
sigma = float(pair.std() / np.sqrt(2))
pair2 = on[30] - on[150]
out['noise'] = dict(sigma_from_0_180=sigma, sigma_from_30_150=float(pair2.std() / np.sqrt(2)),
                    )

# ---------------------------------------------------------------- 1. pump-off
def two_lor(kk, A, k0, g, c):
    return A * (g**2 / ((kk - k0)**2 + g**2) + g**2 / ((kk + k0)**2 + g**2)) + c

mdc = []
for i, e in enumerate(E):
    y = off[i]
    p0 = [y.max() / 2, max(abs(e) / 6, 0.003), 0.015, 0]
    try:
        p, C = curve_fit(two_lor, k, y, p0=p0, maxfev=20000)
        mdc.append(dict(E=float(e), A=p[0], k0=abs(p[1]), g=abs(p[2]), c=p[3]))
    except Exception:
        pass
mdc = pd.DataFrame(mdc); mdc.to_csv(O + 'pumpoff_mdc_fits.csv', index=False)
sel = mdc.E.abs() > 0.06            # avoid the region where the two branches merge
up, lo = mdc[sel & (mdc.E > 0)], mdc[sel & (mdc.E < 0)]
pu = np.polyfit(up.k0, up.E, 1); pl = np.polyfit(lo.k0, lo.E, 1)
out['pumpoff_mdc'] = dict(hv_upper=pu[0], ED_upper=pu[1], hv_lower=-pl[0], ED_lower=pl[1],
                          hwhm_k_mean=float(mdc[sel].g.mean()), hwhm_k_std=float(mdc[sel].g.std()))

def cone(ED, v, g, A, kD=0.0):
    k0 = np.abs(EE - ED) / v
    return A * (g**2 / ((KK - kD - k0)**2 + g**2) + g**2 / ((KK - kD + k0)**2 + g**2))

r = least_squares(lambda p: (cone(p[0], p[1], p[2], p[3], p[4]) + p[5] - off).ravel(),
                  [0.0, 6.0, 0.02, 90, 0.0, 0.0])
out['pumpoff_2D'] = dict(zip(['ED', 'hv_F', 'g_k', 'A', 'kD', 'c'], map(float, r.x)),
                         rms=float(np.std(r.fun)), max_abs_resid=float(np.abs(r.fun).max()))
ED0, V0, G0, A00 = r.x[:4]

# ---------------------------------------------------------------- 2. pump-on global fits
NAMES = ['A0', 'g0', 'A+1', 'A-1', 'g1', 'A+2', 'A-2', 'hw', 'v', 'c']
def floquet_model(p):
    A0, g0, Ap1, Am1, g1, Ap2, Am2, hw, v, c = p
    m = cone(ED0, v, g0, A0)
    m += cone(ED0 + hw, v, g1, Ap1) + cone(ED0 - hw, v, g1, Am1)
    m += cone(ED0 + 2 * hw, v, g1 * np.sqrt(1.5), Ap2) + cone(ED0 - 2 * hw, v, g1 * np.sqrt(1.5), Am2)
    return m + c

def fit_map(img, p0=None):
    p0 = p0 or [100, 1 / 60, 8, 8, 0.024, 0.5, 0.5, 0.248, 6.0, 0]
    res = least_squares(lambda p: (floquet_model(p) - img).ravel() / sigma, p0)
    J = res.jac; cov = np.linalg.pinv(J.T @ J)
    chi2r = float(np.sum(res.fun**2) / (res.fun.size - len(p0)))
    return res.x, np.sqrt(np.diag(cov)), chi2r, (img - floquet_model(res.x))

fits = []; resid = {}
for a in sorted(on):
    p, e, chi2r, rr = fit_map(on[a])
    d = dict(angle=a, chi2_red=chi2r, rms=float(rr.std()))
    for n, pv, ev in zip(NAMES, p, e):
        d[n] = float(pv); d[n + '_err'] = float(ev)
    fits.append(d); resid[a] = rr
fits = pd.DataFrame(fits); fits.to_csv(O + 'floquet_fit_per_angle.csv', index=False)
np.save(O + 'resid_angle0.npy', resid[0])

# Alternative null models for angle 0 (model comparison)
def chi2_of(model_fun, p0):
    res = least_squares(lambda p: (model_fun(p) - on[0]).ravel() / sigma, p0)
    return float(np.sum(res.fun**2)), len(p0), res.x
N = on[0].size
cmp_ = {}
cmp_['no_replica (main cone only, free width/amp + const)'] = chi2_of(
    lambda p: cone(ED0, p[1], p[0], p[2]) + p[3], [1 / 60, 6, 100, 0])
cmp_['main + broad k-Gaussian background (heating-like)'] = chi2_of(
    lambda p: cone(ED0, p[1], p[0], p[2]) + p[3] * np.exp(-KK**2 / (2 * p[4]**2)) + p[5], [1 / 60, 6, 100, 5, 0.1, 0])
cmp_['main + +-1 replicas (6 par)'] = chi2_of(
    lambda p: cone(ED0, p[5], p[0], p[1]) + cone(ED0 + p[3], p[5], p[2], p[4]) + cone(ED0 - p[3], p[5], p[2], p[4]) + p[6],
    [1 / 60, 100, 0.024, 0.248, 8, 6, 0])
cmp_['full (+-1, +-2 orders, 10 par)'] = chi2_of(floquet_model, [100, 1 / 60, 8, 8, 0.024, 0.5, 0.5, 0.248, 6.0, 0])
out['model_comparison_angle0'] = {kname: dict(chi2=c2, npar=npar, chi2_red=c2 / (N - npar),
                                              BIC=c2 + npar * np.log(N)) for kname, (c2, npar, _) in cmp_.items()}

# photon-energy profile scan (angle 0): fix hw, refit the rest
scan = []
for hw in np.linspace(0.20, 0.30, 41):
    res = least_squares(lambda p: (floquet_model(np.r_[p[:7], hw, p[7:]]) - on[0]).ravel() / sigma,
                        [100, 1 / 60, 8, 8, 0.024, 0.5, 0.5, 6.0, 0])
    scan.append(dict(hw=hw, chi2=float(np.sum(res.fun**2))))
pd.DataFrame(scan).to_csv(O + 'hw_scan.csv', index=False)

# ---------------------------------------------------------------- 3. model-independent checks
dmap = on[0] - off
j0 = np.argmin(abs(k))
edc = dmap[:, j0 - 1:j0 + 2].mean(1)
np.save(O + 'edc_k0_diff.npy', np.c_[E, edc, off[:, j0 - 1:j0 + 2].mean(1), on[0][:, j0 - 1:j0 + 2].mean(1)])

# Replica dispersion: fit MDCs of the difference map near the replica Dirac points
# with 4 Lorentzians (2 replica branches + 2 main-band branches whose positions are fixed by pump-off fit)
rep = []
for i, e in enumerate(E):
    for n in (+1, -1):
        Er = ED0 + n * HW_NOM
        if not (0.035 < abs(e - Er) < 0.20):
            continue
        y = dmap[i]   # main band cancels in the difference map (A0 unchanged); both replicas present
        kn = abs(e - Er) / 6.0; ko = abs(e - (ED0 - n * HW_NOM)) / 6.0
        def mfun(kk, A, kr, g, B, ko_, c):
            return two_lor(kk, A, kr, g, 0) + two_lor(kk, B, ko_, g, 0) + c
        try:
            p, C = curve_fit(mfun, k, y, p0=[8, kn, 0.024, 8, ko, 0], maxfev=20000,
                             sigma=np.full_like(y, sigma), absolute_sigma=True)
            rep.append(dict(order=n, E=float(e), k_rep=abs(p[1]), k_rep_err=float(np.sqrt(C[1, 1])), g=abs(p[2])))
        except Exception:
            pass
rep = pd.DataFrame(rep); rep.to_csv(O + 'replica_mdc_fits.csv', index=False)
disp = {}
for n in (+1, -1):
    for br, s in (('upper', 1), ('lower', -1)):
        sub = rep[(rep.order == n) & (np.sign(rep.E - (ED0 + n * HW_NOM)) == s)]
        pp, cc = np.polyfit(sub.k_rep, sub.E, 1, cov=True)
        disp[f'n{n:+d}_{br}'] = dict(slope=float(pp[0]), slope_err=float(np.sqrt(cc[0, 0])),
                                     intercept=float(pp[1]), intercept_err=float(np.sqrt(cc[1, 1])))
out['replica_dispersion'] = disp

# global fit with replica velocity and replica energies free (tests 'copy of the Dirac cone' + hw spacing)
def free_rep(p):
    A0, g0, v0, Ap, Am, g1, v1, Ep, Em, c = p
    return cone(ED0, v0, g0, A0) + cone(Ep, v1, g1, Ap) + cone(Em, v1, g1, Am) + c
rr_ = least_squares(lambda p: (free_rep(p) - on[0]).ravel() / sigma, [100, 1/60, 6, 8, 8, 0.024, 6, 0.25, -0.25, 0])
Jf = rr_.jac; ef = np.sqrt(np.diag(np.linalg.pinv(Jf.T @ Jf)))
out['free_replica_fit_angle0'] = {n: dict(val=float(v), err=float(e_)) for n, v, e_ in
    zip(['A0', 'g0', 'v0', 'A+1', 'A-1', 'g1', 'v1', 'E+1', 'E-1', 'c'], rr_.x, ef)}
out['free_replica_fit_angle0']['g1/g0'] = float(rr_.x[5] / rr_.x[1])

# ---------------------------------------------------------------- 4. Floquet gap test at crossings
# n=0 upper branch (E = v|k|) crosses n=+1 lower branch (E = hw - v|k|) at E=hw/2, k=+-hw/(2v)
# Model: replace the two bare Lorentzian lines by the hybridised two-level spectral function
# (Lorentzian in energy with width Gamma_E = v*g, weights from eigenvectors).
P = fits.set_index('angle').loc[0]
def gap_model(Delta, Ap=None):
    v, hw = P['v'], P['hw']
    m = floquet_model(P[NAMES].values)
    # analytic hybridisation at all four crossings (n=0 with n=+-1, +-k)
    for n in (+1, -1):
        for sgn in (+1, -1):
            e0 = ED0 + n * v * np.abs(KK)            # n=0 branch heading towards replica n
            e1 = ED0 + n * (hw - v * np.abs(KK))     # replica branch heading back
            mask = (np.sign(KK) == sgn)
            # bare contributions (Lorentzian in energy equivalent to Lorentzian in k)
            G0E, G1E = v * P['g0'], v * P['g1']
            A0w, A1w = P['A0'], P['A+1'] if n > 0 else P['A-1']
            Lr = lambda x, g: g**2 / (x**2 + g**2)
            bare = A0w * Lr(EE - e0, G0E) + A1w * Lr(EE - e1, G1E)
            em, dd = (e0 + e1) / 2, (e0 - e1) / 2
            R = np.sqrt(dd**2 + Delta**2)
            cth2 = 0.5 * (1 + dd / np.where(R > 0, R, 1))   # weight of state 0 in upper eigenstate
            Eu, El = em + R, em - R
            # eigenstate widths/weights interpolate between the bare states so Delta=0 recovers the bare model
            Gu, Gl = cth2 * G0E + (1 - cth2) * G1E, (1 - cth2) * G0E + cth2 * G1E
            hyb = (A0w * cth2 + A1w * (1 - cth2)) * Lr(EE - Eu, Gu) + (A0w * (1 - cth2) + A1w * cth2) * Lr(EE - El, Gl)
            loc = mask & (np.abs(np.abs(KK) - hw / (2 * v)) < 0.03) & (np.abs(EE - n * hw / 2) < 0.08)
            m = np.where(loc, m - bare + hyb, m)
    return m
win = (np.abs(np.abs(KK) - P['hw'] / (2 * P['v'])) < 0.03) & (np.abs(np.abs(EE) - P['hw'] / 2) < 0.08)
gscan = []
for Delta in np.r_[np.linspace(0, 0.01, 51), np.linspace(0.012, 0.06, 25)]:
    c2 = float(np.sum(((gap_model(Delta) - on[0])[win] / sigma)**2))
    gscan.append(dict(Delta=Delta, chi2=c2))
gscan = pd.DataFrame(gscan); gscan.to_csv(O + 'gap_scan.csv', index=False)
c2min = gscan.chi2.min()
ub = float(gscan[gscan.chi2 - c2min <= 3.84].Delta.max())
out['gap_test'] = dict(best_Delta=float(gscan.loc[gscan.chi2.idxmin(), 'Delta']), Delta_95CL_upper=ub,
                       n_pixels=int(win.sum()))
# simple EDC check through the crossing point
jc = np.argmin(abs(k - P['hw'] / (2 * P['v'])))
out['gap_test']['k_cross'] = float(k[jc])

# ---------------------------------------------------------------- 5. validate provided products
js = json.load(open(D + 'processed_band_data.json'))
csv = pd.read_csv(D + 'polarization_dependence_data.csv')
val = dict(json_dirac_point=js['dirac_point'], refit_dirac_point_E_k=[float(ED0), float(r.x[4])])
chk = []
for rb in js['replica_bands']:
    e, kk = rb['energy'], rb['kx']
    chk.append(dict(order=rb['order'], E=e, k=kk,
                    dist_to_main_cone_k=float(abs(abs(kk) - abs(e - ED0) / V0)),
                    dist_to_replica_k=float(abs(abs(kk) - abs(e - ED0 - rb['order'] * HW_NOM) / V0))))
val['json_replicas'] = chk
te, tk = csv.target_energy[0], csv.target_kx[0]
val['csv_target'] = dict(E=te, k=tk, main_cone_k_at_E=float(abs(te - ED0) / V0),
                         replica_n1_k_at_E=float(abs(te - ED0 - HW_NOM) / V0))
ii, jj = np.argmin(abs(E - te)), np.argmin(abs(k - tk))
val['csv_vs_raw'] = [dict(angle=int(a), csv=float(csv.intensity[csv.angle_degrees == a].iloc[0]),
                          raw_on=float(on[a][ii, jj]), raw_off=float(off[ii, jj]),
                          on_over_on_plus_off=float(on[a][ii, jj] / (on[a][ii, jj] + off[ii, jj])))
                     for a in sorted(on)]
out['validation'] = val

# model-independent replica weight: integrate difference map in a box around replica Dirac point,
# excluding main-band pixels (|k| within 2.5 g0 of main band)
box = []
for a in sorted(on):
    dm = on[a] - off
    row = dict(angle=a)
    for n in (+1, -1):
        m = (np.abs(EE - (ED0 + n * HW_NOM)) < 0.06) & (np.abs(KK) < 0.03)
        row[f'box_n{n:+d}'] = float(dm[m].sum() * dE * dk)
        row[f'box_n{n:+d}_err'] = float(sigma * np.sqrt(m.sum()) * dE * dk)
    box.append(row)
pd.DataFrame(box).to_csv(O + 'replica_box_integrals.csv', index=False)

json.dump(out, open(O + 'analysis_results.json', 'w'), indent=2, default=float)
print(json.dumps(out, indent=1, default=float))
print(fits[['angle', 'A0', 'A+1', 'A+1_err', 'A-1', 'A+2', 'A-2', 'A+2_err', 'g1', 'hw', 'hw_err', 'v', 'chi2_red']].to_string())
print(pd.DataFrame(box).to_string())
