"""Polarization dependence of the n=+-1 Floquet sideband weight.

Input: outputs/floquet_fit_per_angle.csv (from analysis.py).
Fits a hierarchy of models to A1(theta) = (A+1 + A-1)/2 normalised by A0, and tests a
Floquet-Bloch + Volkov Bessel-dressing model against the simultaneously measured
second-order sideband weight A2.
"""
import json, numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.special import jv

O = 'outputs/'
F = pd.read_csv(O + 'floquet_fit_per_angle.csv')
th = np.deg2rad(F.angle.values)
R1 = 0.5 * (F['A+1'] + F['A-1']) / F.A0
R1e = 0.5 * np.sqrt(F['A+1_err']**2 + F['A-1_err']**2) / F.A0
R2 = 0.5 * (F['A+2'] + F['A-2']) / F.A0
R2e = 0.5 * np.sqrt(F['A+2_err']**2 + F['A-2_err']**2) / F.A0
y, ye = R1.values, R1e.values
N = len(y)

models = {
    'M0 constant': (lambda p, t: p[0] + 0 * t, [0.07]),
    'M1 a cos^2(theta)  [pure parallel coupling: Volkov or FB, lowest order]': (lambda p, t: p[0] * np.cos(t)**2, [0.08]),
    'M2 a + b cos^2(theta)  [p/s-asymmetric quadratic form]': (lambda p, t: p[0] + p[1] * np.cos(t)**2, [0.07, 0.01]),
    'M3 a + b cos2theta + c sin2theta  [most general lowest-order (linear-response) form]':
        (lambda p, t: p[0] + p[1] * np.cos(2 * t) + p[2] * np.sin(2 * t), [0.07, 0, 0]),
    'M4 a + b cos^2(2theta)  [four-fold harmonic]': (lambda p, t: p[0] + p[1] * np.cos(2 * t)**2, [0.05, 0.03]),
    'M5 a + b cos2theta + c cos4theta': (lambda p, t: p[0] + p[1] * np.cos(2 * t) + p[2] * np.cos(4 * t), [0.07, 0, 0.01]),
}
res = {}
for name, (fn, p0) in models.items():
    r = least_squares(lambda p: (fn(p, th) - y) / ye, p0)
    J = r.jac; cov = np.linalg.pinv(J.T @ J)
    chi2 = float(np.sum(r.fun**2)); k_ = len(p0)
    res[name] = dict(params=r.x.tolist(), errs=np.sqrt(np.diag(cov)).tolist(), chi2=chi2, npar=k_,
                     dof=N - k_, AIC=chi2 + 2 * k_, BIC=chi2 + k_ * np.log(N))

# ---------------- Floquet-Bloch + Volkov Bessel model
# FB dressing of a Dirac state with k || x (cut through K):   alpha_F = aF * cos(theta)   (A.k_hat)
# Volkov dressing of the photoelectron, alpha_V = aV*sqrt(cos^2 theta * cp + sin^2 theta * (1-cp)) -> we
# allow the general form alpha_V^2 = aV^2 (cos^2 + r sin^2) (r captures the p/s geometry / Fresnel factors).
# Sideband amplitudes M_n = sum_m J_{n-m}(alpha_V) J_m(alpha_F) e^{i m phi}; phi = relative phase.
def bessel_ratios(p, t, nmax=6):
    aF, aV, r, phi = p
    alF = aF * np.cos(t)
    alV = aV * np.sqrt(np.cos(t)**2 + r * np.sin(t)**2)
    M = {}
    for n in (0, 1, 2):
        M[n] = sum(jv(n - m, alV) * jv(m, alF) * np.exp(1j * m * phi) for m in range(-nmax, nmax + 1))
    I0, I1, I2 = (np.abs(M[n])**2 for n in (0, 1, 2))
    return I1 / I0, I2 / I0
def bessel_resid(p, use_R2=True):
    r1, r2 = bessel_ratios(p, th)
    out = [(r1 - y) / ye]
    if use_R2:
        out.append((r2 - R2.values) / R2e.values)
    return np.concatenate(out)
best = None
for aF0 in [0.2, 0.5, 0.8, 1.2]:
    for aV0 in [0.2, 0.5, 0.8, 1.2]:
        for phi0 in [0, np.pi / 2, np.pi]:
            for use in (False,):
                rr = least_squares(lambda p: bessel_resid(p, use), [aF0, aV0, 1.0, phi0],
                                   bounds=([0, 0, 0, -np.pi], [3, 3, 5, 2 * np.pi]))
                if best is None or rr.cost < best.cost:
                    best = rr
pB = best.x
r1B, r2B = bessel_ratios(pB, th)
res['B1 FB+Volkov Bessel model, fit to A1 only'] = dict(
    params=pB.tolist(), names=['alpha_F', 'alpha_V', 'r_s/p', 'phi'], chi2=float(np.sum(((r1B - y) / ye)**2)),
    npar=4, predicted_R2=r2B.tolist(), observed_R2=R2.tolist(), observed_R2_err=R2e.tolist(),
    chi2_R2_prediction=float(np.sum(((r2B - R2.values) / R2e.values)**2)))
best2 = None
for aF0 in [0.1, 0.3, 0.6, 1.0]:
    for aV0 in [0.1, 0.3, 0.6, 1.0]:
        for phi0 in [0, np.pi / 2, np.pi]:
            rr = least_squares(lambda p: bessel_resid(p, True), [aF0, aV0, 1.0, phi0],
                               bounds=([0, 0, 0, -np.pi], [3, 3, 5, 2 * np.pi]))
            if best2 is None or rr.cost < best2.cost:
                best2 = rr
pB2 = best2.x
r1B2, r2B2 = bessel_ratios(pB2, th)
res['B2 FB+Volkov Bessel model, joint fit to A1 and A2'] = dict(
    params=pB2.tolist(), names=['alpha_F', 'alpha_V', 'r_s/p', 'phi'],
    chi2_total=float(2 * best2.cost), chi2_A1=float(np.sum(((r1B2 - y) / ye)**2)),
    chi2_A2=float(np.sum(((r2B2 - R2.values) / R2e.values)**2)), npar=4, n_data=2 * N)

# Single-parameter dressing consistency check (independent of polarization):
# I1/I0 = (J1/J0)^2 -> alpha ; then predicted I2/I0 = (J2/J0)^2
from scipy.optimize import brentq
chk = []
for t_, r1, r2, r2e in zip(F.angle, y, R2, R2e):
    a = brentq(lambda a: (jv(1, a) / jv(0, a))**2 - r1, 1e-4, 2.0)
    pr = (jv(2, a) / jv(0, a))**2
    chk.append(dict(angle=int(t_), R1=r1, alpha_eff=a, R2_pred=pr, R2_obs=r2, R2_err=r2e,
                    z_score=(r2 - pr) / r2e, main_band_weight_pred=jv(0, a)**2 / (jv(0, a)**2 + 2 * jv(1, a)**2)))
pd.DataFrame(chk).to_csv(O + 'single_alpha_consistency.csv', index=False)

# Provided CSV re-fit (for comparison)
C = pd.read_csv('data/polarization_dependence_data.csv')
thc = np.deg2rad(C.angle_degrees.values)
csvfit = {}
for name in ['M0 constant', 'M3 a + b cos2theta + c sin2theta  [most general lowest-order (linear-response) form]',
             'M4 a + b cos^2(2theta)  [four-fold harmonic]']:
    fn, p0 = models[name]
    p0 = [0.5] + list(p0[1:])
    rr = least_squares(lambda p: fn(p, thc) - C.intensity.values, p0)
    csvfit[name] = dict(params=rr.x.tolist(), rss=float(np.sum(rr.fun**2)))
pol = dict(theta_deg=F.angle.tolist(), R1=y.tolist(), R1_err=ye.tolist(), R2=R2.tolist(), R2_err=R2e.tolist(),
           contrast_R1=(y.max() - y.min()) / (y.max() + y.min()),
           csv_contrast=float((C.intensity.max() - C.intensity.min()) / (C.intensity.max() + C.intensity.min())),
           models=res, csv_refit=csvfit)
json.dump(pol, open(O + 'polarization_results.json', 'w'), indent=2)
for kname, v in res.items():
    print(kname, json.dumps(v, default=float)[:600])
print(pd.DataFrame(chk).round(5).to_string())
print('contrast', pol['contrast_R1'], 'csv contrast', pol['csv_contrast'])
