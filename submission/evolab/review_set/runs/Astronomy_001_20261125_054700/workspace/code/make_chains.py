"""Build seeded Gaussian mock MCMC chains (GetDist format) from the Table II/III means and 1-sigma errors.

Two versions are produced for every model:
  outputs/chains/<model>.txt          -- correlated Gaussian (assumed, physically motivated correlation matrix)
  outputs/chains/<model>_indep.txt    -- independent Gaussian (diagonal covariance)
In both cases the marginal mean and std are rescaled to the tabulated values exactly; physical
priors f_EDE >= 0 and tau > 0 are then imposed by rejection (affects <0.2% of samples).
"""
import numpy as np, os, csv
from common import load_data, MODELS, LABELS, OUT

N = 20000
SEED = 20251125

# --- assumed correlation coefficients (documented in the report) ---------------------------
# Motivation: CMB acoustic scale fixes ~Omega_m h^3 -> strong Om-H0 anticorrelation; the CMB
# amplitude fixes A_s e^{-2 tau} -> ln10As-tau correlation; sigma8 ∝ sqrt(A_s); in EDE, f_EDE
# drives H0, n_s, sigma8 up (well-known EDE degeneracy directions); in w0wa, w0-wa anticorrelate
# and Om grows with w0 while H0 falls.
RHO = {
 'common': {('omega_m','H0'):-0.93, ('ln10As','tau'):0.85, ('sigma8','ln10As'):0.55,
            ('sigma8','tau'):0.45, ('ns','ombh2'):0.35, ('ns','H0'):0.35, ('ombh2','H0'):0.30,
            ('omega_m','sigma8'):0.30, ('omega_m','ns'):-0.30, ('omega_m','ombh2'):-0.25},
 'lcdm': {},
 'ede':  {('omega_m','H0'):-0.45, ('f_EDE','H0'):0.88, ('f_EDE','ns'):0.80, ('H0','ns'):0.75,
          ('f_EDE','sigma8'):0.70, ('H0','sigma8'):0.60, ('sigma8','ns'):0.50,
          ('f_EDE','ombh2'):0.45, ('ombh2','H0'):0.45, ('ns','ombh2'):0.50,
          ('log10_ac','f_EDE'):0.15, ('log10_ac','H0'):0.10, ('omega_m','f_EDE'):-0.20},
 'w0wa': {('omega_m','H0'):-0.97, ('w0','wa'):-0.88, ('w0','omega_m'):0.80, ('w0','H0'):-0.80,
          ('wa','omega_m'):-0.55, ('wa','H0'):0.55, ('omega_m','sigma8'):-0.85,
          ('H0','sigma8'):0.88, ('w0','sigma8'):-0.70, ('wa','sigma8'):0.45},
}

# LCDM matrix is additionally conditioned on the CMB theta_* precision (see theta_calib.py);
# for EDE/w0wa this unconstrained conditioning scrambled the physical correlations, so it is not applied.
THETA_CALIB = ('lcdm',)

def nearest_pd(C, eps=1e-6):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    C2 = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(C2))
    return C2 / np.outer(d, d)

def corr_matrix(model, names):
    C = np.eye(len(names))
    rho = dict(RHO['common']); rho.update(RHO[model])
    for (a, b), r in rho.items():
        if a in names and b in names:
            i, j = names.index(a), names.index(b)
            C[i, j] = C[j, i] = r
    return nearest_pd(C)

def draw(model, params, correlated, rng):
    names = list(params.keys())
    mu = np.array([params[p][0] for p in names]); sig = np.array([params[p][1] for p in names])
    C = corr_matrix(model, names) if correlated else np.eye(len(names))
    if correlated and model in THETA_CALIB:
        from theta_calib import theta_grad, calibrate
        C = calibrate(C, theta_grad(model, params))
    Ntry = int(N * 1.01) + 50
    z = rng.multivariate_normal(np.zeros(len(names)), C, size=Ntry)
    z = (z - z.mean(0)) / z.std(0)                     # exact standardisation
    x = mu + z * sig
    ok = np.ones(Ntry, bool)
    for p in ('f_EDE', 'tau'):
        if p in names: ok &= x[:, names.index(p)] > 0
    x, z = x[ok][:N], z[ok][:N]
    Cinv = np.linalg.inv(C)
    mloglike = 0.5 * np.einsum('ij,jk,ik->i', z, Cinv, z)
    return names, x, mloglike, C

def write_chain(root, names, x, mll):
    arr = np.column_stack([np.ones(len(x)), mll, x])
    np.savetxt(root + '.txt', arr, fmt='%.8e')
    with open(root + '.paramnames', 'w') as f:
        for p in names: f.write(f'{p}\t{LABELS[p]}\n')

if __name__ == '__main__':
    d = load_data()
    rng = np.random.default_rng(SEED)
    rows = []
    for m, key in MODELS.items():
        params = d[key]
        for corr, suffix in [(True, ''), (False, '_indep')]:
            names, x, mll, C = draw(m, params, corr, rng)
            write_chain(os.path.join(OUT, 'chains', m + suffix), names, x, mll)
            if corr:
                np.savetxt(os.path.join(OUT, 'chains', m + '_corr_matrix.txt'), C, fmt='%.4f',
                           header=' '.join(names))
            for i, p in enumerate(names):
                mu, s = params[p]
                mean, std = x[:, i].mean(), x[:, i].std(ddof=1)
                rows.append(dict(model=m + suffix, param=p, table_mean=mu, table_sigma=s,
                                 chain_mean=mean, chain_std=std,
                                 dmean_over_sigma=(mean - mu) / s, std_ratio_minus1=std / s - 1,
                                 pass_=abs(mean - mu) / s < 0.01 and abs(std / s - 1) < 0.02))
    with open(os.path.join(OUT, 'chain_validation.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['model']:11s} {r['param']:9s} table {r['table_mean']:9.5f}±{r['table_sigma']:.5f} "
              f"chain {r['chain_mean']:9.5f}±{r['chain_std']:.5f}  dmu/s={r['dmean_over_sigma']:+.4f} "
              f"ds/s={r['std_ratio_minus1']:+.4f} {'OK' if r['pass_'] else 'FAIL'}")
    print('all pass:', all(r['pass_'] for r in rows))
