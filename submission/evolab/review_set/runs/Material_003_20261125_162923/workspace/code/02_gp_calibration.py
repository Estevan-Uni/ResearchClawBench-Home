"""Step 2: Gaussian-process calibration of MD Tg against experiment.
Primary model: GP on offset dTg = Tg_exp - Tg_MD with a Tanimoto (MinMax) kernel on
Morgan count fingerprints of the repeat unit (+ white noise).  Evaluated by LOOCV.
Comparison rows: raw MD, linear(MD), GP(MD-only), composite GP variants, no-MD GP.
Then: real-data hold-out on epoxy/phenoxy resins and application to 8,424 vitrimers."""
import numpy as np, pandas as pd, json, pickle, sys
sys.path.insert(0, 'code')
from gp_lib import CompositeGP, minmax_kernel
from scipy.stats import norm

cal = pd.read_csv('data/tg_calibration.csv')
vit = pd.read_csv('outputs/vitrimer_repeat_units.csv')
F = np.load('outputs/features.npz')
y_exp = cal.tg_exp.values; md = cal.tg_md.values; sd = cal['std'].values
dT = y_exp - md
n = len(cal)

# fingerprints: raw counts and size-normalised fractions (size-intensive, for transfer)
fp_raw = F['cal_fp']; fp_norm = fp_raw / fp_raw.sum(1, keepdims=True)
vfp_norm = F['vit_fp'] / F['vit_fp'].sum(1, keepdims=True)
desc = F['cal_desc']; vdesc = F['vit_desc']
dmu, dsd = desc.mean(0), desc.std(0) + 1e-6
descz = (desc - dmu) / dsd; vdescz = (vdesc - dmu) / dsd
mdz = ((md - 400) / 100)[:, None]

K_raw = minmax_kernel(fp_raw, fp_raw)
K_norm = minmax_kernel(fp_norm, fp_norm)


def metrics(y, mu, var=None):
    r = y - mu
    out = dict(MAE=float(np.abs(r).mean()), RMSE=float(np.sqrt((r ** 2).mean())),
               R2=float(1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()))
    if var is not None:
        s = np.sqrt(var)
        out['cov95'] = float((np.abs(r) <= 1.96 * s).mean())
        out['NLL'] = float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * r ** 2 / var))
    return out


rows = {}; loo_preds = {}
# raw MD
rows['Raw MD'] = metrics(y_exp, md); loo_preds['Raw MD'] = (md, None)
# linear(MD) LOO via hat matrix
A = np.c_[np.ones(n), md]
H = A @ np.linalg.pinv(A)
res = y_exp - H @ y_exp
loo_r = res / (1 - np.diag(H))
mu_lin = y_exp - loo_r
s2 = (res ** 2).sum() / (n - 2)
var_lin = s2 * (1 + np.diag(H) / (1 - np.diag(H)))
rows['Linear(MD)'] = metrics(y_exp, mu_lin, var_lin); loo_preds['Linear(MD)'] = (mu_lin, var_lin)
# constant offset only
mu_c = md + (dT.sum() - dT) / (n - 1)
rows['MD + const. offset'] = metrics(y_exp, mu_c, np.full(n, dT.var() * (1 + 1 / n)))

L = np.log
configs = {
    # name: (target, blocks, X, hetero, theta0, bounds)
    'GP dTg | Tanimoto(raw FP)': ('d', [('pre', 'Kfp')], dict(Kfp=K_raw), False,
                                  [L(1500), L(400)], [(L(10), L(2e4)), (L(4), L(1e4))]),
    'GP dTg | Tanimoto(norm. FP)': ('d', [('pre', 'Kfp')], dict(Kfp=K_norm), False,
                                    [L(1500), L(400)], [(L(10), L(2e4)), (L(4), L(1e4))]),
    'GP dTg | Tanimoto(norm. FP), hetero': ('d', [('pre', 'Kfp')], dict(Kfp=K_norm), True,
                                            [L(1500), L(200), L(0.5)], [(L(10), L(2e4)), (L(4), L(1e4)), (L(0.01), L(3))]),
    'GP dTg | Tanimoto + RBF(MD)': ('d', [('pre', 'Kfp'), ('rbf', 'md')], dict(Kfp=K_norm, md=mdz), True,
                                    [L(1500), L(300), L(1.0), L(200), L(0.5)],
                                    [(L(10), L(2e4)), (L(10), L(1e4)), (L(0.2), L(5)), (L(4), L(1e4)), (L(0.01), L(3))]),
    'GP dTg | Tanimoto + RBF(MD) + RBF(desc)': ('d', [('pre', 'Kfp'), ('rbf', 'md'), ('rbf', 'desc')],
                                                dict(Kfp=K_norm, md=mdz, desc=descz), True,
                                                [L(1000), L(300), L(1.0), L(300), L(3.0), L(200), L(0.5)],
                                                [(L(10), L(2e4)), (L(10), L(1e4)), (L(0.2), L(5)), (L(10), L(1e4)),
                                                 (L(1.0), L(30)), (L(4), L(1e4)), (L(0.01), L(3))]),
    'GP(MD only) on Tg_exp': ('y', [('rbf', 'md')], dict(md=mdz), False,
                              [L(8000), L(1.0), L(1000)], [(L(100), L(1e5)), (L(0.2), L(10)), (L(4), L(1e4))]),
    'GP no-MD | Tanimoto on Tg_exp': ('y', [('pre', 'Kfp')], dict(Kfp=K_norm), False,
                                      [L(8000), L(400)], [(L(100), L(1e5)), (L(4), L(1e4))]),
}
models = {}
for name, (tgt, blocks, X, het, t0, bd) in configs.items():
    yt = dT if tgt == 'd' else y_exp
    gp = CompositeGP(blocks, hetero=het).fit(X, yt, sd, t0, bd, restarts=4)
    mu, var = gp.loo()
    if tgt == 'd':
        mu = mu + md
    rows[name] = metrics(y_exp, mu, var)
    rows[name]['hyper'] = [float(v) for v in np.exp(gp.theta)]
    rows[name]['mean'] = float(gp.m)
    loo_preds[name] = (mu, var); models[name] = gp
    print(f'{name:45s}', {k: round(v, 3) for k, v in rows[name].items() if k not in ('hyper',)})

tab = pd.DataFrame(rows).T
tab.to_csv('outputs/gp_calibration_loocv.csv')
PRIMARY = 'GP dTg | Tanimoto(norm. FP), hetero'
BEST = min([k for k in rows if k.startswith('GP dTg')], key=lambda k: rows[k]['MAE'])
print('primary', PRIMARY, 'best', BEST)

# ---------- repeated 5-fold CV (refit hyperparameters each fold) for primary and best model -----------
rng = np.random.default_rng(0)
cv5 = {}
for name in sorted({PRIMARY, BEST}):
    tgt, blocks, X, het, t0, bd = configs[name]
    errs = []
    for rep in range(3):
        perm = rng.permutation(n); folds = np.array_split(perm, 5)
        pred = np.zeros(n)
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            Xtr = {k: (v[np.ix_(tr, tr)] if k == 'Kfp' else v[tr]) for k, v in X.items()}
            gp = CompositeGP(blocks, hetero=het).fit(Xtr, dT[tr], sd[tr], t0, bd, restarts=1)
            Xte = {}
            for k, v in X.items():
                if k == 'Kfp': Xte['Kfp'] = v[np.ix_(te, tr)]
                else: Xte[k + '_te'] = v[te]
            mu, _, _ = gp.predict(Xte, sd[te])
            pred[te] = md[te] + mu
        errs.append(np.abs(pred - y_exp).mean())
    cv5[name] = (float(np.mean(errs)), float(np.std(errs)))
    print('5-fold CV (refit)', name, cv5[name])

# ---------- real-data hold-out: epoxy / phenoxy resin family -----------
ho_mask = cal.name.str.contains('Resin|Bisphenol|Phenoxy|resin', regex=True).values
tr = np.where(~ho_mask)[0]; te = np.where(ho_mask)[0]
holdout = {}
for name in sorted({PRIMARY, BEST}):
    tgt, blocks, X, het, t0, bd = configs[name]
    Xtr = {k: (v[np.ix_(tr, tr)] if k == 'Kfp' else v[tr]) for k, v in X.items()}
    gp = CompositeGP(blocks, hetero=het).fit(Xtr, dT[tr], sd[tr], t0, bd, restarts=2)
    Xte = {}
    for k, v in X.items():
        if k == 'Kfp': Xte['Kfp'] = v[np.ix_(te, tr)]
        else: Xte[k + '_te'] = v[te]
    mu, vf, nv = gp.predict(Xte, sd[te])
    holdout[name] = pd.DataFrame(dict(name=cal.name.values[te], smiles=cal.smiles.values[te], tg_exp=y_exp[te],
                                      tg_md=md[te], tg_cal=md[te] + mu, sigma=np.sqrt(vf + nv)))
    print('HOLDOUT', name); print(holdout[name].to_string())
pd.concat(holdout, names=['model']).to_csv('outputs/gp_holdout_resins.csv')

# ---------- apply to vitrimers -----------
Kv = minmax_kernel(vfp_norm, fp_norm)
maxsim_v = Kv.max(1)
Kself = K_norm.copy(); np.fill_diagonal(Kself, 0); maxsim_c = Kself.max(1)
out = vit[['acid', 'epoxide', 'repeat_unit', 'tg', 'std']].copy()
for name in sorted({PRIMARY, BEST}):
    tgt, blocks, X, het, t0, bd = configs[name]
    gp = models[name]
    Xte = {'Kfp': Kv, 'md_te': ((vit.tg.values - 400) / 100)[:, None], 'desc_te': vdescz}
    mu, vf, nv = gp.predict(Xte, vit['std'].values)
    tag = 'primary' if name == PRIMARY else 'best'
    out[f'tg_cal_{tag}'] = vit.tg.values + mu
    out[f'sigma_{tag}'] = np.sqrt(vf + nv)
    out[f'sigma_f_{tag}'] = np.sqrt(vf)
    print(name, 'vitrimer shift mean %.1f K, cal range %.1f-%.1f, mean sigma %.1f' % (
        mu.mean(), out[f'tg_cal_{tag}'].min(), out[f'tg_cal_{tag}'].max(), np.sqrt(vf + nv).mean()))
out['maxsim_cal'] = maxsim_v
out.to_csv('outputs/vitrimer_calibrated.csv', index=False)
np.save('outputs/cal_maxsim_self.npy', maxsim_c)
pickle.dump(dict(rows=rows, loo=loo_preds, cv5=cv5, primary=PRIMARY, best=BEST,
                 models={k: models[k] for k in {PRIMARY, BEST}}, configs_X_keys=None,
                 desc_norm=(dmu, dsd), fp_norm_train=fp_norm, ho_mask=ho_mask),
            open('outputs/gp_calibration.pkl', 'wb'))
print('vitrimer maxsim to cal: median %.2f, calibration LOO-maxsim median %.2f' % (np.median(maxsim_v), np.median(maxsim_c)))
