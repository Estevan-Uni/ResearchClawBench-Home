"""Optional: RPE feature-level corruption (Gaussian SNR and random dropout) -> re-select -> evaluate k=30."""
import sys, json; sys.path.insert(0, '/workspace/code')
import numpy as np, pandas as pd
from core import *
import rpe_eval as E
X0 = E.X.copy(); obs = E.obs
cfgj = json.load(open('/workspace/outputs/config_frozen.json'))
rng = np.random.default_rng(7); sd = X0.std(0)
levels = [('gauss', 0.5), ('gauss', 1.0), ('gauss', 2.0), ('dropout', 0.2), ('dropout', 0.5)]
rows = []
for kind, lv in levels:
    if kind == 'gauss': Xn = X0 + rng.normal(0, 1, X0.shape) * sd * lv
    else: Xn = np.where(rng.random(X0.shape) < lv, 0.0, X0)
    E.X = Xn
    ranks = {n: delve(Xn, cfgj['variants'][n])['ranking'] for n in ['DELVE-faithful', '+dedup', '+whiten']}
    sc = baseline_scores(Xn, 0, 30)
    for n in ['variance', 'dispersion', 'laplacian', 'mcfs', 'hotspot_moran', 'random']: ranks[n] = np.argsort(-sc[n])
    for n, r in ranks.items():
        ev = E.evaluate(r[:30]); rows.append(dict(kind=kind, level=lv, method=n, acc=ev['acc'], nmi=ev['nmi'], dpt_rho_all=ev['dpt_rho_all'], dpt_rho_cyc=ev['dpt_rho_cyc']))
    ev = E.evaluate(np.arange(241)); rows.append(dict(kind=kind, level=lv, method='all_features', acc=ev['acc'], nmi=ev['nmi'], dpt_rho_all=ev['dpt_rho_all'], dpt_rho_cyc=ev['dpt_rho_cyc']))
    pd.DataFrame(rows).to_csv('/workspace/outputs/rpe_noise.csv', index=False); print(kind, lv, flush=True)
print('done')
