"""Propagate mock-chain samples through CAMB: theta*, r_d, h r_d, sigma8 check, and distance bands."""
import os, json, numpy as np, camb
from common import load_data, MODELS, OUT
from cosmo import background, distances, sigma8

d = load_data()
rng = np.random.default_rng(7)
NS = 250
zs = np.linspace(0.02, 2.6, 130)
summary = {}
bands = {}
for m in MODELS:
    for suf in ['', '_indep']:
        root = os.path.join(OUT, 'chains', m + suf)
        names = [l.split('\t')[0] for l in open(root + '.paramnames')]
        arr = np.loadtxt(root + '.txt')[:, 2:]
        idx = rng.choice(len(arr), NS, replace=False)
        th, hrd, rd, curves, om = [], [], [], [], []
        for i in idx:
            p = dict(zip(names, arr[i]))
            try:
                res, der = background(p, m)
            except Exception as e:
                continue
            D = distances(res, der, zs)
            om.append(p['omega_m']); th.append(der['thetastar']); rd.append(der['rdrag']); hrd.append(p['H0'] / 100 * der['rdrag'])
            curves.append(np.concatenate([D['DVrd'], D['FAP'], D['mu'] - 5 * np.log10(1)]))
        th, hrd, rd = map(np.array, (th, hrd, rd))
        summary[m + suf] = dict(n=len(th), theta_mean=th.mean(), theta_std=th.std(), hrd_mean=hrd.mean(),
                                hrd_std=hrd.std(), rd_mean=rd.mean(), rd_std=rd.std())
        print(m + suf, {k: round(float(v), 5) for k, v in summary[m + suf].items()})
        np.save(os.path.join(OUT, f'samples_{m}{suf}.npy'), np.column_stack([om, hrd, th, rd]))
        if suf == '':
            np.save(os.path.join(OUT, f'curves_{m}.npy'), np.array(curves))
np.save(os.path.join(OUT, 'curves_z.npy'), zs)

# sigma8 consistency check at the table means (full CAMB with transfer functions)
s8 = {}
for m, key in MODELS.items():
    p = {k: v[0] for k, v in d[key].items()}
    s, der = sigma8(p, m)
    s8[m] = dict(sigma8_camb=s, sigma8_table=p['sigma8'], sigma_table=d[key]['sigma8'][1],
                 pull=(s - p['sigma8']) / d[key]['sigma8'][1], rd=der['rdrag'], theta_star=der['thetastar'],
                 zdrag=der['zdrag'], zstar=der['zstar'])
    print(m, {k: round(float(v), 5) for k, v in s8[m].items()})
json.dump(dict(propagation=summary, sigma8_check=s8), open(os.path.join(OUT, 'propagation_summary.json'), 'w'),
          indent=1, default=float)
