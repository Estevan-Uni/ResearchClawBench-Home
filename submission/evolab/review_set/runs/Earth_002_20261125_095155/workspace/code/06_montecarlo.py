"""Monte Carlo over SLR quantile, kappa, radius, unit size, basin frequency change, LMI change and warming.
Common random numbers across SSPs and variants."""
import numpy as np, json, time, sys
sys.argv = sys.argv[:1]
import model as M
exec(open(M.W + 'code/03_main.py').read().split("if __name__")[0])
NIT = 60
rng = np.random.default_rng(42)
SLq = SL['quantiles']
draws = []
for i in range(NIT):
    draws.append(dict(u=rng.uniform(0.17, 0.83), kappa=float(np.exp(rng.normal(0, 0.35))),
                      R=int(rng.choice([50, 100, 200], p=[.25, .5, .25])), unit=float(rng.choice([2, 5, 10], p=[.25, .5, .25])),
                      dfA={b: rng.uniform(0.0, 0.20) for b in M.BASINS}, dfB={b: rng.uniform(-0.1, 0.1) for b in M.BASINS},
                      di=rng.uniform(0.02, 0.08), tq=rng.uniform()))
def tri(lo, mode, hi, q):   # inverse CDF of triangular distribution
    c = (mode - lo) / (hi - lo)
    return lo + np.sqrt(q * (hi - lo) * (mode - lo)) if q < c else hi - np.sqrt((1 - q) * (hi - lo) * (hi - mode))
res = {f'{v}|{s}': dict(ge3=[], s5=[], ge3_pop=[], ge3_stock=[], tcshift=[], reg={}) for v in M.VARIANTS for s in M.SSPS}
m0 = S['cls'] == 1
pmap = {k: np.zeros(m0.sum()) for k in res}
REG = {'SE Asia': np.isin(cidx[m0], [i for i, c in enumerate(cw) if c['ISO3'] in ('PHL', 'VNM', 'MYS', 'IDN')]),
       'SW Pacific': (S['lat'][m0] < 0) & ((S['lon'][m0] >= 135) | (S['lon'][m0] < -150)),
       'Central America & Caribbean': (S['lon'][m0] > -100) & (S['lon'][m0] < -59) & (S['lat'][m0] > 5) & (S['lat'][m0] < 30),
       'SE Africa': np.isin(cidx[m0], [i for i, c in enumerate(cw) if c['ISO3'] in ('MOZ', 'MDG', 'TZA', 'KEN', 'ZAF', 'COM', 'MYT')])}
wt = weights(m0); wa = wt['area'][m0]
t0 = time.time()
for i, d in enumerate(draws):
    for ssp in M.SSPS:
        qc = SL[f'{ssp}_qcurve']
        j = np.interp(d['u'], SLq, np.arange(len(SLq))); j0 = int(np.floor(j)); fr = j - j0
        rate = qc[:, j0] * (1 - fr) + qc[:, min(j0 + 1, len(SLq) - 1)] * fr
        lo, best, hi = M.GSAT[ssp]
        dT = tri(lo, best, hi, d['tq']) - M.BASE_WARM
        for v in M.VARIANTS:
            df = d['dfA'] if v == 'A_MIT' else {b: M.VARIANTS[v]['df'][b] + d['dfB'][b] for b in M.BASINS}
            r = run(R=d['R'], unit=d['unit'], variant=v, ssp=ssp, kappa=d['kappa'], dT=dT, df=df, di=d['di'], slr_rate=rate)
            k = f'{v}|{ssp}'; sc = r['score']
            res[k]['ge3'].append(float(wa[sc >= 3].sum() / wa.sum())); res[k]['s5'].append(float(wa[sc == 5].sum() / wa.sum()))
            res[k]['ge3_pop'].append(float(wt['pop'][m0][sc >= 3].sum() / wt['pop'][m0].sum()))
            res[k]['ge3_stock'].append(float(wt['stock'][m0][sc >= 3].sum() / wt['stock'][m0].sum()))
            res[k]['tcshift'].append(float(wa[r['tcl'] > 0].sum() / wa.sum()))
            for rn, rm in REG.items():
                res[k]['reg'].setdefault(rn, []).append(float(wa[rm & (sc >= 3)].sum() / wa[rm].sum()))
            pmap[k] += (sc >= 3)
    print(i, round(time.time() - t0), {k: round(res[k]['ge3'][-1], 3) for k in res}, flush=True)
json.dump(dict(draws=[{k: (v if not isinstance(v, dict) else v) for k, v in d.items()} for d in draws], res=res),
          open(W + 'outputs/montecarlo.json', 'w'), default=float)
np.savez(W + 'outputs/mc_probmap.npz', **{k.replace('|', '_'): v / NIT for k, v in pmap.items()})
print('done')
