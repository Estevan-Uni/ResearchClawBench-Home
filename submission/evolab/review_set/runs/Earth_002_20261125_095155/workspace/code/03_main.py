"""Central runs + deterministic sensitivities for the composite TC-regime-shift + SLR risk score."""
import numpy as np, json, time
from scipy import sparse
import model as M
W = M.W; t0 = time.time()
S, cw = M.load_sites()
cidx = np.load(W + 'outputs/site_cidx.npy')
sbasin = np.load(W + 'outputs/site_basin.npy'); tbasin = np.load(W + 'outputs/tc_basin.npy')
tlon, tlat, wind = M.load_tc()
SL = np.load(W + 'outputs/slr_sites.npz')
A = {R: sparse.load_npz(W + f'outputs/nbr_R{R}.npz') for R in [50, 100, 200]}
cname = np.array([c['Country_2020'] or c['Country'] for c in cw] + ['(none)'])
iso = np.array([c['ISO3'] for c in cw] + ['---'])
BIDX = {b: i for i, b in enumerate(M.BASINS)}

def unit_for(spec, m):
    lon, lat = S['lon'][m], S['lat'][m]
    if spec == 'country_basin':
        return cidx[m].astype(np.int64) * 10 + np.array([BIDX[b] for b in sbasin[m]])
    return M.unit_ids(lon, lat, float(spec))

def run(mask='ref1', R=100, unit=5, variant='A_MIT', ssp='ssp370', slr='q50', kappa=1.0,
        test='unit', newly=True, dT=None, df=None, di=None, slr_rate=None):
    m = (S['cls'] == 1) if mask == 'ref1' else np.ones(len(S['cls']), bool)
    if kappa == 'site': kappa = site_kappa(R)
    w0 = M.point_base_weight(tbasin, R, kappa)
    W0 = M.class_weights(wind, w0, tbasin)
    f0 = M.site_freq(A[R][m], W0)
    V = M.VARIANTS[variant]
    if dT is None: dT = M.GSAT[ssp][0] - M.BASE_WARM
    sc = dT / 2.0
    dfx = {b: (df[b] if df is not None else V['df'][b]) * sc for b in M.BASINS}
    dix = (di if di is not None else V['di']) * sc
    W1 = M.class_weights(wind, w0, tbasin, dfx, dix)
    f1 = M.site_freq(A[R][m], W1)
    uid = unit_for(unit, m)
    rs = M.regime_shift(f0, f1, uid, newly)
    tcl = rs['tclevel'] if test == 'unit' else M.site_regime_shift(f0, f1, rs['state'])
    rate = SL[f'{ssp}_{slr}'][m] if slr_rate is None else slr_rate[m]
    sl = M.slr_level(rate)
    return dict(m=m, f0=f0, f1=f1, rs=rs, tcl=tcl, sl=sl, rate=rate, score=tcl + sl, uid=uid)

def site_kappa(R):
    # per-basin multiplier = geometric mean of (model RP / observed RP) at reference sites, capped to [0.25, 4]
    V = json.load(open(W + 'outputs/validation.json'))['ref']
    from collections import defaultdict
    acc = defaultdict(list)
    REF = json.load(open(W + 'outputs/validation.json'))['ref_coords']
    for n, (lo, la) in REF.items():
        b = M.basin_of(np.array([lo]), np.array([la]))[0]
        rp = V[n][f'rp_R{R}']
        if np.isfinite(rp): acc[b].append(np.log(rp / V[n]['obs_rp']))
    km = {b: float(np.clip(np.exp(np.mean(v)), 0.25, 4.0)) for b, v in acc.items()}
    return km

def weights(m):
    return M.site_weights(cidx, cw, m)

def shares(res, wk='area'):
    w = weights(res['m'])[wk][res['m']]
    tot = w.sum()
    sc = np.array([w[res['score'] == k].sum() / tot for k in range(6)])
    tc = np.array([w[res['tcl'] == k].sum() / tot for k in range(4)])
    sl = np.array([w[res['sl'] == k].sum() / tot for k in range(3)])
    typ = dict(none=w[(res['tcl'] == 0) & (res['sl'] < 2)].sum() / tot,
               tc_only=w[(res['tcl'] > 0) & (res['sl'] < 2)].sum() / tot,
               slr_only=w[(res['tcl'] == 0) & (res['sl'] == 2)].sum() / tot,
               compound=w[(res['tcl'] > 0) & (res['sl'] == 2)].sum() / tot)
    return dict(score=sc, ge3=sc[3:].sum(), tc=tc, slr=sl, typ=typ, total=tot)

if __name__ == '__main__':
    summary = {}; site_out = {}
    # -------- central runs
    for v in M.VARIANTS:
        for ssp in M.SSPS:
            r = run(variant=v, ssp=ssp)
            key = f'{v}|{ssp}'
            summary[key] = {wk: {k: (x.tolist() if hasattr(x, 'tolist') else x) for k, x in shares(r, wk).items()} for wk in ['area', 'pop', 'stock']}
            site_out[f'tcl_{v}_{ssp}'] = r['tcl']; site_out[f'score_{v}_{ssp}'] = r['score']
            site_out[f'sl_{ssp}'] = r['sl']; site_out[f'rate_{ssp}'] = r['rate']
            site_out[f'r_{v}_{ssp}'] = r['rs']['r']; site_out[f'shift_{v}_{ssp}'] = r['rs']['shift']
            site_out[f'f1_{v}_{ssp}'] = r['f1']
            print(key, 'score shares', np.round(summary[key]['area']['score'], 3), 'ge3 %.3f' % summary[key]['area']['ge3'],
                  'TC', np.round(summary[key]['area']['tc'], 3), flush=True)
    site_out['f0'] = r['f0']; site_out['state'] = r['rs']['state']; site_out['uid'] = r['uid']
    m = r['m']; wts = weights(m)
    for k in wts: site_out['w_' + k] = wts[k][m]
    site_out['lon'] = S['lon'][m]; site_out['lat'] = S['lat'][m]; site_out['cidx'] = cidx[m]; site_out['basin'] = sbasin[m]
    np.savez(W + 'outputs/site_results_central.npz', **site_out)
    json.dump(summary, open(W + 'outputs/summary_central.json', 'w'), indent=1)
    # -------- deterministic sensitivities (ge3 area share, both variants, all SSPs)
    sens_specs = {
        'central (5deg unit, R100, q50, ref_cls=1)': {},
        'unit 2deg': dict(unit=2), 'unit 10deg': dict(unit=10), 'unit country x basin': dict(unit='country_basin'),
        'per-site test (G1-like)': dict(test='site'),
        'R = 50 km': dict(R=50), 'R = 200 km': dict(R=200),
        'kappa x 0.5': dict(kappa=0.5), 'kappa x 2': dict(kappa=2.0),
        'kappa site-calibrated per basin': dict(kappa='site'),
        'SLR 17th pct': dict(slr='q17'), 'SLR 83rd pct': dict(slr='q83'), 'SLR 2080-2100 mean': dict(slr='q50_late'),
        'all GMW sample points': dict(mask='all'),
        'exclude newly-affected rule': dict(newly=False),
    }
    sens = {}
    for name, kw in sens_specs.items():
        sens[name] = {}
        for v in M.VARIANTS:
            for ssp in M.SSPS:
                r = run(variant=v, ssp=ssp, **kw); sh = shares(r)
                sens[name][f'{v}|{ssp}'] = dict(ge3=sh['ge3'], sev=sh['score'][5], tcshift=1 - sh['tc'][0])
        print(name, {k: round(x['ge3'], 3) for k, x in sens[name].items()}, round(time.time() - t0), flush=True)
    # SLR threshold sensitivity: thresholds +-0.5 mm/yr
    for dth in [-0.5, 0.5]:
        name = f'SLR thresholds {4+dth:.1f}/{7+dth:.1f} mm/yr'; sens[name] = {}
        for v in M.VARIANTS:
            for ssp in M.SSPS:
                r = run(variant=v, ssp=ssp, slr_rate=SL[f'{ssp}_q50'] - dth); sh = shares(r)
                sens[name][f'{v}|{ssp}'] = dict(ge3=sh['ge3'], sev=sh['score'][5], tcshift=1 - sh['tc'][0])
        print(name, {k: round(x['ge3'], 3) for k, x in sens[name].items()}, flush=True)
    # frequency-only / intensity-only decompositions for variant A
    for name, kw in {'A: frequency change only': dict(di=0.0), 'A: intensity change only': dict(df={b: 0.0 for b in M.BASINS})}.items():
        sens[name] = {}
        for ssp in M.SSPS:
            r = run(variant='A_MIT', ssp=ssp, **kw); sh = shares(r)
            sens[name][f'A_MIT|{ssp}'] = dict(ge3=sh['ge3'], sev=sh['score'][5], tcshift=1 - sh['tc'][0])
        print(name, {k: round(x['ge3'], 3) for k, x in sens[name].items()}, flush=True)
    json.dump(sens, open(W + 'outputs/sensitivity.json', 'w'), indent=1)
    print('done', time.time() - t0)
