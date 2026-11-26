"""Batch runner: generates LP jobs, solves them on 2 cores, appends results to CSV incrementally.
usage: python run_batch.py main|second|hourly
"""
import sys, os, json, time, csv
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
import params as P, model as M

OUT = '/workspace/outputs'
HEX = pd.read_csv(f'{OUT}/hex_prepared.csv', keep_default_na=False, na_values=[''])
NE2C = {'NA': 'NA', 'BW': 'BW', 'AO': 'AO', 'ZA': 'ZA', 'SEA': 'NA'}
_cache = {}

def profiles(kind, key):
    if (kind, key) in _cache: return _cache[(kind, key)]
    if kind == 'hex':
        r = HEX.set_index('hex_id').loc[key]
        pv = M.pv_profile(r.lat, r.cf_pv, 'africa'); wd = M.wind_profile(r.cf_wind, 'africa')
    else:
        s = P.EU_SITES[key]
        pv = M.pv_profile(s['lat'], s['pv_cf'], 'eu'); wd = M.wind_profile(s['wind_cf'], 'eu')
    _cache[(kind, key)] = (pv, wd)
    return pv, wd

def run(job):
    t0 = time.time()
    pv, wd = profiles(job['kind'], job['site'])
    pv = pv * job.get('pv_mult', 1.0); wd = wd * job.get('wind_mult', 1.0)
    country = job['country']
    if 'country_over' in job:
        P.COUNTRY['X'] = job['country_over']; country = 'X'
    if job['scen'] == 'Zero':
        w = M.wacc_set(country, 'Commercial', 0, flat=0.0)
    elif 'flat' in job:
        w = M.wacc_set(country, 'Commercial', 0, flat=job['flat'])
    else:
        w = M.wacc_set(country, job['scen'], job['ir'])
    try:
        o = M.solve_plant(pv, wd, w, mode=job['mode'], tech=job.get('tech', '2030'), dt=job.get('dt', P.DT),
                          cavern=job.get('cavern', False), grid_km=job.get('grid_km'), w_infra=w['infra'],
                          tech_over=job.get('tech_over'), battery=job.get('battery', False))
        o['status'] = 'ok'
    except Exception as ex:
        o = dict(status=f'err:{ex}')
    rec = {k: v for k, v in job.items() if k not in ('tech_over', 'country_over')}
    rec['tech_over'] = json.dumps(job.get('tech_over', {})); rec['country_over'] = json.dumps(job.get('country_over', {}))
    for a in P.ASSET: rec[f'w_{a}'] = w[a]
    rec.update({k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in o.items()})
    rec['secs'] = time.time() - t0
    return rec

def base(kind, site, country, mode, scen, rname, ir, group, **kw):
    d = dict(group=group, kind=kind, site=site, country=country, mode=mode, scen=scen, rate=rname, ir=ir)
    d.update(kw); return d

def jobs_main():
    J = []
    order = list(HEX.sort_values('onshore', ascending=False).hex_id)
    for mode in ['flex', 'const']:
        for h in order:
            for rn, ir in P.BASE_RATES.items():
                for s in P.FIN_SCEN:
                    J.append(base('hex', h, 'NA', mode, s, rn, ir, 'af'))
            J.append(base('hex', h, 'NA', mode, 'Zero', 'Zero', 0.0, 'af'))
    for site, s in P.EU_SITES.items():
        for mode in ['flex', 'const']:
            rates = dict(P.BASE_RATES); rates.update({'ir 0%': 0.0, 'ir 1%': 0.01, 'ir 3%': 0.03, 'ir 5%': 0.05, 'ir 6%': 0.06})
            for rn, ir in rates.items():
                for sc in ['Commercial', 'Offtake']:
                    J.append(base('eu', site, 'EU', mode, sc, rn, ir, 'eu', cavern=s['cavern']))
            J.append(base('eu', site, 'EU', mode, 'Zero', 'Zero', 0.0, 'eu', cavern=s['cavern']))
            J.append(base('eu', site, 'EU', mode, 'Commercial', 'Moderate 2.15%', 0.0215, 'eu_tank', cavern=False))
    # flex first (fast), then const sorted so EU const (slow) are spread
    J.sort(key=lambda j: (j['mode'] != 'flex', j['group'] == 'eu'))
    return J

def jobs_second(rep, best):
    J = []
    mid = ('Moderate 2.15%', 0.0215)
    for h in HEX.hex_id:
        for s in ['Commercial', 'Full de-risking']:
            r = HEX.set_index('hex_id').loc[h]
            J.append(base('hex', h, 'NA', 'const', s, *mid, 'grid', grid_km=float(r.grid_dist_km)))
        J.append(base('hex', h, 'NA', 'const', 'flat6', 'flat 6%', 0.0, 'geoh2', tech='geoh2', flat=0.06))
        J.append(base('hex', h, 'NA', 'flex', 'flat6', 'flat 6%', 0.0, 'geoh2', tech='geoh2', flat=0.06))
        c = NE2C[HEX.set_index('hex_id').loc[h].ne_iso]
        if c != 'NA':
            for mode in ['flex', 'const']:
                J.append(base('hex', h, c, mode, 'Commercial', *mid, 'country'))
    # sensitivities on representative hex (const + flex)
    t = P.TECH['2030']
    sens = {
        'PV capex -20%': dict(tech_over=dict(pv_capex=t['pv_capex'] * 0.8)), 'PV capex +20%': dict(tech_over=dict(pv_capex=t['pv_capex'] * 1.2)),
        'Wind capex -20%': dict(tech_over=dict(wind_capex=t['wind_capex'] * 0.8)), 'Wind capex +20%': dict(tech_over=dict(wind_capex=t['wind_capex'] * 1.2)),
        'Electrolyser capex -30%': dict(tech_over=dict(el_capex=t['el_capex'] * 0.7)), 'Electrolyser capex +30%': dict(tech_over=dict(el_capex=t['el_capex'] * 1.3)),
        'H2 storage capex -30%': dict(tech_over=dict(h2s_capex=t['h2s_capex'] * 0.7)), 'H2 storage capex +30%': dict(tech_over=dict(h2s_capex=t['h2s_capex'] * 1.3)),
        'Efficiency 0.60': dict(tech_over=dict(el_eff=0.60)), 'Efficiency 0.70': dict(tech_over=dict(el_eff=0.70)),
        'PV CF -10%': dict(pv_mult=0.9), 'PV CF +10%': dict(pv_mult=1.1),
        'Wind CF -15%': dict(wind_mult=0.85), 'Wind CF +15%': dict(wind_mult=1.15),
        'Battery allowed': dict(battery=True), 'Battery allowed, -50% cost': dict(battery=True, tech_over=dict(bat_e_capex=75e3, bat_p_capex=30e3)),
    }
    for h in [rep]:
        for name, kw in sens.items():
            for mode in ['flex', 'const']:
                if mode == 'flex' and ('Battery' in name or 'storage' in name): continue
                J.append(base('hex', h, 'NA', mode, 'Commercial', *mid, 'sens', sens=name, **kw))
    # CRP sweep (country spread on debt = 0.7 x CRP), representative + best hex
    for h in sorted({rep, best}):
        for crp in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]:
            for mode in ['flex', 'const']:
                J.append(base('hex', h, 'X', mode, 'Commercial', *mid, 'crp', crp=crp, country_over=dict(spread=0.7 * crp, crp=crp)))
    # interest-rate sweep for representative hex
    for h in [rep]:
        for rn, ir in {'ir 0%': 0.0, 'ir 1%': 0.01, 'ir 3%': 0.03, 'ir 5%': 0.05, 'ir 6%': 0.06}.items():
            for s in ['Commercial', 'Offtake', 'Full de-risking']:
                for mode in ['flex', 'const']:
                    J.append(base('hex', h, 'NA', mode, s, rn, ir, 'ratesweep'))
    J.sort(key=lambda j: j['mode'] != 'flex')
    return J

def jobs_eu2():
    J = []
    for site, s in P.EU_SITES.items():
        for mode in ['flex', 'const']:
            rates = dict(P.BASE_RATES); rates.update({'ir 0%': 0.0, 'ir 1%': 0.01, 'ir 3%': 0.03, 'ir 5%': 0.05, 'ir 6%': 0.06})
            for rn, ir in rates.items():
                for sc in ['Commercial', 'Offtake']:
                    J.append(base('eu', site, 'EU', mode, sc, rn, ir, 'eu', cavern=s['cavern'], tech_over=s['capex']))
            J.append(base('eu', site, 'EU', mode, 'Zero', 'Zero', 0.0, 'eu', cavern=s['cavern'], tech_over=s['capex']))
            J.append(base('eu', site, 'EU', mode, 'Commercial', 'Moderate 2.15%', 0.0215, 'eu_tank', cavern=False, tech_over=s['capex']))
    J.sort(key=lambda j: j['mode'] == 'flex')
    return J

def jobs_hourly(rep):
    J = []
    for dt in [1, 3]:
        J.append(base('hex', rep, 'NA', 'const', 'Commercial', 'Moderate 2.15%', 0.0215, 'tres', dt=dt))
        J.append(base('hex', rep, 'NA', 'flex', 'Commercial', 'Moderate 2.15%', 0.0215, 'tres', dt=dt))
        for m in ['const', 'flex']:
            J.append(base('eu', 'Germany (north)', 'EU', m, 'Commercial', 'Moderate 2.15%', 0.0215, 'tres', dt=dt, cavern=True, tech_over=P.EU_SITES['Germany (north)']['capex']))
    return J

if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'main':
        J = jobs_main()
    elif which == 'eu2':
        J = jobs_eu2()
    else:
        rep, best = sys.argv[2], sys.argv[3]
        J = jobs_second(rep, best) if which == 'second' else jobs_hourly(rep)
    fn = f'{OUT}/lp_{which}.csv'
    done = set()
    if os.path.exists(fn):
        old = pd.read_csv(fn)
        done = set(old.jobkey)
    for j in J: j['jobkey'] = json.dumps({k: j[k] for k in sorted(j) if k != 'jobkey'}, sort_keys=True, default=str)
    J = [j for j in J if j['jobkey'] not in done]
    print(f'{len(J)} jobs', flush=True)
    header = None
    with Pool(int(os.environ.get('NPROC', 8))) as pool:
        for i, rec in enumerate(pool.imap_unordered(run, J)):
            df = pd.DataFrame([rec])
            write_header = not os.path.exists(fn)
            if not write_header:
                cols = pd.read_csv(fn, nrows=0).columns
                df = df.reindex(columns=list(cols) + [c for c in df.columns if c not in cols])
                if len(df.columns) > len(cols):   # new columns -> rewrite file
                    old = pd.read_csv(fn); pd.concat([old, df]).to_csv(fn, index=False); continue
            df.to_csv(fn, mode='a', header=write_header, index=False)
            if i % 10 == 0:
                print(i, rec['group'], rec['site'], rec['mode'], rec['scen'], rec['rate'], rec.get('lcoh'), round(rec['secs'], 1), flush=True)
    print('done', flush=True)
