"""Regional diagnosis tables, country rankings, Kropf state summary, basin-direction comparison."""
import numpy as np, json, csv
import model as M
W = M.W
R = np.load(W + 'outputs/site_results_central.npz')
cw = json.load(open(W + 'outputs/cwon.json'))
iso = np.array([c['ISO3'] for c in cw]); name = np.array([c['Country_2020'] or c['Country'] for c in cw])
lon, lat, ci, wa, wp, ws = R['lon'], R['lat'], R['cidx'], R['w_area'], R['w_pop'], R['w_stock']
siso = np.where(ci >= 0, iso[np.clip(ci, 0, None)], '---')
REG = {
 'Philippines': siso == 'PHL', 'Vietnam': siso == 'VNM', 'Malaysia': siso == 'MYS', 'Indonesia': siso == 'IDN',
 'SE Asia (PHL+VNM+MYS+IDN)': np.isin(siso, ['PHL', 'VNM', 'MYS', 'IDN']),
 'SW Pacific (SH, 135E-150W)': (lat < 0) & ((lon >= 135) | (lon < -150)),
 'Central America & Caribbean': (lon > -100) & (lon < -59) & (lat > 5) & (lat < 30),
 'SE Africa (MOZ,MDG,TZA,KEN,ZAF,COM,MYT)': np.isin(siso, ['MOZ', 'MDG', 'TZA', 'KEN', 'ZAF', 'COM', 'MYT']),
 'South Asia (IND,BGD,MMR,LKA,PAK)': np.isin(siso, ['IND', 'BGD', 'MMR', 'LKA', 'PAK']),
 'East Asia (CHN,TWN,JPN)': np.isin(siso, ['CHN', 'TWN', 'JPN']),
 'Australia': siso == 'AUS',
 'West & Central Africa': (lon > -20) & (lon < 15) & (lat > -15) & (lat < 20),
 'South America (Atlantic+Pacific, lat<5)': (lon < -34) & (lon > -82) & (lat < 5) & (lat > -35),
 'Global': np.ones(len(lon), bool)}
rows = []
for reg, m in REG.items():
    w = wa[m]; tot = w.sum()
    if tot == 0: continue
    st = R['state'][m]
    for v in M.VARIANTS:
        for ssp in M.SSPS:
            tcl = R[f'tcl_{v}_{ssp}'][m]; sl = R[f'sl_{ssp}'][m]; sc = R[f'score_{v}_{ssp}'][m]
            f = lambda c: float(w[c].sum() / tot)
            rows.append(dict(region=reg, variant=v, ssp=ssp, area_ha=float(tot), n=int(m.sum()),
                             tc0=f(tcl == 0), tc1=f(tcl == 1), tc2=f(tcl == 2), tc3=f(tcl == 3),
                             slr0=f(sl == 0), slr1=f(sl == 1), slr2=f(sl == 2),
                             **{f's{k}': f(sc == k) for k in range(6)}, ge3=f(sc >= 3),
                             base_affected=f(R['f0'][m].max(1) >= M.F_MIN),
                             st_vuln=f(st.max(1) == 1), st_res=f(st.max(1) == 2), st_dep=f(st.max(1) == 3),
                             mean_rate=float(np.average(R[f'rate_{ssp}'][m], weights=w))))
with open(W + 'outputs/regional_diagnosis.csv', 'w') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
for r in rows:
    if r['ssp'] == 'ssp370':
        print(f"{r['region'][:32]:32s} {r['variant'][:6]} A={r['area_ha']/1e6:5.2f}Mha baseAff={r['base_affected']:.2f} TC0={r['tc0']:.2f} TC3={r['tc3']:.2f} SLR2={r['slr2']:.2f} ge3={r['ge3']:.2f} s5={r['s5']:.2f}")
# ---------- country table
crow = []
for k in np.unique(ci[ci >= 0]):
    m = ci == k
    d = dict(iso=iso[k], country=name[k], n=int(m.sum()), area_ha=float(wa[m].sum()), pop=float(wp[m].sum()), stock=float(ws[m].sum()))
    for v in M.VARIANTS:
        for ssp in M.SSPS:
            sc = R[f'score_{v}_{ssp}'][m]
            d[f'ge3_{v}_{ssp}'] = float(wa[m][sc >= 3].sum() / wa[m].sum())
            d[f'meanscore_{v}_{ssp}'] = float(sc.mean())
            d[f'area_ge3_{v}_{ssp}'] = float(wa[m][sc >= 3].sum())
            d[f'pop_ge3_{v}_{ssp}'] = float(wp[m][sc >= 3].sum()); d[f'stock_ge3_{v}_{ssp}'] = float(ws[m][sc >= 3].sum())
    crow.append(d)
crow.sort(key=lambda d: -d['area_ge3_A_MIT_ssp370'])
with open(W + 'outputs/country_table.csv', 'w') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(crow[0].keys())); wr.writeheader(); wr.writerows(crow)
print('\nTop countries by area at score>=3 (A, SSP3-7.0):')
for d in crow[:15]:
    print(f"{d['country'][:20]:20s} area {d['area_ha']/1e3:8.0f} kha  ge3 A245/370/585 {d['ge3_A_MIT_ssp245']:.2f}/{d['ge3_A_MIT_ssp370']:.2f}/{d['ge3_A_MIT_ssp585']:.2f}  B370 {d['ge3_B_Knutson_ssp370']:.2f}")
# monotonicity check per country (variant A)
nonmono = [d['country'] for d in crow if not (d['meanscore_A_MIT_ssp245'] <= d['meanscore_A_MIT_ssp370'] + 1e-9 <= d['meanscore_A_MIT_ssp585'] + 2e-9)]
print('\nNon-monotonic mean score countries (A):', nonmono)
# ---------- basin-level damage-weighted change in TC exposure vs Mo et al. (2023)
bas = R['basin']; mo = {'NA': '+', 'EP': '+', 'NI': '+', 'SI': '+', 'WP': '-', 'SP': '-'}
f0 = R['f0']; bt = []
for b in ['NA', 'EP', 'WP', 'NI', 'SI', 'SP']:
    m = bas == b
    if m.sum() == 0: continue
    # damage-weighted frequency: weights 0.05/0.10/0.20 (Mo et al. EVI reductions per Cat1-2/3/4-5 proxy)
    dw = np.array([0.05, 0.10, 0.20])
    d0 = np.average(f0[m] @ dw, weights=wa[m])
    out = dict(basin=b, mo2023=mo[b], n=int(m.sum()))
    for v in M.VARIANTS:
        d1 = np.average(R[f'f1_{v}_ssp245'][m] @ dw, weights=wa[m])
        out[v] = float(d1 / d0 - 1)
    bt.append(out); print(out)
json.dump(bt, open(W + 'outputs/basin_direction.json', 'w'), indent=1)
# Kropf state area summary per class
st = R['state']; sm = {}
for c, cn in enumerate('LMH'):
    sm[cn] = {M.STATE_NAMES[k]: float(wa[st[:, c] == k].sum() / wa.sum()) for k in range(4)}
print('state area shares', sm); json.dump(sm, open(W + 'outputs/state_shares.json', 'w'), indent=1)
