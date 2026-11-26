"""Combine LP plant results with infrastructure / export-chain terms -> per-hex, per-scenario cost tables."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import params as P, chain as C, model as M

OUT = '/workspace/outputs'
HEX = pd.read_csv(f'{OUT}/hex_prepared.csv', keep_default_na=False, na_values=['']).set_index('hex_id')
KG = P.H2_DEMAND_KG

def eu_scen(scen):
    return 'Commercial' if scen == 'Commercial' else 'Offtake'

def wsets(row):
    """WACC dicts for African side, EU import side, ship."""
    if row['scen'] == 'Zero':
        z = {a: 0.0 for a in P.ASSET}; return z, z, 0.0
    if row['scen'] == 'flat6':
        z = {a: 0.06 for a in P.ASSET}; return z, z, 0.06
    country = row['country']
    if country == 'X':
        P.COUNTRY['X'] = json.loads(row['country_over'])
    w_af = {a: P.wacc(a, country, row['scen'], row['ir']) for a in P.ASSET}
    w_eu = {a: P.wacc(a, 'EU', eu_scen(row['scen']), row['ir']) for a in P.ASSET}
    return w_af, w_eu, w_af['ship']

def site_terms(h, w_af, lcoe, allow_fresh=True):
    r = HEX.loc[h]
    road = C.road_cost(r.road_dist_km, w_af['infra'])
    water, src = C.water_cost(r.waterbody_dist_km, r.ocean_dist_km, w_af['infra'], lcoe, allow_fresh)
    return road, water, src

def delivered(row, ports=P.PRIMARY_PORTS, allow_fresh=True, nmi_override=None):
    """Least-cost delivered H2 at Rotterdam for one const-mode LP row. Returns dict (EUR/kg delivered)."""
    h = row['site']; r = HEX.loc[h]
    w_af, w_eu, w_ship = wsets(row)
    lcoe = row['lcoe']
    road, water, src = site_terms(h, w_af, lcoe, allow_fresh)
    best = None
    for p in ports:
        km = r[f'km_{p}']
        pipe = C.pipeline_cost(km, w_af['infra'], lcoe)
        truck = C.truck_nh3_cost(km, w_af['infra'])
        inland, mode = (pipe, 'H2 pipeline') if pipe <= truck else (truck, 'NH3 truck')
        nmi = P.PORTS[p]['nmi'] if nmi_override is None else nmi_override
        ch = C.export_chain(nmi, w_af, w_eu, w_ship, lcoe)
        tot = row['total'] + road + water + inland + ch['hb'] + ch['term_export'] + ch['ship'] + ch['term_import'] + ch['crack']
        d = tot / ch['h2_delivered_kg']
        if best is None or d < best['delivered']:
            hd = ch['h2_delivered_kg']
            best = dict(delivered=d, port=p, inland_mode=mode, water_src=src, inland_km=km, nmi=nmi,
                        c_power=(row['cost_pv'] + row['cost_wind'] + row['cost_bat'] + row.get('cost_grid', 0) * 0) / KG,
                        c_pv=row['cost_pv'] / KG, c_wind=row['cost_wind'] / KG, c_bat=row['cost_bat'] / KG,
                        c_el=row['cost_el'] / KG, c_h2s=row['cost_h2s'] / KG, c_grid=row.get('cost_grid', 0) / KG,
                        c_water=water / KG, c_road=road / KG, c_inland=inland / KG, c_hb=ch['hb'] / KG,
                        c_terminals=(ch['term_export'] + ch['term_import']) / KG, c_ship=ch['ship'] / KG,
                        c_crack=ch['crack'] / KG, c_losses=tot / hd - tot / KG,
                        plant_gate=row['total'] / KG, onsite=(row['total'] + road + water) / KG,
                        at_port=(row['total'] + road + water + inland) / KG,
                        nh3_rotterdam_eur_t=(tot - ch['crack']) / ch['nh3_arrived_t'],
                        nh3_port_eur_t=(row['total'] + road + water + inland + ch['hb'] + ch['term_export']) / C.NH3_T,
                        loss_frac=1 - hd / KG)
    return best

def onsite_flex(row, allow_fresh=True):
    w_af, _, _ = wsets(row)
    road, water, src = site_terms(row['site'], w_af, row['lcoe'], allow_fresh)
    return dict(onsite=(row['total'] + road + water) / KG, c_water=water / KG, c_road=road / KG, water_src=src,
                plant_gate=row['total'] / KG)

def eu_cost(row):
    """European green H2 delivered to industrial demand (same constant-supply basis)."""
    s = P.EU_SITES[row['site']]
    if row['scen'] == 'Zero': wi = 0.0
    else: wi = P.wacc('infra', 'EU', row['scen'], row['ir'])
    m3 = KG * P.WATER_L_PER_KG / 1e3
    water = m3 * (P.WATER_SPEC_COST * 1.6 + P.FRESH_EL * row['lcoe'] / 1e3) / KG    # EU water tariff ~2 EUR/m3
    pipe = C.eu_pipeline_tariff(s['pipe_km'], wi) if s['pipe_km'] > 0 else 0.0
    # European green ammonia (Haber-Bosch in Europe, EU finance) for comparison with imported NH3
    if row['scen'] == 'Zero': wh = 0.0
    else: wh = P.wacc('h2', 'EU', row['scen'], row['ir'])
    hb = C.hb_cost(wh, row['lcoe'])
    nh3 = (row['total'] + water * KG + hb) / C.NH3_T
    return dict(plant_gate=row['total'] / KG, c_water=water, c_pipe=pipe, delivered=row['total'] / KG + water + pipe,
                nh3_eu_eur_t=nh3)

def build():
    L = pd.read_csv(f'{OUT}/lp_main.csv')
    L = L[L.status == 'ok']
    L['country'] = L['country'].fillna('NA')
    for extra in [f'{OUT}/lp_second.csv', f'{OUT}/lp_eu2.csv', f'{OUT}/lp_hourly.csv']:
      if os.path.exists(extra):
        L2 = pd.read_csv(extra, keep_default_na=False, na_values=['']); L2['country'] = L2['country'].fillna('NA'); L = pd.concat([L, L2[L2.status == 'ok']], ignore_index=True)
    recs = []
    for _, row in L.iterrows():
        row = row.to_dict()
        base = {k: row.get(k, np.nan) for k in ['group', 'kind', 'site', 'country', 'mode', 'scen', 'rate', 'ir', 'Cpv', 'Cw', 'Cel', 'Eb', 'Eh',
                                    'el_util', 'curtail', 'lcoe', 'lcoh', 'h2s_hours', 'w_pv', 'w_wind', 'w_h2', 'w_infra']}
        for k in ['sens', 'crp', 'grid_km', 'cost_grid', 'grid_import_MWh', 'grid_export_MWh', 'tech', 'dt']:
            base[k] = row.get(k, np.nan)
        if row['kind'] == 'eu':
            base.update(eu_cost(row))
        elif row['mode'] == 'const':
            base.update(delivered(row))
            ext = delivered(row, ports=list(P.PORTS))
            base['delivered_allports'] = ext['delivered']; base['port_allports'] = ext['port']
            base['delivered_seaonly'] = delivered(row, allow_fresh=False)['delivered']
        else:
            base.update(onsite_flex(row))
            if row['kind'] == 'hex' and row['group'] == 'af':
                # lower bound: fully flexible Haber-Bosch sized for peak H2 output (no H2 buffer)
                peak_ratio = row['Cel'] * P.TECH['2030']['el_eff'] / P.D_MW
                old = P.HB_CAPEX; P.HB_CAPEX = old * peak_ratio
                try:
                    dd = delivered(row)
                finally:
                    P.HB_CAPEX = old
                base['delivered_flexHB'] = dd['delivered']; base['hb_peak_ratio'] = peak_ratio
        recs.append(base)
    R = pd.DataFrame(recs)
    R = R.merge(HEX[['lat', 'lon', 'ne_iso', 'onshore', 'cf_pv', 'cf_wind', 'grid_dist_km', 'road_dist_km', 'ocean_dist_km',
                     'waterbody_dist_km']], left_on='site', right_index=True, how='left')
    R.to_csv(f'{OUT}/results_all.csv', index=False)
    return R

if __name__ == '__main__':
    R = build()
    print(R.groupby(['group', 'mode', 'scen', 'rate']).agg(n=('site', 'size'), lcoh=('lcoh', 'median'), onsite=('onsite', 'median'),
          deliv=('delivered', 'median')).round(2).to_string())
