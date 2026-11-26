"""Synthetic weather + plant LP (full year, cyclic storage) for green H2 production.

Two production metrics:
  mode='flex'  : electrolyser follows renewable output, no storage, annual H2 quantity fixed.
  mode='const' : constant hourly H2 supply (for Haber-Bosch), battery + compressed H2 storage (GeoH2-style).
Optional grid connection (import capped, surplus export) in const mode.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.stats import norm
import params as P
METHOD = 'highs-ds'

# ---------------------------------------------------------------------------------- weather
def _ar1(n, rho, rng):
    e = rng.standard_normal(n)
    z = np.empty(n); z[0] = e[0]
    s = np.sqrt(1 - rho ** 2)
    for i in range(1, n):
        z[i] = rho * z[i - 1] + s * e[i]
    return z

def pv_profile(lat, target_cf, region='africa', seed=P.SEED):
    rng = np.random.default_rng(seed)
    h = np.arange(P.YEAR_H); d = h // 24; hr = h % 24 + 0.5
    dec = np.radians(23.45) * np.sin(2 * np.pi * (284 + d + 1) / 365)
    phi = np.radians(lat); om = np.radians(15 * (hr - 12))
    cosz = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(om)
    # fixed tilt = |lat| facing equator -> incidence equivalent to latitude 0
    cost = np.cos(dec) * np.cos(om)
    beam = np.clip(cost, 0, None) * (cosz > 0)
    diffuse = 0.15 * np.clip(cosz, 0, None)
    # daily clearness index (AR1), more variable in Europe; Southern-African wet season (DJF) dimming
    if region == 'africa':
        sd, rho = 0.12, 0.6
        seas = 1 - 0.08 * np.cos(2 * np.pi * (np.arange(365) - 15) / 365)  # lowest in Jan
    else:
        sd, rho = 0.33, 0.55
        seas = np.ones(365)
    kt = np.clip(1 + sd * _ar1(365, rho, rng), 0.08, 1.25) * seas
    raw = (beam * kt[d] + diffuse * (0.6 + 0.4 * kt[d]))
    cf = raw / raw.mean() * target_cf
    for _ in range(20):
        cf = np.clip(cf, 0, 0.92); cf *= target_cf / cf.mean()
    return np.clip(cf, 0, 0.92)

def _power_curve(v):
    vin, vr, vout = 3.0, 12.0, 25.0
    p = np.where((v >= vin) & (v < vr), (v ** 3 - vin ** 3) / (vr ** 3 - vin ** 3), 0.0)
    p = np.where((v >= vr) & (v < vout), 1.0, p)
    return p * 0.95  # wake/availability losses

def wind_profile(target_cf, region='africa', seed=P.SEED + 1):
    rng = np.random.default_rng(seed)
    n = P.YEAR_H; h = np.arange(n); doy = h / 24.0; hr = h % 24
    # synoptic (multi-day) + hourly components; calibrated so weekly P5 CF ~ 0.4-0.6 x mean
    z = 0.62 * _ar1(n, 0.985, rng) + 0.78 * _ar1(n, 0.90, rng)
    z = z / z.std()
    u = norm.cdf(z)
    k = 2.2
    base = (-np.log(1 - np.clip(u, 1e-9, 1 - 1e-9))) ** (1 / k)
    if region == 'africa':   # southern summer (Oct-Mar) stronger, afternoon/evening maximum
        mod = 1 + 0.10 * np.cos(2 * np.pi * (doy - 15) / 365) + 0.12 * np.cos(2 * np.pi * (hr - 18) / 24)
    else:                    # northern winter stronger, weak diurnal cycle
        mod = 1 + 0.15 * np.cos(2 * np.pi * (doy - 15) / 365) + 0.04 * np.cos(2 * np.pi * (hr - 15) / 24)
    lo, hi = 1.0, 30.0
    for _ in range(60):
        lam = 0.5 * (lo + hi)
        cf = _power_curve(lam * base * mod).mean()
        if cf < target_cf: lo = lam
        else: hi = lam
    return _power_curve(lam * base * mod)

def aggregate(x, dt):
    return x.reshape(-1, dt).mean(axis=1)

# ---------------------------------------------------------------------------------- LP
def annual_cost_coeffs(tech, w, eu_cavern=False):
    """Annualized cost per unit capacity (EUR/yr per MW or MWh). w: dict asset->wacc."""
    t = tech
    c = {}
    c['pv'] = t['pv_capex'] * (P.crf(w['pv'], t['pv_life']) + t['pv_fom'])
    c['wind'] = t['wind_capex'] * (P.crf(w['wind'], t['wind_life']) + t['wind_fom'])
    c['el'] = t['el_capex'] * (P.crf(w['h2'], t['el_life']) + t['el_fom'])
    c['bat_e'] = t['bat_e_capex'] * (P.crf(w['h2'], t['bat_life']) + t['bat_fom'])
    c['bat_p'] = t['bat_p_capex'] * (P.crf(w['h2'], t['bat_life']) + t['bat_fom'])
    sc = t['cavern_capex'] if eu_cavern else t['h2s_capex']
    c['h2s'] = sc * (P.crf(w['h2'], t['h2s_life']) + t['h2s_fom'])
    return c

def solve_plant(cfpv, cfw, w, mode='const', tech='2030', dt=P.DT, cavern=False,
                grid_km=None, w_infra=None, tech_over=None, battery=False):
    t = dict(P.TECH[tech]);
    if tech_over: t.update(tech_over)
    c = annual_cost_coeffs(t, w, cavern)
    cpv = aggregate(cfpv, dt); cw = aggregate(cfw, dt); T = len(cpv)
    eta = t['el_eff']; D = P.D_MW; Hann = P.H2_DEMAND_KG * P.LHV_H2 / 1e3  # MWh H2/yr
    if mode == 'flex':
        # x = [Cpv, Cw, Cel, e_1..e_T]
        nv = 3 + T
        cost = np.zeros(nv); cost[:3] = [c['pv'], c['wind'], c['el']]
        I = sp.identity(T, format='csr')
        A1 = sp.hstack([sp.csr_matrix(-cpv[:, None]), sp.csr_matrix(-cw[:, None]), sp.csr_matrix((T, 1)), I])
        A2 = sp.hstack([sp.csr_matrix((T, 2)), sp.csr_matrix(-np.ones((T, 1))), I])
        A_ub = sp.vstack([A1, A2]).tocsr(); b_ub = np.zeros(2 * T)
        A_eq = sp.csr_matrix(np.r_[np.zeros(3), np.full(T, eta * dt)][None, :]); b_eq = [Hann]
        res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
        if res.status != 0: raise RuntimeError(res.message)
        x = res.x; Cpv, Cw, Cel = x[:3]; e = x[3:]
        out = dict(Cpv=Cpv, Cw=Cw, Cel=Cel, Eb=0, Pb=0, Eh=0, Cg=0,
                   cost_pv=Cpv * c['pv'], cost_wind=Cw * c['wind'], cost_el=Cel * c['el'],
                   cost_bat=0.0, cost_h2s=0.0, cost_grid=0.0,
                   el_util=e.sum() * dt / (Cel * T * dt) if Cel > 0 else 0,
                   elec_used=e.sum() * dt,
                   curtail=1 - e.sum() / max(1e-9, (cpv * Cpv + cw * Cw).sum()))
        out['total'] = res.fun
        out['lcoh'] = res.fun / P.H2_DEMAND_KG
        out['lcoe'] = (out['cost_pv'] + out['cost_wind']) / out['elec_used']
        return out
    # ------------------------- constant supply (reduced formulation)
    # vars: caps [Cpv,Cw,Cel,Eb,Pb,Eh,Cg] + blocks e, hc, sh (+ bc, bd, sb if battery) (+ g, x if grid)
    # H2 balance: hd = D - eta*e + hc >= 0 ; storage: sh[t+1] = sh[t] + dt*(eta*e - D); hc >= eta*e - D (compression)
    grid = grid_km is not None
    ncap = 7
    blocks = ['e', 'hc', 'sh'] + (['bc', 'bd', 'sb'] if battery else []) + (['g', 'x'] if grid else [])
    nv = ncap + len(blocks) * T
    def idx(b): return ncap + blocks.index(b) * T
    kap = t['comp_el']; etab = np.sqrt(t['bat_eta'])
    cost = np.zeros(nv)
    cost[:6] = [c['pv'], c['wind'], c['el'], c['bat_e'], c['bat_p'], c['h2s']]
    if grid:
        cost[6] = P.GRID_LINE_PER_MW * grid_km * P.DETOUR * (P.crf(w_infra, P.INFRA_LIFE) + 0.015)
        cost[idx('g'):idx('g') + T] = P.GRID_TARIFF * dt
        cost[idx('x'):idx('x') + T] = -P.GRID_EXPORT_PRICE * dt
    tt = np.arange(T); tn = (tt + 1) % T
    rows, cols, vals = [], [], []
    def add(R, C, v):
        rows.append(np.asarray(R).ravel()); cols.append(np.asarray(C).ravel()); vals.append(np.broadcast_to(v, np.shape(C)).ravel())
    r = 0
    R = r + tt   # power balance
    add(R, idx('e') + tt, 1.0); add(R, idx('hc') + tt, kap); add(R, np.zeros(T, int), -cpv); add(R, np.ones(T, int), -cw)
    if battery: add(R, idx('bc') + tt, 1.0); add(R, idx('bd') + tt, -1.0)
    if grid: add(R, idx('x') + tt, 1.0); add(R, idx('g') + tt, -1.0)
    r += T
    R = r + tt; add(R, idx('e') + tt, 1.0); add(R, np.full(T, 2), -1.0); r += T          # e <= Cel
    R = r + tt; add(R, idx('sh') + tt, 1.0); add(R, np.full(T, 5), -1.0); r += T         # sh <= Eh
    R = r + tt; add(R, idx('e') + tt, eta); add(R, idx('hc') + tt, -1.0); b2 = (r, D); r += T   # eta e - hc <= D
    if battery:
        R = r + tt; add(R, idx('bc') + tt, 1.0); add(R, np.full(T, 4), -1.0); r += T
        R = r + tt; add(R, idx('bd') + tt, 1.0); add(R, np.full(T, 4), -1.0); r += T
        R = r + tt; add(R, idx('sb') + tt, 1.0); add(R, np.full(T, 3), -1.0); r += T
    if grid:
        R = r + tt; add(R, idx('g') + tt, 1.0); add(R, np.full(T, 6), -1.0); r += T
        R = r + tt; add(R, idx('x') + tt, 1.0); add(R, np.full(T, 6), -1.0); r += T
        add(np.full(T, r), idx('g') + tt, dt); add(np.full(T, r), idx('e') + tt, -P.GRID_MAX_IMPORT_SHARE * dt); r += 1
    A_ub = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, nv))
    b_ub = np.zeros(r); b_ub[b2[0]:b2[0] + T] = D
    rows, cols, vals = [], [], []
    r = 0
    R = r + tt; add(R, idx('sh') + tn, 1.0); add(R, idx('sh') + tt, -1.0); add(R, idx('e') + tt, -dt * eta); r += T
    beq = [np.full(T, -dt * D)]
    if battery:
        R = r + tt; add(R, idx('sb') + tn, 1.0); add(R, idx('sb') + tt, -1.0); add(R, idx('bc') + tt, -dt * etab); add(R, idx('bd') + tt, dt / etab); r += T
        beq.append(np.zeros(T))
    A_eq = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, nv))
    b_eq = np.concatenate(beq)
    bounds = [(0, None)] * nv
    bounds[6] = (0, P.GRID_MAX_MW) if grid else (0, 0)
    if not battery: bounds[3] = bounds[4] = (0, 0)
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=METHOD)
    if res.status != 0: raise RuntimeError(res.message)
    x = res.x
    Cpv, Cw, Cel, Eb, Pb, Eh, Cg = x[:7]
    e = x[idx('e'):idx('e') + T]; hc = x[idx('hc'):idx('hc') + T]
    out = dict(Cpv=Cpv, Cw=Cw, Cel=Cel, Eb=Eb, Pb=Pb, Eh=Eh, Cg=Cg,
               cost_pv=Cpv * c['pv'], cost_wind=Cw * c['wind'], cost_el=Cel * c['el'],
               cost_bat=Eb * c['bat_e'] + Pb * c['bat_p'], cost_h2s=Eh * c['h2s'])
    if grid:
        g = x[idx('g'):idx('g') + T]; xx = x[idx('x'):idx('x') + T]
        out['cost_grid'] = Cg * cost[6] + g.sum() * dt * P.GRID_TARIFF - xx.sum() * dt * P.GRID_EXPORT_PRICE
        out['grid_import_MWh'] = g.sum() * dt; out['grid_export_MWh'] = xx.sum() * dt
    else:
        out['cost_grid'] = 0.0
    out['elec_used'] = (e + kap * hc).sum() * dt
    out['el_util'] = e.mean() / Cel if Cel > 0 else 0
    gen = (cpv * Cpv + cw * Cw).sum()
    out['curtail'] = 1 - min(1, (e + kap * hc).sum() / max(gen, 1e-9))
    out['total'] = res.fun
    out['lcoh'] = res.fun / P.H2_DEMAND_KG
    out['lcoe'] = (out['cost_pv'] + out['cost_wind'] + out['cost_bat']) / out['elec_used']
    out['h2s_hours'] = Eh / P.D_MW
    return out

def wacc_set(country, scen, ir, flat=None):
    if flat is not None:
        return {a: flat for a in P.ASSET}
    return {a: P.wacc(a, country, scen, ir) for a in P.ASSET}
