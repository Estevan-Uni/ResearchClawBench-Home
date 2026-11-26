"""Post-LP cost terms: road, water, inland transport, NH3 synthesis, terminals, shipping, cracking;
fossil and European benchmarks. All return EUR per year unless noted."""
import numpy as np
import params as P

KG = P.H2_DEMAND_KG
NH3_T = KG * P.HB_H2_YIELD * P.NH3_PER_H2 / 1e3   # t NH3 / yr produced

def road_cost(km, w_infra):
    if km <= 0: return 0.0
    capex = (P.ROAD_CAPEX_SHORT if km < 10 else P.ROAD_CAPEX_LONG) * km
    return capex * P.crf(w_infra, P.INFRA_LIFE) + P.ROAD_OPEX * km

def water_cost(fresh_km, ocean_km, w_infra, lcoe, allow_fresh=True):
    """Returns (annual cost, source). lcoe in EUR/MWh."""
    m3 = KG * P.WATER_L_PER_KG / 1e3; cap = m3 / 365 * 1.2
    a = P.crf(w_infra, P.INFRA_LIFE) + P.WATER_FOM
    a_plant = P.crf(w_infra, 25) + P.WATER_FOM
    fresh = (P.FRESH_TREAT_CAPEX * cap * a_plant + P.WATER_PIPE_CAPEX * fresh_km * P.DETOUR * a
             + m3 * (P.WATER_SPEC_COST + P.FRESH_EL * lcoe / 1e3 + P.WATER_TRANSPORT_VAR * fresh_km * P.DETOUR / 100))
    sea = (P.DESAL_CAPEX * cap * a_plant + P.WATER_PIPE_CAPEX * ocean_km * P.DETOUR * a
           + m3 * (P.SEA_EL * lcoe / 1e3 + P.WATER_TRANSPORT_VAR * ocean_km * P.DETOUR / 100))
    if not allow_fresh or sea <= fresh:
        return sea, 'seawater'
    return fresh, 'freshwater'

def pipeline_cost(km, w_infra, lcoe):
    """H2 pipeline (GeoH2 small) + small HV line for HB/ASU power, plant -> port."""
    capex = (P.H2PIPE_CAPEX + P.HVLINE_SMALL) * km
    return capex * (P.crf(w_infra, P.H2PIPE_LIFE) + P.H2PIPE_FOM) + KG * P.H2PIPE_EL * km * lcoe / 1e3

def truck_nh3_cost(km, w_infra):
    t = P.TRUCK
    trips = KG / t['cap_kgH2']
    trip_h = 2 * km / t['speed'] + 2 * t['load_h']
    n = trips * trip_h / 8760
    truck = n * t['capex'] * (P.crf(w_infra, t['life']) + t['opex'])
    trailer = n * t['trailer_capex'] * (P.crf(w_infra, t['trailer_life']) + t['trailer_opex'])
    fuel = trips * 2 * km * t['cons'] * t['diesel']
    wage = trips * trip_h * t['wage']
    return truck + trailer + fuel + wage

def hb_cost(w_h2, lcoe):
    capex = P.HB_CAPEX * NH3_T
    return capex * (P.crf(w_h2, P.HB_LIFE) + P.HB_FOM) + KG * P.HB_EL * lcoe / 1e3

def terminal_cost(nh3_t, w_infra):
    return P.TERM_CAPEX * nh3_t * (P.crf(w_infra, P.TERM_LIFE) + P.TERM_FOM)

def shipping(nmi, w_ship):
    s = P.SHIP
    sail_days = nmi / (s['speed_kn'] * 24)
    voyage = 2 * sail_days + s['port_days']
    voyages = NH3_T / s['cap_t']
    ships = voyages * voyage / 365
    cost = ships * s['capex'] * (P.crf(w_ship, s['life']) + s['fom']) + voyages * s['canal_port_fee']
    fuel_t = voyages * s['fuel_t_per_day'] * 2 * sail_days
    boil_t = voyages * s['cap_t'] * s['boiloff_per_day'] * sail_days
    loss_frac = (fuel_t + boil_t) / NH3_T
    return cost, loss_frac

def cracking(nh3_t_in, w_h2_eu):
    h2_out = nh3_t_in * 1e3 / P.NH3_PER_H2 * P.CRACK_YIELD
    mw = h2_out * P.LHV_H2 / 1e3 / 8760 / 0.95
    cost = P.CRACK_CAPEX * mw * (P.crf(w_h2_eu, P.CRACK_LIFE) + P.CRACK_FOM) + h2_out * P.CRACK_EL * P.EU_ELEC / 1e3
    return cost, h2_out

def export_chain(port_nmi, w_af, w_eu, w_ship, lcoe):
    """Costs of HB (at port), export terminal, ship, import terminal, cracker. Returns dict EUR/yr + delivered kg."""
    hb = hb_cost(w_af['h2'], lcoe)
    term_x = terminal_cost(NH3_T, w_af['infra'])
    ship, loss = shipping(port_nmi, w_ship)
    nh3_arr = NH3_T * (1 - loss)
    term_m = terminal_cost(nh3_arr, w_eu['infra'])
    crack, h2_out = cracking(nh3_arr, w_eu['h2'])
    return dict(hb=hb, term_export=term_x, ship=ship, term_import=term_m, crack=crack,
                loss_frac=loss, nh3_arrived_t=nh3_arr, h2_delivered_kg=h2_out)

# --------------------------------------------------------------------------- benchmarks
def grey_h2(co2, gas=P.GAS_PRICE):
    return P.SMR['gas'] * gas + P.SMR['capex_om'] + P.SMR['co2'] * co2 / 1e3

def blue_h2(co2, gas=P.GAS_PRICE):
    b = P.BLUE
    return b['gas'] * gas + b['capex_om'] + b['captured'] * b['ts'] + b['co2'] * co2 / 1e3

def grey_nh3(co2):
    """EUR/kg NH3, linear fit of Mueller et al. Table 5 (0.40 EUR/kg + 1.65 kgCO2/kgNH3)."""
    return 0.40 + 0.00165 * co2

def eu_pipeline_tariff(km, w_infra, util=0.7):
    """EUR/kg for a share of a large backbone pipeline (GeoH2/EHB costs)."""
    flow = P.H2PIPE_LARGE_CAP_GW * 1e3 * util * 8760 / P.LHV_H2 * 1e3   # kg/yr
    ann = P.H2PIPE_LARGE * km * (P.crf(w_infra, P.H2PIPE_LIFE) + P.H2PIPE_FOM)
    return ann / flow + P.H2PIPE_EL * km * P.EU_ELEC / 1e3
