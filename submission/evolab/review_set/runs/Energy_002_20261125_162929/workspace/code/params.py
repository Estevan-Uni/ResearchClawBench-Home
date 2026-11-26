"""Central parameter file. Every number used in the model is defined here with its source.

Sources (abbrev.):
  GeoH2  = Halloran et al. 2024, MethodsX 12:102660 (paper_000), Appendix A
  MUL    = Mueller et al. 2023, Applied Energy 343:121219 (paper_001)
  STE    = Steffen 2020, Energy Economics 88:104783 (paper_002), Table 2
  SCH    = Schmidt et al. 2019, Nature Sustainability (paper_003)
  ASSUMP = documented modelling assumption (2030 projection from public literature ranges,
           e.g. IEA Global Hydrogen Review / DEA technology catalogue / IRENA 2020); flagged in report.
All costs in EUR (constant, ~2023 money).
"""
import numpy as np

YEAR_H = 8760
LHV_H2 = 33.33  # kWh/kg

# ----------------------------------------------------------------------------------------
# Demand (GeoH2 case study: median of planned SSA projects online by 2030)
H2_DEMAND_KT = 60.6                     # kt H2 / yr  [GeoH2]
H2_DEMAND_KG = H2_DEMAND_KT * 1e6
D_MW = H2_DEMAND_KG * LHV_H2 / 1e3 / YEAR_H   # constant H2 output, MW(LHV) ~230.6

# ----------------------------------------------------------------------------------------
# Plant technologies. Two cost sets: 'geoh2' (replication) and '2030' (main analysis)
TECH = {
    '2030': dict(
        pv_capex=450e3, pv_fom=0.015, pv_life=25,          # EUR/MW  [ASSUMP: IEA/DEA 2030 utility PV 400-550 EUR/kW]
        wind_capex=1150e3, wind_fom=0.025, wind_life=25,   # EUR/MW  [ASSUMP: DEA/IEA 2030 onshore 1,050-1,300]
        el_capex=700e3, el_fom=0.035, el_life=20,          # EUR/MW_el incl. stack repl. in FOM [ASSUMP: IEA NZE 2030 ~ 600-900]
        el_eff=0.65,                                       # MWh_H2(LHV)/MWh_el  [ASSUMP; GeoH2 uses 0.59 for today]
        bat_e_capex=150e3, bat_p_capex=60e3, bat_fom=0.02, bat_life=15, bat_eta=0.95,  # EUR/MWh, EUR/MW [ASSUMP: BNEF/NREL ATB 2030]
        h2s_capex=17.0e3, h2s_fom=0.01, h2s_life=25,       # EUR/MWh_H2 compressed tank [GeoH2 21.7 today -> ASSUMP 2030]
        comp_el=0.051,                                     # MWh_el/MWh_H2 into storage [GeoH2]
        cavern_capex=1.0e3,                                # EUR/MWh_H2 salt cavern (Europe only) [ASSUMP: lit. 0.3-2 EUR/kWh]
    ),
    'geoh2': dict(
        pv_capex=1470e3, pv_fom=0.015, pv_life=20,
        wind_capex=1580e3, wind_fom=0.025, wind_life=20,
        el_capex=1250e3, el_fom=0.035, el_life=20, el_eff=0.59,
        bat_e_capex=95e3, bat_p_capex=0.0, bat_fom=0.02, bat_life=20, bat_eta=0.95,
        h2s_capex=21.7e3, h2s_fom=0.01, h2s_life=20, comp_el=0.051, cavern_capex=1.0e3,
    ),
}

# ----------------------------------------------------------------------------------------
# Resource-index -> capacity-factor mapping (dataset indices are unitless 0-1 "theoretical potential")
# ASSUMP: linear scaling so that ranges match Global Solar/Wind Atlas values for Namibia
#   PV  : 0.58-0.85 -> CF 0.19-0.255  (PVOUT ~1,650-2,250 kWh/kWp, single-axis-free fixed tilt)
#   wind: 0.29-0.74 -> CF 0.17-0.44  (interior to Luederitz coastal belt)
PV_CF_SCALE = 0.30
WIND_CF_SCALE = 0.60
def pv_cf(idx):   return PV_CF_SCALE * np.asarray(idx)
def wind_cf(idx): return WIND_CF_SCALE * np.asarray(idx)

# European benchmark sites (annual CF, ASSUMP from Global Solar/Wind Atlas / Renewables.ninja typical)
EU_SITES = {
    'Germany (north)': dict(lat=53.5, lon=9.0, pv_cf=0.110, wind_cf=0.300, cavern=True, pipe_km=0, elec_note='DE',
                            capex=dict(pv_capex=520e3, wind_capex=1400e3)),   # DE cost premium [ASSUMP: Fraunhofer ISE LCOE 2024 ranges]
    'Spain (+2,000 km pipe)': dict(lat=39.0, lon=-3.5, pv_cf=0.205, wind_cf=0.260, cavern=True, pipe_km=2000, elec_note='ES',
                                   capex=dict(pv_capex=430e3, wind_capex=1200e3)),
}

# ----------------------------------------------------------------------------------------
# Finance  (Egli/Schmidt structure:  Kd = IR + DM (+ country default spread),  Ke = Kd + EP (+ CRP))
BASE_RATES = {'Flat 0.49%': 0.0049, 'Moderate 2.15%': 0.0215, 'Extreme 4.29%': 0.0429}   # [SCH]
TAX = {'NA': 0.31, 'EU': 0.30}       # corporate tax [ASSUMP: Namibia 31% (2024), EU ~30%]
# Asset-class parameters: debt margin DM, equity premium EP, debt share D/V
# PV / wind DM, EP, D/V calibrated so that German after-tax WACC at 2017 LIBOR (-0.18%) reproduces STE Table 2
# (PV 2.7%, onshore wind 3.0%);
# H2 assets (electrolyser, battery, storage, Haber-Bosch, cracker) carry a first-of-a-kind / merchant premium.
ASSET = {
    'pv':    dict(DM=0.0130, EP=0.073, DV=0.75),
    'wind':  dict(DM=0.0140, EP=0.082, DV=0.75),
    'h2':    dict(DM=0.0250, EP=0.100, DV=0.60),
    'infra': dict(DM=0.0120, EP=0.050, DV=0.70),   # roads, pipelines, lines, water, terminals
    'ship':  dict(DM=0.0180, EP=0.060, DV=0.65),   # internationally financed, no country risk
}
# Country risk (Namibia, Moody's B1/Ba3 region) [ASSUMP approx. Damodaran 2024 ratings-based spreads]
COUNTRY = {
    'NA': dict(spread=0.035, crp=0.050),
    'BW': dict(spread=0.012, crp=0.018),   # A3 (sensitivity only)
    'ZA': dict(spread=0.035, crp=0.052),   # Ba2
    'AO': dict(spread=0.055, crp=0.080),   # B3
    'EU': dict(spread=0.000, crp=0.000),
}
# Finance scenarios for African projects. Each modifies the parameters above.
#  offtake: long-term EUR hard-currency offtake (e.g. H2Global/EU Hydrogen Bank) removes merchant & FX risk:
#           H2-asset EP -> renewable-like, DV -> 0.70.
#  blended: offtake + DFI/MIGA political-risk & credit guarantees: country default spread on debt removed,
#           CRP on equity halved, DV -> 0.75.
#  full   : offtake + guarantees + concessional/public (EIB/ECA) debt: no country risk, DM halved, DV 0.80.
FIN_SCEN = ['Commercial', 'Offtake', 'Blended', 'Full de-risking']
OFFTAKE_H2_EP = 0.080
def wacc(asset, country, scen, ir):
    a = dict(ASSET[asset]); c = dict(COUNTRY[country]); tax = TAX['EU' if country == 'EU' else 'NA']
    if asset == 'ship':
        c = dict(spread=0.0, crp=0.0)           # ships financed internationally
    if scen in ('Offtake', 'Blended', 'Full de-risking') and asset == 'h2':
        a['EP'] = OFFTAKE_H2_EP; a['DV'] = 0.70
    if scen in ('Offtake', 'Blended', 'Full de-risking') and asset in ('pv', 'wind'):
        a['EP'] = a['EP'] - 0.005               # contracted revenue also helps power assets
    if scen == 'Blended':
        c['spread'] = 0.0; c['crp'] = c['crp'] * 0.5; a['DV'] = max(a['DV'], 0.75)
    if scen == 'Full de-risking':
        c['spread'] = 0.0; c['crp'] = 0.0; a['DM'] = a['DM'] * 0.5; a['DV'] = max(a['DV'], 0.80)
    kd = ir + a['DM'] + c['spread']
    ke = kd + a['EP'] + c['crp']
    return a['DV'] * kd * (1 - tax) + (1 - a['DV']) * ke

def crf(r, n):
    if r < 1e-9:
        return 1.0 / n
    return r * (1 + r) ** n / ((1 + r) ** n - 1)

# ----------------------------------------------------------------------------------------
# Infrastructure / water  [GeoH2 unless noted]
ROAD_CAPEX_SHORT = 626478.45; ROAD_CAPEX_LONG = 481866.6; ROAD_OPEX = 7149.7   # EUR/km, EUR/km/yr
INFRA_LIFE = 50
WATER_L_PER_KG = 21.0                       # L/kg H2 [GeoH2]
WATER_SPEC_COST = 1.25                      # EUR/m3 freshwater purchase [GeoH2]
FRESH_EL = 0.4; SEA_EL = 3.7                # kWh/m3 treatment [GeoH2]
WATER_TRANSPORT_VAR = 0.1                   # EUR/100km/m3 [GeoH2]
DESAL_CAPEX = 1500.0                        # EUR per m3/day SWRO [ASSUMP: 1,000-2,000]
FRESH_TREAT_CAPEX = 300.0                   # EUR per m3/day [ASSUMP]
WATER_PIPE_CAPEX = 350e3                    # EUR/km for ~3,500 m3/d (DN300) incl. pumping stations [ASSUMP]
WATER_FOM = 0.03
# Grid connection
GRID_LINE_FIXED = 250e3                     # EUR/km (220 kV single circuit, right-of-way) [ASSUMP]
GRID_LINE_PER_MW = 600.0                    # EUR/MW/km capacity-dependent part [ASSUMP]
GRID_SUBSTATION = 8e6                       # EUR fixed [ASSUMP]
GRID_TARIFF = 0.10465e3                     # EUR/MWh import [GeoH2, Namibia electricity price]
GRID_EXPORT_PRICE = 0.020e3                 # EUR/MWh value of midday surplus sold to NamPower/SAPP [ASSUMP]
GRID_MAX_MW = 150.0                         # max grid-tie capacity (network absorption limit) [ASSUMP]
GRID_MAX_IMPORT_SHARE = 0.10                # RFNBO-type cap on grid share of electricity [ASSUMP]
# H2 pipeline to port (GeoH2 'small' pipeline: <1.2 GW)
H2PIPE_CAPEX = 90e3 + 90e3                  # EUR/km pipe + compressor [GeoH2]
H2PIPE_FOM = 0.0125; H2PIPE_LIFE = 42.5; H2PIPE_EL = 0.000614  # kWh/kg/km
H2PIPE_LARGE = 2.8e6 + 0.62e6; H2PIPE_LARGE_CAP_GW = 13.0      # EU backbone [GeoH2/EHB]
HVLINE_SMALL = 150e3                        # EUR/km small line for HB+ASU power at port [ASSUMP]
DETOUR = 1.25                               # route length / great-circle [ASSUMP]
# Trucks (GeoH2)
TRUCK = dict(speed=70, capex=160e3, opex=0.12, life=8, diesel=1.5, cons=0.35, wage=2.85,
             trailer_capex=210e3, trailer_opex=0.02, trailer_life=12, cap_kgH2=2600, load_h=1.5)

# ----------------------------------------------------------------------------------------
# Ammonia export chain
NH3_PER_H2 = 17.031 / (3 * 1.008)           # 5.63 kg NH3 per kg H2 (stoichiometric)
HB_H2_YIELD = 0.98                          # H2 converted to NH3 [ASSUMP]
HB_CAPEX = 750.0                            # EUR per t NH3/yr capacity incl. ASU [ASSUMP IEA 2030]
HB_FOM = 0.02; HB_LIFE = 25                 # [GeoH2 opex 1.5% -> 2% incl. ASU]
HB_EL = 2.809                               # kWh/kg H2 [GeoH2]
TERM_CAPEX = 90.0                           # EUR per t NH3/yr throughput (storage + jetty), each end [ASSUMP]
TERM_FOM = 0.03; TERM_LIFE = 30
SHIP = dict(cap_t=53000, capex=80e6, fom=0.04, life=25, speed_kn=16.0, port_days=4.0,
            fuel_t_per_day=78.0,  # NH3-fuelled; ~35 t HFO-eq/day energy [ASSUMP]
            boiloff_per_day=0.0005, canal_port_fee=0.4e6)
NH3_PRICE_PROXY = None                      # fuel NH3 valued at its own delivered cost (endogenous)
CRACK_CAPEX = 750e3                         # EUR/MW_H2 output [ASSUMP 2030; lit. 500-1,500]
CRACK_FOM = 0.03; CRACK_LIFE = 25
CRACK_YIELD = 0.85                          # H2 out / H2 in NH3 (heat by NH3 combustion + PSA) [ASSUMP]
CRACK_EL = 0.6                              # kWh/kg H2 out [ASSUMP]
EU_ELEC = 120.0                             # EUR/MWh industrial [ASSUMP]

# Ports (great-circle based; nautical miles to Rotterdam via Atlantic) [ASSUMP: sea-distances.org approx.]
PORTS = {
    'Walvis Bay': dict(lat=-22.95, lon=14.50, nmi=5450, country='NA'),
    'Lüderitz':   dict(lat=-26.65, lon=15.16, nmi=5700, country='NA'),
    'Saldanha Bay': dict(lat=-33.03, lon=17.94, nmi=6080, country='ZA'),
    'Namibe':     dict(lat=-15.19, lon=12.15, nmi=4980, country='AO'),
}
PRIMARY_PORTS = ['Walvis Bay', 'Lüderitz']

# ----------------------------------------------------------------------------------------
# Fossil benchmarks (Europe 2030)
GAS_PRICE = 35.0          # EUR/MWh(HHV) TTF 2030 forward-ish [ASSUMP]; sweep 20-60
SMR = dict(gas=0.050, capex_om=0.30, co2=9.0)              # MWh_HHV gas/kg H2 (~165 MJ_LHV/kg incl. fuel), EUR/kg, kgCO2/kg (incl. fuel) [ASSUMP IEA]
BLUE = dict(gas=0.055, capex_om=0.60, co2=1.0, captured=8.9, ts=0.040)   # 90% capture, T&S 40 EUR/t [ASSUMP]
# Conventional ammonia (MUL Table 5): 0.40 EUR/kg NH3 at 0 EUR/t, +0.0825 EUR/kg per 50 EUR/t -> 1.65 tCO2/tNH3
MUL_NH3 = {0: 0.40, 50: 0.48, 100: 0.57, 150: 0.65, 200: 0.73}
MUL_BE_LCOH = {0: 1.05, 50: 1.52, 100: 1.98, 150: 2.44, 200: 2.91}

# ----------------------------------------------------------------------------------------
# Time resolution
DT = 3                     # hours per step for the main LP
SEED = 7
