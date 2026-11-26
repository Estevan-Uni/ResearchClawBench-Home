"""Compare the 20-node nodal model with a 2-zone (North/South) aggregation and a 1-node copper plate,
across boundary capacities: does spatial aggregation hide the downstream Bus10-Bus11 bottleneck?"""
import numpy as np, pandas as pd, os
from model import load_data, solve, summarize, NORTH, SOUTH, BOUNDARY

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
d = load_data()


def two_zone(d):
    z = {b: ("North" if b in NORTH else "South") for b in d["buses"]["name"]}
    buses = pd.DataFrame({"name": ["North", "South"], "v_nom": 400.0, "carrier": "AC", "x": 0, "y": 0})
    gens = d["gens"].copy(); gens["bus"] = gens["bus"].map(z)
    # aggregate wind per zone as capacity-weighted CF
    wind = gens[gens.carrier.str.contains("wind")]
    cf = pd.DataFrame({zz: (d["wind_cf"][wind[wind.bus == zz].index.map(lambda i: d["gens"].loc[i, "bus"])].values
                            * wind[wind.bus == zz]["p_nom"].values).sum(1) / wind[wind.bus == zz]["p_nom"].sum()
                       for zz in ["North", "South"]})
    gw = wind.groupby("bus")["p_nom"].sum()
    other = gens[~gens.carrier.str.contains("wind")].groupby(["bus", "carrier"], as_index=False).agg(p_nom=("p_nom", "sum"), marginal_cost=("marginal_cost", "first"))
    g2 = pd.concat([pd.DataFrame({"bus": gw.index, "carrier": "onshore wind", "p_nom": gw.values, "marginal_cost": 0.0}), other], ignore_index=True)
    g2["name"] = g2["bus"] + " " + g2["carrier"]
    dem = pd.DataFrame({"North": d["demand"][NORTH].sum(1), "South": d["demand"][SOUTH].sum(1)})
    st = d["storage"].copy(); st["bus"] = st["bus"].map(z); st["name"] = st["bus"] + " " + st.index.astype(str)
    links = pd.DataFrame({"bus0": ["North"], "bus1": ["South"], "p_nom": [7500.0], "length": [200], "carrier": ["AC"], "name": ["North-South"]})
    return dict(buses=buses, links=links, gens=g2, demand=dem, wind_cf=cf, storage=st)


def zsum(o):
    g = o["gens"]; wm = g.carrier.str.contains("wind").to_numpy()
    av = o["pmax"][:, wm].sum(); dp = o["p"].values[:, wm].sum()
    return dict(curt_pct=100 * (av - dp) / av, shed_TWh=o["shed"].values.sum() / 1e6,
                op_cost_GBPm=(o["p"].values * o["mc"]).sum() / 1e6)


d2 = two_zone(d)
rows = []
for fct in [1, 2, 2.5, 3, 4, 8]:
    on = solve(d, mode="ntc", link_scale=lambda l: np.where(l["name"].isin(BOUNDARY), fct, 1.0)); sn = summarize(on)
    oz = solve(d2, mode="ntc", link_scale=fct); sz = zsum(oz)
    rows.append(dict(boundary_GW=7.5 * fct, nodal_curt_pct=sn["curt_pct"], zonal_curt_pct=sz["curt_pct"],
                     nodal_shed_TWh=sn["shed_TWh"], zonal_shed_TWh=sz["shed_TWh"],
                     nodal_fuel_cost_GBPm=sn["op_cost_GBPm"], zonal_fuel_cost_GBPm=sz["op_cost_GBPm"]))
oc = summarize(solve(d, mode="copper"))
df = pd.DataFrame(rows); df["copper_curt_pct"] = oc["curt_pct"]; df["copper_shed_TWh"] = oc["shed_TWh"]
df.to_csv(f"{OUT}/zonal_vs_nodal.csv", index=False)
print(df.round(4).to_string())
