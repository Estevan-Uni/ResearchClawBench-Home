"""Re-implement the same 20-bus LOPF in PyPSA (open-source framework, linopy + HiGHS) and
compare objective, curtailment and LMPs with the custom scipy/HiGHS LP in model.py."""
import numpy as np, pandas as pd, json, os, time, warnings
warnings.filterwarnings("ignore")
import pypsa
from model import load_data, solve, summarize, NORTH, BOUNDARY, VOLL

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
d = load_data()


def build(mode="ntc", boundary_factor=1.0):
    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(168, name="snapshot"))
    buses = d["buses"]["name"].tolist()
    if mode == "copper":
        n.add("Bus", "GB")
        busmap = {b: "GB" for b in buses}
    else:
        n.add("Bus", buses, x=d["buses"]["x"].values, y=d["buses"]["y"].values, v_nom=400.0)
        busmap = {b: b for b in buses}
    for b in buses:
        n.add("Load", f"{b} load", bus=busmap[b], p_set=d["demand"][b].values)
        n.add("Generator", f"{b} shed", bus=busmap[b], p_nom=1e6, marginal_cost=VOLL, carrier="load shedding")
    for _, g in d["gens"].iterrows():
        kw = dict(bus=busmap[g.bus], p_nom=g.p_nom, marginal_cost=g.marginal_cost, carrier=g.carrier)
        if "wind" in g.carrier:
            kw["p_max_pu"] = d["wind_cf"][g.bus].values
        n.add("Generator", g["name"], **kw)
    for _, s in d["storage"].iterrows():
        eta = np.sqrt(s.efficiency)
        n.add("StorageUnit", s["name"], bus=busmap[s.bus], p_nom=s.p_nom, max_hours=s.e_nom / s.p_nom,
              efficiency_store=eta, efficiency_dispatch=eta, cyclic_state_of_charge=True,
              marginal_cost=0.01, carrier="PHS")
    if mode != "copper":
        for _, l in d["links"].iterrows():
            f = boundary_factor if l["name"] in BOUNDARY else 1.0
            n.add("Link", l["name"], bus0=l.bus0, bus1=l.bus1, p_nom=l.p_nom * f, p_min_pu=-1.0,
                  efficiency=1.0, length=l.length, carrier="AC")
    return n


res = {"pypsa_version": pypsa.__version__}
for key, mode, bf in [("ntc", "ntc", 1.0), ("copper", "copper", 1.0), ("x2", "ntc", 2.0)]:
    n = build(mode, bf)
    t0 = time.time()
    status, cond = n.optimize(solver_name="highs", log_to_console=False)
    tt = time.time() - t0
    gp = n.generators_t.p; wind = [g for g in n.generators.index if "wind" in g]
    avail = (n.generators.loc[wind, "p_nom"] * n.get_switchable_as_dense("Generator", "p_max_pu")[wind]).sum().sum()
    disp = gp[wind].sum().sum()
    shed = gp[[g for g in n.generators.index if g.endswith("shed")]].sum().sum()
    # custom LP for comparison
    o = solve(d, mode=mode, link_scale=(lambda l: np.where(l["name"].isin(BOUNDARY), bf, 1.0)) if bf != 1 else None)
    s = summarize(o)
    lmp_p = n.buses_t.marginal_price
    if mode == "copper":
        lmp_diff = float(np.abs(lmp_p["GB"].values - o["lmp"].values[:, 0]).max())
    else:
        lmp_diff = float(np.abs(lmp_p[o["buses"]].values - o["lmp"].values).max())
    r = dict(status=status, pypsa_objective_GBPm=float(n.objective) / 1e6, custom_objective_GBPm=o["objective"] / 1e6,
             pypsa_curt_TWh=(avail - disp) / 1e6, custom_curt_TWh=s["curt_wind_TWh"],
             pypsa_curt_pct=100 * (avail - disp) / avail, custom_curt_pct=s["curt_pct"],
             pypsa_shed_TWh=shed / 1e6, custom_shed_TWh=s["shed_TWh"], lmp_max_abs_diff=lmp_diff,
             pypsa_solve_s=tt)
    if mode != "copper":
        ld = n.links_t.p0[BOUNDARY].abs() / n.links.loc[BOUNDARY, "p_nom"]
        r["pypsa_boundary_hours_ge99_pct"] = float(100 * (ld >= 0.99).values.mean())
        r["custom_boundary_hours_ge99_pct"] = s["boundary_hours_ge99_pct"]
    r["obj_rel_diff"] = abs(r["pypsa_objective_GBPm"] - r["custom_objective_GBPm"]) / r["custom_objective_GBPm"]
    res[key] = r
    print(key, json.dumps(r, indent=1, default=str))
    if key == "ntc":
        n.export_to_netcdf(f"{OUT}/pypsa_network_ntc_solved.nc")
json.dump(res, open(f"{OUT}/pypsa_crosscheck.json", "w"), indent=2, default=str)
