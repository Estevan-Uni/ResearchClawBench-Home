"""Run all scenarios and validation checks; write results to ../outputs."""
import numpy as np, pandas as pd, pickle, json, os, time
from model import load_data, solve, summarize, residuals, NORTH, SOUTH, BOUNDARY

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)
d = load_data()
results, rows = {}, []


def run(name, **kw):
    o = solve(d, **kw)
    s = summarize(o); s.update(residuals(o)); s["scenario"] = name
    s.update({k: (v if not callable(v) else "fn") for k, v in kw.items() if k != "flex"})
    if "flex" in kw: s["flex_GW"] = sum(kw["flex"].values()) / 1e3
    rows.append(s); results[name] = o
    print(f"{name:32s} curt={s['curt_pct']:6.2f}%  shed={s['shed_TWh']:.3f}TWh  cost={s['total_cost_GBPm']:.0f}M  t={s['solve_time_s']:.2f}s")
    return o


def boundary_scale(f):
    return lambda links: np.where(links["name"].isin(BOUNDARY), f, 1.0)


t0 = time.time()
# --- headline runs
run("constrained_NTC", mode="ntc")
run("unconstrained_copperplate", mode="copper")
run("unconstrained_links_x100", mode="ntc", link_scale=100.0)
run("constrained_DCOPF", mode="dcopf")
run("constrained_NTC_ipm", mode="ntc", method="highs-ipm")
# --- robustness
run("NTC_boundary_x2", mode="ntc", link_scale=boundary_scale(2.0))
run("NTC_noStorage", mode="ntc", storage_scale=0.0)
run("NTC_PHS_x10", mode="ntc", storage_scale=10.0)
run("NTC_carbon150", mode="ntc", carbon_price=150.0)
run("DCOPF_boundary_x2", mode="dcopf", link_scale=boundary_scale(2.0))

# --- boundary capacity sweep
sweep = []
for fct in [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8]:
    o = solve(d, mode="ntc", link_scale=boundary_scale(fct)); s = summarize(o)
    ld = np.abs(o["f"].values) / o["F"]
    hrs99 = (ld >= 0.99).mean(0) * 100
    s.update(factor=fct, boundary_GW=7.5 * fct,
             n_internal_binding=int(((hrs99 > 5) & ~o["links"]["name"].isin(BOUNDARY).values).sum()),
             max_internal_hours99=float(hrs99[~o["links"]["name"].isin(BOUNDARY).values].max()),
             worst_internal=o["links"]["name"].values[~o["links"]["name"].isin(BOUNDARY).values][np.argmax(hrs99[~o["links"]["name"].isin(BOUNDARY).values])])
    sweep.append(s)
    print(f"sweep x{fct}: curt={s['curt_pct']:.2f}% internal-binding={s['n_internal_binding']} worst={s['worst_internal']} {s['max_internal_hours99']:.0f}%")
pd.DataFrame(sweep).to_csv(f"{OUT}/boundary_sweep.csv", index=False)

# --- flexibility / sector-coupling: northern electrolysers bidding 5 GBP/MWh (below nuclear)
flex_rows = []
for gw in [0, 1, 2.5, 5, 7.5, 10, 15, 20, 30]:
    flex = {b: gw * 1e3 / 5 for b in NORTH} if gw > 0 else None
    o = solve(d, mode="ntc", flex=flex, flex_value=5.0); s = summarize(o)
    fl = o["flex"].values.sum(1) if gw > 0 else np.zeros(o["T"])
    s.update(electrolyser_GW=gw, electrolyser_CF=(fl.mean() / (gw * 1e3)) if gw > 0 else np.nan,
             H2_TWh_LHV=0.7 * fl.sum() / 1e6)
    flex_rows.append(s)
    if gw == 10: results["NTC_electrolyser_10GW"] = o
    print(f"electrolyser {gw} GW: curt={s['curt_pct']:.2f}% flex={s['flex_TWh']:.3f}TWh CF={s['electrolyser_CF']}")
pd.DataFrame(flex_rows).to_csv(f"{OUT}/electrolyser_sweep.csv", index=False)

# --- combined options
run("NTC_boundary_x2_electrolyser5GW", mode="ntc", link_scale=boundary_scale(2.0),
    flex={b: 1000.0 for b in NORTH})

# --- demand-level sensitivity (NOT the headline; illustrates regime change when the south is not short)
dem = []
for ds in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
    for m in ["ntc", "copper"]:
        for stc in [1.0, 10.0]:
            o = solve(d, mode=m, demand_scale=ds, storage_scale=stc); s = summarize(o)
            s.update(demand_scale=ds, mode=m, storage_scale=stc); dem.append(s)
pd.DataFrame(dem).to_csv(f"{OUT}/demand_sensitivity.csv", index=False)
print(pd.DataFrame(dem)[["demand_scale", "mode", "storage_scale", "curt_pct", "shed_TWh", "storage_dis_GWh"]])

# =====================  VALIDATION  =====================
val = {}
o = results["constrained_NTC"]
gens = o["gens"]; wm = gens["carrier"].str.contains("wind").values
wbus = gens.loc[wm, "bus"].values
WN = o["pmax"][:, wm][:, np.isin(wbus, NORTH)].sum(1)
DN = o["D"][:, :5].sum(1)
cf_curt = np.maximum(0, WN - DN - 7500.0)
lp_curt = (o["pmax"][:, wm] - o["p"].values[:, wm]).sum(1)
val["closed_form_curt_TWh"] = cf_curt.sum() / 1e6
val["lp_curt_TWh"] = lp_curt.sum() / 1e6
val["closed_form_max_abs_hourly_diff_MW"] = float(np.abs(cf_curt - lp_curt).max())
val["hours_with_north_surplus"] = int((WN - DN - 7500 > 0).sum())
val["min_north_surplus_MW"] = float((WN - DN - 7500).min())
pd.DataFrame({"WN": WN, "DN": DN, "closed_form": cf_curt, "lp": lp_curt}).to_csv(f"{OUT}/closed_form_check.csv", index=False)

# copper plate analytic merit order: wind (0) -> nuclear (10) -> gas (50) -> shed (VoLL)
oc = results["unconstrained_copperplate"]
W = oc["pmax"][:, wm].sum(1); Dt = oc["D"].sum(1)
nuc_cap = gens.loc[gens.carrier == "nuclear", "p_nom"].sum(); gas_cap = gens.loc[gens.carrier == "gas", "p_nom"].sum()
w_an = np.minimum(W, Dt); r1 = Dt - w_an
n_an = np.minimum(nuc_cap, r1); r2 = r1 - n_an
g_an = np.minimum(gas_cap, r2); sh_an = r2 - g_an
lmp_an = np.select([r1 <= 1e-9, r2 <= 1e-9, sh_an <= 1e-9], [0.0, 10.0, 50.0], 6000.0)
pc = oc["p"].values
val["copper_wind_maxdiff_MW"] = float(np.abs(pc[:, wm].sum(1) - w_an).max())
val["copper_nuc_maxdiff_MW"] = float(np.abs(pc[:, (gens.carrier == "nuclear").values].sum(1) - n_an).max())
val["copper_gas_maxdiff_MW"] = float(np.abs(pc[:, (gens.carrier == "gas").values].sum(1) - g_an).max())
val["copper_shed_maxdiff_MW"] = float(np.abs(oc["shed"].values.sum(1) - sh_an).max())
val["copper_lmp_maxdiff"] = float(np.abs(oc["lmp"].values[:, 0] - lmp_an).max())
val["copper_min_residual_after_wind_MW"] = float(r1.min())
# NTC: analytic LMP pattern: north 0 when surplus>0, south VoLL when southern shedding
val["ntc_lmp_north_unique"] = sorted(set(np.round(o["lmp"][NORTH].values.ravel(), 6)))
val["ntc_lmp_south_unique"] = sorted(set(np.round(o["lmp"][SOUTH].values.ravel(), 6)))
# simplex vs interior point
oi = results["constrained_NTC_ipm"]
val["simplex_vs_ipm_obj_reldiff"] = abs(o["objective"] - oi["objective"]) / abs(o["objective"])
val["simplex_vs_ipm_curt_diff_MWh"] = float(abs(lp_curt.sum() - (oi["pmax"][:, wm] - oi["p"].values[:, wm]).sum()))
val["simplex_vs_ipm_lmp_maxdiff"] = float(np.abs(o["lmp"].values - oi["lmp"].values).max())
# duality: congestion rent = sum (lmp_j - lmp_i) * f
f = o["f"].values; L = o["links"]
li = [o["buses"].index(b) for b in L["bus0"]]; lj = [o["buses"].index(b) for b in L["bus1"]]
lmp = o["lmp"].values
cr = ((lmp[:, lj] - lmp[:, li]) * f).sum()
val["congestion_rent_GBPm"] = cr / 1e6
val["congestion_rent_boundary_share"] = float(((lmp[:, lj] - lmp[:, li]) * f)[:, L["name"].isin(BOUNDARY).values].sum() / cr)
with open(f"{OUT}/validation.json", "w") as fh:
    json.dump(val, fh, indent=2, default=float)
print(json.dumps(val, indent=2, default=float))

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/scenario_summary.csv", index=False)
# hourly outputs of headline runs
for k in ["constrained_NTC", "unconstrained_copperplate", "constrained_DCOPF", "NTC_boundary_x2", "NTC_electrolyser_10GW"]:
    r = results[k]
    r["p"].to_csv(f"{OUT}/{k}_generation.csv", index_label="hour")
    r["lmp"].to_csv(f"{OUT}/{k}_lmp.csv", index_label="hour")
    r["shed"].to_csv(f"{OUT}/{k}_shed.csv", index_label="hour")
    r["soc"].to_csv(f"{OUT}/{k}_storage_soc.csv", index_label="hour")
    if r["f"].shape[1]: r["f"].to_csv(f"{OUT}/{k}_link_flows.csv", index_label="hour")
slim = {k: {kk: v for kk, v in r.items() if kk not in ("A", "x", "beq")} for k, r in results.items()}
with open(f"{OUT}/results.pkl", "wb") as fh:
    pickle.dump(slim, fh)
print(f"total runtime {time.time()-t0:.1f}s")
