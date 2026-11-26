"""
Transparent linear optimal power flow (LOPF) model for the 20-bus GB-style test system.

Formulation (hourly, T = 168 snapshots, single-period economic dispatch):

  min  sum_t [ sum_g c_g p_{g,t} + VoLL * sum_n shed_{n,t} - v_flex * sum_k flex_{k,t} ]

  s.t. nodal balance (dual = locational marginal price, LMP)
         sum_{g@n} p_{g,t} + shed_{n,t} + sum_{s@n}(dis_{s,t} - ch_{s,t}) - sum_{k@n} flex_{k,t}
         - sum_l K_{n,l} f_{l,t} = d_{n,t}
       0 <= p_{g,t} <= p_nom_g * p_max_pu_{g,t}       (wind: p_max_pu = capacity factor)
       -F_l <= f_{l,t} <= F_l                          (bidirectional links, p_min_pu = -1;
                                                         f = f+ - f-, tiny cost eps_flow on |f| removes loop-flow degeneracy)
       0 <= dis, ch <= P_s ; 0 <= soc <= E_s   (tiny cost eps_storage on discharge removes cycling degeneracy)
       soc_{s,t} = soc_{s,t-1} + eta_s ch_{s,t} - dis_{s,t}/eta_s     (cyclic: soc_{-1} = soc_{T-1})
       DC-OPF mode only: f_{l,t} = (theta_{i,t} - theta_{j,t}) / x_l, theta_slack = 0
       copper-plate mode: one aggregated balance per hour (no network)

Solved with HiGHS through scipy.optimize.linprog (sparse matrices).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog
import os, time

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
VOLL = 6000.0  # GBP/MWh value of lost load


def load_data(datadir=DATA):
    buses = pd.read_csv(f"{datadir}/buses.csv")
    links = pd.read_csv(f"{datadir}/links.csv")
    gens = pd.read_csv(f"{datadir}/generators.csv")
    demand = pd.read_csv(f"{datadir}/demand.csv")
    wind_cf = pd.read_csv(f"{datadir}/wind_cf.csv")
    storage = pd.read_csv(f"{datadir}/storage.csv")
    gens["name"] = gens["bus"] + " " + gens["carrier"]
    links["name"] = links["bus0"] + "-" + links["bus1"]
    storage["name"] = storage["bus"] + " " + storage["carrier"]
    return dict(buses=buses, links=links, gens=gens, demand=demand, wind_cf=wind_cf, storage=storage)


def solve(data, mode="ntc", link_scale=None, storage_scale=1.0, demand_scale=1.0,
          flex=None, flex_value=5.0, voll=VOLL, method="highs", carbon_price=0.0, verbose=False,
          eps_storage=0.01, eps_flow=0.001):
    """
    mode: 'ntc' (transport model), 'dcopf' (KVL with x ~ length) or 'copper' (single node).
    link_scale: dict {link_name: factor} or float applied to all links, or callable(links_df)->array.
    flex: dict {bus: MW} flexible load (e.g. electrolyser) willing to pay flex_value GBP/MWh.
    """
    buses = data["buses"]["name"].tolist()
    N = len(buses); bidx = {b: i for i, b in enumerate(buses)}
    links = data["links"].copy(); gens = data["gens"].copy(); st = data["storage"].copy()
    D = data["demand"][buses].values * demand_scale  # T x N
    CF = data["wind_cf"][buses].values
    T = D.shape[0]

    # link capacities
    F = links["p_nom"].values.astype(float).copy()
    if link_scale is not None:
        if callable(link_scale):
            F = F * np.asarray(link_scale(links), dtype=float)
        elif isinstance(link_scale, dict):
            for k, v in link_scale.items():
                F[links["name"].values == k] *= v
        else:
            F = F * float(link_scale)
    L = len(links)
    G = len(gens)
    S = len(st)
    Pst = st["p_nom"].values * storage_scale
    Est = st["e_nom"].values * storage_scale
    eta = np.sqrt(st["efficiency"].values)  # round-trip split equally between charge & discharge
    flex = flex or {}
    K = len(flex); flex_bus = list(flex.keys()); flex_cap = np.array([flex[b] for b in flex_bus], float)

    # emissions intensity for optional carbon price (tCO2/MWh_el, CCGT ~0.37)
    ef = np.where(gens["carrier"].values == "gas", 0.37, 0.0)
    mc = gens["marginal_cost"].values + carbon_price * ef

    # generator availability
    pmax = np.zeros((T, G))
    for g in range(G):
        b = bidx[gens.loc[g, "bus"]]
        pmax[:, g] = gens.loc[g, "p_nom"] * (CF[:, b] if "wind" in gens.loc[g, "carrier"] else 1.0)

    use_links = mode in ("ntc", "dcopf")
    use_theta = mode == "dcopf"
    # variable blocks per snapshot
    blocks = [("p", G), ("shed", N), ("fp", L if use_links else 0), ("fn", L if use_links else 0), ("dis", S), ("ch", S), ("soc", S),
              ("flex", K), ("theta", N if use_theta else 0)]
    off = {}; o = 0
    for name, n in blocks:
        off[name] = o; o += n
    nv_t = o; nv = nv_t * T

    def vi(name, t, k):
        return t * nv_t + off[name] + k

    c = np.zeros(nv); lb = np.zeros(nv); ub = np.full(nv, np.inf)
    for t in range(T):
        s = slice(vi("p", t, 0), vi("p", t, 0) + G); c[s] = mc; ub[s] = pmax[t]
        s = slice(vi("shed", t, 0), vi("shed", t, 0) + N); c[s] = voll; ub[s] = D[t] if mode != "copper" else np.inf
        if use_links:
            for nm in ("fp", "fn"):
                s = slice(vi(nm, t, 0), vi(nm, t, 0) + L); ub[s] = F; c[s] = eps_flow
        if S:
            s = slice(vi("dis", t, 0), vi("dis", t, 0) + S); ub[s] = Pst; c[s] = eps_storage
            s = slice(vi("ch", t, 0), vi("ch", t, 0) + S); ub[s] = Pst
            s = slice(vi("soc", t, 0), vi("soc", t, 0) + S); ub[s] = Est
        if K:
            s = slice(vi("flex", t, 0), vi("flex", t, 0) + K); ub[s] = flex_cap; c[s] = -flex_value
        if use_theta:
            s = slice(vi("theta", t, 0), vi("theta", t, 0) + N); lb[s] = -np.inf; ub[s] = np.inf
            lb[vi("theta", t, 0)] = 0.0; ub[vi("theta", t, 0)] = 0.0  # slack bus angle

    rows, cols, vals = [], [], []; beq = []; r = 0
    gbus = np.array([bidx[b] for b in gens["bus"]]); sbus = np.array([bidx[b] for b in st["bus"]], int)
    fbus = np.array([bidx[b] for b in flex_bus], int)
    l0 = np.array([bidx[b] for b in links["bus0"]]); l1 = np.array([bidx[b] for b in links["bus1"]])
    bal_rows = np.zeros((T, N if mode != "copper" else 1), int)
    for t in range(T):
        nb = N if mode != "copper" else 1
        base = r
        bal_rows[t] = base + np.arange(nb)
        m = (lambda b: base + b) if mode != "copper" else (lambda b: base)
        for g in range(G):
            rows.append(m(gbus[g])); cols.append(vi("p", t, g)); vals.append(1.0)
        for n in range(N):
            rows.append(m(n)); cols.append(vi("shed", t, n)); vals.append(1.0)
        for s_ in range(S):
            rows += [m(sbus[s_]), m(sbus[s_])]; cols += [vi("dis", t, s_), vi("ch", t, s_)]; vals += [1.0, -1.0]
        for k in range(K):
            rows.append(m(fbus[k])); cols.append(vi("flex", t, k)); vals.append(-1.0)
        if use_links:
            for l in range(L):  # flow positive from bus0 to bus1
                rows += [m(l0[l]), m(l1[l]), m(l0[l]), m(l1[l])]
                cols += [vi("fp", t, l)] * 2 + [vi("fn", t, l)] * 2; vals += [-1.0, 1.0, 1.0, -1.0]
        if mode != "copper":
            beq += list(D[t]); r += N
        else:
            beq.append(D[t].sum()); r += 1
    # storage SOC recursion
    soc_rows = np.zeros((T, S), int)
    for t in range(T):
        tp = (t - 1) % T
        for s_ in range(S):
            soc_rows[t, s_] = r
            rows += [r, r, r, r]
            cols += [vi("soc", t, s_), vi("soc", tp, s_), vi("ch", t, s_), vi("dis", t, s_)]
            vals += [1.0, -1.0, -eta[s_], 1.0 / eta[s_]]
            beq.append(0.0); r += 1
    # KVL via angles: f*x - (theta0 - theta1) = 0 (x in per-unit of length; scaling is irrelevant in LP)
    if use_theta:
        x = links["length"].values / 50.0
        for t in range(T):
            for l in range(L):
                rows += [r, r, r, r]
                cols += [vi("fp", t, l), vi("fn", t, l), vi("theta", t, l0[l]), vi("theta", t, l1[l])]
                vals += [x[l], -x[l], -1.0, 1.0]; beq.append(0.0); r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))
    t0 = time.time()
    res = linprog(c, A_eq=A, b_eq=np.array(beq), bounds=np.c_[lb, ub], method=method,
                  options={"presolve": True, "disp": verbose, "primal_feasibility_tolerance": 1e-9,
                           "dual_feasibility_tolerance": 1e-9})
    solve_time = time.time() - t0
    if res.status != 0:
        raise RuntimeError(res.message)
    xsol = res.x.reshape(T, nv_t)
    def blk(name, n): return xsol[:, off[name]:off[name] + n]
    out = dict(mode=mode, T=T, buses=buses, links=links.assign(p_nom_used=F), gens=gens, storage=st,
               D=D, CF=CF, pmax=pmax, F=F, objective=res.fun, solve_time=solve_time,
               p=pd.DataFrame(blk("p", G), columns=gens["name"]),
               shed=pd.DataFrame(blk("shed", N), columns=buses),
               dis=pd.DataFrame(blk("dis", S), columns=st["name"]),
               ch=pd.DataFrame(blk("ch", S), columns=st["name"]),
               soc=pd.DataFrame(blk("soc", S), columns=st["name"]),
               flex=pd.DataFrame(blk("flex", K), columns=flex_bus),
               eta=eta, Pst=Pst, Est=Est, mc=mc, voll=voll, flex_value=flex_value, flex_cap=flex_cap)
    out["f"] = pd.DataFrame(blk("fp", L) - blk("fn", L), columns=links["name"]) if use_links else pd.DataFrame(np.zeros((T, 0)))
    if use_theta:
        out["theta"] = pd.DataFrame(blk("theta", N), columns=buses)
    duals = res.eqlin.marginals
    lmp = duals[bal_rows]
    if mode == "copper":
        lmp = np.repeat(lmp, N, axis=1)
    out["lmp"] = pd.DataFrame(lmp, columns=buses)
    out["A"] = A; out["beq"] = np.array(beq); out["x"] = res.x; out["bal_rows"] = bal_rows
    return out


NORTH = [f"Bus{i}" for i in range(1, 6)]
SOUTH = [f"Bus{i}" for i in range(6, 21)]
BOUNDARY = [f"Bus{i}-Bus{i+5}" for i in range(1, 6)]


def summarize(o):
    gens = o["gens"]; p = o["p"]
    wind_mask = gens["carrier"].str.contains("wind").values
    avail = o["pmax"][:, wind_mask]; disp = p.values[:, wind_mask]
    wb = gens.loc[wind_mask, "bus"].values
    nmask = np.isin(wb, NORTH)
    curt = avail - disp
    gas = p.values[:, (gens["carrier"] == "gas").values].sum()
    nuc = p.values[:, (gens["carrier"] == "nuclear").values].sum()
    shed = o["shed"].values.sum()
    op_cost = (p.values * o["mc"]).sum()  # excluding VoLL
    s = dict(
        avail_wind_TWh=avail.sum() / 1e6, disp_wind_TWh=disp.sum() / 1e6, curt_wind_TWh=curt.sum() / 1e6,
        curt_pct=100 * curt.sum() / avail.sum(),
        curt_north_TWh=curt[:, nmask].sum() / 1e6, curt_south_TWh=curt[:, ~nmask].sum() / 1e6,
        avail_north_TWh=avail[:, nmask].sum() / 1e6, avail_south_TWh=avail[:, ~nmask].sum() / 1e6,
        curt_north_pct=100 * curt[:, nmask].sum() / avail[:, nmask].sum(),
        curt_south_pct=100 * curt[:, ~nmask].sum() / avail[:, ~nmask].sum(),
        nuclear_TWh=nuc / 1e6, gas_TWh=gas / 1e6, shed_TWh=shed / 1e6,
        shed_pct_demand=100 * shed / o["D"].sum(),
        demand_TWh=o["D"].sum() / 1e6,
        storage_dis_GWh=o["dis"].values.sum() / 1e3,
        flex_TWh=o["flex"].values.sum() / 1e6 if o["flex"].shape[1] else 0.0,
        op_cost_GBPm=op_cost / 1e6, shed_cost_GBPm=shed * o["voll"] / 1e6,
        total_cost_GBPm=(op_cost + shed * o["voll"]) / 1e6,
        objective_GBPm=o["objective"] / 1e6, solve_time_s=o["solve_time"],
        mean_lmp_north=o["lmp"][NORTH].values.mean(), mean_lmp_south=o["lmp"][SOUTH].values.mean(),
    )
    if o["f"].shape[1]:
        bl = [c for c in BOUNDARY if c in o["f"].columns]
        Fb = o["links"].set_index("name").loc[bl, "p_nom_used"].values
        load = np.abs(o["f"][bl].values) / Fb
        s["boundary_flow_TWh"] = o["f"][bl].values.sum() / 1e6
        s["boundary_mean_loading"] = load.mean()
        s["boundary_hours_ge99_pct"] = 100 * (load >= 0.99).mean()
    return s


def residuals(o):
    """Independent post-hoc checks computed from the primal solution (not from the solver)."""
    A, x, b = o["A"], o["x"], o["beq"]
    r = A @ x - b
    bal = np.abs(r[o["bal_rows"].ravel()]).max()
    # recompute SOC recursion independently
    soc, ch, dis, eta = o["soc"].values, o["ch"].values, o["dis"].values, o["eta"]
    soc_prev = np.roll(soc, 1, axis=0)
    soc_res = np.abs(soc - soc_prev - eta * ch + dis / eta).max() if soc.size else 0.0
    over = 0.0
    if o["f"].shape[1]:
        over = (np.abs(o["f"].values) - o["F"]).max()
    simult = (np.minimum(ch, dis)).max() if ch.size else 0.0
    return dict(balance_resid_MW=bal, soc_resid_MWh=soc_res, max_line_overload_MW=max(over, 0.0),
                max_simultaneous_ch_dis_MW=simult)
