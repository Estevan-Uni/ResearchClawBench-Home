"""Generate all report figures from outputs/ (run after run_experiments.py)."""
import numpy as np, pandas as pd, pickle, os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from model import load_data, NORTH, SOUTH, BOUNDARY

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT, IMG = f"{ROOT}/outputs", f"{ROOT}/report/images"
os.makedirs(IMG, exist_ok=True)
R = pickle.load(open(f"{OUT}/results.pkl", "rb"))
d = load_data()

# palette (validated reference categorical palette, fixed order)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                         "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
hours = np.arange(168)


def save(fig, name):
    fig.savefig(f"{IMG}/{name}.png", dpi=150); plt.close(fig); print("saved", name)


def wind_series(o, zone=None):
    g = o["gens"]; wm = g["carrier"].str.contains("wind").to_numpy().copy()
    if zone is not None: wm = wm & g["bus"].isin(zone).to_numpy()
    av = o["pmax"][:, wm].sum(axis=1); dp = o["p"].values[:, wm].sum(axis=1)
    return av, dp, av - dp


# ---------------- 1. network schematic ----------------
def pos():
    p = {}
    for i in range(1, 6): p[f"Bus{i}"] = (i - 1, 2.0)
    for i in range(6, 11): p[f"Bus{i}"] = (i - 6, 0.6)
    for i in range(11, 16): p[f"Bus{i}"] = (4 - (i - 11), -0.6)    # snake back
    for i in range(16, 21): p[f"Bus{i}"] = (i - 16, -1.8)
    return p


P = pos()
fig, ax = plt.subplots(figsize=(9, 5.4)); ax.grid(False); ax.axis("off")
L = d["links"]
for _, l in L.iterrows():
    (x0, y0), (x1, y1) = P[l.bus0], P[l.bus1]
    bnd = l["name"] in BOUNDARY
    if l["name"] == "Bus15-Bus16":
        continue
    if l["name"] == "Bus10-Bus11":  # snake turn
        xm = x0 + 0.45
        ax.plot([x0, xm, xm, x1], [y0, y0, y1, y1], color=INK2, lw=2.2, zorder=1)
        if l["name"] == "Bus10-Bus11":
            ax.text(xm + 0.06, (y0 + y1) / 2, "Bus10–Bus11\n5 GW", fontsize=8, color=ORANGE, va="center")
        continue
    if l["name"] == "Bus15-Bus16":
        continue
    ax.plot([x0, x1], [y0, y1], color=RED if bnd else INK2, lw=3.2 if bnd else 2.2, zorder=1,
            ls="-" if not bnd else (0, (4, 1.5)))
# fix Bus15-Bus16 (x=0 at y=-0.6 to x=0 at y=-1.8): left-side turn
ax.plot([0, -0.45, -0.45, 0], [-0.6, -0.6, -1.8, -1.8], color=INK2, lw=2.2, zorder=1)
dem = d["demand"].mean()
wcap = d["gens"][d["gens"].carrier.str.contains("wind")].set_index("bus")["p_nom"]
for b, (x, y) in P.items():
    n = b in NORTH
    ax.scatter(x, y, s=120 + wcap[b] / 18, color=GREEN if n else BLUE, edgecolor="white", lw=2, zorder=3)
    ax.text(x, y + 0.27, b, ha="center", fontsize=8, color=INK)
    ax.text(x, y - 0.33, f"{dem[b]/1e3:.1f} GW", ha="center", fontsize=7, color=INK2)
for b in ["Bus2", "Bus8", "Bus14"]:
    ax.text(P[b][0] + 0.17, P[b][1] + 0.12, "N", fontsize=7, color=VIOLET, weight="bold")
for b in ["Bus1", "Bus3", "Bus12"]:
    ax.text(P[b][0] - 0.27, P[b][1] + 0.12, "PHS", fontsize=7, color=MAGENTA, weight="bold")
ax.text(4.75, 1.3, "Boundary: 5 × 1.5 GW\n= 7.5 GW (200 km)", color=RED, fontsize=9, va="center")
ax.text(-0.9, 2.0, "North\n(Scotland-like)\n50 GW wind\n3.9% of demand", fontsize=8.5, color=GREEN, ha="right", va="center")
ax.text(-0.9, -0.6, "South\n(England & Wales-like)\n7.5 GW wind\n96.1% of demand", fontsize=8.5, color=BLUE, ha="right", va="center")
ax.set_xlim(-2.6, 6.2); ax.set_ylim(-2.3, 2.5)
ax.set_title("20-bus GB-style test network (schematic, non-geographic layout)\n"
             "node area ∝ wind capacity; label below node = mean demand; intra-zone links 5 GW / 50 km", fontsize=10)
save(fig, "fig01_network_schematic")

# ---------------- 2. data overview ----------------
o = R["constrained_NTC"]
D = d["demand"]; CF = d["wind_cf"]
fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
ax = axs[0, 0]
ax.fill_between(hours, 0, D[SOUTH].sum(axis=1) / 1e3, color=BLUE, alpha=0.8, label="South (Bus6–20)", lw=0)
ax.fill_between(hours, D[SOUTH].sum(axis=1) / 1e3, D.sum(axis=1) / 1e3, color=GREEN, alpha=0.9, label="North (Bus1–5)", lw=0)
ax.set(title="(a) Hourly demand by zone", xlabel="hour of week", ylabel="GW"); ax.legend(loc="upper right")
ax = axs[0, 1]
mcf = CF.mean(); ax.bar(range(20), mcf.values, color=[GREEN if b in NORTH else BLUE for b in mcf.index], width=0.75)
ax.set_xticks(range(20)); ax.set_xticklabels([b.replace("Bus", "") for b in mcf.index], fontsize=8)
ax.set(title="(b) Mean wind capacity factor by bus", xlabel="bus", ylabel="mean CF")
ax.text(1, 0.48, "North", color=GREEN); ax.text(12, 0.33, "South", color=BLUE)
ax = axs[1, 0]
avN, _, _ = wind_series(o, NORTH)
DN = D[NORTH].sum(axis=1).values
ax.plot(hours, avN / 1e3, color=GREEN, lw=1.8, label="available northern wind")
ax.plot(hours, (DN + 7500) / 1e3, color=RED, lw=1.8, ls="--", label="northern demand + 7.5 GW export limit")
ax.fill_between(hours, (DN + 7500) / 1e3, avN / 1e3, where=avN > DN + 7500, color=RED, alpha=0.15, lw=0,
                label="surplus that cannot be exported")
ax.set(title="(c) Northern wind vs. local demand + boundary capacity", xlabel="hour of week", ylabel="GW", ylim=(0, 36))
ax.legend(loc="upper center", fontsize=8, ncol=2, frameon=True, framealpha=0.95)
ax = axs[1, 1]
av, _, _ = wind_series(o)
firm = 3600 + d["gens"].loc[d["gens"].carrier == "gas", "p_nom"].sum()
ax.fill_between(hours, 0, av / 1e3, color=GREEN, alpha=0.85, lw=0, label="available wind (all buses)")
ax.fill_between(hours, av / 1e3, av / 1e3 + firm / 1e3, color=ORANGE, alpha=0.8, lw=0, label="nuclear + gas capacity (14.2 GW)")
ax.plot(hours, D.sum(axis=1) / 1e3, color=INK, lw=1.8, label="total demand")
ax.set(title="(d) System adequacy: maximum supply vs. demand", xlabel="hour of week", ylabel="GW", ylim=(0, 175))
ax.legend(loc="upper center", fontsize=8, ncol=3, frameon=True, framealpha=0.95)
fig.tight_layout(); save(fig, "fig02_data_overview")

# ---------------- 3. wind dispatched vs curtailed (headline) ----------------
fig, axs = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, sharey=True)
for ax, (k, ttl) in zip(axs, [("constrained_NTC", "(a) Network-constrained LOPF (boundary = 7.5 GW)"),
                              ("unconstrained_copperplate", "(b) Unconstrained (copper plate)")]):
    av, dp, cu = wind_series(R[k])
    ax.fill_between(hours, 0, dp / 1e3, color=GREEN, lw=0, label=f"dispatched wind ({dp.sum()/1e6:.2f} TWh)", step="mid")
    ax.fill_between(hours, dp / 1e3, av / 1e3, color=RED, lw=0, alpha=0.85, label=f"curtailed wind ({cu.sum()/1e6:.2f} TWh, {100*cu.sum()/av.sum():.1f}%)", step="mid")
    ax.plot(hours, av / 1e3, color=INK, lw=1.0, label=f"available wind ({av.sum()/1e6:.2f} TWh)", drawstyle="steps-mid")
    ax.set(title=ttl, ylabel="GW"); ax.legend(loc="upper right", fontsize=8.5, ncol=3)
    ax.set_ylim(0, 40)
axs[1].set_xlabel("hour of week"); axs[1].set_xlim(0, 167)
fig.suptitle("Hourly wind generation: dispatched + curtailed = available (all 20 buses)", y=0.995)
fig.tight_layout(); save(fig, "fig03_wind_dispatched_vs_curtailed")

# zonal split of curtailment (supporting)
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), sharey=False)
for ax, z, nm in zip(axs, [NORTH, SOUTH], ["North (Bus1–5)", "South (Bus6–20)"]):
    av, dp, cu = wind_series(R["constrained_NTC"], z)
    ax.fill_between(hours, 0, dp / 1e3, color=GREEN, lw=0, step="mid", label="dispatched")
    ax.fill_between(hours, dp / 1e3, av / 1e3, color=RED, lw=0, alpha=0.85, step="mid", label="curtailed")
    ax.set(title=f"Constrained case – {nm}: curtailment {100*cu.sum()/av.sum():.1f}%", xlabel="hour of week", ylabel="GW")
    ax.legend(loc="upper right", fontsize=8); ax.set_xlim(0, 167)
fig.tight_layout(); save(fig, "fig04_wind_curtailment_by_zone")

# ---------------- 5. boundary link loading ----------------
import model as M
extra = {}
for fct, c in [(3.0, AQUA), (4.0, VIOLET)]:
    extra[fct] = M.solve(d, mode="ntc", link_scale=lambda l, f=fct: np.where(l["name"].isin(BOUNDARY), f, 1.0))
R["NTC_boundary_x3"] = extra[3.0]
def loading(o):
    return np.abs(o["f"].values) / o["F"]

fig, axs = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)
cols = [BLUE, ORANGE, AQUA, VIOLET, MAGENTA]
for ax, k, ttl in [(axs[0], "constrained_NTC", "(a) Base network (boundary 7.5 GW): all five boundary links saturated in all 168 h"),
                   (axs[1], "NTC_boundary_x3", "(b) Sensitivity: boundary capacity tripled (22.5 GW): boundary binds intermittently, downstream Bus10–Bus11 starts to bind")]:
    oo = R[k]; ld = pd.DataFrame(loading(oo), columns=oo["f"].columns)
    for i, (b, c) in enumerate(zip(BOUNDARY, cols)):
        lw = 2.6 if b == "Bus3-Bus8" else 1.3
        h99 = 100 * (ld[b] >= 0.99).mean()
        ax.plot(hours, ld[b] + (0.004 * (i - 2) if k == "constrained_NTC" else 0), color=c, lw=lw,
                label=f"{b} ({h99:.0f}% h ≥99%)", drawstyle="steps-mid")
    ax.plot(hours, ld["Bus10-Bus11"], color=INK2, lw=1.0, ls=":", label="Bus10–Bus11 (downstream)")
    ax.axhline(1.0, color=RED, lw=1, ls="--"); ax.text(168.5, 1.0, "limit", color=RED, va="center", fontsize=8)
    ax.set(ylim=(0, 1.05), ylabel="|flow| / capacity", title=ttl, xlim=(0, 167))
    ax.legend(loc="lower right" if k == "constrained_NTC" else "lower left", ncol=3, fontsize=8, framealpha=0.9, frameon=True)
axs[0].text(2, 0.55, "lines offset by ±0.008 for visibility;\nthe key link Bus3–Bus8 is drawn thick", fontsize=8, color=INK2)
axs[1].set_xlabel("hour of week"); axs[1].set_xlim(0, 167)
fig.tight_layout(); save(fig, "fig05_boundary_link_loading")

# heatmap all links
fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
for ax, k, ttl in zip(axs, ["constrained_NTC", "NTC_boundary_x3", "constrained_DCOPF"],
                      ["(a) NTC, base (7.5 GW)", "(b) NTC, boundary ×3 (22.5 GW)", "(c) DC-OPF (KVL), base"]):
    oo = R[k]; ld = loading(oo)
    im = ax.imshow(ld.T, aspect="auto", cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(oo["f"].columns))); lbls = ax.set_yticklabels(list(oo["f"].columns), fontsize=7.5)
    ax.set_title(ttl); ax.set_xlabel("hour of week"); ax.grid(False)
for t_ in axs[0].get_yticklabels():
    if t_.get_text() in BOUNDARY: t_.set_color(RED)
fig.colorbar(im, ax=axs, label="|flow| / capacity", shrink=0.8)
fig.suptitle("Hourly loading of all 23 links (boundary links in red)")
save(fig, "fig06_link_loading_heatmap")

# loading duration curves
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
sw_keys = [("constrained_NTC", "boundary 7.5 GW", BLUE), ("NTC_boundary_x2", "boundary 15 GW", ORANGE)]
for k, lab, c in sw_keys:
    ld = pd.DataFrame(loading(R[k]), columns=R[k]["f"].columns)
    axs[0].plot(np.linspace(0, 100, 168 * 5), np.sort(ld[BOUNDARY].values.ravel())[::-1], color=c, lw=2, label=lab)
    axs[1].plot(np.linspace(0, 100, 168), np.sort(ld["Bus10-Bus11"].values)[::-1], color=c, lw=2, label=lab)
for fct, c in [(3.0, AQUA), (4.0, VIOLET)]:
    ld = pd.DataFrame(loading(extra[fct]), columns=extra[fct]["f"].columns)
    axs[0].plot(np.linspace(0, 100, 168 * 5), np.sort(ld[BOUNDARY].values.ravel())[::-1], color=c, lw=2, label=f"boundary {7.5*fct:.1f} GW")
    axs[1].plot(np.linspace(0, 100, 168), np.sort(ld["Bus10-Bus11"].values)[::-1], color=c, lw=2, label=f"boundary {7.5*fct:.1f} GW")
for ax, t in zip(axs, ["(a) Five boundary links (pooled link-hours)", "(b) Downstream line Bus10–Bus11"]):
    ax.axhline(1, color=RED, ls="--", lw=1); ax.set(ylim=(0, 1.05), xlabel="% of hours exceeded", ylabel="|flow| / capacity", title=t)
    ax.legend(fontsize=8, loc="lower left")
axs[0].text(50, 1.02, "7.5 GW and 15 GW curves coincide at 1.0 (100% of hours)", ha="center", fontsize=8, color=INK2)
fig.suptitle("Loading-duration curves", y=1.02); fig.tight_layout(); save(fig, "fig07_loading_duration")

# ---------------- 8. nodal price heatmap ----------------
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
x4 = extra[4.0]
for ax, oo, ttl in zip(axs, [R["constrained_NTC"], R["NTC_boundary_x2"], x4],
                       ["(a) base (7.5 GW)", "(b) boundary ×2 (15 GW)", "(c) boundary ×4 (30 GW)"]):
    im = ax.imshow(oo["lmp"].values.T, aspect="auto", cmap=SEQ, vmin=0, vmax=6000, interpolation="nearest")
    ax.set_yticks(range(20)); ax.set_yticklabels(oo["buses"], fontsize=8); ax.set_title(ttl); ax.set_xlabel("hour of week"); ax.grid(False)
    ax.axhline(4.5, color=RED, lw=1.2, ls="--"); ax.axhline(9.5, color=ORANGE, lw=1, ls=":")
fig.colorbar(im, ax=axs, label="LMP (£/MWh)", shrink=0.85)
fig.suptitle("Locational marginal prices (dual of nodal balance). Red dash: Scotland–England boundary; orange dot: Bus10–Bus11")
save(fig, "fig08_lmp_heatmap")

# ---------------- 9. system dispatch stack ----------------
fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
for ax, k, ttl in zip(axs, ["constrained_NTC", "unconstrained_copperplate"], ["(a) Network-constrained", "(b) Copper plate"]):
    oo = R[k]; g = oo["gens"]; p = oo["p"].values
    nuc = p[:, (g.carrier == "nuclear").values].sum(axis=1); gas = p[:, (g.carrier == "gas").values].sum(axis=1)
    wnd = p[:, g.carrier.str.contains("wind").values].sum(axis=1); sto = (oo["dis"].values - oo["ch"].values).sum(axis=1)
    shed = oo["shed"].values.sum(axis=1)
    stack = [("nuclear", nuc, VIOLET), ("wind", wnd, GREEN), ("gas", gas, ORANGE), ("storage (net)", sto, MAGENTA), ("unserved (load shed)", shed, "#c9c8c2")]
    base = np.zeros(168)
    for nm, v, c in stack:
        ax.fill_between(hours, base / 1e3, (base + v) / 1e3, color=c, lw=0, label=f"{nm} {v.sum()/1e6:.2f} TWh", step="mid")
        base = base + v
    ax.plot(hours, oo["D"].sum(axis=1) / 1e3, color=INK, lw=1.2, label="demand", drawstyle="steps-mid")
    ax.set(title=ttl, ylabel="GW", ylim=(0, 175)); ax.legend(ncol=6, fontsize=7.5, loc="upper center", frameon=True, framealpha=0.95)
axs[1].set_xlabel("hour of week"); axs[1].set_xlim(0, 167)
fig.suptitle("System-wide hourly dispatch by carrier"); fig.tight_layout(); save(fig, "fig09_dispatch_stack")

# ---------------- 10. boundary sweep ----------------
sw = pd.read_csv(f"{OUT}/boundary_sweep.csv")
fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
axs[0].plot(sw.boundary_GW, sw.curt_pct, "-o", color=RED, lw=2, ms=6)
axs[0].set(xlabel="boundary capacity (GW)", ylabel="wind curtailment (%)", title="(a) Curtailment")
axs[0].axvline(7.5, color=INK2, ls=":", lw=1); axs[0].text(8, 55, "base", color=INK2, fontsize=8)
axs[1].plot(sw.boundary_GW, sw.shed_TWh, "-o", color=INK2, lw=2, ms=6)
axs[1].set(xlabel="boundary capacity (GW)", ylabel="unserved energy (TWh/week)", title="(b) Unserved energy")
axs[2].plot(sw.boundary_GW, sw.max_internal_hours99, "-o", color=ORANGE, lw=2, ms=6)
axs[2].set(xlabel="boundary capacity (GW)", ylabel="% hours ≥99% loaded", title="(c) Most-loaded internal line (Bus10–Bus11)")
fig.tight_layout(); save(fig, "fig10_boundary_sweep")

# ---------------- 11. electrolyser / flexibility ----------------
el = pd.read_csv(f"{OUT}/electrolyser_sweep.csv")
fig, axs = plt.subplots(1, 2, figsize=(12, 3.9))
axs[0].plot(el.electrolyser_GW, el.curt_pct, "-o", color=RED, lw=2, ms=6, label="northern electrolyser (GW)")
axs[0].plot(sw.boundary_GW - 7.5, sw.curt_pct, "-s", color=BLUE, lw=2, ms=6, label="extra boundary capacity (GW)")
axs[0].set(xlim=(-0.5, 31), xlabel="added sink / transfer capacity (GW)", ylabel="wind curtailment (%)",
           title="(a) Curtailment: electrolysis vs. reinforcement")
axs[0].legend(fontsize=8)
axs[1].bar(el.electrolyser_GW.astype(str), 100 * el.electrolyser_CF.fillna(0), color=AQUA, width=0.6)
for i, (cf, h2) in enumerate(zip(el.electrolyser_CF.fillna(0), el.H2_TWh_LHV)):
    if h2 > 0: axs[1].text(i, 100 * cf + 2, f"{h2:.2f}\nTWh H₂", ha="center", fontsize=7, color=INK2)
axs[1].set(ylim=(0, 118), xlabel="electrolyser capacity in North (GW)", ylabel="utilisation (%)", title="(b) Electrolyser utilisation and H₂ output (η = 70%)")
fig.tight_layout(); save(fig, "fig11_flexibility_electrolyser")

# ---------------- 12. validation ----------------
cf = pd.read_csv(f"{OUT}/closed_form_check.csv")
val = json.load(open(f"{OUT}/validation.json"))
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].plot(hours, cf.closed_form / 1e3, color=INK, lw=3, alpha=0.35, label="closed form max(0, W_N − D_N − 7.5 GW)")
axs[0].plot(hours, cf.lp / 1e3, color=RED, lw=1.2, label="LP (HiGHS)")
axs[0].set(xlabel="hour", ylabel="curtailment (GW)", title=f"(a) Curtailment: max |Δ| = {val['closed_form_max_abs_hourly_diff_MW']:.1e} MW")
axs[0].legend(fontsize=8)
oc = R["unconstrained_copperplate"]; g = oc["gens"]
W = oc["pmax"][:, g.carrier.str.contains("wind").values].sum(axis=1); Dt = oc["D"].sum(axis=1)
axs[1].scatter(Dt / 1e3, (Dt - W) / 1e3, s=10, color=BLUE, label="residual load after wind")
axs[1].scatter(Dt / 1e3, oc["shed"].values.sum(axis=1) / 1e3, s=10, color=ORANGE, label="LP load shed")
xx = np.linspace(Dt.min(), Dt.max(), 10)
axs[1].set(xlabel="total demand (GW)", ylabel="GW", title="(b) Copper plate: shed = residual − 14.2 GW firm")
axs[1].legend(fontsize=8)
# solver cross-check bars
pp = f"{OUT}/pypsa_crosscheck.json"
if os.path.exists(pp):
    pc = json.load(open(pp))
    labs = ["NTC", "copper plate", "boundary×2"]
    cust = [pc[k]["custom_curt_TWh"] for k in ["ntc", "copper", "x2"]]; pyp = [pc[k]["pypsa_curt_TWh"] for k in ["ntc", "copper", "x2"]]
    xi = np.arange(3)
    axs[2].bar(xi - 0.2, cust, 0.38, color=BLUE, label="custom LP (scipy/HiGHS)")
    axs[2].bar(xi + 0.2, pyp, 0.38, color=ORANGE, label=f"PyPSA {pc.get('pypsa_version','')} (linopy/HiGHS)")
    axs[2].set_xticks(xi); axs[2].set_xticklabels(labs); axs[2].set(ylabel="curtailed wind (TWh)", title="(c) Independent implementation cross-check")
    axs[2].legend(fontsize=8)
else:
    sim = R["constrained_NTC"]; ipm = R["constrained_NTC_ipm"]
    axs[2].plot(hours, wind_series(sim)[2] / 1e3, color=BLUE, lw=3, alpha=0.4, label="dual simplex")
    axs[2].plot(hours, wind_series(ipm)[2] / 1e3, color=ORANGE, lw=1, label="interior point")
    axs[2].set(title="(c) Simplex vs IPM", xlabel="hour", ylabel="curtailment (GW)"); axs[2].legend(fontsize=8)
fig.tight_layout(); save(fig, "fig12_validation")

# ---------------- 13. demand sensitivity ----------------
ds = pd.read_csv(f"{OUT}/demand_sensitivity.csv")
fig, ax = plt.subplots(figsize=(7.5, 4))
for (m, stc), c, ls, lab in [(("ntc", 1.0), RED, "-", "constrained, PHS ×1"), (("ntc", 10.0), RED, "--", "constrained, PHS ×10"),
                             (("copper", 1.0), BLUE, "-", "copper plate, PHS ×1"), (("copper", 10.0), BLUE, "--", "copper plate, PHS ×10")]:
    q = ds[(ds["mode"] == m) & (ds.storage_scale == stc)]
    ax.plot(q.demand_scale, q.curt_pct, ls, marker="o", color=c, lw=2, ms=5, label=lab)
ax.set(xlabel="demand scaling factor (1.0 = data as provided)", ylabel="wind curtailment (%)",
       title="Sensitivity: curtailment vs. demand level"); ax.legend(fontsize=8)
fig.tight_layout(); save(fig, "fig13_demand_sensitivity")

# ---------------- 14. curtailment by bus ----------------
o = R["constrained_NTC"]; g = o["gens"]; wm = g.carrier.str.contains("wind").values
av = o["pmax"][:, wm].sum(axis=0) / 1e3; cu = av - o["p"].values[:, wm].sum(axis=0) / 1e3
fig, ax = plt.subplots(figsize=(11, 3.6))
b = g.bus[wm].str.replace("Bus", "").values
ax.bar(b, av - cu, color=GREEN, width=0.7, label="dispatched"); ax.bar(b, cu, bottom=av - cu, color=RED, width=0.7, label="curtailed")
ax.set(xlabel="bus", ylabel="GWh / week", title="Constrained case: weekly wind energy by bus"); ax.legend()
fig.tight_layout(); save(fig, "fig14_curtailment_by_bus")
