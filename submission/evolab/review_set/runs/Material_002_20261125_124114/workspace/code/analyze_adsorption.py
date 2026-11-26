import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, IMG
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open(f"{OUT}/adsorption.json"))
M = d["metals"]; core = [m for m in M if not M[m]["extension"]]; ext = [m for m in M if M[m]["extension"]]
EXP_A = {"Ni": 3.524, "Cu": 3.615, "Rh": 3.803, "Pd": 3.891, "Ir": 3.839, "Pt": 3.924, "Co": 3.544, "Ag": 4.086, "Au": 4.078}  # room-T experimental fcc lattice constants (CRC Handbook)
def fit(x, y):
    r = stats.linregress(x, y); pred = r.intercept + r.slope * np.array(x)
    return {"slope": r.slope, "slope_se": r.stderr, "intercept": r.intercept, "r": r.rvalue, "MAE": float(np.mean(np.abs(pred - y))), "n": len(x)}
out = {"fits": {}, "table": {}}
for ref, kO, kOH, kC in [("gas", "dE_gas", "dE_gas", "dE_gas"), ("mol", "dE_mol", "dE_mol", "dE_mol")]:
    for lab, ms in [("core6", core), ("all9", core + ext), ("core6_minus_Ni", [m for m in core if m != "Ni"])]:
        x = np.array([M[m]["O"][kO] for m in ms]); y = np.array([M[m]["OH"][kOH] for m in ms]); z = np.array([M[m]["C"][kC] for m in ms])
        out["fits"][f"OH_vs_O_{ref}_{lab}"] = fit(x, y); out["fits"][f"C_vs_O_{ref}_{lab}"] = fit(x, z)
for m in M:
    out["table"][m] = {"a_input": M[m]["a_input"], "a_model": M[m]["a_model"], "a_exp": EXP_A[m], "dev_pct": 100 * (M[m]["a_model"] / EXP_A[m] - 1),
                       "dE_O_gas": M[m]["O"]["dE_gas"], "dE_OH_gas": M[m]["OH"]["dE_gas"], "dE_C_gas": M[m]["C"]["dE_gas"],
                       "dE_O_mol": M[m]["O"]["dE_mol"], "dE_OH_mol": M[m]["OH"]["dE_mol"], "dE_C_mol": M[m]["C"]["dE_mol"], "dE_O_halfO2": M[m]["O"]["dE_halfO2"],
                       "drift_O": M[m]["O"]["lateral_drift"], "drift_OH": M[m]["OH"]["lateral_drift"], "drift_C": M[m]["C"]["lateral_drift"],
                       "h_O": M[m]["O"]["height"], "h_C": M[m]["C"]["height"], "OH_tilt": M[m]["OH"]["OH_tilt_deg"], "OH_bond": M[m]["OH"]["OH_bond"],
                       "conv": all(M[m][a]["conv"] for a in ["O", "OH", "C"]), "extension": M[m]["extension"]}
# site-stability sensitivity: use lowest-energy OH after perturbation (tilted / off-hollow) where lower
if os.path.exists(f"{OUT}/adsorption_perturb.json"):
    pt = json.load(open(f"{OUT}/adsorption_perturb.json")); out["perturb"] = pt
    x = np.array([M[m]["O"]["dE_gas"] + min(0, pt[f"{m}_O"]["E_relaxed_minus_E_fcc"]) for m in core])
    y = np.array([M[m]["OH"]["dE_gas"] + min(0, pt[f"{m}_OH"]["E_relaxed_minus_E_fcc"]) for m in core])
    out["fits"]["OH_vs_O_gas_core6_lowestOH"] = fit(x, y); out["dE_OH_lowest"] = dict(zip(core, y.tolist()))
# reference-invariance: slope identical for gas vs mol reference
out["slope_gas_minus_mol"] = out["fits"]["OH_vs_O_gas_core6"]["slope"] - out["fits"]["OH_vs_O_mol_core6"]["slope"]
json.dump(out, open(f"{OUT}/adsorption_analysis.json", "w"), indent=1)
for k, v in out["fits"].items(): print(k, {a: round(b, 3) for a, b in v.items()})
for m, t in out["table"].items(): print(m, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in t.items()})
# ---- figures ----
cols = {m: "#1f5f8b" for m in core}; cols.update({m: "#e67e22" for m in ext})
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
x = np.array([M[m]["O"]["dE_gas"] for m in core]); y = np.array([M[m]["OH"]["dE_gas"] for m in core])
f = out["fits"]["OH_vs_O_gas_core6"]; xx = np.linspace(-5.3, -1.9, 10)
ax[0].plot(xx, f["intercept"] + f["slope"] * xx, color="#1f5f8b", lw=1.6, label=f"MACE fit, 6 metals: slope {f['slope']:.2f}±{f['slope_se']:.2f}, r={f['r']:.2f}")
b05 = np.mean(y - 0.5 * x)
ax[0].plot(xx, b05 + 0.5 * xx, color="gray", ls="--", lw=1.3, label="literature: bond-order scaling γ=0.5\n(Abild-Pedersen et al., PRL 99, 016105, 2007),\nintercept fitted to MACE points")
fn = out["fits"]["OH_vs_O_gas_core6_minus_Ni"]
ax[0].plot(xx, fn["intercept"] + fn["slope"] * xx, color="#1f5f8b", ls=":", lw=1.2, label=f"MACE fit without Ni: slope {fn['slope']:.2f}, r={fn['r']:.2f}")
for m in M:
    ax[0].scatter(M[m]["O"]["dE_gas"], M[m]["OH"]["dE_gas"], color=cols[m], s=45, zorder=3, marker="o" if m in core else "s")
    ax[0].annotate(m, (M[m]["O"]["dE_gas"], M[m]["OH"]["dE_gas"]), xytext=(4, 4), textcoords="offset points", fontsize=9)
if "dE_OH_lowest" in out:
    fl = out["fits"]["OH_vs_O_gas_core6_lowestOH"]
    for m in core:
        if out["dE_OH_lowest"][m] < M[m]["OH"]["dE_gas"] - 0.01:
            ax[0].scatter(M[m]["O"]["dE_gas"], out["dE_OH_lowest"][m], facecolors="none", edgecolors="#1f5f8b", s=45, zorder=3)
            ax[0].plot([M[m]["O"]["dE_gas"]] * 2, [M[m]["OH"]["dE_gas"], out["dE_OH_lowest"][m]], color="#1f5f8b", lw=0.6, ls=":")
    ax[0].plot(xx, fl["intercept"] + fl["slope"] * xx, color="#8e44ad", ls="-.", lw=1.1, label=f"fit using lowest OH (open symbols, tilted): slope {fl['slope']:.2f}, r={fl['r']:.2f}")
ax[0].set_xlabel("ΔE$_O$ (eV, vs. gas O atom)"); ax[0].set_ylabel("ΔE$_{OH}$ (eV, vs. gas OH)"); ax[0].legend(fontsize=7.2, loc="upper left"); ax[0].set_title("OH vs O scaling, fcc(111) fcc-hollow", fontsize=10)
fc = out["fits"]["C_vs_O_gas_core6"]; fca = out["fits"]["C_vs_O_gas_all9"]
z = np.array([M[m]["C"]["dE_gas"] for m in core])
ax[1].plot(xx, fc["intercept"] + fc["slope"] * xx, color="#1f5f8b", lw=1.4, label=f"6 metals: slope {fc['slope']:.2f}±{fc['slope_se']:.2f}, r={fc['r']:.2f}")
ax[1].plot(xx, fca["intercept"] + fca["slope"] * xx, color="#e67e22", lw=1.2, ls="--", label=f"9 metals: slope {fca['slope']:.2f}±{fca['slope_se']:.2f}, r={fca['r']:.2f}")
for m in M:
    ax[1].scatter(M[m]["O"]["dE_gas"], M[m]["C"]["dE_gas"], color=cols[m], s=45, zorder=3, marker="o" if m in core else "s")
    ax[1].annotate(m, (M[m]["O"]["dE_gas"], M[m]["C"]["dE_gas"]), xytext=(4, 4), textcoords="offset points", fontsize=9)
ax[1].set_xlabel("ΔE$_O$ (eV, vs. gas O atom)"); ax[1].set_ylabel("ΔE$_C$ (eV, vs. gas C atom)"); ax[1].legend(fontsize=8); ax[1].set_title("C vs O (different adsorbate valence)", fontsize=10)
plt.tight_layout(); plt.savefig(f"{IMG}/adsorption_scaling.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ms = list(M); xs = np.arange(len(ms))
ax.bar(xs - 0.2, [EXP_A[m] for m in ms], 0.4, color="#bbbbbb", label="experiment")
ax.bar(xs + 0.2, [M[m]["a_model"] for m in ms], 0.4, color="#1f5f8b", label="MACE-MP-0b3 (EOS)")
for i, m in enumerate(ms): ax.text(i + 0.2, M[m]["a_model"] + 0.01, f"{out['table'][m]['dev_pct']:+.1f}%", ha="center", fontsize=7.5)
ax.set_xticks(xs); ax.set_xticklabels(ms); ax.set_ylim(3.3, 4.3); ax.set_ylabel("fcc lattice constant (Å)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{IMG}/lattice_constants.png", dpi=150); plt.close()
