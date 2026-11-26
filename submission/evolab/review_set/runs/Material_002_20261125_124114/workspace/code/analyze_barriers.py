import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, IMG
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = json.load(open(f"{OUT}/barriers_as_specified.json")); B = json.load(open(f"{OUT}/barriers_neb.json"))
v2 = json.load(open(f"{OUT}/neb_rxn20_v2.json")) if os.path.exists(f"{OUT}/neb_rxn20_v2.json") else None
pbe = {}
for f in ["barriers_pbe_rxn1_11.json", "barriers_pbe_rxn20.json", "barriers_pbe_ft.json"]:
    if os.path.exists(f"{OUT}/{f}"): pbe.update({(k if f != "barriers_pbe_ft.json" else k + "_ft"): v for k, v in json.load(open(f"{OUT}/{f}")).items()})
keys = ["rxn1", "rxn11", "rxn20"]; names = {"rxn1": "Rxn 1\ncyclobutene→butadiene", "rxn11": "Rxn 11\nCH$_3$O→CH$_2$O+H", "rxn20": "Rxn 20\ncyclopropane→propene"}
REF = {k: A[k]["ref"] for k in keys}
final = {}
for k in keys:
    b = dict(B[k])
    if k == "rxn20" and v2 is not None:
        b["first_attempt"] = {x: B[k][x] for x in ["barrier", "neb_converged", "neb_fmax_proj"]}
        for x in ["E_path", "barrier", "reverse_barrier", "climbing_image", "neb_converged", "neb_fmax_proj", "neb_steps"]: b[x] = v2[x]
    final[k] = b
summ = {"as_specified": {k: {"barrier": A[k]["barrier"], "ref": REF[k], "fmax_R": A[k]["reactant"]["fmax"], "fmax_TS": A[k]["ts"]["fmax"],
                              "mind_R": A[k]["reactant"]["min_dist"], "mind_TS": A[k]["ts"]["min_dist"], "formula": A[k]["reactant"]["formula"]} for k in keys},
        "neb": {k: {x: final[k][x] for x in ["barrier", "reverse_barrier", "dE_rxn", "climbing_image", "neb_converged", "neb_fmax_proj", "fmax_R", "fmax_P", "formula_R", "formula_P", "map_rmsd"]} for k in keys}}
summ["MAE_as_specified"] = float(np.mean([abs(A[k]["barrier"] - REF[k]) for k in keys]))
summ["MAE_neb_vs_ref"] = float(np.mean([abs(final[k]["barrier"] - REF[k]) for k in keys]))
for k in keys:
    if k in pbe:
        summ["neb"][k].update({"pbe_on_mace_path": pbe[k]["barrier_pbe_on_mace_path"], "pbe_upper_bound": pbe[k]["barrier_pbe_upper_bound"], "dE_rxn_pbe_relaxed": pbe[k]["dE_rxn_pbe_relaxed"]})
pk = [k for k in keys if k in pbe]
if pk:
    summ["MAE_neb_vs_pbe_on_path"] = float(np.mean([abs(final[k]["barrier"] - pbe[k]["barrier_pbe_on_mace_path"]) for k in pk]))
    summ["MAE_dErxn_vs_pbe"] = float(np.mean([abs(final[k]["dE_rxn"] - pbe[k]["dE_rxn_pbe_relaxed"]) for k in pk]))
summ["order_mace_neb"] = sorted(keys, key=lambda k: final[k]["barrier"]); summ["order_ref"] = sorted(keys, key=lambda k: REF[k])
if "rxn1_ft" in pbe: summ["ft"] = pbe["rxn1_ft"]
json.dump(summ, open(f"{OUT}/barriers_summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
# Fig: as specified
x = np.arange(3); fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(x - 0.2, [A[k]["barrier"] for k in keys], 0.4, color="#1f5f8b", label="MACE-MP-0b3 on data-file geometries")
ax[0].bar(x + 0.2, [REF[k] for k in keys], 0.4, color="#bbbbbb", label="data-file reference")
for i, k in enumerate(keys): ax[0].text(i - 0.2, A[k]["barrier"] + (0.2 if A[k]["barrier"] > 0 else -0.6), f"{A[k]['barrier']:.2f}", ha="center", fontsize=8)
ax[0].axhline(0, color="k", lw=0.7); ax[0].set_xticks(x); ax[0].set_xticklabels([names[k] for k in keys], fontsize=8); ax[0].set_ylabel("E(TS) − E(R) (eV)"); ax[0].legend(fontsize=8); ax[0].set_title("As specified (single points)", fontsize=10)
w = 0.2
ax[1].bar(x - 1.5 * w, [A[k]["reactant"]["fmax"] for k in keys], w, color="#c0392b", label="max|F| reactant (eV/Å)")
ax[1].bar(x - 0.5 * w, [A[k]["ts"]["fmax"] for k in keys], w, color="#e67e22", label="max|F| 'TS' (eV/Å)")
ax2 = ax[1].twinx()
ax2.bar(x + 0.5 * w, [A[k]["reactant"]["min_dist"] for k in keys], w, color="#2e8b57", label="min distance reactant (Å)")
ax2.bar(x + 1.5 * w, [A[k]["ts"]["min_dist"] for k in keys], w, color="#7fbf7f", label="min distance 'TS' (Å)")
ax[1].set_yscale("log"); ax[1].set_ylabel("max |F| (eV/Å)"); ax2.set_ylabel("shortest interatomic distance (Å)"); ax2.set_ylim(0, 1.6)
ax[1].set_xticks(x); ax[1].set_xticklabels([f"{names[k]}\n[{A[k]['reactant']['formula']}]" for k in keys], fontsize=7.5)
h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[1].legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right"); ax[1].set_title("Diagnostics: inputs are not stationary points", fontsize=10)
plt.tight_layout(); plt.savefig(f"{IMG}/barriers_as_specified.png", dpi=150); plt.close()
# Fig: NEB profiles
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
for a_, k in zip(ax, keys):
    E = np.array(final[k]["E_path"]); s = np.linspace(0, 1, len(E))
    a_.plot(s, E, "o-", color="#1f5f8b", label=f"MACE-MP-0b3 CI-NEB: E$_a$={final[k]['barrier']:.2f} eV")
    if k in pbe:
        Ep = np.array(pbe[k]["E_path_pbe"]); a_.plot(np.linspace(0, 1, len(Ep)), Ep, "s--", color="#c0392b", ms=4, label=f"PBE/6-31G* on MACE path: max {pbe[k]['barrier_pbe_on_mace_path']:.2f} eV")
    if k == "rxn1" and "rxn1_ft" in pbe:
        Ef = np.array(json.load(open(f"{OUT}/neb_rxn1_ft.json"))["E_path"]); a_.plot(np.linspace(0, 1, len(Ef)), Ef, "^-", color="#2e8b57", ms=4, label=f"fine-tuned (N=48) CI-NEB: {Ef.max():.2f} eV")
        Efp = np.array(pbe["rxn1_ft"]["E_path_pbe"]); a_.plot(np.linspace(0, 1, len(Efp)), Efp, "v:", color="#8e44ad", ms=4, label=f"PBE on fine-tuned path: max {Efp.max():.2f} eV")
    a_.axhline(REF[k], color="gray", ls=":", lw=1, label=f"data-file reference {REF[k]:.2f} eV")
    a_.set_xlabel("reaction coordinate (image index, normalised)"); a_.set_ylabel("E − E$_R$ (eV)"); a_.set_title(names[k].replace("\n", ": "), fontsize=10); a_.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{IMG}/barriers_neb_profiles.png", dpi=150); plt.close()
# Fig: summary bars
fig, ax = plt.subplots(figsize=(7.5, 4))
sets = [("MACE CI-NEB", [final[k]["barrier"] for k in keys], "#1f5f8b")]
if len(pk) == 3: sets.append(("PBE/6-31G* max on MACE path", [pbe[k]["barrier_pbe_on_mace_path"] for k in keys], "#c0392b"))
sets.append(("data-file reference", [REF[k] for k in keys], "#bbbbbb"))
w = 0.8 / len(sets)
for j, (lab, vals, c) in enumerate(sets):
    ax.bar(x + (j - (len(sets) - 1) / 2) * w, vals, w, color=c, label=lab)
    for i, v in enumerate(vals): ax.text(i + (j - (len(sets) - 1) / 2) * w, v + 0.05, f"{v:.2f}", ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels([names[k] for k in keys], fontsize=8); ax.set_ylabel("forward barrier (eV)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{IMG}/barriers_summary.png", dpi=150); plt.close()
