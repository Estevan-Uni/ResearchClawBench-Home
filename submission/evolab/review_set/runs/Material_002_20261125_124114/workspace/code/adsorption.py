"""Exp. 2: O, OH, C adsorption at fcc hollow of fcc(111) 2x2x3 slabs with MACE-MP-0b3-medium (float64)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import get_calc, OUT
from ase import Atoms
from ase.build import fcc111, add_adsorbate, bulk, molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.eos import EquationOfState
calc = get_calc("float64")
LAT = {"Ni": 3.52, "Cu": 3.61, "Rh": 3.80, "Pd": 3.89, "Ir": 3.84, "Pt": 3.92}
EXT = {"Co": 3.54, "Ag": 4.09, "Au": 4.08}   # extension (not in data file); experimental fcc lattice constants
res = {"gas": {}, "metals": {}}
def relax(a, fmax=0.05, steps=500):
    a.calc = calc
    opt = BFGS(a, logfile=None); conv = opt.run(fmax=fmax, steps=steps)
    return a.get_potential_energy(), bool(conv), int(opt.nsteps)
def box(sym, pos):
    a = Atoms(sym, positions=pos, cell=[10, 10, 10], pbc=False); a.center(); return a
# gas references
g = box("O", [[0, 0, 0]]); g.calc = calc; res["gas"]["O"] = g.get_potential_energy()
g = box("C", [[0, 0, 0]]); g.calc = calc; res["gas"]["C"] = g.get_potential_energy()
g = box("OH", [[0, 0, 0], [0, 0, 1.0]]); res["gas"]["OH"], _, _ = relax(g, 0.01); res["gas"]["OH_bond"] = float(g.get_distance(0, 1))
for m in ["H2O", "H2", "O2", "CH4"]:
    g = molecule(m); g.cell = [10, 10, 10]; g.center(); res["gas"][m], _, _ = relax(g, 0.01)
print("gas", res["gas"], flush=True)
for met, a0 in list(LAT.items()) + list(EXT.items()):
    r = {"a_input": a0, "extension": met in EXT}
    # bulk EOS sanity check
    vols, ens = [], []
    for s in np.linspace(0.95, 1.05, 9):
        b = bulk(met, "fcc", a=a0 * s); b.calc = calc
        vols.append(b.get_volume()); ens.append(b.get_potential_energy())
    v0, e0, B = EquationOfState(vols, ens).fit(); r["a_model"] = float((4 * v0) ** (1 / 3)); r["B_GPa"] = float(B / 1e-21 * 1.602176634e-19 / 1e9 * 1e-9) if False else float(B * 160.21766)
    slab = fcc111(met, (2, 2, 3), a=a0, vacuum=10.0)
    slab.set_constraint(FixAtoms(mask=slab.get_tags() >= 2))
    clean = slab.copy(); r["E_slab"], r["conv_slab"], _ = relax(clean)
    for ads in ["O", "OH", "C"]:
        s = slab.copy()
        if ads == "OH":
            add_adsorbate(s, Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.98]]), 1.5, "fcc")
        else:
            add_adsorbate(s, ads, 1.5, "fcc")
        nads = len(s) - len(slab); x0 = s.positions[len(slab)].copy()
        E, conv, n = relax(s)
        x1 = s.positions[len(slab)]
        d = x1 - x0; d = s.cell.cartesian_positions(np.mod(s.cell.scaled_positions(d[None]) + 0.5, 1) - 0.5)[0] if False else d
        top = s.positions[[i for i in range(len(slab)) if slab.get_tags()[i] == 1], 2].max()
        out = {"E": E, "conv": conv, "nsteps": n, "lateral_drift": float(np.linalg.norm(d[:2])), "height": float(x1[2] - top)}
        if ads == "OH":
            v = s.positions[-1] - s.positions[-2]; out["OH_tilt_deg"] = float(np.degrees(np.arccos(v[2] / np.linalg.norm(v)))); out["OH_bond"] = float(np.linalg.norm(v))
        out["dE_gas"] = E - r["E_slab"] - res["gas"][ads]
        G = res["gas"]
        mref = {"O": G["H2O"] - G["H2"], "OH": G["H2O"] - 0.5 * G["H2"], "C": G["CH4"] - 2 * G["H2"]}[ads]
        out["dE_mol"] = E - r["E_slab"] - mref
        if ads == "O": out["dE_halfO2"] = E - r["E_slab"] - 0.5 * G["O2"]
        if ads == "OH": out["dE_halfO2_halfH2"] = E - r["E_slab"] - 0.5 * G["O2"] - 0.5 * G["H2"]
        r[ads] = out
        s.write(f"{OUT}/ads_{met}_{ads}.xyz")
    res["metals"][met] = r
    print(met, {k: (round(v["dE_gas"], 3) if isinstance(v, dict) else v) for k, v in r.items()}, flush=True)
    json.dump(res, open(f"{OUT}/adsorption.json", "w"), indent=1)
print("FINISHED")
