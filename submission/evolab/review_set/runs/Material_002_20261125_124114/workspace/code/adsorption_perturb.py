"""Site-stability check: displace the adsorbate 0.2 A laterally (and tilt OH by 15 deg) from the relaxed fcc-hollow geometry, re-relax."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import get_calc, OUT
from ase.io import read
from ase.constraints import FixAtoms
from ase.optimize import BFGS
calc = get_calc("float64"); res = {}
for m in ["Ni", "Cu", "Rh", "Pd", "Ir", "Pt"]:
    for ads in ["O", "OH", "C"]:
        a = read(f"{OUT}/ads_{m}_{ads}.xyz"); n = 12
        a.set_constraint(FixAtoms(mask=a.get_tags() >= 2) if len(set(a.get_tags())) > 1 else FixAtoms(indices=list(range(8))))
        a.calc = calc; E0 = a.get_potential_energy(); x0 = a.positions[n].copy()
        a.positions[n:] += np.array([0.2 * np.cos(0.3), 0.2 * np.sin(0.3), 0])
        if ads == "OH": a.positions[n + 1] += np.array([0.98 * np.sin(np.radians(15)), 0, 0.98 * (np.cos(np.radians(15)) - 1)])
        BFGS(a, logfile=None).run(fmax=0.05, steps=500)
        E1 = a.get_potential_energy(); d = a.positions[n] - x0
        r = {"E_relaxed_minus_E_fcc": float(E1 - E0), "lateral_return_dist": float(np.linalg.norm(d[:2]))}
        if ads == "OH":
            v = a.positions[n + 1] - a.positions[n]; r["tilt_deg"] = float(np.degrees(np.arccos(v[2] / np.linalg.norm(v))))
        res[f"{m}_{ads}"] = r; print(m, ads, r, flush=True)
json.dump(res, open(f"{OUT}/adsorption_perturb.json", "w"), indent=1)
print("FINISHED")
