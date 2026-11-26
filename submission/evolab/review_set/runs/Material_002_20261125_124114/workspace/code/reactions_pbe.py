"""usage: reactions_pbe.py out.json key:neb_file.xyz ...
PBE/6-31G* (PySCF) single points along the MACE CI-NEB paths and PBE-relaxed endpoints."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from pyscf_calc import pbe, PySCFPBE
from common import OUT
from ase.io import read
from ase.optimize import BFGS
outname = sys.argv[1]
res = {}
for spec in sys.argv[2:]:
    k, fn = spec.split(":")
    imgs = read(f"{OUT}/{fn}", ":")
    E = np.array([pbe(a) for a in imgs])
    r = {"E_path_pbe": (E - E[0]).tolist(), "barrier_pbe_on_mace_path": float(E.max() - E[0]), "dE_rxn_pbe_at_mace_geoms": float(E[-1] - E[0]), "imax_pbe": int(E.argmax())}
    # PBE relaxation of endpoints (looser fmax because of cost)
    ends = []
    for a in (imgs[0].copy(), imgs[-1].copy()):
        a.calc = PySCFPBE(); BFGS(a, logfile=None).run(fmax=0.03, steps=60); ends.append(a.get_potential_energy())
    r["E_R_pbe_relaxed_minus_mace_geom"] = float(ends[0] - E[0]); r["E_P_pbe_relaxed_minus_mace_geom"] = float(ends[1] - E[-1])
    r["dE_rxn_pbe_relaxed"] = float(ends[1] - ends[0]); r["barrier_pbe_upper_bound"] = float(E.max() - ends[0])
    res[k] = r; print(k, r, flush=True)
    json.dump(res, open(f"{OUT}/{outname}", "w"), indent=1)
print("FINISHED")
