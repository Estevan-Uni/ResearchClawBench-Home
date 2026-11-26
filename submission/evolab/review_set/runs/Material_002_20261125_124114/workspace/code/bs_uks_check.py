"""Broken-symmetry UKS-PBE/6-31G* check for diradicaloid images (restricted PBE overestimates singlet diradical energies)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import pyscf_calc
from pyscf import gto, dft
from common import OUT
from ase.io import read
HA = 27.211386245988
def energies(a):
    mol = gto.M(atom=[(s, p) for s, p in zip(a.get_chemical_symbols(), a.positions)], basis="6-31g*", unit="Angstrom", verbose=0)
    r = dft.RKS(mol).density_fit(); r.xc = "pbe"; r.grids.level = 2; er = r.kernel()
    u = dft.UKS(mol).density_fit(); u.xc = "pbe"; u.grids.level = 2
    mo = r.mo_coeff; occ = r.mo_occ; h = int(np.where(occ > 0)[0][-1]); l = h + 1
    ca, cb = mo.copy(), mo.copy(); th = np.pi / 4
    ca[:, h], ca[:, l] = np.cos(th) * mo[:, h] + np.sin(th) * mo[:, l], -np.sin(th) * mo[:, h] + np.cos(th) * mo[:, l]
    cb[:, h], cb[:, l] = np.cos(th) * mo[:, h] - np.sin(th) * mo[:, l], np.sin(th) * mo[:, h] + np.cos(th) * mo[:, l]
    oa = np.zeros_like(occ); oa[: h + 1] = 1
    dm = u.make_rdm1((ca, cb), (oa, oa)); eu = u.kernel(dm0=dm)
    s2 = u.spin_square()[0]
    return er * HA, eu * HA, float(s2)
res = {}
for key, fn, idxs in [("rxn20", "neb_rxn20_v2.xyz", [0, 3, 4, 5, 6, 7, 10]), ("rxn1", "neb_rxn1.xyz", [0, 2, 3]), ("rxn1_ft", "neb_rxn1_ft.xyz", [0, 3])]:
    im = read(f"{OUT}/{fn}", ":"); out = {}
    for i in idxs:
        er, eu, s2 = energies(im[i]); out[i] = {"E_RKS": er, "E_BSUKS": eu, "S2": s2}; print(key, i, round(eu - er, 3), round(s2, 2), flush=True)
    e0 = min(out[idxs[0]]["E_RKS"], out[idxs[0]]["E_BSUKS"])
    for i in idxs: out[i]["rel_RKS"] = out[i]["E_RKS"] - e0; out[i]["rel_BS"] = min(out[i]["E_RKS"], out[i]["E_BSUKS"]) - e0
    res[key] = out
    json.dump(res, open(f"{OUT}/bs_uks_check.json", "w"), indent=1)
print("FINISHED")
