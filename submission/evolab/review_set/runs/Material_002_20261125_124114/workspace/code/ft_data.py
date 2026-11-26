"""PBE/6-31G* labels for the few-shot fine-tuning demo (Rxn 1, cyclobutene ring opening).
Configurations: the 9 MACE CI-NEB images + Gaussian-rattled copies."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from pyscf_calc import pbe
from common import OUT
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
rng = np.random.default_rng(7)
imgs = read(f"{OUT}/neb_rxn1.xyz", ":")
confs = []
for i, a in enumerate(imgs):
    b = a.copy(); b.info = {"set": "test", "kind": "neb", "img": i, "sigma": 0.0}; confs.append(b)
for i, a in enumerate(imgs):
    for j, s in enumerate([0.02, 0.04, 0.06, 0.02, 0.04, 0.06]):
        b = a.copy(); b.positions += rng.normal(0, s, b.positions.shape)
        b.info = {"set": "train", "kind": "rattle", "img": i, "sigma": s}; confs.append(b)
# 6 rattled configs moved to the test set (one per sigma, images 1,4,7)
k = 0
for c in confs:
    if c.info["kind"] == "rattle" and c.info["img"] in (1, 4, 7) and k < 6 and (c.info["sigma"] in (0.04, 0.06)):
        c.info["set"] = "test"; k += 1
out = []
for n, c in enumerate(confs):
    e, f = pbe(c, grad=True)
    c.calc = SinglePointCalculator(c, energy=e, forces=f); out.append(c)
    print(n, c.info, round(e, 4), flush=True)
    write(f"{OUT}/ft_data_rxn1.extxyz", out)
print("FINISHED")
