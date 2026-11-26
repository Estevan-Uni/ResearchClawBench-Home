import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, IMG
from ase import Atoms
from ase.io import read
from ase.visualize.plot import plot_atoms
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 5, figsize=(17, 3.8))
w = read(f"{OUT}/water_L12.traj", 0); plot_atoms(w, ax[0], rotation="10x,10y", radii=0.4); ax[0].set_title("Exp.1: 32 H$_2$O, L=12 Å\n(initial grid, random orientations)", fontsize=9)
s = read(f"{OUT}/ads_Pt_OH.xyz"); s = s.repeat((2, 2, 1)); plot_atoms(s, ax[1], rotation="-75x", radii=0.8); ax[1].set_title("Exp.2: OH at fcc hollow, Pt(111) 2×2×3\n(shown repeated 2×2)", fontsize=9)
A = Atoms("C4H4", positions=[[0,0,0],[1.5,0,0],[1.5,1.5,0],[0,1.5,0],[-.5,-.5,0],[2,-.5,0],[2,2,0],[-.5,2,0]])
plot_atoms(A, ax[2], radii=0.5); ax[2].set_title("Exp.3 as specified: 'cyclobutene'\n(planar C$_4$H$_4$ square, C–C 1.5 Å)", fontsize=9)
for a_, k, t in [(ax[3], "rxn1", "relaxed cyclobutene → s-cis butadiene\n(MACE CI-NEB endpoints, overlaid)"), (ax[4], "rxn20_v2", "relaxed cyclopropane → propene\n(MACE CI-NEB endpoints, overlaid)")]:
    im = read(f"{OUT}/neb_{k}.xyz", ":"); both = im[0].copy(); P = im[-1].copy(); P.positions += [5.0, 0, 0]; both += P; both.cell = [0, 0, 0]; both.pbc = False
    plot_atoms(both, a_, rotation="20x,10y", radii=0.45); a_.set_title(t, fontsize=9)
for a_ in ax: a_.set_xticks([]); a_.set_yticks([])
plt.tight_layout(); plt.savefig(f"{IMG}/overview_structures.png", dpi=150); plt.close()
