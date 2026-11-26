"""Robust CI-NEB re-run (more images, stiffer springs, small FIRE steps) for a given reaction; endpoints from first NEB.
usage: neb_retry.py rxnKEY nimages k tag [model_path]"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import get_calc, OUT, MODEL
from ase.io import read, write
from ase.optimize import FIRE
try:
    from ase.mep import NEB
except ImportError:
    from ase.neb import NEB
key, nim, kspr, tag = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
mp = sys.argv[5] if len(sys.argv) > 5 else MODEL
old = read(f"{OUT}/neb_{key}.xyz", ":")
R, P = old[0].copy(), old[-1].copy()
images = [R] + [R.copy() for _ in range(nim - 2)] + [P]
for im in images: im.calc = get_calc("float64", mp)
neb = NEB(images, k=kspr, climb=False, method="improvedtangent", allow_shared_calculator=False)
neb.interpolate("idpp")
def mind(a):
    D = a.get_all_distances(); np.fill_diagonal(D, 9); return float(D.min())
print("init min dists", [round(mind(a), 2) for a in images], flush=True)
log = f"{OUT}/logs/neb_{tag}.log"
FIRE(neb, maxstep=0.05, dt=0.05, logfile=log).run(fmax=0.1, steps=1500)
neb.climb = True
opt = FIRE(neb, maxstep=0.03, dt=0.05, logfile=log); conv = opt.run(fmax=0.05, steps=2000)
E = np.array([im.get_potential_energy() for im in images])
res = {"E_path": (E - E[0]).tolist(), "barrier": float(E.max() - E[0]), "reverse_barrier": float(E.max() - E[-1]), "dE_rxn": float(E[-1] - E[0]),
       "climbing_image": int(E.argmax()), "neb_converged": bool(conv), "neb_steps": int(opt.nsteps), "nimages": nim, "k": kspr,
       "neb_fmax_proj": float(np.sqrt((neb.get_forces() ** 2).sum(axis=1).max())), "min_dist_path": [mind(a) for a in images],
       "CC_dists": [[float(a.get_distance(i, j)) for i, j in [(0, 1), (0, 2), (1, 2)]] if key == "rxn20" else None for a in images], "model": os.path.basename(mp)}
json.dump(res, open(f"{OUT}/neb_{tag}.json", "w"), indent=1)
write(f"{OUT}/neb_{tag}.xyz", images)
print(json.dumps({k: v for k, v in res.items() if k not in ("CC_dists", "min_dist_path")}))
