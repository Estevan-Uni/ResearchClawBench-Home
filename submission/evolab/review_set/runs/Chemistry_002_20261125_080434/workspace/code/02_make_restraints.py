"""Step 2: generate ambiguous interaction restraints (AIRs) for different information scenarios.

Scenarios (all written in HADDOCK .tbl format, outputs/restraints/):
  ideal      active = crystal interface residues (BSA>=20 A^2, rel. SASA>=15% in free form);
             passive = surface neighbours (<=6.5 A) of active residues
  hotspot    active = residues with experimental single-mutant ddG >= 2 kcal/mol (SKEMPI 2.0)
             filtered to rel. SASA>=15%; passive = surface neighbours
  noisy      hotspot active set + an equal number of decoy surface residues far (>12 A CA-CA) from
             the true interface on each partner (50% false-positive rate)
  onesided   barnase hotspot residues active; all barstar surface residues passive
  abinitio   no AIRs (centre-of-mass restraints only, set in the workflow)
"""
import os, json, copy
import numpy as np, pandas as pd
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.SASA import ShrakeRupley

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/outputs/restraints"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

# max ASA (Tien et al. 2013, theoretical) for relative accessibility
MAXASA = dict(ALA=129, ARG=274, ASN=195, ASP=193, CYS=167, GLN=225, GLU=223, GLY=104, HIS=224,
              ILE=197, LEU=201, LYS=236, MET=224, PHE=240, PRO=159, SER=155, THR=172, TRP=285,
              TYR=263, VAL=174)

s = PDBParser(QUIET=True).get_structure("c", f"{ROOT}/data/1brs_AD.pdb")[0]
for ch in s:
    for r in list(ch):
        if r.id[0] != " ":
            ch.detach_child(r.id)
        else:
            for a in list(r):
                if a.element == "H":
                    r.detach_child(a.id)

# relative SASA in the free (separated) partners
sr = ShrakeRupley()
rel = {}
free = {}
for cid in "AD":
    sub = copy.deepcopy(s)
    sub.detach_child("D" if cid == "A" else "A")
    sr.compute(sub, level="R")
    free[cid] = sub[cid]
    for r in sub[cid]:
        rel[(cid, r.id[1])] = r.sasa / MAXASA[r.get_resname()]

iface = pd.read_csv(f"{ROOT}/outputs/interface_residues.csv")
sk = pd.read_csv(f"{ROOT}/outputs/skempi_1brs.csv")

surface = {c: [k[1] for k in rel if k[0] == c and rel[k] >= 0.15] for c in "AD"}


def neighbours(cid, actives, cutoff=6.5):
    atoms = [a for a in free[cid].get_atoms()]
    ns = NeighborSearch(atoms)
    out = set()
    for r in free[cid]:
        if r.id[1] in actives:
            for a in r:
                for b in ns.search(a.coord, cutoff):
                    out.add(b.get_parent().id[1])
    return sorted(x for x in out if x not in actives and x in surface[cid])


def write_tbl(fname, act, pas):
    """Standard HADDOCK AIR: each active residue of one partner to all active+passive of the other."""
    lines = []
    for c1, c2 in (("A", "D"), ("D", "A")):
        for r in act[c1]:
            partners = sorted(set(act[c2]) | set(pas[c2]))
            if not partners:
                continue
            lines.append(f"assign (resid {r} and segid {c1})")
            lines.append("(\n" + "\n     or\n".join(f"  (resid {p} and segid {c2})" for p in partners) + "\n) 2.0 2.0 0.0\n")
    open(fname, "w").write("\n".join(lines))


def filt(cid, lst):
    return sorted(x for x in set(lst) if x in surface[cid])


scen = {}
# ideal
act = {c: filt(c, iface[(iface.chain == c) & (iface.bsa >= 20)].resid) for c in "AD"}
scen["ideal"] = (act, {c: neighbours(c, act[c]) for c in "AD"})

# hotspot: residue where any single substitution has mean ddG >= 2 kcal/mol
single = sk[sk.n_mut == 1].copy()
single["chain"] = single.mutations.str[1]
single["resid"] = single.mutations.str[2:-1].astype(int)
# mean over replicate measurements of the same substitution, then max over substitutions
hs = single.groupby(["chain", "resid", "mutations"]).ddG.mean().reset_index()
hs = hs.groupby(["chain", "resid"]).ddG.max().reset_index()
hs = hs[hs.ddG >= 2.0]
act = {c: filt(c, hs[hs.chain == c].resid) for c in "AD"}
scen["hotspot"] = (act, {c: neighbours(c, act[c]) for c in "AD"})

# noisy: add decoys far from the true interface
ca = {(c, r.id[1]): r["CA"].coord for c in "AD" for r in free[c]}
true_if = {c: set(iface[(iface.chain == c) & iface.interface].resid) for c in "AD"}
act_noisy = {}
for c in "AD":
    dmin = {r: min(np.linalg.norm(ca[(c, r)] - ca[(c, t)]) for t in true_if[c]) for r in surface[c]}
    far = [r for r in surface[c] if dmin[r] > 12]
    if len(far) < 2 * len(scen["hotspot"][0][c]):  # small protein: take the most distant residues
        far = sorted(surface[c], key=lambda r: -dmin[r])[:2 * len(scen["hotspot"][0][c])]
    dec = sorted(rng.choice(far, size=len(scen["hotspot"][0][c]), replace=False).tolist())
    act_noisy[c] = sorted(set(scen["hotspot"][0][c]) | set(dec))
scen["noisy"] = (act_noisy, {c: neighbours(c, act_noisy[c]) for c in "AD"})

# one-sided
scen["onesided"] = ({"A": scen["hotspot"][0]["A"], "D": []}, {"A": [], "D": surface["D"]})

summary = {}
for name, (act, pas) in scen.items():
    write_tbl(f"{OUT}/{name}.tbl", act, pas)
    summary[name] = dict(active=act, passive=pas)
    print(f"{name:9s} active A={act['A']} D={act['D']}  | n_passive A={len(pas['A'])} D={len(pas['D'])}")
open(f"{OUT}/hotspot_table.csv", "w").write(hs.to_csv(index=False))
json.dump(summary, open(f"{OUT}/scenarios.json", "w"), indent=1, default=int)
# also store true interface for precision analysis
json.dump({c: sorted(int(x) for x in true_if[c]) for c in "AD"}, open(f"{OUT}/true_interface.json", "w"))
json.dump({c: [int(x) for x in surface[c]] for c in "AD"}, open(f"{OUT}/surface.json", "w"))
