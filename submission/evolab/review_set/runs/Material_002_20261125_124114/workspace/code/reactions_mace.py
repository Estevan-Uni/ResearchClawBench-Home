"""Exp. 3: reaction barriers with MACE-MP-0b3-medium.
Part A: single points on the data-file 'simplified' geometries (as specified) + diagnostics.
Part B: relaxed endpoints + CI-NEB for Rxn 1, 11, 20."""
import sys, os, json, itertools, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import get_calc, OUT
from ase import Atoms
from ase.build import molecule
from ase.optimize import BFGS, FIRE
from ase.constraints import FixAtoms
from ase.io import write
try:
    from ase.mep import NEB
except ImportError:
    from ase.neb import NEB
calc = get_calc("float64")
REF = {"rxn1": 1.72, "rxn11": 1.74, "rxn20": 1.77}
# ---------------- Part A ----------------
D = {
 "rxn1": (["C"]*4+["H"]*4,
   [[0,0,0],[1.5,0,0],[1.5,1.5,0],[0,1.5,0],[-.5,-.5,0],[2,-.5,0],[2,2,0],[-.5,2,0]],
   [[0,0,0],[1.4,.2,0],[1.4,1.3,0],[0,1.5,0],[-.5,-.5,0],[1.9,-.3,0],[1.9,1.8,0],[-.5,2,0]]),
 "rxn11": (["C","H","H","H","O"],
   [[0,0,0],[0,1,0],[.9,-.5,0],[-.9,-.5,0],[1.2,0,0]],
   [[0,0,0],[0,1,0],[.9,-.5,0],[-.9,-.5,0],[1.5,0,0]]),
 "rxn20": (["C"]*3+["H"]*6,
   [[0,0,0],[1.5,0,0],[.75,1.3,0],[-.5,-.5,0],[2,-.5,0],[.75,2,0],[0,0,1],[1.5,0,1],[.75,1.3,1]],
   [[0,0,0],[1.5,0,0],[.75,1.3,0],[-.5,-.5,0],[2,-.5,0],[.75,2,0],[0,0,1.5],[1.5,0,1.5],[.75,1.3,1.5]]),
}
def mk(sym, pos):
    a = Atoms(sym, positions=pos); a.center(vacuum=8.0); return a
def mindist(a):
    d = a.get_all_distances(); np.fill_diagonal(d, 99); i, j = np.unravel_index(d.argmin(), d.shape)
    return float(d[i, j]), f"{a[i].symbol}{i}-{a[j].symbol}{j}"
partA = {}
for k, (sym, r, ts) in D.items():
    out = {}
    for lab, p in [("reactant", r), ("ts", ts)]:
        a = mk(sym, p); a.calc = calc
        e = a.get_potential_energy(); f = a.get_forces()
        md, pair = mindist(a)
        out[lab] = {"E": float(e), "fmax": float(np.linalg.norm(f, axis=1).max()), "min_dist": md, "min_pair": pair, "formula": a.get_chemical_formula()}
    out["barrier"] = out["ts"]["E"] - out["reactant"]["E"]; out["ref"] = REF[k]
    partA[k] = out
    print("A", k, round(out["barrier"], 3), out["reactant"]["fmax"], out["ts"]["fmax"], flush=True)
json.dump(partA, open(f"{OUT}/barriers_as_specified.json", "w"), indent=1)
# ---------------- Part B ----------------
def relax(a, fmax=0.01, steps=2000, fix=None):
    a = a.copy(); a.calc = calc
    if fix is not None: a.set_constraint(FixAtoms(indices=fix))
    BFGS(a, logfile=None).run(fmax=fmax, steps=steps)
    a.set_constraint(); a.calc = calc
    return a
def kabsch_rmsd(P, Q):
    P = P - P.mean(0); Q = Q - Q.mean(0)
    H = P.T @ Q; U, S, Vt = np.linalg.svd(H); dd = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, dd]) @ U.T
    return float(np.sqrt(((P @ Rm.T - Q) ** 2).sum(1).mean())), Rm
def align(prod, react):
    rm, Rm = kabsch_rmsd(prod.positions, react.positions)
    p = prod.copy(); p.positions = (prod.positions - prod.positions.mean(0)) @ Rm.T + react.positions.mean(0); return p, rm
def topo(a):
    C = [i for i in range(len(a)) if a[i].symbol == "C"]; H = [i for i in range(len(a)) if a[i].symbol == "H"]
    d = a.get_all_distances()
    hs = {c: [] for c in C}
    for h in H: hs[C[int(np.argmin(d[h, C]))]].append(h)
    bonds = {(i, j) for i in C for j in C if i < j and d[i, j] < 1.75}
    return C, hs, bonds, d
def best_mapping(react, prod, candidates):
    """candidates: iterable of lists 'order' with prod index for each reactant index."""
    best = (1e9, None)
    for mirror in [False, True]:
        P = prod.copy()
        if mirror: P.positions[:, 0] *= -1
        for order in candidates:
            rm, _ = kabsch_rmsd(P.positions[order], react.positions)
            if rm < best[0]: best = (rm, (mirror, list(order)))
    rm, (mirror, order) = best
    P = prod.copy()
    if mirror: P.positions[:, 0] *= -1
    P = P[order]
    P, rm = align(P, react)
    return P, rm
def perms_groups(groups):
    """groups: list of (react_idx_list, prod_idx_list); yields full order arrays."""
    n = sum(len(g[0]) for g in groups)
    for combo in itertools.product(*[itertools.permutations(g[1]) for g in groups]):
        order = [None] * n
        for (ri, _), pp in zip(groups, combo):
            for r_, p_ in zip(ri, pp): order[r_] = p_
        yield order
endpoints = {}
# ---- Rxn 1: cyclobutene -> gauche (s-cis) 1,3-butadiene ----
R1 = molecule("cyclobutene"); R1.center(vacuum=8.0); R1 = relax(R1)
bd = molecule("butadiene"); bd.center(vacuum=8.0)
Cb, hsb, bb, _ = topo(bd)
term = [c for c in Cb if len(hsb[c]) == 2]; inner = [c for c in Cb if len(hsb[c]) == 1]
t1 = term[0]; i1 = [c for c in inner if (min(t1, c), max(t1, c)) in bb][0]; i2 = [c for c in inner if c != i1][0]; t2 = [c for c in term if c != t1][0]
# rotate t2 side about i1-i2 axis to dihedral ~ +38 deg (gauche s-cis)
from ase.geometry.geometry import get_dihedrals
dih0 = bd.get_dihedral(t1, i1, i2, t2)
side = [i2, t2] + hsb[i2] + hsb[t2]
bd.set_dihedral(t1, i1, i2, t2, 38.0, indices=side)
P1 = relax(bd)
Cr, hsr, br, _ = topo(R1)
sp3 = [c for c in Cr if len(hsr[c]) == 2]; sp2 = [c for c in Cr if len(hsr[c]) == 1]
A, B = sp3
Pa = [c for c in sp2 if (min(A, c), max(A, c)) in br][0]; Qb = [c for c in sp2 if c != Pa][0]
cands = []
for (T1, I1, I2, T2) in [(t1, i1, i2, t2), (t2, i2, i1, t1)]:
    groups = [([A], [T1]), ([Pa], [I1]), ([Qb], [I2]), ([B], [T2]), (hsr[A], hsb[T1]), (hsr[Pa], hsb[I1]), (hsr[Qb], hsb[I2]), (hsr[B], hsb[T2])]
    cands += list(perms_groups(groups))
P1m, rm1 = best_mapping(R1, P1, cands); P1m.calc = calc
endpoints["rxn1"] = (R1, P1m, rm1, [(A, Pa), (Pa, Qb), (Qb, B)], (A, B))
# ---- Rxn 11: CH3O -> CH2O + H ----
R2 = molecule("CH3O"); R2.center(vacuum=8.0); R2 = relax(R2)
iC = [i for i in range(len(R2)) if R2[i].symbol == "C"][0]; iO = [i for i in range(len(R2)) if R2[i].symbol == "O"][0]
Hs = [i for i in range(len(R2)) if R2[i].symbol == "H"]
# leaving H: the one most anti to... choose any (C3v-ish); pick first
hl = Hs[0]; v = R2.positions[hl] - R2.positions[iC]; v /= np.linalg.norm(v)
P2 = R2.copy(); P2.positions[hl] = R2.positions[iC] + 3.0 * v
P2r = relax(P2)
h_fixed = False
if P2r.get_distance(iC, hl) < 2.5:
    P2r = relax(P2, fix=[hl]); h_fixed = True
P2m, rm2 = align(P2r, R2); P2m.calc = calc
endpoints["rxn11"] = (R2, P2m, rm2, [(iC, iO)] + [(iC, h) for h in Hs if h != hl], (iC, hl))
# ---- Rxn 20: cyclopropane -> propene ----
R3 = molecule("C3H6_D3h"); R3.center(vacuum=8.0); R3 = relax(R3)
P3 = molecule("C3H6_Cs"); P3.center(vacuum=8.0); P3 = relax(P3)
Cr, hsr, br, _ = topo(R3); Cp, hsp, bp, _ = topo(P3)
M = [c for c in Cp if len(hsp[c]) == 3][0]; X = [c for c in Cp if len(hsp[c]) == 1][0]; Y = [c for c in Cp if len(hsp[c]) == 2][0]
cands = []
for (c1, c2, c3) in itertools.permutations(Cr):
    for hm in hsr[c2]:
        hk = [h for h in hsr[c2] if h != hm]
        groups = [([c1], [M]), ([c2], [X]), ([c3], [Y]), (hsr[c1] + [hm], hsp[M]), (hk, hsp[X]), (hsr[c3], hsp[Y])]
        cands += [(o, (c1, c2, c3, hm)) for o in perms_groups(groups)]
best = None
for o, info in cands:
    for mirror in [1, -1]:
        Q = P3.copy(); Q.positions[:, 0] *= mirror
        rm, _ = kabsch_rmsd(Q.positions[o], R3.positions)
        if best is None or rm < best[0]: best = (rm, o, info, mirror)
rm3, o3, (c1, c2, c3, hm), mir = best
Q = P3.copy(); Q.positions[:, 0] *= mir; P3m, rm3 = align(Q[o3], R3); P3m.calc = calc
endpoints["rxn20"] = (R3, P3m, rm3, [(c1, c2), (c2, c3)], (c1, c3))
# ---- NEB ----
partB = {}
for k, (R, P, rm, keep_bonds, broken) in endpoints.items():
    R.calc = calc; P.calc = calc
    ER, EP = R.get_potential_energy(), P.get_potential_energy()
    info = {"formula_R": R.get_chemical_formula(), "formula_P": P.get_chemical_formula(), "fmax_R": float(np.linalg.norm(R.get_forces(), axis=1).max()),
            "fmax_P": float(np.linalg.norm(P.get_forces(), axis=1).max()), "map_rmsd": rm, "E_R": ER, "E_P": EP, "dE_rxn": EP - ER}
    if k == "rxn11": info["H_fixed_in_product"] = h_fixed; info["CH_dist_product"] = float(P.get_distance(*broken))
    nim = 9
    images = [R.copy()] + [R.copy() for _ in range(nim - 2)] + [P.copy()]
    for im in images: im.calc = get_calc("float64")
    neb = NEB(images, climb=False, method="improvedtangent", allow_shared_calculator=False)
    neb.interpolate("idpp")
    FIRE(neb, logfile=f"{OUT}/logs/neb_{k}.log").run(fmax=0.15, steps=600)
    neb.climb = True
    opt = FIRE(neb, logfile=f"{OUT}/logs/neb_{k}.log"); conv = opt.run(fmax=0.05, steps=1500)
    E = np.array([im.get_potential_energy() for im in images])
    fm = [float(np.linalg.norm(im.get_forces(), axis=1).max()) for im in images[1:-1]]
    # bond-length tracking along path
    bl = {f"{a}-{b}": [float(im.get_distance(a, b)) for im in images] for a, b in keep_bonds + [broken]}
    ci = int(np.argmax(E))
    info.update({"E_path": (E - E[0]).tolist(), "barrier": float(E.max() - E[0]), "reverse_barrier": float(E.max() - E[-1]), "climbing_image": ci,
                 "neb_converged": bool(conv), "neb_steps": int(opt.nsteps), "neb_fmax_proj": float(np.sqrt((neb.get_forces() ** 2).sum(axis=1).max())),
                 "bond_lengths": bl, "ref": REF[k]})
    partB[k] = info
    write(f"{OUT}/neb_{k}.xyz", images)
    print("B", k, {x: info[x] for x in ["formula_R", "formula_P", "fmax_R", "fmax_P", "map_rmsd", "dE_rxn", "barrier", "climbing_image", "neb_converged", "neb_fmax_proj"]}, flush=True)
    json.dump(partB, open(f"{OUT}/barriers_neb.json", "w"), indent=1)
print("FINISHED")
