"""Step 7: unbound -> bound conformational change of barnase (1A2P) and barstar (1A19) relative to
the 1BRS complex: CA RMSD (all residues) and interface heavy-atom RMSD after optimal superposition."""
import os, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f"{ROOT}/outputs/docking/inputs"
iface = json.load(open(f"{ROOT}/outputs/restraints/true_interface.json"))


def atoms(fn, chain):
    d = {}
    for l in open(fn):
        if l.startswith("ATOM") and l[21] == chain:
            d[(int(l[22:26]), l[12:16].strip())] = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
    return d


def kabsch_rmsd(P, Q):
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return np.sqrt(((P @ R - Q) ** 2).sum(1).mean())


rows = []
for name, ch in (("barnase", "A"), ("barstar", "D")):
    b, u = atoms(f"{D}/ref_1brs.pdb", ch), atoms(f"{D}/{name}_unbound.pdb", ch)
    common = sorted(set(b) & set(u))
    ca = [k for k in common if k[1] == "CA"]
    ifa = [k for k in common if k[0] in iface[ch]]
    rows.append(dict(protein=name, n_CA=len(ca), CA_rmsd=kabsch_rmsd(np.array([u[k] for k in ca]), np.array([b[k] for k in ca])),
                     n_if_atoms=len(ifa), interface_heavy_rmsd=kabsch_rmsd(np.array([u[k] for k in ifa]), np.array([b[k] for k in ifa]))))
df = pd.DataFrame(rows)
df.to_csv(f"{ROOT}/outputs/conformational_change.csv", index=False)
print(df.round(2))
