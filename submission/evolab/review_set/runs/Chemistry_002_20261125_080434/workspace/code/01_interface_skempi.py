"""Step 1: characterise the barnase-barstar interface and parse SKEMPI 2.0 data for 1BRS.

Outputs
  outputs/interface_residues.csv  per-residue contacts + buried SASA
  outputs/skempi_1brs.csv         parsed SKEMPI entries with ddG (kcal/mol)
"""
import numpy as np, pandas as pd
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Polypeptide import three_to_index, index_to_one
import copy, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDB = f"{ROOT}/data/1brs_AD.pdb"
OUT = f"{ROOT}/outputs"
R, T = 1.98720425864083e-3, 298.0  # kcal/(mol K)

s = PDBParser(QUIET=True).get_structure("c", PDB)[0]
for ch in s:
    for res in list(ch):
        if res.id[0] != " ":
            ch.detach_child(res.id)
# strip hydrogens
for ch in s:
    for res in ch:
        for a in list(res):
            if a.element == "H":
                res.detach_child(a.id)

atoms = [a for a in s.get_atoms()]
ns = NeighborSearch(atoms)
contacts = {}
for a, b in ns.search_all(5.0):
    ca, cb = a.get_parent().get_parent().id, b.get_parent().get_parent().id
    if ca == cb:
        continue
    for x, y in ((a, b), (b, a)):
        r = x.get_parent()
        key = (r.get_parent().id, r.id[1])
        contacts.setdefault(key, set()).add((y.get_parent().get_parent().id, y.get_parent().id[1]))

sr = ShrakeRupley()
sr.compute(s, level="R")
sasa_cplx = {(r.get_parent().id, r.id[1]): r.sasa for r in s.get_residues()}
sasa_free = {}
for cid in ("A", "D"):
    sub = copy.deepcopy(s)
    for other in [c.id for c in sub if c.id != cid]:
        sub.detach_child(other)
    sr.compute(sub, level="R")
    for r in sub.get_residues():
        sasa_free[(cid, r.id[1])] = r.sasa

rows = []
for r in s.get_residues():
    k = (r.get_parent().id, r.id[1])
    rows.append(dict(chain=k[0], resid=k[1], resname=r.get_resname(),
                     aa=index_to_one(three_to_index(r.get_resname())),
                     n_partner_res=len(contacts.get(k, [])),
                     sasa_free=sasa_free[k], sasa_cplx=sasa_cplx[k],
                     bsa=sasa_free[k] - sasa_cplx[k]))
df = pd.DataFrame(rows)
df["interface"] = df.n_partner_res > 0
df.to_csv(f"{OUT}/interface_residues.csv", index=False)
print("Residues per chain:", df.groupby("chain").size().to_dict())
print("Interface residues (5 A):", df[df.interface].groupby("chain").size().to_dict())
print("Total BSA (A^2): %.0f" % df.bsa.sum())
print(df[df.interface].sort_values(["chain", "resid"]).to_string())

# ---------------- SKEMPI -----------------
sk = pd.read_csv(f"{ROOT}/data/skempi_v2.csv", sep=";")
sk = sk[sk["#Pdb"] == "1BRS_A_D"].copy()
sk["T"] = pd.to_numeric(sk["Temperature"].astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(298)
sk["ddG"] = R * sk["T"] * np.log(sk["Affinity_mut_parsed"] / sk["Affinity_wt_parsed"])
sk["n_mut"] = sk["Mutation(s)_PDB"].str.count(",") + 1
sk["mutations"] = sk["Mutation(s)_PDB"]  # PDB numbering matches 1brs_AD.pdb
sk["is_ala"] = sk["mutations"].apply(lambda m: all(x[-1] == "A" for x in m.split(",")))
sk["salt"] = np.where(sk["Notes"].fillna("").str.contains("NaCl"), "low(1mM NaCl)", "standard")
keep = ["mutations", "n_mut", "is_ala", "iMutation_Location(s)", "Affinity_mut_parsed",
        "Affinity_wt_parsed", "T", "ddG", "Method", "Reference", "salt"]
sk[keep].to_csv(f"{OUT}/skempi_1brs.csv", index=False)
print("\nSKEMPI 1BRS entries:", len(sk), " singles:", (sk.n_mut == 1).sum(),
      " doubles:", (sk.n_mut == 2).sum())
print(sk[sk.n_mut == 1].groupby("mutations").ddG.agg(["mean", "std", "count"]).round(2))
