"""Step 12: interface-residue precision/recall of refined models of the ideal-AIR run vs the true 1BRS interface
(tests whether incorrect top-ranked models are orientation decoys sitting on the correct binding faces)."""
import gzip, json, os
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
true = json.load(open(f"{ROOT}/outputs/restraints/true_interface.json"))
mods = pd.read_csv(f"{ROOT}/outputs/docking_models.csv")
rows = []
for run in ["unbound_ideal", "unbound_hotspot"]:
    t = mods[(mods.run == run) & (mods.stage == "refined")]
    for _, r in t.iterrows():
        fn = f"{ROOT}/outputs/docking/run_{run}/5_emref/" + os.path.basename(r.model) + ".gz"
        xyz, key = [], []
        for l in gzip.open(fn, "rt"):
            if l.startswith("ATOM") and l[76:78].strip() != "H":
                xyz.append([float(l[30:38]), float(l[38:46]), float(l[46:54])]); key.append((l[21], int(l[22:26])))
        xyz = np.array(xyz); ch = np.array([k[0] for k in key]); rs = np.array([k[1] for k in key])
        a = ch == "A"; pa, pb = set(), set()
        for i, js in enumerate(cKDTree(xyz[a]).query_ball_tree(cKDTree(xyz[~a]), 5.0)):
            if js:
                pa.add(int(rs[a][i])); pb.update(int(x) for x in rs[~a][js])
        ta, tb = set(true["A"]), set(true["D"])
        rows.append(dict(run=run, model=os.path.basename(r.model), rank=r["rank"], dockq=r.dockq, irmsd=r.irmsd,
                         lrmsd=r.lrmsd, air=r.air, cluster=r.cluster, capri=r.capri,
                         prec_A=len(pa & ta) / len(pa), rec_A=len(pa & ta) / len(ta),
                         prec_D=len(pb & tb) / len(pb), rec_D=len(pb & tb) / len(tb)))
df = pd.DataFrame(rows)
df.to_csv(f"{ROOT}/outputs/decoy_interface_check.csv", index=False)
print(df.groupby(["run", "capri"])[["prec_A", "rec_A", "prec_D", "rec_D", "air", "lrmsd"]].median().round(2))
