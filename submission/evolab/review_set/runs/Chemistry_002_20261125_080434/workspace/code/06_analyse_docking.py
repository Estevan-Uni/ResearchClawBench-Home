"""Step 6: analyse the HADDOCK3 docking runs.

 - parses caprieval tables after rigid-body (it0) and after flexref+emref (refined)
 - assigns CAPRI quality classes (DockQ thresholds: >=0.23 acceptable, >=0.49 medium, >=0.80 high)
 - FCC clustering (Rodrigues et al. 2012) of refined models implemented in Python
   (cutoff 0.60, min. cluster size 2 for 20 refined models); clusters ranked by mean score of
   their top-4 members, as in HADDOCK
 - top-N success rates, score/i-RMSD funnels
 - unbound->bound conformational change of the partners
Outputs: outputs/docking_summary.csv, outputs/docking_models.csv, outputs/docking_clusters.csv
"""
import os, gzip, glob
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f"{ROOT}/outputs/docking"
RUNS = ["unbound_ideal", "unbound_hotspot", "unbound_noisy", "unbound_onesided", "unbound_abinitio",
        "bound_hotspot", "bound_abinitio"]


def capri_class(dq):
    return np.select([dq >= 0.80, dq >= 0.49, dq >= 0.23], ["high", "medium", "acceptable"], "incorrect")


def read_pdb(fn):
    op = gzip.open if fn.endswith(".gz") else open
    xyz, key = [], []
    with op(fn, "rt") as fh:
        for l in fh:
            if l.startswith("ATOM") and l[76:78].strip() != "H" and not l[12:16].strip().startswith("H"):
                xyz.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
                # segid (cols 73-76) is used by HADDOCK, fall back to chain id
                seg = l[72:76].strip() or l[21]
                key.append((seg, int(l[22:26])))
    return np.array(xyz), key


def contacts(fn, cut=5.0):
    xyz, key = read_pdb(fn)
    segs = np.array([k[0] for k in key])
    res = np.array([k[1] for k in key])
    a = segs == segs[0]
    ta, tb = cKDTree(xyz[a]), cKDTree(xyz[~a])
    pairs = ta.query_ball_tree(tb, cut)
    ra, rb = res[a], res[~a]
    return {(int(ra[i]), int(rb[j])) for i, js in enumerate(pairs) for j in js}


def fcc_cluster(cons, cutoff=0.6, min_size=2):
    """HADDOCK FCC clustering: greedy, largest neighbourhood first; FCC must hold in both directions."""
    n = len(cons)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = len(cons[i] & cons[j]) / max(len(cons[i]), 1)
    nb = (M >= cutoff) & (M.T >= cutoff)
    left = set(range(n))
    clusters = []
    while left:
        cnt = {i: sum(nb[i, j] for j in left) for i in left}
        c = max(cnt, key=lambda i: (cnt[i], -i))
        mem = [j for j in left if nb[c, j]]
        if len(mem) < min_size:
            break
        clusters.append(mem)
        left -= set(mem)
    lab = np.full(n, -1)
    for k, mem in enumerate(clusters):
        lab[mem] = k
    return lab


rows, models, clrows = [], [], []
for run in RUNS:
    rd = f"{D}/run_{run}"
    if not os.path.exists(f"{rd}/6_caprieval/capri_ss.tsv"):
        print("missing", run); continue
    for stage, cap in (("rigid", "2_caprieval"), ("refined", "6_caprieval")):
        t = pd.read_csv(f"{rd}/{cap}/capri_ss.tsv", sep="\t").sort_values("score").reset_index(drop=True)
        t["capri"] = capri_class(t.dockq)
        t["run"], t["stage"], t["rank"] = run, stage, np.arange(1, len(t) + 1)
        if stage == "refined":
            files = [f"{rd}/6_caprieval/{m}" for m in t.model]
            files = [f if os.path.exists(f) else f + ".gz" for f in files]
            cons = [contacts(f) for f in files]
            lab = fcc_cluster(cons)
            t["cluster"] = lab
            for k in sorted(set(lab) - {-1}):
                sub = t[t.cluster == k].sort_values("score")
                top4 = sub.head(4)
                clrows.append(dict(run=run, cluster=k, size=len(sub), score_top4=top4.score.mean(),
                                   score_top4_sd=top4.score.std(), irmsd=top4.irmsd.mean(),
                                   lrmsd=top4.lrmsd.mean(), fnat=top4.fnat.mean(), dockq=top4.dockq.mean(),
                                   best_capri=capri_class(top4.dockq.max())[()] if False else
                                   str(capri_class(np.array([top4.dockq.mean()]))[0])))
        models.append(t)
        succ = lambda n, q=0.23: bool((t.head(n).dockq >= q).any())
        rows.append(dict(run=run, stage=stage, n_models=len(t),
                         top1_dockq=t.dockq.iloc[0], top1_irmsd=t.irmsd.iloc[0], top1_lrmsd=t.lrmsd.iloc[0],
                         top1_fnat=t.fnat.iloc[0], top1_class=t.capri.iloc[0],
                         best_dockq=t.dockq.max(), best_irmsd=t.irmsd.min(),
                         frac_acceptable=(t.dockq >= 0.23).mean(), frac_medium=(t.dockq >= 0.49).mean(),
                         frac_high=(t.dockq >= 0.80).mean(),
                         T1=succ(1), T5=succ(5), T10=succ(10), T1_med=succ(1, .49), T5_med=succ(5, .49),
                         T10_med=succ(10, .49)))

summ = pd.DataFrame(rows)
mods = pd.concat(models)
cl = pd.DataFrame(clrows)
if len(cl):
    cl = cl.sort_values(["run", "score_top4"])
    cl["cluster_rank"] = cl.groupby("run").cumcount() + 1
summ.to_csv(f"{ROOT}/outputs/docking_summary.csv", index=False)
mods.to_csv(f"{ROOT}/outputs/docking_models.csv", index=False)
cl.to_csv(f"{ROOT}/outputs/docking_clusters.csv", index=False)
pd.set_option("display.width", 250)
print(summ.round(3).to_string())
print(cl.round(2).to_string())
