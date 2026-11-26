"""Is entanglement the root cause of inconsistent per-attack performance?
(1) per-class entanglement score (max best-5-feature overlap with any other class) vs per-class F1
(2) pairwise overlap vs pairwise confusion rate
(3) per-class representation |corr| vs per-class F1 (3D-IDS Fig.2e/f analogue)
(4) duplicate-aware evaluation: macro-F1 on test flows whose feature vector never occurs in training."""
import sys, json, glob, os, numpy as np
sys.path.insert(0, '/workspace/code')
from common import *
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
style()
ent = json.load(open(f'{OUT}/entangle.json'))
M = np.array(ent['overlap_best5_F']); np.fill_diagonal(M, np.nan)
score = np.nanmax(M, 1)
RUN = f'{OUT}/runs'
meths = [('RF', 'rf'), ('MLP', 'mlp'), ('E-GraphSAGE', 'egs'), ('TGN', 'tgn'), ('3D-IDS', '3dids'), ('DIDS-MFL', 'dids_mfl')]
res = {'entanglement_score': dict(zip(CLASSES, score.tolist()))}
fig, axs = plt.subplots(1, 3, figsize=(14, 3.9))
att = [c for c in range(10) if c != BENIGN]
for j, (lab, m) in enumerate(meths):
    fs = sorted(glob.glob(f'{RUN}/known_{m}_s[0-9].json'))
    if not fs: continue
    R = [json.load(open(f)) for f in fs]
    pc = np.mean([r['test']['per_class_f1'] for r in R], 0)
    rho = spearmanr(score[att], pc[att]).correlation
    axs[0].scatter(score[att], pc[att], color=PAL[j], s=22, label=f'{lab} (ρ={rho:.2f})', zorder=3)
    cm = np.mean([np.array(r['test']['cm'], float) for r in R], 0)
    n = cm.sum(1)
    xs, ys = [], []
    for a in att:
        for b in att:
            if a < b:
                xs.append(M[a, b]); ys.append((cm[a, b] + cm[b, a]) / max(n[a] + n[b], 1))
    rho2 = spearmanr(xs, ys).correlation
    res[lab] = {'rho_entanglement_vs_classF1': float(rho), 'rho_pair_overlap_vs_confusion': float(rho2),
                'per_class_f1': dict(zip(CLASSES, pc.tolist()))}
    if lab in ('RF', 'E-GraphSAGE', 'DIDS-MFL'):
        axs[1].scatter(xs, np.array(ys) + 1e-4, color=PAL[j], s=16, label=f'{lab} (ρ={rho2:.2f})', alpha=0.85, zorder=3)
for c in att:
    axs[0].annotate(CLASSES[c], (score[c], 1.0 - 0.035 * (att.index(c) % 3)), fontsize=6, color=INK2, ha='center')
axs[0].set_xlabel('class entanglement score (max pair overlap, raw F)'); axs[0].set_ylabel('test F1 of the class (known)')
axs[0].legend(fontsize=7); axs[0].set_title('Class entanglement vs per-class F1 (Spearman ρ, 9 attacks)')
axs[1].set_yscale('log'); axs[1].set_xlabel('pair overlap (best-5 features)'); axs[1].set_ylabel('pair confusion rate (+1e-4)')
axs[1].legend(fontsize=7); axs[1].set_title('Pair overlap vs pair confusion (36 attack pairs)')
vs = json.load(open(f'{OUT}/viz_stats.json')) if os.path.exists(f'{OUT}/viz_stats.json') else {}
for j, lab in enumerate(['E-GraphSAGE', 'TGN', 'DIDS-MFL']):
    cc = vs.get('corr_by_class', {}).get(lab)
    if not cc or lab not in res: continue
    cl = [c for c in att if CLASSES[c] in cc]
    x = [cc[CLASSES[c]] for c in cl]; yv = [res[lab]['per_class_f1'][CLASSES[c]] for c in cl]
    rho = spearmanr(x, yv).correlation; res[lab]['rho_repr_corr_vs_classF1'] = float(rho)
    axs[2].scatter(x, yv, color=PAL[[m[0] for m in meths].index(lab)], s=22, label=f'{lab} (ρ={rho:.2f})', zorder=3)
axs[2].set_xlabel('mean |corr| of class representations'); axs[2].set_ylabel('test F1 of the class'); axs[2].legend(fontsize=7)
axs[2].set_title('Representation |corr| vs per-class F1')
plt.tight_layout(); plt.savefig(f'{IMG}/rootcause.png'); plt.close()

# ---- duplicate-aware evaluation
D = np.load(f'{OUT}/data.npz'); X = D['msg']; y = D['attack']
tr, va, te = chrono_split(len(y))
trset = set(map(bytes, X[tr]))
novel_vec = np.array([bytes(r) not in trset for r in X[te]])
res['test_novel_vector_frac'] = float(novel_vec.mean())
res['test_novel_vector_class_counts'] = np.bincount(y[te][novel_vec], minlength=10).tolist()
dup = {}
for lab, m in meths + [('TGN (20 ep)', 'tgn', ), ('3D-IDS (20 ep)', '3dids'), ('DIDS-MFL (20 ep)', 'dids_mfl')]:
    fs = sorted(glob.glob(f'{RUN}/known_{m}_s[0-9]{"_long" if "20 ep" in lab else ""}_testpred.npz'))
    if not fs: continue
    a, b = [], []
    for f in fs:
        P = np.load(f); assert (P['gid'] == te).all()
        p = P['pred'].astype(int)
        a.append(f1_score(y[te][~novel_vec], p[~novel_vec], average='macro', labels=range(10), zero_division=0))
        b.append(f1_score(y[te][novel_vec], p[novel_vec], average='macro', labels=range(10), zero_division=0))
    dup[lab] = {'macroF1_seen_vectors': float(np.mean(a)), 'macroF1_unseen_vectors': float(np.mean(b)),
                'std_unseen': float(np.std(b)), 'n_seeds': len(fs)}
res['duplicate_aware'] = dup
save_json(res, f'{OUT}/rootcause.json')
print(json.dumps({k: v for k, v in res.items() if k != 'entanglement_score'}, indent=1)[:4000])
