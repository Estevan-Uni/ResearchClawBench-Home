"""Apply identical heads (Euclidean prototype, cosine prototype, logistic-regression fine-tune) to
saved novel-class embeddings of every model (4-way episodes, pool=all and dedup), so embedding quality
is compared independently of the head."""
import sys, glob, json, os, re, numpy as np
sys.path.insert(0, '/workspace/code')
from common import *
from fewshot_eval import evaluate, head_euclid, head_cosine
from sklearn.linear_model import LogisticRegression


def head_lr(S, Qe):
    N, K, d = S.shape
    mu = S.reshape(-1, d).mean(0); sd = S.reshape(-1, d).std(0) + 1e-6
    if K == 1:
        return head_cosine(S, Qe)
    lr = LogisticRegression(C=1.0, max_iter=300).fit((S.reshape(N * K, d) - mu) / sd, np.repeat(np.arange(N), K))
    return lr.predict_proba((Qe - mu) / sd)


LAB = {'mlp': 'MLP (supervised pretrain)', 'dids_mfl': 'DIDS-MFL', '3dids': '3D-IDS', 'tgn': 'TGN', 'egs': 'E-GraphSAGE', 'bsnet': 'BSNet-style',
       'mfl_noSD': 'DIDS-MFL w/o SD', 'mfl_noDiff': 'DIDS-MFL w/o diffusion', 'mfl_noEp': 'DIDS-MFL w/o episodes'}
ONLY = sys.argv[1] if len(sys.argv) > 1 else None
out = {}
os.makedirs(f'{OUT}/posthoc', exist_ok=True)
if ONLY in (None, 'raw'):
    Fq = np.load(f'{OUT}/Fq.npy'); Hq = np.load(f'{OUT}/Hq.npy'); raw = np.c_[Fq, Hq]
    for head_name, head in [('proto', head_euclid), ('LR', head_lr)]:
        r = evaluate(lambda gi: raw[gi], head, pools=('all', 'dedup'), settings=('4way',))
        out[f'Raw [F||h] + {head_name}'] = {f'{p}_K{K}': {'f1': [r[f'{p}_4way_K{K}']['f1']], 'nmi': [r[f'{p}_4way_K{K}']['nmi']]}
                                           for p in ('all', 'dedup') for K in (1, 5, 10)}
for m, lab in LAB.items():
    if ONLY is not None and m != ONLY: continue
    files = sorted(glob.glob(f'{OUT}/runs/fewshot_{m}_s[0-9]_emb.npz'))
    if not files: continue
    for head_name, head in [('proto', head_euclid), ('LR', head_lr)]:
        acc = {}
        for f in files:
            E = np.load(f); pos = {g: i for i, g in enumerate(E['gid'])}; Z = E['z']
            r = evaluate(lambda gi: Z[[pos[g] for g in gi]], head, pools=('all', 'dedup'), settings=('4way',), kmeans_ep=0)
            for p in ('all', 'dedup'):
                for K in (1, 5, 10):
                    a = acc.setdefault(f'{p}_K{K}', {'f1': [], 'nmi': []})
                    a['f1'].append(r[f'{p}_4way_K{K}']['f1']); a['nmi'].append(r[f'{p}_4way_K{K}']['nmi'])
        out[f'{lab} + {head_name}'] = acc
        print(lab, head_name, {k: round(np.mean(v['f1']), 3) for k, v in acc.items()}, flush=True)
save_json(out, f'{OUT}/posthoc/{ONLY or "all"}.json')
