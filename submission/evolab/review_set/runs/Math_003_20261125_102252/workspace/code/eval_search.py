"""Evaluate candidate orderings (random / heuristic / learned) on the exhaustive
IMO depth-1 labels: rank of first success, solved-vs-budget curves."""
import glob
import json
import os
import pickle
import sys

import numpy as np

from prior import first_rank, p_solved_random, expected_first_rank_random

BUDGETS = [1, 3, 10, 30, 100, 300, 1000, 2000]


def load_labels():
    """Merge base (quota 200) and extended (quota 1500, small problems) pools."""
    L = {}
    for f in sorted(glob.glob('../outputs/aux_labels/*.json')):
        d = json.load(open(f))
        nm = d['problem']
        F = [np.load(f.replace('.json', '_feat.npy'))]
        cands = d['cands']
        pf = [('../outputs/proofs_aux/%s/cand%05d.json' % (nm, i)) if c['ok'] else None for i, c in enumerate(cands)]
        fe = f.replace('aux_labels', 'aux_labels_ext')
        if os.path.exists(fe):
            de = json.load(open(fe))
            if de['cands']:
                F.append(np.load(fe.replace('.json', '_feat.npy')))
                pf += [('../outputs/proofs_aux_ext/%s/cand%05d.json' % (nm, i)) if c['ok'] else None
                       for i, c in enumerate(de['cands'])]
                cands = cands + de['cands']
        F = np.vstack(F)
        y = np.array([c['ok'] for c in cands], float)
        heur = np.array([c['heur'] for c in cands])
        L[nm] = dict(F=F, y=y, heur=heur, cands=cands, t=np.array([c['t'] for c in cands]), proof=pf,
                     coords=d['coords'])
    return L


def evaluate(L, model=None, n_ddar=15, budgets=BUDGETS):
    rows = {}
    for nm, d in L.items():
        y = d['y']
        n, s = len(y), int(y.sum())
        r_h = first_rank(d['heur'], y)
        r_l = first_rank(model.score(d['F']), y) if model is not None else None
        rows[nm] = dict(pool=n, successes=s, rank_heuristic=r_h, rank_learned=r_l,
                        rank_random_expected=expected_first_rank_random(n, s) if s else None,
                        families=sorted(set(c['family'] for c, yy in zip(d['cands'], y) if yy)))
    curves = {}
    for name in ('random', 'heuristic', 'learned'):
        c = []
        for k in budgets:
            tot = n_ddar
            for nm, r in rows.items():
                if name == 'random':
                    tot += p_solved_random(r['pool'], r['successes'], k)
                else:
                    rk = r['rank_' + name]
                    tot += rk is not None and rk <= k
            c.append(tot)
        curves[name] = c
    return rows, curves


if __name__ == '__main__':
    L = load_labels()
    model = pickle.load(open(sys.argv[1], 'rb')) if len(sys.argv) > 1 else None
    rows, curves = evaluate(L, model)
    for nm, r in rows.items():
        print(nm, r)
    print(curves)
