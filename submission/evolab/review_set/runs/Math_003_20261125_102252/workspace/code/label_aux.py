"""Exhaustive depth-1 auxiliary-point labelling on IMO-AG-30.

For every problem not solved by DD+AR, every candidate of the (quota-capped) pool is
tried once; success/failure, time and features are stored, so that any ordering
(random / heuristic / learned) can be evaluated offline on identical labels.
Proofs of all successful candidates are traced back and written to disk.
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from geo.problem import load_problems, Problem, build_diagram
from geo.dd import Engine
from geo.aux import generate, features, heuristic_score
from geo.trace import make_proof

OUT = '../outputs/aux_labels'
PROOFS = '../outputs/proofs_aux'
QUOTA = int(os.environ.get('QUOTA', 200))
EXT = os.environ.get('EXT', '')
if EXT:
    OUT, PROOFS = OUT + '_ext', PROOFS + '_ext'
G = {}


def init(nm, text, coords):
    G['pr'] = Problem(nm, text)
    G['P'] = coords


def run_one(c):
    pr, P = G['pr'], G['P']
    pr2 = pr.with_aux(c['clause'])
    P2 = dict(P)
    P2['aux'] = c['z']
    t0 = time.time()
    try:
        E = Engine(pr2, P2)
        g = E.saturate()
    except Exception as e:  # report, never hide
        return dict(ok=False, err=repr(e), t=time.time() - t0)
    res = dict(ok=g is not None, t=time.time() - t0, nfacts=len(E.facts))
    if g is not None:
        pf = make_proof(E, g, pr2, P2)
        res['proof'] = pf
    return res


def main(names, workers=2):
    probs = load_problems()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PROOFS, exist_ok=True)
    for nm in names:
        if os.path.exists(os.path.join(OUT, nm + '.json')):
            continue
        pr = Problem(nm, probs[nm])
        P = build_diagram(pr, seed=0)
        C = generate(pr, P, quota=QUOTA)
        if EXT:  # only candidates not already labelled in the base pool
            base = json.load(open('../outputs/aux_labels/%s.json' % nm))
            seen = set((round(c['z'][0], 6), round(c['z'][1], 6)) for c in base['cands'])
            C = [c for c in C if (round(c['z'].real, 6), round(c['z'].imag, 6)) not in seen]
        F = features(C, pr, P)
        t0 = time.time()
        with mp.Pool(workers, initializer=init, initargs=(nm, probs[nm], P)) as pool:
            R = pool.map(run_one, C, chunksize=8)
        rec = []
        os.makedirs(os.path.join(PROOFS, nm), exist_ok=True)
        for i, (c, r) in enumerate(zip(C, R)):
            if 'proof' in r:
                json.dump(r.pop('proof'), open(os.path.join(PROOFS, nm, 'cand%05d.json' % i), 'w'))
            rec.append(dict(family=c['family'], clause=c['clause'], args=c['args'], z=[c['z'].real, c['z'].imag],
                            heur=heuristic_score(c, pr), **r))
        np.save(os.path.join(OUT, nm + '_feat.npy'), F)
        json.dump(dict(problem=nm, n=len(C), coords={k: [v.real, v.imag] for k, v in P.items()},
                       time=time.time() - t0, cands=rec), open(os.path.join(OUT, nm + '.json'), 'w'))
        print(nm, 'pool', len(C), 'succ', sum(r['ok'] for r in rec), 'errors', sum('err' in r for r in rec),
              '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    names = sys.argv[1:]
    main(names)
