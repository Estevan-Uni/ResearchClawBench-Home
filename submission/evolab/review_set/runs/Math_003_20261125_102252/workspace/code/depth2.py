"""Depth-2 beam search for problems unsolved with one auxiliary point.
Beam: top-B first-level candidates (by the given ordering) x top-W second-level
candidates (re-generated on the augmented diagram, name 'aux2')."""
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

from geo.problem import load_problems, Problem
from geo.dd import Engine
from geo.aux import generate, features, heuristic_score
from geo.trace import make_proof

G = {}


def init(text, coords):
    G['pr'] = Problem('d2', text)
    G['P'] = coords


def run(c):
    pr2 = G['pr'].with_aux(c['clause'])
    P2 = dict(G['P'])
    P2['aux2'] = c['z']
    E = Engine(pr2, P2)
    g = E.saturate()
    if g is None:
        return None
    return make_proof(E, g, pr2, P2)


def main(nm, order='learned', B=8, W=250, model_path='../outputs/prior_model.pkl', out='../outputs/proofs_depth2'):
    probs = load_problems()
    from eval_search import load_labels
    d = load_labels()[nm]
    F = d['F']
    model = pickle.load(open(model_path, 'rb')) if order == 'learned' else None
    pr = Problem(nm, probs[nm])
    P = {k: complex(*v) for k, v in d['coords'].items()}
    sc = model.score(F) if model is not None else np.array([c['heur'] for c in d['cands']])
    top = np.argsort(-sc)[:B]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    n_tried = 0
    for bi, i in enumerate(top):
        c1 = d['cands'][i]
        pr1 = pr.with_aux(c1['clause'])
        P1 = dict(P)
        P1['aux'] = complex(*c1['z'])
        C2 = generate(pr1, P1, name='aux2', quota=200)
        F2 = features(C2, pr1, P1)
        s2 = model.score(F2) if model is not None else np.array([heuristic_score(c, pr1) for c in C2])
        C2 = [C2[j] for j in np.argsort(-s2)[:W]]
        with mp.Pool(2, initializer=init, initargs=(pr1.text, P1)) as pool:
            for j, pf in enumerate(pool.imap(run, C2, chunksize=4)):
                n_tried += 1
                if pf is not None:
                    pool.terminate()
                    json.dump(pf, open(os.path.join(out, '%s_%s.json' % (nm, order)), 'w'))
                    res = dict(problem=nm, order=order, solved=True, beam_index=bi, second_rank=j + 1,
                               aux1=c1['clause'], aux2=C2[j]['clause'], tried=n_tried, time=time.time() - t0)
                    print(res, flush=True)
                    return res
        print(nm, 'beam', bi, 'done', n_tried, '%.0fs' % (time.time() - t0), flush=True)
    res = dict(problem=nm, order=order, solved=False, tried=n_tried, time=time.time() - t0)
    print(res, flush=True)
    return res


if __name__ == '__main__':
    order = sys.argv[1]
    allres = []
    outf = '../outputs/depth2_%s.json' % order
    if os.path.exists(outf):
        allres = json.load(open(outf))
    done = set(r['problem'] for r in allres)
    for nm in sys.argv[2:]:
        if nm in done:
            continue
        allres.append(main(nm, order))
        json.dump(allres, open(outf, 'w'), indent=1)
