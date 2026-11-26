"""Cross-diagram transfer: every successful auxiliary construction found on diagram seed 0
is re-tested on 4 fresh random diagrams (new coordinates, re-sampled until the goal holds)."""
import json, sys
from eval_search import load_labels
from geo.problem import load_problems, Problem, build_diagram
from geo import numeric as N
from geo.dd import Engine
from geo.trace import make_proof
from checker import check_proof, CheckError
probs = load_problems()
L = load_labels()
res = {}
for nm, d in L.items():
    for c, ok in zip(d['cands'], d['y']):
        if not ok:
            continue
        pr = Problem(nm, probs[nm]).with_aux(c['clause'])
        r = []
        for s in range(1, 5):
            try:
                P = build_diagram(pr, seed=s)
            except N.DegenerateError:
                r.append('nodiagram'); continue
            E = Engine(pr, P, seed=s); g = E.saturate()
            if g is None:
                r.append('fail'); continue
            try:
                check_proof(make_proof(E, g, pr, P)); r.append('ok')
            except CheckError:
                r.append('reject')
        res.setdefault(nm, []).append(dict(clause=c['clause'], family=c['family'], results=r))
        print(nm, c['clause'], r, flush=True)
extra = json.load(open('../outputs/depth2_learned.json')) if len(sys.argv) > 1 else []
for e in extra:
    if e.get('solved'):
        nm = e['problem']
        pr = Problem(nm, probs[nm]).with_aux(e['aux1']).with_aux(e['aux2'])
        r = []
        for s in range(1, 5):
            try:
                P = build_diagram(pr, seed=s)
                E = Engine(pr, P, seed=s); g = E.saturate()
                if g is None: r.append('fail'); continue
                check_proof(make_proof(E, g, pr, P)); r.append('ok')
            except (N.DegenerateError, CheckError) as ex:
                r.append('fail')
        res.setdefault(nm, []).append(dict(clause=e['aux1'] + '; ' + e['aux2'], family='depth2', results=r))
        print(nm, 'depth2', r)
json.dump(res, open('../outputs/transfer.json', 'w'), indent=1)
