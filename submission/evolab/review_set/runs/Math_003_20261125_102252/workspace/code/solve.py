"""Run DD+AR on a problem (optionally with aux clauses) and write a proof JSON."""
import json, os, sys, time
from geo.problem import load_problems, Problem, build_diagram
from geo.dd import Engine, ALL_RULES
from geo.trace import make_proof, proof_text

def solve(prob, seed=0, rules=ALL_RULES, use_ar=True, coords=None, max_rounds=40):
    P = coords or build_diagram(prob, seed=seed)
    E = Engine(prob, P, seed=seed, rules=rules, use_ar=use_ar)
    g = E.saturate(max_rounds=max_rounds)
    return E, P, g

if __name__ == '__main__':
    probs = load_problems()
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for nm, t in probs.items():
        pr = Problem(nm, t)
        t0 = time.time()
        E, P, g = solve(pr)
        if g is None:
            continue
        pf = make_proof(E, g, pr, P)
        json.dump(pf, open(os.path.join(out, nm + '.json'), 'w'), indent=1)
        print(nm, 'steps', pf['stats'], '%.2fs' % (time.time() - t0))
