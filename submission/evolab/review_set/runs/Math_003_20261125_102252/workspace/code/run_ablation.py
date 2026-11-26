"""DD+AR over 5 diagram seeds, rule ablations and the DD-without-AR baseline."""
import json, sys, time
from geo.problem import load_problems, Problem, build_diagram
from geo.dd import Engine, ALL_RULES
from geo.trace import make_proof
from checker import check_proof, CheckError
probs = load_problems()
configs = {'full': dict(rules=ALL_RULES), 'no_AR (DD only)': dict(rules=ALL_RULES, ar_mode='uf')}
for r in ALL_RULES:
    configs['-' + r] = dict(rules=tuple(x for x in ALL_RULES if x != r))
seeds = range(5)
res = {}
for cname, cfg in configs.items():
    res[cname] = {}
    for s in seeds:
        solved = []
        for nm, t in probs.items():
            pr = Problem(nm, t)
            P = build_diagram(pr, seed=s)
            E = Engine(pr, P, seed=s, **cfg)
            g = E.saturate()
            if g is not None:
                pf = make_proof(E, g, pr, P)
                try:
                    check_proof(pf); solved.append(nm)
                except CheckError as e:
                    print('REJECT', cname, s, nm, e, flush=True)
        res[cname][s] = solved
        print(cname, s, len(solved), flush=True)
json.dump(res, open('../outputs/ddar_ablation.json', 'w'), indent=1)
