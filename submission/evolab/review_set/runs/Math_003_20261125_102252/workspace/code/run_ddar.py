import sys, time, json
from geo.problem import load_problems, Problem, build_diagram
from geo.dd import Engine
probs = load_problems()
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
res = {}
tot = 0
for nm, t in probs.items():
    pr = Problem(nm, t)
    P = build_diagram(pr, seed=seed)
    t0 = time.time()
    E = Engine(pr, P, seed=seed)
    g = E.saturate()
    dt = time.time() - t0
    tot += g is not None
    res[nm] = dict(solved=g is not None, time=dt, facts=len(E.facts), rows=len(E.rows), rounds=E.rounds,
                   nsim=len(E.cand_sim))
    print('%-14s %-5s %6.2fs facts=%5d rows=%5d rounds=%2d sim=%d' % (nm, g is not None, dt, len(E.facts), len(E.rows), E.rounds, len(E.cand_sim)), flush=True)
print('solved', tot)
