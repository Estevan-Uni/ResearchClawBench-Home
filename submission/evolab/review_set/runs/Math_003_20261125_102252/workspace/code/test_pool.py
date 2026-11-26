import time, collections, random
from geo.problem import load_problems, Problem, build_diagram
from geo.dd import Engine
from geo.aux import generate
probs = load_problems()
for nm in ['imo_2004_p1','imo_2009_p2','imo_2018_p1','imo_2000_p6','imo_2011_p6','imo_2008_p6']:
    pr = Problem(nm, probs[nm]); P = build_diagram(pr, seed=0)
    t0=time.time(); C = generate(pr, P); tg=time.time()-t0
    cnt = collections.Counter(c['family'] for c in C)
    rng=random.Random(0); sample=rng.sample(C, min(30,len(C)))
    t0=time.time(); ns=0
    for c in sample:
        pr2 = pr.with_aux(c['clause']); P2=dict(P); P2['aux']=c['z']
        E=Engine(pr2,P2); g=E.saturate(); ns+= g is not None
    print(nm, len(P), 'pool', len(C), 'gen %.1fs'%tg, 'per-cand %.3fs'%((time.time()-t0)/len(sample)), 'succ', ns, dict(cnt))
