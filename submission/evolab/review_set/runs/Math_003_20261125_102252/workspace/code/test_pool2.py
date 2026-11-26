import time, collections, random, sys
from geo.problem import load_problems, Problem, build_diagram
from geo.dd import Engine
from geo.aux import generate
probs = load_problems()
nm=sys.argv[1]
pr = Problem(nm, probs[nm]); P = build_diagram(pr, seed=0)
C = generate(pr, P, quota=200)
print(len(C), collections.Counter(c['family'] for c in C))
t0=time.time(); succ=[]
for i,c in enumerate(C):
    pr2 = pr.with_aux(c['clause']); P2=dict(P); P2['aux']=c['z']
    E=Engine(pr2,P2); g=E.saturate()
    if g is not None: succ.append(c['clause'])
print('time %.1f'%(time.time()-t0), 'succ', len(succ), succ[:10])
