"""Is the ideal conservative structure (seed + (1,3) / seed + (1,4)) thermally stable at
T*=0.22 for 200 tau?  If yes, the class-raising outcome of growth is a growth-pathway
(kinetic) effect.  -> ../outputs/thermal_stability.json"""
import json, numpy as np, md_growth as md
J = json.load(open('../outputs/joint_seeds.json'))
cases = [('p_01112', [(0, 0), (0, 1), (1, 1), (1, 2)], (1, 3), 0.0, [(1, 3), (3, 1), (2, 2), (0, 3)]),
         ('p_01112', [(0, 0), (0, 1), (1, 1), (1, 2)], (1, 3), 0.03, [(1, 3), (3, 1), (2, 2), (0, 3)]),
         ('p_0111213', [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3)], (1, 4), 0.02, [(2, 3), (3, 2), (1, 4), (4, 1), (0, 4)])]
import os, sys
fn = '../outputs/thermal_stability.json'
out = json.load(open(fn)) if os.path.exists(fn) else []
done = {(o['shell'], o['delta']) for o in out}
for key, sp, shell, dl, cands in cases:
    if (str(shell), dl) in done:
        continue
    sd = J[key]['diams']; dn = sd[-1] * (1 + dl)
    E0, X, D, I = md.relaxed_template(sp + [shell], list(sd) + [dn])
    idf = md.Identifier(sp, sd, dn, cands)
    lay = np.where(I == len(sp))[0]
    r = md.run_growth(X, D, dn, 0, T=0.22, rng=7, anneal_tau=200.0, cool_tau=100.0)
    Xa = np.array(r['X_anneal'])
    ida = idf.identify(Xa, 0, lay)
    rec = dict(seed=key, shell=str(shell), delta=dl, E_ideal=E0, E_after=r['E_anneal'],
               f1={k: round(v, 3) for k, v in ida['f1'].items()}, cov={k: round(v, 3) for k, v in ida['cov'].items()})
    out.append(rec); print(rec, flush=True)
    json.dump(out, open(fn, 'w'), indent=1)
