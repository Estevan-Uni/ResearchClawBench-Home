"""Cross-coverage of identification templates on ideal relaxed shells (validation)."""
import json, numpy as np, md_growth as md
J = json.load(open('../outputs/joint_seeds.json'))
out = {}
cases = {'E1': ([(0,0),(0,1)], [1.0,1.0], 1.0, [(0,2),(1,1)]),
         'E3b': ([(0,0),(0,1),(1,1)], J['p_011']['diams'], J['p_011']['diams'][-1]*1.09, [(1,2),(2,1),(0,2)]),
         'E2_d0.08': ([(0,0),(0,1),(1,1),(1,2)], J['p_01112']['diams'], J['p_01112']['diams'][-1]*1.08, [(1,3),(3,1),(2,2),(0,3)]),
         'E2_d0.03': ([(0,0),(0,1),(1,1),(1,2)], J['p_01112']['diams'], J['p_01112']['diams'][-1]*1.03, [(1,3),(3,1),(2,2),(0,3)]),
         'E2b_d0.08': ([(0,0),(0,1),(1,1),(1,2),(1,3)], J['p_0111213']['diams'], J['p_0111213']['diams'][-1]*1.08, [(2,3),(3,2),(1,4),(4,1),(0,4)])}
for name,(sp,sd,dn,cands) in cases.items():
    idf = md.Identifier(sp, sd, dn, cands)
    res = {}
    for c in cands:
        E,X1,D,I = md.relaxed_template(sp+[c], list(sd)+[dn])
        res[str(c)] = idf.identify(X1, 0, np.where(I==len(sp))[0])
        print(name, c, {k: round(v,2) for k,v in res[str(c)]['cov'].items()}, {k:v for k,v in res[str(c)].items() if k.startswith('chi_') and 'n_' not in k})
    out[name] = res
json.dump(out, open('../outputs/template_crosscheck.json','w'), indent=1)
