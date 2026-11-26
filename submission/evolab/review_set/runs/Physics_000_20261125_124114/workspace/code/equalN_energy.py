"""Equal-N energy test: annealed grown structure vs the ideal relaxed conservative
shell with the same number of atoms (same truncated LJ).  -> ../outputs/equalN_energy.json"""
import json, glob, os
from growth_analysis import summary
TARGET = {'E1': '(0, 2)', 'E2': '(1, 3)', 'E2b': '(1, 4)', 'E3a': '(1, 1)', 'E3b': '(1, 2)'}
out = []
for n, s in summary.items():
    fam = n.split('_')[0]
    if fam not in TARGET:
        continue
    for r in s['runs']:
        js = json.load(open('../outputs/growth/%s_s%d.json' % (n, r['seed'])))
        ndep = len(js['D']) - js['nseed']
        t = TARGET[fam]
        Nt = r['ideal'][t]['N']
        if ndep != Nt:
            continue
        out.append(dict(run=n, seed=r['seed'], conservative_shell=t, E_anneal=r['E_anneal'], E_grown=r['E_grown'],
                        E_ideal=r['ideal'][t]['E'], dE=r['E_anneal'] - r['ideal'][t]['E'],
                        grown_best=str(r['id_anneal']['best']), resolved=r['id_anneal']['resolved']))
json.dump(out, open('../outputs/equalN_energy.json', 'w'), indent=1)
for o in out:
    if not o['run'].startswith('E1'):
        print(o['run'], o['seed'], o['grown_best'], 'E_anneal-E_ideal(%s) = %.2f' % (o['conservative_shell'], o['dE']))
