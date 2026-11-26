"""Generate result-dependent text snippets from outputs (slow-deposition E2, E2b delta=0, E2b s1)."""
import json, os, glob
G = '../outputs/growth/'
def reid(name, s):
    f = G + 'reid_%s_s%d.json' % (name, s)
    return json.load(open(f))['X_anneal'] if os.path.exists(f) else None
eq = {(o['run'], o['seed']): o for o in json.load(open('../outputs/equalN_energy.json'))}
# slow E2
n = 'E2_d=0.03_T=0.22_tau=160_n=1'
r = reid(n, 0)
if r:
    best = max(r['f1'], key=r['f1'].get)
    e = eq.get((n, 0))
    txt = ('The slow-deposition run (τ_dep = 160 τ, four times slower; 1 run) also grows %s (annealed F1 %.2f, recall %.2f)%s. '
           'A four-fold slower flux therefore does not restore the conservative step on this seed.' %
           (best, r['f1'][best], r['cov'][best], (', %.1f ε above the ideal seed+(1,3) cluster' % e['dE']) if e else ''))
else:
    last = [json.loads(l) for l in open(G + n + '.log') if l.startswith('{')][-1]
    best = max(last['f1'], key=last['f1'].get)
    txt = ('The slow-deposition run (τ_dep = 160 τ, four times slower; 1 run, still running when the report was built; '
           'partial result after %d of 132 deposits) is also growing %s (quenched F1 %.2f, recall %.2f). '
           'A four-fold slower flux therefore does not restore the conservative step on this seed within the accessible time.'
           % (last['n'], best, last['f1'][best], last['cov'][best]))
open('txt_slow_e2.md', 'w').write(txt)
# E2b delta 0
n = 'E2b_d=0.0_T=0.22_tau=30_n=1_N=212'
r = reid(n, 0)
if r:
    best = max(r['f1'], key=r['f1'].get)
    e = eq.get((n, 0))
    s = '**%s** (F1 %.2f, recall %.2f, χ₂₃ %+.2f)%s' % (best, r['f1'][best], r['cov'][best], r.get('chi_23', 0),
                                                     ('; %.1f ε above ideal seed+(1,4) at equal N' % e['dE']) if e else '')
else:
    s = 'not finished'
open('txt_e2b_d0.md', 'w').write(s)
n = 'E2b_d=0.06_T=0.22_tau=30_n=2_N=192'
r = reid(n, 1)
s = ('%s again (F1 %.2f, recall %.2f, χ₂₃ %+.2f)' % (max(r['f1'], key=r['f1'].get), max(r['f1'].values()),
                                                   r['cov'][max(r['f1'], key=r['f1'].get)], r.get('chi_23', 0))) if r else 'not finished'
open('txt_e2b_s1.md', 'w').write(s)
print(open('txt_slow_e2.md').read()); print(open('txt_e2b_d0.md').read()); print(s)
