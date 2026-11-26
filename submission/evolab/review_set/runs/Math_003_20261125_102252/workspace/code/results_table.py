"""Main results table: one checker-verified proof per solved problem (DD+AR proof, else the
first successful candidate under the learned ordering, else depth-2)."""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from eval_search import load_labels
from prior import first_rank, LogReg
from checker import check_proof, CheckError
from geo.problem import load_problems
probs = load_problems()
L = load_labels(); m = pickle.load(open('../outputs/prior_model.pkl', 'rb'))
d2 = {r['problem']: r for r in json.load(open('../outputs/depth2_learned.json'))} if os.path.exists('../outputs/depth2_learned.json') else {}
rows = []
os.makedirs('../outputs/final_proofs', exist_ok=True)
for nm in probs:
    r = dict(problem=nm)
    f = '../outputs/proofs_ddar/%s.json' % nm
    if os.path.exists(f):
        r.update(method='DD+AR', aux=None, rank=0)
    elif nm in L and L[nm]['y'].sum() > 0:
        sc = m.score(L[nm]['F']); order = np.argsort(-(sc + np.random.RandomState(0).rand(len(sc)) * 1e-9))
        i = [j for j in order if L[nm]['y'][j]][0]
        f = L[nm]['proof'][i]
        r.update(method='DD+AR + 1 aux', aux=L[nm]['cands'][i]['clause'], rank=int(list(order).index(i)) + 1)
    elif d2.get(nm, {}).get('solved'):
        f = '../outputs/proofs_depth2/%s_learned.json' % nm
        r.update(method='DD+AR + 2 aux (beam)', aux=d2[nm]['aux1'] + '; ' + d2[nm]['aux2'], rank=d2[nm]['tried'])
    else:
        r.update(method='unsolved'); rows.append(r); continue
    pf = json.load(open(f))
    try:
        info = check_proof(pf); r['checker'] = 'accepted'; r['nonintegral_angle_ar'] = info['nonintegral_ar']
    except CheckError as e:
        r['checker'] = 'REJECTED: %s' % e
    st = pf['steps']
    r.update(steps=len(st), premises=sum(s['rule'] == 'premise' for s in st), ar=sum(s['rule'] == 'ar' for s in st),
             dd=sum(s['rule'] not in ('premise', 'ar') for s in st), proof_file=f)
    json.dump(pf, open('../outputs/final_proofs/%s.json' % nm, 'w'))
    from geo.trace import proof_text
    open('../outputs/final_proofs/%s.txt' % nm, 'w').write(pf['statement'] + '\n\n' + proof_text(pf))
    rows.append(r)
    print(r)
json.dump(rows, open('../outputs/main_results.json', 'w'), indent=1)
print('solved', sum(r['method'] != 'unsolved' for r in rows), 'accepted', sum(r.get('checker') == 'accepted' for r in rows))
