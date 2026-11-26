"""IMO 2004 P1: from the traceback premise audit to a more general theorem.

Variants (all with the same goal `coll p b c`):
  V0 original           o = midpoint of BC, M,N on circle(O,OB)
  V1 bline              o on the perpendicular bisector of BC (OB = OC kept, coll(O,B,C) dropped)
  V2 free-centre        o arbitrary point, M,N on circle(O,OB) (both midpoint predicates dropped)
  V3 only OM=ON         o arbitrary, M arbitrary on AB, N on AC with ON = OM (circle need not pass through B)
For each: numeric validity on many random diagrams, proof search (DD+AR + depth-1 aux),
independent checker, and transfer of the found construction to fresh diagrams.
"""
import json
import math
import os
import random
import sys

from geo.problem import Problem, build_diagram
from geo import numeric as N
from geo.dd import Engine
from geo.trace import make_proof, proof_text
from geo.aux import generate, heuristic_score
from checker import check_proof, CheckError

TAIL = ('r = angle_bisector r b a c, angle_bisector r m o n; o1 = circle o1 b m r; o2 = circle o2 c n r; '
        'p = on_circle p o1 r, on_circle p o2 r ? coll p b c')
VARIANTS = {
    'V0_original': 'a b c = triangle a b c; o = midpoint o b c; m = on_circle m o b, on_line m a b; '
                   'n = on_circle n o b, on_line n a c; ' + TAIL,
    'V1_bline': 'a b c = triangle a b c; o = on_bline o b c; m = on_circle m o b, on_line m a b; '
                'n = on_circle n o b, on_line n a c; ' + TAIL,
    'V2_free_centre': 'a b c = triangle a b c; o = free o; m = on_circle m o b, on_line m a b; '
                      'n = on_circle n o b, on_line n a c; ' + TAIL,
    'V3_only_OM_eq_ON': 'a b c = triangle a b c; o = free o; m = on_line m a b; '
                        'n = on_line n a c, on_circle n o m; ' + TAIL,
}
OUT = '../outputs/gen2004'


def numeric_validity(prob, n=200, seed0=1000):
    ok = tot = 0
    far = far_ok = 0
    rec = []
    for s in range(seed0, seed0 + 4 * n):
        if tot >= n:
            break
        try:
            P = build_diagram(prob, seed=s, check_goal=False, max_tries=50)
        except N.DegenerateError:
            continue
        tot += 1
        g = N.check_pred('coll', ['p', 'b', 'c'], P, tol=1e-6)
        ok += g
        # distance of O from line BC relative to |BC|
        b, c, o = P['b'], P['c'], P['o']
        dist = abs(((c - b).conjugate() * (o - b)).imag) / abs(c - b) / abs(c - b)
        ob_oc = abs(abs(o - b) - abs(o - c)) / abs(c - b)
        if dist > 0.2:
            far += 1
            far_ok += g
        rec.append(dict(seed=s, goal=bool(g), dist_O_BC=dist, OB_minus_OC=ob_oc))
    return dict(n=tot, holds=ok, far_from_BC=far, far_holds=far_ok, records=rec)


def search(prob, P, max_cands=3000):
    E = Engine(prob, P)
    g = E.saturate()
    if g is not None:
        return None, E, g, P, prob
    C = generate(prob, P, quota=200)
    C.sort(key=lambda c: -heuristic_score(c, prob))
    for i, c in enumerate(C[:max_cands]):
        pr2 = prob.with_aux(c['clause'])
        P2 = dict(P)
        P2['aux'] = c['z']
        E = Engine(pr2, P2)
        g = E.saturate()
        if g is not None:
            return (i, c), E, g, P2, pr2
    return None, None, None, None, None


def transfer(prob_with_aux, seeds):
    res = []
    for s in seeds:
        try:
            P = build_diagram(prob_with_aux, seed=s)
        except N.DegenerateError:
            res.append(dict(seed=s, built=False))
            continue
        E = Engine(prob_with_aux, P)
        g = E.saturate()
        ok = False
        if g is not None:
            pf = make_proof(E, g, prob_with_aux, P)
            try:
                check_proof(pf)
                ok = True
            except CheckError:
                ok = False
        res.append(dict(seed=s, built=True, proved=g is not None, checked=ok))
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    for vn, text in VARIANTS.items():
        prob = Problem('imo_2004_p1_' + vn, text)
        val = numeric_validity(prob)
        print(vn, 'numeric: %d/%d diagrams satisfy goal; O far from BC: %d/%d' %
              (val['holds'], val['n'], val['far_holds'], val['far_from_BC']), flush=True)
        P = build_diagram(prob, seed=0)
        found, E, g, P2, pr2 = search(prob, P)
        entry = dict(text=text, numeric={k: v for k, v in val.items() if k != 'records'})
        if g is None:
            entry['proved'] = False
            print('   no proof found', flush=True)
        else:
            pf = make_proof(E, g, pr2, P2)
            try:
                info = check_proof(pf)
                acc = True
            except CheckError as e:
                acc = False
                info = {'err': str(e)}
            json.dump(pf, open(os.path.join(OUT, vn + '_proof.json'), 'w'), indent=1)
            open(os.path.join(OUT, vn + '_proof.txt'), 'w').write(pr2.text + '\n\n' + proof_text(pf))
            used = sorted(set((s['src'][2], ' '.join(s['src'][3])) for s in pf['steps'] if s['rule'] == 'premise'))
            allp = sorted(set((p, ' '.join(a)) for ci, cn, p, a in pr2.premises()
                              if p in ('coll', 'cong', 'perp', 'para', 'eqangle', 'cyclic')))
            unused = [u for u in allp if u not in used]
            tr = transfer(pr2, range(1, 11)) if found else []
            entry.update(proved=True, checker_accept=acc, aux=found[1]['clause'] if found else None,
                         aux_rank=found[0] if found else None, n_steps=pf['stats']['n_steps'],
                         used_premises=used, unused_premises=unused,
                         transfer=dict(n=len(tr), proved=sum(t.get('checked', False) for t in tr)))
            print('   proof: aux=%s rank=%s steps=%d checker=%s unused=%s transfer=%d/%d' % (
                entry['aux'], entry['aux_rank'], entry['n_steps'], acc, unused,
                entry['transfer']['proved'], entry['transfer']['n']), flush=True)
        json.dump(val['records'], open(os.path.join(OUT, vn + '_numeric.json'), 'w'))
        summary[vn] = entry
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
