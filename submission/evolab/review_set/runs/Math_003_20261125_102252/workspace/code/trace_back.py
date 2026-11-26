"""Traceback premise audit + automatic relaxation attempts.

For every solved IMO problem and every checker-accepted proof we list which
construction predicates of the ORIGINAL statement are cited by the proof
(premise steps) and which are not.  Where a predicate is unused, we try to
relax the statement (replace the construction by a weaker one that omits the
unused predicate), then (i) test the goal numerically on fresh random diagrams
of the relaxed statement, (ii) search for a proof of the relaxed statement
(re-using the same auxiliary construction first, then depth-1 search), (iii) run
the independent checker, (iv) test transfer to further diagrams.
A relaxation is reported as a *generalisation* only if (i)-(iii) all succeed.
"""
import glob
import json
import os
import sys

from geo.problem import load_problems, Problem, build_diagram
from geo import numeric as N
from geo.dd import Engine
from geo.trace import make_proof
from geo.aux import generate, heuristic_score
from checker import check_proof, CheckError

KINDS = ('coll', 'cong', 'perp', 'para', 'eqangle', 'cyclic', 'eqratio', 'rconst')

# construction -> {frozenset(unused predicate indices): replacement template}
# templates use the construction's own argument names
RELAX = {
    'midpoint': {frozenset([0]): 'on_bline {x} {a} {b}', frozenset([1]): 'on_line {x} {a} {b}', frozenset([0, 1]): 'free {x}'},
    'foot': {frozenset([1]): 'on_tline {x} {a} {b} {c}', frozenset([0]): 'on_line {x} {b} {c}'},
    'circle': {frozenset([0]): 'on_bline {x} {b} {c}', frozenset([1]): 'on_bline {x} {a} {b}'},
    'mirror': {frozenset([0]): 'on_circle {x} {b} {a}', frozenset([1]): 'on_line {x} {a} {b}'},
    'on_circle': {frozenset([0]): None},
    'on_line': {frozenset([0]): None},
    'on_tline': {frozenset([0]): None},
    'on_pline': {frozenset([0]): None},
    'on_bline': {frozenset([0, 1]): None},
    'on_dia': {frozenset([0]): None},
    'angle_bisector': {frozenset([0]): None},
    'on_aline': {frozenset([0]): None},
    'eqdistance': {frozenset([0]): None},
}


# predicates implied by the other predicates of the same construction (index in defs line 4)
REDUNDANT = {('on_bline', 1), ('iso_triangle', 0), ('orthocenter', 2), ('incenter', 2), ('excenter', 2),
             ('incenter2', 2), ('excenter2', 2), ('incenter2', 9), ('incenter2', 10), ('excenter2', 9),
             ('excenter2', 10), ('parallelogram', 2), ('parallelogram', 3), ('eq_triangle', 2), ('eq_triangle', 3),
             ('reflect', 2)}


def pred_index(cname, pred, args, prob, ci):
    from geo.problem import DEFS
    d = DEFS[cname]
    for cn, full in prob.clauses[ci].constructs:
        if cn != cname:
            continue
        sub = dict(zip(d.args, full))
        for i, (p, pa) in enumerate(d.preds):
            if p == pred and tuple(sub.get(x, x) for x in pa) == tuple(args):
                return i
    return None


def goal_closure(prob):
    deps = {}
    for cl in prob.clauses:
        a = set(x for _, full in cl.constructs for x in full) - set(cl.outs)
        for o in cl.outs:
            deps[o] = a
    seen, st = set(), list(prob.goal[1])
    while st:
        p = st.pop()
        if p not in seen:
            seen.add(p); st += list(deps.get(p, ()))
    return seen


def classify(prob, p):
    ci, cname, pred, args = p
    i = pred_index(cname, pred, args, prob, ci)
    if (cname, i) in REDUNDANT:
        return 'redundant (implied by the other predicates of its construction)'
    if not set(prob.clauses[ci].outs) & goal_closure(prob):
        return 'point irrelevant to the goal'
    return 'independent'


def proofs_for(nm):
    fs = []
    f = '../outputs/proofs_ddar/%s.json' % nm
    if os.path.exists(f):
        fs.append(f)
    fs += sorted(glob.glob('../outputs/proofs_aux/%s/*.json' % nm))
    fs += sorted(glob.glob('../outputs/proofs_aux_ext/%s/*.json' % nm))
    fs += sorted(glob.glob('../outputs/proofs_depth2/%s*.json' % nm))
    return fs


def audit(nm, text):
    prob = Problem(nm, text)
    ncl = len(prob.clauses)
    prem = [(ci, cn, p, tuple(a)) for ci, cn, p, a in prob.premises() if p in KINDS]
    res = []
    for f in proofs_for(nm):
        pf = json.load(open(f))
        try:
            check_proof(pf)
        except CheckError:
            continue
        used = set()
        for s in pf['steps']:
            if s['rule'] == 'premise' and s['src'] and s['src'][0] < ncl:
                used.add((s['src'][0], s['src'][1], s['src'][2], tuple(s['src'][3])))
        unused = [p for p in prem if p not in used]
        aux = pf['statement'].split('?')[0].split(';')[ncl:]
        res.append(dict(file=f, aux=[a.strip() for a in aux], n_steps=len(pf['steps']),
                        used=sorted(used), unused=unused))
    return prob, prem, res


def relaxations(prob, unused):
    """Yield (description, relaxed_text) for clauses whose predicates are (partly) unused."""
    body, goal = prob.text.split('?')
    clauses = [c.strip() for c in body.split(';') if c.strip()]
    from geo.problem import DEFS
    by_clause = {}
    for ci, cn, p, a in unused:
        by_clause.setdefault(ci, []).append((cn, p, a))
    for ci, items in sorted(by_clause.items()):
        cl = prob.clauses[ci]
        for k, (cname, full) in enumerate(cl.constructs):
            d = DEFS[cname]
            sub = dict(zip(d.args, full))
            inst = [(p, tuple(sub.get(x, x) for x in pa)) for p, pa in d.preds]
            un_idx = frozenset(i for i, (p, a) in enumerate(inst) if (cname, p, a) in [(c, pp, aa) for c, pp, aa in items])
            if not un_idx:
                continue
            table = RELAX.get(cname, {})
            if un_idx not in table:
                continue
            tpl = table[un_idx]
            names = dict(zip(['x', 'a', 'b', 'c', 'd', 'e'], full))
            parts = []
            for j, (cn2, full2) in enumerate(cl.constructs):
                if j == k:
                    if tpl is None:
                        continue
                    parts.append(tpl.format(**names))
                else:
                    parts.append(cn2 + ' ' + ' '.join(full2))
            if not parts:
                parts = ['free ' + cl.outs[0]]
            lhs = ' '.join(cl.outs)
            new_cl = '%s = %s' % (lhs, ', '.join(parts))
            new = clauses[:ci] + [new_cl] + clauses[ci + 1:]
            yield ('clause %d: "%s" -> "%s"' % (ci, clauses[ci], new_cl), '; '.join(new) + ' ? ' + goal.strip())


def test_relaxed(name, text, aux_clauses, n_numeric=30, search_budget=600):
    prob = Problem(name, text)
    # (i) numeric validity
    ok = tot = 0
    for s in range(500, 500 + 4 * n_numeric):
        if tot >= n_numeric:
            break
        try:
            P = build_diagram(prob, seed=s, check_goal=False, max_tries=60)
        except N.DegenerateError:
            continue
        tot += 1
        ok += N.check_pred(prob.goal[0], prob.goal[1], P, tol=1e-6)
    out = dict(text=text, numeric_n=tot, numeric_holds=ok)
    if tot == 0 or ok < tot:
        out['verdict'] = 'not a theorem (goal fails numerically)' if tot else 'degenerate (no diagram)'
        return out
    # (ii) proof: same aux first, then search
    P = build_diagram(prob, seed=0)
    tries = []
    pr_aux = prob
    for a in aux_clauses:
        pr_aux = pr_aux.with_aux(a)
    try:
        P2 = build_diagram(pr_aux, seed=0)
        E = Engine(pr_aux, P2)
        g = E.saturate()
        if g is not None:
            tries.append((pr_aux, P2, E, g, 'same aux'))
    except N.DegenerateError:
        pass
    if not tries:
        E = Engine(prob, P)
        g = E.saturate()
        if g is not None:
            tries.append((prob, P, E, g, 'DD+AR'))
    if not tries:
        C = generate(prob, P, quota=200)
        C.sort(key=lambda c: -heuristic_score(c, prob))
        for c in C[:search_budget]:
            pr2 = prob.with_aux(c['clause'])
            P2 = dict(P)
            P2['aux'] = c['z']
            E = Engine(pr2, P2)
            g = E.saturate()
            if g is not None:
                tries.append((pr2, P2, E, g, 'new aux: ' + c['clause']))
                break
    if not tries:
        out['verdict'] = 'numerically true but no proof found'
        return out
    pr2, P2, E, g, how = tries[0]
    pf = make_proof(E, g, pr2, P2)
    try:
        check_proof(pf)
        acc = True
    except CheckError as e:
        acc = False
    # (iv) transfer
    tr = 0
    for s in range(1, 6):
        try:
            Pt = build_diagram(pr2, seed=s)
            Et = Engine(pr2, Pt)
            gt = Et.saturate()
            if gt is not None:
                check_proof(make_proof(Et, gt, pr2, Pt))
                tr += 1
        except (N.DegenerateError, CheckError):
            pass
    out.update(proof=how, checker=acc, transfer='%d/5' % tr, n_steps=len(pf['steps']),
               verdict='GENERALISATION (verified)' if acc else 'proof rejected by checker')
    os.makedirs('../outputs/generalisations', exist_ok=True)
    json.dump(pf, open('../outputs/generalisations/%s.json' % name, 'w'))
    return out


def main():
    probs = load_problems()
    table = {}
    relax_results = []
    for nm, text in probs.items():
        prob, prem, res = audit(nm, text)
        if not res:
            continue
        best = min(res, key=lambda r: (len(r['used']), r['n_steps']))
        never_used = [p for p in prem if all(p in r['unused'] for r in res)]
        some_unused = [p for p in prem if any(p in r['unused'] for r in res)]
        fmt = lambda p: '%s %s' % (p[2], ' '.join(p[3]))
        table[nm] = dict(n_premises=len(prem), n_proofs=len(res),
                         min_used=len(best['used']), best_proof=best['file'], best_aux=best['aux'],
                         unused_in_best=[fmt(p) for p in best['unused']],
                         unused_in_best_class={fmt(p): classify(prob, p) for p in best['unused']},
                         unused_in_all=[fmt(p) for p in never_used],
                         unused_in_some=[fmt(p) for p in some_unused])
        print('%-14s premises=%2d proofs=%3d min_used=%2d unused(best)=%s' % (
            nm, len(prem), len(res), len(best['used']), table[nm]['unused_in_best']), flush=True)
        # relaxation attempts based on the proof with the most unused predicates
        seen = set()
        for r in sorted(res, key=lambda r: len(r['used']))[:3]:
            for desc, rtext in relaxations(prob, r['unused']):
                if rtext in seen:
                    continue
                seen.add(rtext)
                tag = '%s_relax%d' % (nm, len(seen))
                out = test_relaxed(tag, rtext, r['aux'])
                out.update(problem=nm, change=desc, from_proof=r['file'])
                relax_results.append(out)
                print('    %s | %s | numeric %d/%d | %s' % (tag, desc, out['numeric_holds'], out['numeric_n'],
                                                         out['verdict']), flush=True)
    json.dump(dict(table=table, relaxations=relax_results), open('../outputs/premise_usage.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
