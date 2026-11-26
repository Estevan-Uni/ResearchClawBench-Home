"""Synthetic data generation.

Corpus A (AlphaGeometry-style): random premises -> DD+AR saturation -> every
derived (non-premise) fact of type coll/para/perp/cong/cyclic/simtri is a synthetic
theorem; its proof is obtained by traceback.  Points used by the minimal proof but
not in the dependency closure of the conclusion are auxiliary points.
Corpus B (self-play): numerically true facts that DD+AR cannot prove ("hard
conjectures"); up to K aux candidates are tried and labelled success/failure.
"""
import itertools
import json
import math
import os
import random
import sys
import time

import numpy as np

from geo import numeric as N
from geo.problem import Problem, DEFS, _apply_clause
from geo.dd import Engine, fact_key
from geo.trace import Tracer
from geo.aux import generate, features

# (template, n_input_points); 'X' is the new point
POINT_CONS = [
    ('X = midpoint X {0} {1}', 2, 3),
    ('X = foot X {0} {1} {2}', 3, 3),
    ('X = circle X {0} {1} {2}', 3, 2),
    ('X = orthocenter X {0} {1} {2}', 3, 1),
    ('X = incenter X {0} {1} {2}', 3, 1),
    ('X = mirror X {0} {1}', 2, 1.5),
    ('X = reflect X {0} {1} {2}', 3, 1),
    ('X = parallelogram {0} {1} {2} X', 3, 1),
    ('X = on_line X {0} {1}, on_line X {2} {3}', 4, 3),
    ('X = on_line X {0} {1}, on_circle X {2} {3}', 4, 2),
    ('X = on_circle X {0} {1}, on_circle X {2} {3}', 4, 1),
    ('X = on_tline X {0} {1} {2}, on_line X {3} {4}', 5, 1.5),
    ('X = on_pline X {0} {1} {2}, on_line X {3} {4}', 5, 1.5),
    ('X = on_bline X {0} {1}, on_line X {2} {3}', 4, 1.5),
    ('X = angle_bisector X {0} {1} {2}, on_line X {3} {4}', 5, 1),
    ('X = on_circum X {0} {1} {2}, on_line X {3} {4}', 5, 2),
    ('X = on_dia X {0} {1}, on_line X {2} {3}', 4, 1),
    ('X = on_line X {0} {1}', 2, 1.5),
    ('X = on_circle X {0} {1}', 2, 1.5),
    ('X = on_circum X {0} {1} {2}', 3, 1.5),
    ('X = on_tline X {0} {1} {2}', 3, 0.7),
]
BASES = [('a b c = triangle a b c', 0.7), ('a b c d = quadrangle a b c d', 0.1),
         ('a b c = r_triangle a b c', 0.1), ('a b c = iso_triangle a b c', 0.1)]


def random_problem(rng, n_points):
    r = rng.random()
    acc = 0
    for base, w in BASES:
        acc += w
        if r <= acc:
            break
    clauses = [base]
    P = {}
    prob = Problem('synth', base)
    _apply_clause(prob.clauses[0], P, rng, DEFS)
    names = list(P)
    k = 0
    weights = [w for _, _, w in POINT_CONS]
    while len(names) < n_points:
        nm = 'p%d' % k
        ok = False
        for attempt in range(30):
            tpl, nin, _ = rng.choices(POINT_CONS, weights)[0]
            if nin > len(names):
                continue
            # recency-biased sampling of distinct argument points
            w = [1.0 + 2.0 * (i / max(1, len(names) - 1)) ** 2 for i in range(len(names))]
            args = []
            pool = list(names)
            ww = list(w)
            for _ in range(nin):
                x = rng.choices(range(len(pool)), ww)[0]
                args.append(pool.pop(x)); ww.pop(x)
            if args is None:
                continue
            cl = tpl.replace('X', nm).format(*args)
            try:
                pr = Problem('synth', cl, )
            except Exception:
                continue
            P2 = dict(P)
            try:
                _apply_clause(pr.clauses[0], P2, rng, DEFS)
                z = P2[nm]
                if not (abs(z) < 8) or min(abs(z - w) for k_, w in P.items()) < 2e-2:
                    raise N.DegenerateError('bad point')
                # all premises numerically valid
                full = Problem('synth', '; '.join(clauses + [cl]))
                for ci, cname, pred, pa in full.premises():
                    if ci == len(clauses) and not N.check_pred(pred, pa, P2, tol=1e-8):
                        raise N.DegenerateError('premise')
                # avoid accidental near-coincidences (collinear with a pair it was not built on)
            except (N.DegenerateError, ZeroDivisionError, ValueError, KeyError):
                continue
            P = P2
            clauses.append(cl)
            names.append(nm)
            ok = True
            break
        k += 1
        if not ok:
            return None, None
    return Problem('synth', '; '.join(clauses)), P


def dependency_closure(prob, pts):
    deps = {}
    for cl in prob.clauses:
        args = set(a for _, full in cl.constructs for a in full) - set(cl.outs)
        for o in cl.outs:
            deps[o] = args
    seen = set()
    stack = list(pts)
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack += list(deps.get(p, ()))
    return seen


def enumerate_theorems(E):
    """Facts in the deduction closure that are not premises."""
    th = []
    prem_keys = set(fact_key(f.kind, f.args, f.const) for f in E.facts if f.rule == 'premise')
    P = E.coords
    # DD facts (coll, cyclic, simtri, midp)
    for f in E.facts:
        if f.rule in ('premise', 'ar'):
            continue
        if f.kind in ('coll', 'cyclic', 'simtri', 'simtri2', 'midp', 'eqangle', 'eqratio'):
            th.append(('fact', f.id))
    # AR-closure para / perp / cong via the random-solution classes
    names = E.names
    xa, xb = E.ar['a'].xa, E.ar['a'].xb
    la, lb = E.ar['l'].xa, E.ar['l'].xb
    # numeric line id of each pair
    line_of = {}
    for L in E.nlines:
        for a, b in itertools.combinations(L, 2):
            line_of[(a, b)] = tuple(L)
    groups = {}
    lgroups = {}
    for i, j in itertools.combinations(range(len(names)), 2):
        a, b = names[i], names[j]
        v = E.th(a, b)
        groups.setdefault((xa[v], xb[v]), []).append((a, b))
        lgroups.setdefault((la[v], lb[v]), []).append((a, b))
    for key, segs in groups.items():
        if len(segs) < 2:
            continue
        lines = {}
        for a, b in segs:
            L = line_of.get((a, b), (a, b))
            d = N.ang_dir(P[b] - P[a])
            lines.setdefault(L, (a, b, d))
        Ls = list(lines.values())
        for (a, b, d1), (c, d, d2) in itertools.combinations(Ls, 2):
            t = (d1 - d2) % 1.0
            if min(t, 1 - t) < 1e-7:
                kind = 'para'
            elif abs(t - 0.5) < 1e-7:
                kind = 'perp'
            else:
                continue
            if fact_key(kind, (a, b, c, d)) in prem_keys:
                continue
            th.append((kind, (a, b, c, d)))
    for key, segs in lgroups.items():
        if len(segs) < 2:
            continue
        byl = {}
        for a, b in segs:
            byl.setdefault(round(abs(P[a] - P[b]), 7), []).append((a, b))
        for lst in byl.values():
            for (a, b), (c, d) in itertools.combinations(lst, 2):
                if fact_key('cong', (a, b, c, d)) in prem_keys:
                    continue
                th.append(('cong', (a, b, c, d)))
    return th


def canon_string(prob_text_premises, concl):
    order = {}
    for s in prob_text_premises + [concl]:
        for t in s.split()[1:]:
            order.setdefault(t, 'v%d' % len(order))
    ren = lambda s: ' '.join([s.split()[0]] + [order[t] for t in s.split()[1:]])
    return '|'.join(sorted(ren(s) for s in prob_text_premises)) + '=>' + ren(concl)


def corpus_A(seed, n_points=None, max_traced=400, check_aux_needed=3):
    rng = random.Random(seed)
    n_points = n_points or rng.randint(10, 14)
    prob, P = random_problem(rng, n_points)
    if prob is None:
        return None
    t0 = time.time()
    E = Engine(prob, P, seed=seed)
    E.saturate(stop_at_goal=False, max_rounds=30)
    tsat = time.time() - t0
    ths = enumerate_theorems(E)
    tr = Tracer(E)
    recs = []
    if len(ths) > max_traced:
        ths = rng.sample(ths, max_traced)
    n_aux_checked = 0
    for t in ths:
        if t[0] == 'fact':
            fid = t[1]
            f = E.facts[fid]
            kind, args = f.kind, f.args
        else:
            kind, args = t
            const = 0.5 if kind == 'perp' else 0
            from fractions import Fraction
            fid = E.arf(kind, args, Fraction(1, 2) if kind == 'perp' else 0)
        try:
            steps = tr.proof(fid)
        except RuntimeError:
            continue
        prem = [s for s in steps if s['rule'] == 'premise']
        prem_pts = set(x for s in prem for x in s['args'])
        clo = dependency_closure(prob, args)
        aux_pts = sorted(prem_pts - clo)
        prem_str = sorted(set('%s %s' % (s['kind'], ' '.join(s['args'])) for s in prem))
        concl = '%s %s' % (kind, ' '.join(args))
        import hashlib
        rec = dict(kind=kind, n_steps=len(steps), n_prem=len(prem),
                   n_dd=sum(s['rule'] not in ('premise', 'ar') for s in steps),
                   n_ar=sum(s['rule'] == 'ar' for s in steps), n_aux=len(aux_pts),
                   h=hashlib.md5(canon_string(prem_str, concl).encode()).hexdigest()[:12])
        # Is the aux point really needed?  Re-run DD+AR on the statement restricted to the closure.
        if aux_pts and n_aux_checked < check_aux_needed:
            n_aux_checked += 1
            keep = [cl for cl in prob.text.split(';') if set(cl.split('=')[0].split()) <= clo]
            sub = Problem('sub', '; '.join(keep) + ' ? ' + concl)
            Psub = {k: v for k, v in P.items() if k in clo}
            try:
                E2 = Engine(sub, Psub, seed=seed)
                g = E2.saturate()
                rec['aux_needed'] = g is None
            except Exception as e:
                rec['aux_needed'] = None
            if rec.get('aux_needed'):
                rec['aux_example'] = dict(statement='; '.join(keep), goal=concl, aux_pts=aux_pts,
                                          coords={k: [v.real, v.imag] for k, v in P.items()},
                                          aux_clauses=[cl.strip() for cl in prob.text.split(';')
                                                       if set(cl.split('=')[0].split()) & set(aux_pts)])
        recs.append(rec)
    return dict(seed=seed, n_points=len(P), n_facts=len(E.facts), t_sat=tsat, t_total=time.time() - t0,
                n_theorems=len(enumerate_theorems(E)) if False else None, n_enumerated=len(ths), theorems=recs)


def hard_conjectures(E, max_n=50):
    """Numerically true (tol 1e-10) facts that the saturated engine does not know."""
    P = E.coords
    names = E.names
    out = []
    for o, groups in E.eqd.items():
        for g in groups:
            for a, b in itertools.combinations(g, 2):
                if abs(abs(P[o] - P[a]) - abs(P[o] - P[b])) < 1e-10 and not E.k_cong(o, a, o, b):
                    out.append(('cong', (o, a, o, b)))
    for L in E.nlines:
        for a, b, c in itertools.combinations(L, 3):
            if not E.k_coll(a, b, c):
                out.append(('coll', (a, b, c)))
    for S, center in E.ncircles:
        for q in itertools.combinations(S, 4):
            a, b, c, d = q
            if not E.k_eqangle(c, a, c, b, d, a, d, b):
                out.append(('cyclic', q))
    pairs = list(itertools.combinations(names, 2))
    dirs = {(a, b): N.ang_dir(P[b] - P[a]) for a, b in pairs}
    for (a, b), (c, d) in itertools.combinations(pairs, 2):
        if len({a, b, c, d}) < 3:
            continue
        t = (dirs[(a, b)] - dirs[(c, d)]) % 1
        if min(t, 1 - t) < 1e-10 and not N.n_coll(P[a], P[b], P[c], 1e-6):
            if not E.k_para(a, b, c, d):
                out.append(('para', (a, b, c, d)))
        elif abs(t - 0.5) < 1e-10:
            if not E.k_para(a, b, c, d):
                out.append(('perp', (a, b, c, d)))
    return out[:max_n]


def corpus_B(seed, n_points=None, K=150, quota=60):
    rng = random.Random(seed)
    n_points = n_points or rng.randint(10, 14)
    prob, P = random_problem(rng, n_points)
    if prob is None:
        return None
    E = Engine(prob, P, seed=seed)
    E.saturate(stop_at_goal=False, max_rounds=30)
    conj = hard_conjectures(E)
    if not conj:
        return dict(seed=seed, n_conj=0)
    kind, args = rng.choice(conj)
    goal_text = prob.text + ' ? ' + kind + ' ' + ' '.join(args)
    gp = Problem('selfplay', goal_text)
    C = generate(gp, P, quota=quota, seed=seed)
    rng.shuffle(C)
    C = C[:K]
    F = features(C, gp, P)
    y = np.zeros(len(C), np.int8)
    for i, c in enumerate(C):
        pr2 = gp.with_aux(c['clause'])
        P2 = dict(P)
        P2['aux'] = c['z']
        try:
            E2 = Engine(pr2, P2, seed=seed)
            y[i] = E2.saturate() is not None
        except Exception as e:
            print('ERR', seed, c['clause'], repr(e), file=sys.stderr)
    return dict(seed=seed, n_conj=len(conj), goal=kind + ' ' + ' '.join(args), statement=prob.text,
                F=F, y=y, fam=[c['family'] for c in C])


if __name__ == '__main__':
    mode, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    out = sys.argv[4]
    os.makedirs(out, exist_ok=True)
    if mode == 'A':
        fh = open(os.path.join(out, 'A_%d_%d.jsonl' % (a, b)), 'w')
        for s in range(a, b):
            r = corpus_A(s)
            if r is not None:
                fh.write(json.dumps(r) + '\n')
                fh.flush()
    else:
        Fs, ys, groups, meta = [], [], [], []
        for s in range(a, b):
            r = corpus_B(s)
            if r is None:
                continue
            meta.append({k: v for k, v in r.items() if k not in ('F', 'y', 'fam')})
            if 'F' in r and len(r['y']):
                Fs.append(r['F']); ys.append(r['y']); groups.append(np.full(len(r['y']), s))
                fam = r['fam']
            if (s - a) % 10 == 9 or s == b - 1:
                if Fs:
                    np.savez_compressed(os.path.join(out, 'B_%d_%d.npz' % (a, b)), F=np.vstack(Fs),
                                        y=np.concatenate(ys), g=np.concatenate(groups))
                json.dump(meta, open(os.path.join(out, 'B_%d_%d_meta.json' % (a, b)), 'w'))
