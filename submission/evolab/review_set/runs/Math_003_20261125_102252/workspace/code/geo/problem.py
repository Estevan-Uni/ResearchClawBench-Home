"""Parsing of defs.txt / problem files and numeric diagram construction."""
import os
import random

from . import numeric as N

DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


class Construction:
    def __init__(self, name, args, outputs, ndgs, preds, sketch):
        self.name, self.args, self.outputs = name, args, outputs
        self.ndgs, self.preds, self.sketch = ndgs, preds, sketch


def _parse_pred_list(line):
    preds = []
    for grp in line.split(';'):
        for item in grp.split(','):
            item = item.strip()
            if ':' in item:
                item = item.split(':', 1)[1].strip()
            if not item:
                continue
            toks = item.split()
            preds.append((toks[0], toks[1:]))
    return preds


def load_defs(path=None):
    path = path or os.path.join(DATA, 'defs.txt')
    lines = open(path).read().split('\n')
    defs = {}
    for i in range(0, len(lines) - 4, 6):
        head = lines[i].split()
        if not head:
            continue
        name, args = head[0], head[1:]
        outs = []
        for grp in lines[i + 1].split(','):
            if ':' in grp:
                outs += grp.split(':')[0].split()
        ndg_line = lines[i + 2]
        ndgs = _parse_pred_list(ndg_line.split('=', 1)[1]) if '=' in ndg_line else []
        preds = _parse_pred_list(lines[i + 3])
        defs[name] = Construction(name, args, outs, ndgs, preds, lines[i + 4].strip())
    return defs


DEFS = load_defs()


def load_problems(path=None):
    path = path or os.path.join(DATA, 'imo_ag_30.txt')
    lines = [l.strip() for l in open(path) if l.strip()]
    probs = {}
    for i in range(0, len(lines), 2):
        probs[lines[i].replace('translated_', '')] = lines[i + 1]
    return probs


class Clause:
    def __init__(self, outs, fixed, constructs):
        self.outs = outs          # list of point names
        self.fixed = fixed        # dict name -> complex (fixed coordinates)
        self.constructs = constructs  # list of (name, full_args)


class Problem:
    def __init__(self, name, text, defs=DEFS):
        self.name, self.text = name, text
        body, goal = text.split('?') if '?' in text else (text, '')
        self.clauses = []
        for cl in body.split(';'):
            cl = cl.strip()
            if not cl:
                continue
            lhs, rhs = cl.split('=', 1)
            outs, fixed = [], {}
            for tok in lhs.split():
                if '@' in tok:
                    nm, xy = tok.split('@')
                    x, y = xy.split('_')
                    fixed[nm] = complex(float(x), float(y))
                    tok = nm
                outs.append(tok)
            cons = []
            for part in rhs.split(','):
                toks = part.split()
                if not toks:
                    continue
                cname, given = toks[0], toks[1:]
                d = defs[cname]
                if len(given) == len(d.args):
                    full = given
                else:
                    missing = len(d.args) - len(given)
                    # fill output slots (in def-argument order) with clause outputs
                    full, oi, gi = [], 0, 0
                    out_slots = [a for a in d.args if a in d.outputs] or d.args
                    nfill = 0
                    for a in d.args:
                        if a in out_slots and nfill < missing:
                            full.append(outs[oi]); oi += 1; nfill += 1
                        else:
                            full.append(given[gi]); gi += 1
                cons.append((cname, full))
            self.clauses.append(Clause(outs, fixed, cons))
        g = goal.split()
        self.goal = (g[0], g[1:]) if g else None
        self.points = [p for c in self.clauses for p in c.outs]

    def premises(self, defs=DEFS):
        """List of (clause_index, construction_name, pred, args) premise predicates."""
        res = []
        for ci, cl in enumerate(self.clauses):
            for cname, full in cl.constructs:
                d = defs[cname]
                sub = dict(zip(d.args, full))
                for pred, pargs in d.preds:
                    res.append((ci, cname, pred, [sub.get(a, a) for a in pargs]))
        return res

    def with_aux(self, clause_text):
        """Return a new Problem with an extra clause inserted before the goal."""
        body, goal = self.text.split('?')
        return Problem(self.name, body.strip() + '; ' + clause_text + ' ? ' + goal.strip())


def build_diagram(prob, seed=0, defs=DEFS, max_tries=400, check_goal=True, rng=None):
    """Sample coordinates for all points. Returns dict name -> complex."""
    rng = rng or random.Random(seed)
    last_err = None
    for attempt in range(max_tries):
        P = {}
        try:
            for cl in prob.clauses:
                _apply_clause(cl, P, rng, defs)
            # premises numerically valid
            for ci, cname, pred, args in prob.premises(defs):
                if not N.check_pred(pred, args, P, tol=1e-6):
                    raise N.DegenerateError('premise %s %s fails' % (pred, args))
            # point separation
            names = list(P)
            for i in range(len(names)):
                for j in range(i):
                    if abs(P[names[i]] - P[names[j]]) < 1e-3:
                        raise N.DegenerateError('coincident %s %s' % (names[i], names[j]))
            if max(abs(z) for z in P.values()) > 60:
                raise N.DegenerateError('too large')
            if check_goal and prob.goal is not None:
                if not N.check_pred(prob.goal[0], prob.goal[1], P, tol=1e-6):
                    raise N.DegenerateError('goal fails numerically')
            return P
        except (N.DegenerateError, ZeroDivisionError, ValueError) as e:
            last_err = e
            continue
    raise N.DegenerateError('could not build diagram for %s: %s' % (prob.name, last_err))


def _apply_clause(cl, P, rng, defs):
    loci, explicit = [], None
    for cname, full in cl.constructs:
        d = defs[cname]
        sub = dict(zip(d.args, full))
        for pred, pargs in d.ndgs:
            pa = [sub.get(a, a) for a in pargs]
            if all(x in P for x in pa) and not N.check_pred(pred, pa, P):
                raise N.DegenerateError('ndg')
        kind, val = N.sketch(cname, full, P, rng)
        if kind == 'pts':
            outs_in_order = [a for a in full if a in cl.outs and a not in P]
            # unique while preserving order
            seen = []
            for a in outs_in_order:
                if a not in seen:
                    seen.append(a)
            if cname in ('eqangle2',):
                seen = [full[0]]
            explicit = dict(zip(seen, val))
        else:
            loci += val
    if explicit is not None:
        for k, v in explicit.items():
            P[k] = cl.fixed.get(k, v)
        for k in cl.outs:
            if k not in P:
                raise N.DegenerateError('missing output %s' % k)
        return
    out = cl.outs[0]
    if out in cl.fixed:
        P[out] = cl.fixed[out]
        return
    if len(loci) == 1:
        L = loci[0]
        if isinstance(L, N.Line):
            P[out] = L.p + L.d * rng.uniform(-1.2, 1.2)
        else:
            P[out] = L.sample(rng)
        return
    cands = N.intersect(loci[0], loci[1])
    cands = [z for z in cands if all(abs(z - q) > 1e-4 for q in P.values())]
    for extra in loci[2:]:
        pass
    if not cands:
        raise N.DegenerateError('no new intersection')
    P[out] = rng.choice(cands)
