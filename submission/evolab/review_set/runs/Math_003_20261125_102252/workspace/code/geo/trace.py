"""Traceback: minimal AR supports and proof extraction."""
from fractions import Fraction as Fr

import numpy as np

from .dd import PMOD, HALF

ANGLE_KINDS = ('coll', 'para', 'perp', 'eqangle')


def gf_solve(rows, target):
    """Find c (mod p) with sum c_i rows_i == target.  rows/target: lists of (var, coeff).
    Returns list of coefficients or None."""
    vars_ = {}
    for r in rows + [target]:
        for v, _ in r:
            if v not in vars_:
                vars_[v] = len(vars_)
    m, n = len(rows), len(vars_)
    if m == 0:
        return None
    # Solve A^T c = t  (n equations, m unknowns): augmented matrix n x (m+1)
    M = np.zeros((n, m + 1), np.int64)
    for i, r in enumerate(rows):
        for v, c in r:
            M[vars_[v], i] = (M[vars_[v], i] + c) % PMOD
    for v, c in target:
        M[vars_[v], m] = (M[vars_[v], m] + c) % PMOD
    piv_cols = []
    row = 0
    for col in range(m):
        if row >= n:
            break
        nz = np.flatnonzero(M[row:, col])
        if len(nz) == 0:
            continue
        r = row + int(nz[0])
        if r != row:
            M[[row, r]] = M[[r, row]]
        M[row] = (M[row] * pow(int(M[row, col]), PMOD - 2, PMOD)) % PMOD
        colv = M[:, col].copy()
        colv[row] = 0
        nzr = np.flatnonzero(colv)
        if len(nzr):
            M[nzr] = (M[nzr] - (np.outer(colv[nzr], M[row]) % PMOD)) % PMOD
        piv_cols.append(col)
        row += 1
    if np.any(M[row:, m] % PMOD):
        return None
    c = [0] * m
    for i, col in enumerate(piv_cols):
        c[col] = int(M[i, m])
    return c


def frac_solve(rows, target):
    """Exact rational solve; rows/target as lists of (var, int coeff)."""
    vars_ = {}
    for r in rows + [target]:
        for v, _ in r:
            vars_.setdefault(v, len(vars_))
    m, n = len(rows), len(vars_)
    M = [[Fr(0)] * (m + 1) for _ in range(n)]
    for i, r in enumerate(rows):
        for v, c in r:
            M[vars_[v]][i] += c
    for v, c in target:
        M[vars_[v]][m] += c
    piv = []
    row = 0
    for col in range(m):
        p = next((r for r in range(row, n) if M[r][col] != 0), None)
        if p is None:
            continue
        M[row], M[p] = M[p], M[row]
        inv = 1 / M[row][col]
        M[row] = [x * inv for x in M[row]]
        for r in range(n):
            if r != row and M[r][col] != 0:
                f = M[r][col]
                M[r] = [a - f * b for a, b in zip(M[r], M[row])]
        piv.append(col)
        row += 1
    if any(M[r][m] != 0 for r in range(row, n)):
        return None
    c = [Fr(0)] * m
    for i, col in enumerate(piv):
        c[col] = M[i][m]
    return c


class Tracer:
    def __init__(self, E):
        self.E = E
        self.var_rows = {'a': {}, 'l': {}}
        for i, (s, terms, c, fid) in enumerate(E.rows):
            for v, _ in terms:
                self.var_rows[s].setdefault(v, []).append(i)
        self.memo = {}
        self.stats = {'ar_resolved': 0, 'nonintegral': 0}

    def support(self, sysn, target, T):
        E = self.E
        V = set(v for v, _ in target)
        R = set()
        for it in range(12):
            newR = set()
            for v in V:
                for i in self.var_rows[sysn].get(v, ()):
                    if i < T:
                        newR.add(i)
            if newR == R and it > 0:
                break
            R = newR
            Rl = sorted(R)
            c = gf_solve([E.rows[i][1] for i in Rl], target)
            if c is not None:
                break
            V = set(v for i in R for v, _ in E.rows[i][1])
        else:
            c = None
        if c is None:
            Rl = [i for i in range(T) if E.rows[i][0] == sysn]
            c = gf_solve([E.rows[i][1] for i in Rl], target)
            if c is None:
                raise RuntimeError('AR traceback failed for %s' % (target,))
        sup = [i for i, ci in zip(Rl, c) if ci]
        # greedy pruning (try to drop late rows first)
        for i in sorted(sup, reverse=True):
            trial = [j for j in sup if j != i]
            if gf_solve([E.rows[j][1] for j in trial], target) is not None:
                sup = trial
        coef = frac_solve([E.rows[j][1] for j in sup], target)
        return sup, coef

    def resolve(self, fid):
        """Return premise fact ids of fact fid (resolving AR facts)."""
        f = self.E.facts[fid]
        if f.rule != 'ar':
            return f.prem
        if fid in self.memo:
            return self.memo[fid]
        E = self.E
        prem = set()
        integral = True
        for sysn, terms, const in E.fact_rows(f.kind, f.args, f.const):
            sup, coef = self.support(sysn, terms, f.T)
            for i in sup:
                prem.add(E.rows[i][3])
            if sysn == 'a' and (coef is None or any(x.denominator != 1 for x in coef)):
                integral = False
        self.stats['ar_resolved'] += 1
        if not integral:
            self.stats['nonintegral'] += 1
        self.memo[fid] = sorted(prem)
        return self.memo[fid]

    def proof(self, goal_fid):
        order, seen = [], set()
        stack = [(goal_fid, False)]
        while stack:
            fid, done = stack.pop()
            if done:
                order.append(fid)
                continue
            if fid in seen:
                continue
            seen.add(fid)
            stack.append((fid, True))
            for p in self.resolve(fid):
                if p not in seen:
                    stack.append((p, False))
        steps = []
        for fid in order:
            f = self.E.facts[fid]
            steps.append(dict(id=fid, kind=f.kind, args=list(f.args), const=str(Fr(f.const)),
                              rule=f.rule, prem=list(self.resolve(fid)),
                              src=list(f.src) if f.src else None))
        return steps


def make_proof(E, goal_fid, prob, coords):
    tr = Tracer(E)
    steps = tr.proof(goal_fid)
    return dict(problem=prob.name, statement=prob.text,
                coords={k: [v.real, v.imag] for k, v in coords.items()},
                goal=[prob.goal[0]] + list(prob.goal[1]),
                steps=steps, goal_step=goal_fid,
                stats=dict(n_steps=len(steps), n_premise=sum(s['rule'] == 'premise' for s in steps),
                           n_ar=sum(s['rule'] == 'ar' for s in steps), **tr.stats))


# ---------------------------------------------------------------- pretty print
def fmt_fact(kind, args, const='0'):
    A = [x.upper() for x in args]
    c = Fr(const)
    if kind == 'coll':
        return '%s are collinear' % ', '.join(A)
    if kind == 'para':
        return '%s%s ∥ %s%s' % tuple(A)
    if kind == 'perp':
        return '%s%s ⊥ %s%s' % tuple(A)
    if kind == 'cong':
        return '%s%s = %s%s' % tuple(A)
    if kind == 'eqangle':
        s = '∠(%s%s,%s%s) = ∠(%s%s,%s%s)' % tuple(A)
        if c:
            s += ' + %d°' % int(c * 180)
        return s
    if kind == 'eqratio':
        return '%s%s:%s%s = %s%s:%s%s' % tuple(A)
    if kind == 'rconst':
        return '%s%s:%s%s = 2^%s' % (tuple(A) + (c,))
    if kind == 'cyclic':
        return '%s are concyclic' % ', '.join(A)
    if kind == 'midp':
        return '%s is the midpoint of %s%s' % tuple(A)
    if kind == 'simtri':
        return '△%s%s%s ∼ △%s%s%s (same orientation)' % tuple(A)
    if kind == 'simtri2':
        return '△%s%s%s ∼ △%s%s%s (opposite orientation)' % tuple(A)
    return '%s %s' % (kind, ' '.join(A))


RULE_TEXT = {
    'premise': 'given', 'ar': 'angle/ratio chasing', 'para_coll': 'AB ∥ AC ⇒ A, B, C collinear', 'midp': 'collinear + equal halves ⇒ midpoint',
    'iso': 'isosceles triangle base angles', 'iso_conv': 'equal base angles ⇒ isosceles',
    'central': 'inscribed angle = half central angle', 'central_conv': 'converse of inscribed/central angle',
    'cyclic_eqangle': 'equal inscribed angles ⇒ concyclic', 'cyclic_center': 'equidistant from a centre ⇒ concyclic',
    'cyclic_merge': 'two circles sharing three points coincide', 'cyclic_sub': 'subset of concyclic points',
    'sim_aa': 'similar triangles (AA)', 'sim_sas': 'similar triangles (SAS)', 'sim_sss': 'similar triangles (SSS)',
    'bisector': 'angle-bisector theorem', 'bisector_conv': 'converse angle-bisector theorem',
    'perp_bisector': 'two equidistant points span the perpendicular bisector',
    'chord': 'equal inscribed angles subtend equal chords', 'chord_conv': 'equal chords subtend equal inscribed angles',
}


def proof_text(proof):
    lines = []
    num = {}
    k = 0
    for s in proof['steps']:
        k += 1
        num[s['id']] = k
        refs = ', '.join('(%d)' % num[p] for p in s['prem'])
        lines.append('(%d) %s   [%s%s]' % (k, fmt_fact(s['kind'], s['args'], s['const']),
                                          RULE_TEXT.get(s['rule'], s['rule']), (' from ' + refs) if refs else ''))
    return '\n'.join(lines)
