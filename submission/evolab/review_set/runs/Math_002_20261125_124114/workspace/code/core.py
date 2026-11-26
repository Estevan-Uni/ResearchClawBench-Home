"""Core numba kernels: grid graph, BFS distances, reservation tables,
soft/hard space-time A* (lexicographic-ish weighted collision cost), collision bookkeeping.
Conventions: cells are flattened r*W+c; moves k: 0 stay,1 up,2 down,3 left,4 right; OPP=[0,2,1,4,3].
Paths are stored time-major P[t, a] for t=0..T (padded with goal after arrival).
"""
import numpy as np
from jit import njit

OPP = np.array([0, 2, 1, 4, 3], dtype=np.int64)

def build_graph(grid):
    H, W = grid.shape
    C = H * W
    nbr = -np.ones((C, 5), np.int64)
    for r in range(H):
        for c in range(W):
            u = r * W + c
            if grid[r, c]:
                continue
            nbr[u, 0] = u
            for k, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and not grid[rr, cc]:
                    nbr[u, k + 1] = rr * W + cc
    return nbr

@njit(cache=True)
def bfs_all(nbr, goals):
    C = nbr.shape[0]; N = goals.shape[0]
    D = np.full((N, C), 30000, np.int32)
    q = np.empty(C, np.int64)
    for a in range(N):
        g = goals[a]
        D[a, g] = 0; q[0] = g; h = 0; tl = 1
        while h < tl:
            u = q[h]; h += 1
            du = D[a, u]
            for k in range(1, 5):
                v = nbr[u, k]
                if v >= 0 and D[a, v] > du + 1:
                    D[a, v] = du + 1; q[tl] = v; tl += 1
    return D

# ---------------- heap -----------------
@njit(cache=True, inline='always')
def hpush(hk, hs, n, key, s):
    i = n
    hk[i] = key; hs[i] = s
    while i > 0:
        p = (i - 1) >> 1
        if hk[p] > hk[i]:
            tk = hk[p]; hk[p] = hk[i]; hk[i] = tk
            ts = hs[p]; hs[p] = hs[i]; hs[i] = ts
            i = p
        else:
            break
    return n + 1

@njit(cache=True, inline='always')
def hpop(hk, hs, n):
    k0 = hk[0]; s0 = hs[0]
    n -= 1
    if n > 0:
        hk[0] = hk[n]; hs[0] = hs[n]
        i = 0
        while True:
            l = 2 * i + 1
            if l >= n:
                break
            r = l + 1
            m = l
            if r < n and hk[r] < hk[l]:
                m = r
            if hk[m] < hk[i]:
                tk = hk[m]; hk[m] = hk[i]; hk[i] = tk
                ts = hs[m]; hs[m] = hs[i]; hs[i] = ts
                i = m
            else:
                break
    return k0, s0, n

class Buffers:
    def __init__(self, C, T):
        S = (T + 2) * C + T + 2
        self.stamp = np.zeros(S, np.int32)
        self.closed = np.zeros(S, np.int32)
        self.gbest = np.zeros(S, np.int32)
        self.parent = np.zeros(S, np.int64)
        cap = min(6 * S, 20_000_000)
        self.hk = np.zeros(cap, np.int64)
        self.hs = np.zeros(cap, np.int64)
        self.hold = np.zeros(T + 2, np.int32)
        self.gen = np.zeros(1, np.int64)
        self.gen2 = np.zeros(1, np.int64)
        self.cstamp = np.zeros(C, np.int64); self.ibeg = np.zeros(C, np.int64); self.iend = np.zeros(C, np.int64)
        Pn = C * (T + 2) + 2 * (T + 2)
        self.ilo = np.zeros(Pn, np.int64); self.ihi = np.zeros(Pn, np.int64); self.icol = np.zeros(Pn, np.int64)
        self.icell = np.zeros(Pn, np.int64); self.nta = np.zeros(Pn, np.int64); self.nc = np.zeros(Pn, np.int64)
        self.npar = np.zeros(Pn, np.int64); self.nclosed = np.zeros(Pn, np.int64)

@njit(cache=True)
def astar(s0, t0, goal, nbr, distg, occ, eout, T, Tcap, w, hard, maxexp,
          gen_arr, stamp, closed, gbest, parent, hk, hs, hold, out_path):
    """Plan from (s0,t0) to goal, minimizing t + w*collisions (hard: collisions forbidden).
    Collisions counted against occ/eout tables (which must NOT contain this agent).
    Writes out_path[t0..T]. Returns (arrival_time, n_collisions) or (-1,-1)."""
    C = nbr.shape[0]
    gen_arr[0] += 1
    if gen_arr[0] > 2000000000:
        gen_arr[0] = 1
        stamp[:] = 0; closed[:] = 0
    gen = gen_arr[0]
    # hold[t] = collisions incurred by sitting at goal for all t' > t
    hold[T] = 0
    for t in range(T - 1, -1, -1):
        hold[t] = hold[t + 1] + occ[t + 1, goal]
    if distg[s0] >= 30000:
        return -1, -1
    base_term = (T + 1) * C
    s = t0 * C + s0
    n = 0
    stamp[s] = gen; gbest[s] = 0; parent[s] = -1
    f = t0 + distg[s0]
    n = hpush(hk, hs, n, f * 8192 + (8191 - t0), s)
    cap = hk.shape[0] - 8
    nexp = 0
    while n > 0:
        key, s, n = hpop(hk, hs, n)
        if s >= base_term:
            # terminal node reached: reconstruct
            ta = s - base_term
            st = ta * C + goal
            # parent of terminal is goal-state at time ta
            tt = ta
            while st >= 0:
                cell = st % C
                out_path[tt] = cell
                st = parent[st]
                tt -= 1
            for t in range(ta + 1, T + 1):
                out_path[t] = goal
            return ta, gbest[s]
        if closed[s] == gen:
            continue
        closed[s] = gen
        nexp += 1
        if nexp > maxexp:
            return -1, -1
        u = s % C; t = s // C
        c = gbest[s]
        if u == goal:
            hc = hold[t]
            if (not hard) or hc == 0:
                ts = base_term + t
                ctot = c + hc
                if stamp[ts] != gen or gbest[ts] > ctot:
                    stamp[ts] = gen; gbest[ts] = ctot; parent[ts] = s
                    closed[ts] = 0
                    n = hpush(hk, hs, n, (t + w * ctot) * 8192 + (8191 - t), ts)
        if t >= Tcap:
            continue
        nt = t + 1
        for k in range(5):
            v = nbr[u, k]
            if v < 0:
                continue
            h = distg[v]
            if nt + h > Tcap:
                continue
            dc = occ[nt, v]
            if k > 0:
                # swap: some agent moves v->u, i.e. direction opp(k) out of v; slot index opp(k)-1
                dc += eout[t, v, k if (k % 2 == 1) else (k - 2)]
            if hard and dc > 0:
                continue
            nc = c + dc
            ns = nt * C + v
            if stamp[ns] == gen and gbest[ns] <= nc:
                continue
            stamp[ns] = gen; gbest[ns] = nc; parent[ns] = s; closed[ns] = 0
            if n >= cap:
                return -1, -1
            n = hpush(hk, hs, n, (nt + h + w * nc) * 8192 + (8191 - nt), ns)
    return -1, -1

# ---------------- reservation tables & collision bookkeeping -----------------
@njit(cache=True)
def move_dir(nbr, u, v):
    for k in range(1, 5):
        if nbr[u, k] == v:
            return k
    return 0

@njit(cache=True)
def table_update(P, a, occ, eout, nbr, T, sign, occ2):
    for t in range(T + 1):
        occ[t, P[t, a]] += sign
        occ2[P[t, a], t] += sign
    for t in range(T):
        u = P[t, a]; v = P[t + 1, a]
        if u != v:
            k = move_dir(nbr, u, v)
            eout[t, u, k - 1] += sign

@njit(cache=True)
def find_partners(P, a, occ, eout, nbr, T, mark, out):
    """Agents in collision (vertex or swap) with agent a. occ/eout must include a. Returns count."""
    N = P.shape[1]
    cnt = 0
    for t in range(T + 1):
        v = P[t, a]
        if occ[t, v] > 1:
            for b in range(N):
                if b != a and P[t, b] == v and mark[b] == 0:
                    mark[b] = 1; out[cnt] = b; cnt += 1
    for t in range(T):
        u = P[t, a]; v = P[t + 1, a]
        if u != v:
            k = move_dir(nbr, u, v)
            sl = k if (k % 2 == 1) else (k - 2)
            if eout[t, v, sl] > 0:
                for b in range(N):
                    if b != a and P[t, b] == v and P[t + 1, b] == u and mark[b] == 0:
                        mark[b] = 1; out[cnt] = b; cnt += 1
    for i in range(cnt):
        mark[out[i]] = 0
    return cnt

@njit(cache=True)
def detach(S, coll, deg):
    """remove all collision pairs incident to agents in S; return number removed"""
    N = coll.shape[0]; rem = 0
    for i in range(S.shape[0]):
        a = S[i]
        if deg[a] == 0:
            continue
        for b in range(N):
            if coll[a, b]:
                coll[a, b] = 0; coll[b, a] = 0; deg[a] -= 1; deg[b] -= 1; rem += 1
    return rem

@njit(cache=True)
def attach(S, P, occ, eout, nbr, T, coll, deg, mark, buf):
    add = 0
    for i in range(S.shape[0]):
        a = S[i]
        c = find_partners(P, a, occ, eout, nbr, T, mark, buf)
        for j in range(c):
            b = buf[j]
            if not coll[a, b]:
                coll[a, b] = 1; coll[b, a] = 1; deg[a] += 1; deg[b] += 1; add += 1
    return add

@njit(cache=True)
def arrival_time(P, a, goal, T):
    for t in range(T, -1, -1):
        if P[t, a] != goal:
            return t + 1
    return 0

# ---------------- soft SIPP (safe-interval search with soft obstacles) -----------------
@njit(cache=True)
def _build_iv(u, occ2, T, ibeg, iend, ilo, ihi, icol, icell, npool):
    """decompose timeline of cell u into maximal free intervals and unit occupied intervals"""
    ibeg[u] = npool
    t = 0
    while t <= T:
        if occ2[u, t] > 0:
            ilo[npool] = t; ihi[npool] = t; icol[npool] = occ2[u, t]; icell[npool] = u; npool += 1
            t += 1
        else:
            t2 = t
            while t2 + 1 <= T and occ2[u, t2 + 1] == 0:
                t2 += 1
            ilo[npool] = t; ihi[npool] = t2; icol[npool] = 0; icell[npool] = u; npool += 1
            t = t2 + 1
    iend[u] = npool
    return npool

@njit(cache=True)
def _find_iv(ibeg, iend, ilo, t):
    # index of interval of cell containing time t (binary search on ilo)
    lo = ibeg; hi = iend - 1
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if ilo[mid] <= t:
            lo = mid
        else:
            hi = mid - 1
    return lo

@njit(cache=True)
def sipp(s0, t0, goal, nbr, distg, occ2, eout, T, Tcap, w, hard, maxexp,
         gen_arr, cstamp, ibeg, iend, ilo, ihi, icol, icell, nta, nc, npar, nclosed, hk, hs, hold, out_path):
    """Soft SIPP: minimise (w*collisions + arrival time); hard=True forbids collisions.
    Node = interval (free maximal interval or unit occupied timestep) of a cell.
    Returns (arrival time, collisions) or (-1,-1)."""
    C = nbr.shape[0]
    gen_arr[0] += 1
    gen = gen_arr[0]
    hold[T] = 0
    for t in range(T - 1, -1, -1):
        hold[t] = hold[t + 1] + occ2[goal, t + 1]
    if distg[s0] >= 30000:
        return -1, -1
    npool = 0
    cap_pool = ilo.shape[0] - (T + 2)
    # start node
    cstamp[s0] = gen
    npool = _build_iv(s0, occ2, T, ibeg, iend, ilo, ihi, icol, icell, npool)
    i0 = _find_iv(ibeg[s0], iend[s0], ilo, t0)
    TERM = cap_pool + 1    # terminal pseudo-node ids = TERM + arrival time  (stored in extra arrays slot)
    nta[i0] = t0; nc[i0] = 0; npar[i0] = -1; nclosed[i0] = -gen
    n = 0
    n = hpush(hk, hs, n, (t0 + distg[s0]) * 8192 + (8191 - t0), i0)
    hcap = hk.shape[0] - 16
    nexp = 0
    best_term = -1; best_term_c = 0; best_term_t = 0; best_term_par = -1
    while n > 0:
        key, x, n = hpop(hk, hs, n)
        if x < 0:
            # terminal
            ta = -x - 1
            # reconstruct
            node = best_term_par
            # fill from goal arrival backward
            cur_t = ta
            out_path[ta] = goal
            for t in range(ta + 1, T + 1):
                out_path[t] = goal
            while node >= 0:
                p = npar[node]
                tn = nta[node]; v = icell[node]
                # agent at v from tn until cur_t (exclusive of the arrival of the child)
                for t in range(tn, cur_t):
                    out_path[t] = v
                cur_t = tn
                node = p
            return ta, best_term_c
        if nclosed[x] == gen:
            continue
        nclosed[x] = gen
        nexp += 1
        if nexp > maxexp:
            return -1, -1
        u = icell[x]; ta = nta[x]; c = nc[x]; hi = ihi[x]
        if hi > Tcap: hi = Tcap
        # goal terminal
        if u == goal:
            hc = hold[ta]
            if (not hard) or hc == 0:
                ctot = c + hc
                kt = (ta + w * ctot) * 8192 + (8191 - ta)
                if best_term < 0 or kt < best_term:
                    best_term = kt; best_term_c = ctot; best_term_t = ta; best_term_par = x
                    n = hpush(hk, hs, n, kt, -(ta + 1))
        # wait into next interval of the same cell
        if ihi[x] < Tcap and x + 1 < iend[u]:
            y = x + 1
            dc = icol[y]
            if not (hard and dc > 0):
                ny = ilo[y]; ncy = c + dc
                if ny + distg[u] <= Tcap:
                    if cstamp_iv_better(y, ny, ncy, gen, nta, nc, nclosed, npar):
                        npar[y] = x
                        n = hpush(hk, hs, n, (ny + distg[u] + w * ncy) * 8192 + (8191 - ny), y)
        # move to neighbours: depart at d in [ta, hi], arrive d+1
        if ta >= Tcap:
            continue
        amax = hi + 1
        if amax > Tcap: amax = Tcap
        for k in range(1, 5):
            v = nbr[u, k]
            if v < 0:
                continue
            h = distg[v]
            if ta + 1 + h > Tcap:
                continue
            if cstamp[v] != gen:
                cstamp[v] = gen
                if npool + T + 2 >= cap_pool:
                    return -1, -1
                npool = _build_iv(v, occ2, T, ibeg, iend, ilo, ihi, icol, icell, npool)
            sl = k if (k % 2 == 1) else (k - 2)   # slot of the opposite move v->u
            j = _find_iv(ibeg[v], iend[v], ilo, ta + 1)
            while j < iend[v] and ilo[j] <= amax:
                a0 = ilo[j] if ilo[j] > ta + 1 else ta + 1
                if a0 + h > Tcap:
                    break
                if a0 <= ihi[j]:
                    dcv = icol[j]
                    if hard and dcv > 0:
                        j += 1; continue
                    sw = eout[a0 - 1, v, sl]
                    arr = a0
                    if sw > 0:
                        # try one step later inside the same free interval to avoid the swap
                        if dcv == 0 and a0 + 1 <= ihi[j] and a0 + 1 <= amax and eout[a0, v, sl] == 0:
                            arr = a0 + 1; sw = 0
                        elif hard:
                            j += 1; continue
                    ncy = c + dcv + sw
                    if cstamp_iv_better(j, arr, ncy, gen, nta, nc, nclosed, npar):
                        npar[j] = x
                        if n >= hcap:
                            return -1, -1
                        n = hpush(hk, hs, n, (arr + h + w * ncy) * 8192 + (8191 - arr), j)
                j += 1
    return -1, -1

@njit(cache=True, inline='always')
def cstamp_iv_better(y, t, c, gen, nta, nc, nclosed, npar):
    # node data validity tracked via nclosed: value gen (closed), -gen (open, valid), other (stale)
    if nclosed[y] == gen:
        return False
    if nclosed[y] == -gen:
        if nc[y] < c or (nc[y] == c and nta[y] <= t):
            return False
    nclosed[y] = -gen; nta[y] = t; nc[y] = c
    return True

# ---------------- exact backward space-time DP planner -----------------
@njit(cache=True)
def dp_plan(s0, t0, goal, nbr, distg, dists, occ, eout, T, hard, V, hold, out_path, tie):
    """Exact minimiser of key = collisions*4096 + arrival-time-offset over the horizon T (same objective as the
    lexicographic soft A*), by backward DP over (t, cell) restricted to cells reachable from s0 by time t
    (dists = BFS distance from s0, valid when s0 is the agent's start; pass distg-of-zeros to disable) and able
    to reach goal by T. hard=True forbids collisions. Returns (arrival, collisions) or (-1,-1)."""
    C = nbr.shape[0]; INF = 2000000000
    hold[T] = 0
    for t in range(T - 1, -1, -1):
        hold[t] = hold[t + 1] + occ[t + 1, goal]
    for v in range(C):
        V[T, v] = INF
    if not (hard and hold[T] > 0):
        V[T, goal] = 0
    for t in range(T - 1, t0 - 1, -1):
        rem = T - t; el = t - t0
        for v in range(C):
            if distg[v] > rem or dists[v] > el or nbr[v, 0] < 0:
                V[t, v] = INF; continue
            best = INF
            if v == goal:
                if not (hard and hold[t] > 0):
                    best = hold[t] * 4096
            for k in range(5):
                u = nbr[v, k]
                if u < 0: continue
                nb = V[t + 1, u]
                if nb >= INF: continue
                cost = occ[t + 1, u]
                if k > 0:
                    cost += eout[t, u, k if (k % 2 == 1) else (k - 2)]
                if hard and cost > 0: continue
                cand = nb + cost * 4096 + 1
                if cand < best: best = cand
            V[t, v] = best
    if V[t0, s0] >= INF:
        return -1, -1
    # forward extraction
    x = s0; t = t0; coll = 0
    out_path[t0] = s0
    while True:
        key = V[t, x]
        if x == goal and key == hold[t] * 4096 and not (hard and hold[t] > 0):
            coll += hold[t]
            for tt in range(t + 1, T + 1):
                out_path[tt] = goal
            return t, coll
        found = False
        for kk in range(5):
            k = (kk + tie) % 5
            u = nbr[x, k]
            if u < 0: continue
            nb = V[t + 1, u]
            if nb >= INF: continue
            cost = occ[t + 1, u]
            if k > 0:
                cost += eout[t, u, k if (k % 2 == 1) else (k - 2)]
            if hard and cost > 0: continue
            if nb + cost * 4096 + 1 == key:
                coll += cost; x = u; t += 1; out_path[t] = x; found = True
                break
        if not found:
            return -1, -1
