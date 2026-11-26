"""PIBT (Okumura et al. 2022) and LaCAM (Okumura 2023) baselines; PIBT kernel in numba, LaCAM DFS in Python."""
import time, numpy as np
from jit import njit
from collections import deque

@njit(cache=True)
def _pibt(root, Q, Qn, occ_now, occ_next, nbr, D, tie, sa, sp_, sc, sm, si):
    """iterative PIBT(root, None) with an explicit stack (numba recursion is unreliable)."""
    top = 0
    _push(0, root, -1, Q, nbr, D, tie, sa, sp_, sc, sm, si)
    ret = -1   # -1: no pending child result, 0 fail, 1 success
    while top >= 0:
        i = sa[top]; pusher = sp_[top]
        if ret == 1:
            ret = 1; top -= 1; continue
        ret = -1
        found = False; child = False
        while si[top] < sm[top]:
            v = sc[top, si[top]]; si[top] += 1
            if occ_next[v] != -1:
                continue
            if pusher >= 0 and v == Q[pusher]:
                continue
            Qn[i] = v; occ_next[v] = i
            j = occ_now[v]
            if j != -1 and j != i and Qn[j] == -1:
                top += 1
                _push(top, j, i, Q, nbr, D, tie, sa, sp_, sc, sm, si)
                child = True
                break
            found = True
            break
        if child:
            continue
        if found:
            ret = 1; top -= 1; continue
        u = Q[i]; Qn[i] = u; occ_next[u] = i
        ret = 0; top -= 1
    return ret == 1

@njit(cache=True)
def _push(top, i, pusher, Q, nbr, D, tie, sa, sp_, sc, sm, si):
    u = Q[i]; m = 0
    key = np.empty(5, np.float64)
    for k in range(5):
        v = nbr[u, k]
        if v >= 0:
            sc[top, m] = v; key[m] = D[i, v] + tie[i, k]; m += 1
    for x in range(1, m):
        y = x
        while y > 0 and key[y - 1] > key[y]:
            tk = key[y]; key[y] = key[y - 1]; key[y - 1] = tk
            tc = sc[top, y]; sc[top, y] = sc[top, y - 1]; sc[top, y - 1] = tc
            y -= 1
    sa[top] = i; sp_[top] = pusher; sm[top] = m; si[top] = 0

@njit(cache=True)
def pibt_step(Q, order, nbr, D, tie, cwho, cwhere, ncon, occ_now, occ_next, Qn):
    """one PIBT configuration step with LaCAM constraints (agents cwho[:ncon] fixed to cwhere). Returns success."""
    N = Q.shape[0]
    for i in range(N):
        occ_now[Q[i]] = i
        Qn[i] = -1
    ok = True
    for c in range(ncon):
        i = cwho[c]; v = cwhere[c]
        if occ_next[v] != -1:
            ok = False; break
        j = occ_now[v]
        if j != -1 and j != i and Qn[j] == Q[i]:  # swap
            ok = False; break
        Qn[i] = v; occ_next[v] = i
    if ok:
        sa = np.empty(N + 1, np.int64); sp_ = np.empty(N + 1, np.int64); sc = np.empty((N + 1, 5), np.int64)
        sm = np.empty(N + 1, np.int64); si = np.empty(N + 1, np.int64)
        for x in range(N):
            i = order[x]
            if Qn[i] == -1:
                _pibt(i, Q, Qn, occ_now, occ_next, nbr, D, tie, sa, sp_, sc, sm, si)
    # verify (vertex conflicts impossible by construction; check swaps for constrained agents)
    if ok:
        for i in range(N):
            if occ_next[Qn[i]] != i:   # two agents share a vertex
                ok = False; break
    if ok:
        for i in range(N):
            j = occ_now[Qn[i]]
            if j != -1 and j != i and Qn[j] == Q[i]:
                ok = False; break
    for i in range(N):
        occ_now[Q[i]] = -1
    for i in range(N):
        if Qn[i] >= 0:
            occ_next[Qn[i]] = -1
    for x in range(ncon):
        occ_next[cwhere[x]] = -1
    return ok

def _configs_to_paths(configs, inst):
    X = np.array(configs)  # (L, N)
    G = inst.goals
    out = []; soc = 0
    for a in range(inst.N):
        col = X[:, a]
        nz = np.flatnonzero(col != G[a])
        arr = 0 if len(nz) == 0 else nz[-1] + 1
        soc += arr
        p = col[:arr + 1]
        out.append(np.stack([p // inst.W, p % inst.W], 1))
    return out, int(soc)

def run_pibt(inst, time_limit, seed=0, max_steps=None):
    """plain PIBT with dynamic priorities (incomplete; runs until all agents at goals or step/time limit)."""
    t0 = time.process_time(); rng = np.random.default_rng(seed)
    N, C = inst.N, inst.C
    Q = inst.starts.copy(); G = inst.goals
    prio = inst.sp / max(1, inst.sp.max()) * 0.0 + rng.random(N) * 1e-3 + inst.sp / (10 * max(1, inst.sp.max()))
    occ_now = -np.ones(C, np.int64); occ_next = -np.ones(C, np.int64); Qn = np.empty(N, np.int64)
    e = np.zeros(1, np.int64)
    configs = [Q.copy()]
    if max_steps is None: max_steps = 20 * inst.T
    while time.process_time() - t0 < time_limit and len(configs) < max_steps:
        if (Q == G).all(): break
        order = np.argsort(-prio, kind='stable')
        tie = rng.random((N, 5)) * 0.5
        pibt_step(Q, order, inst.nbr, inst.D, tie, e, e, 0, occ_now, occ_next, Qn)
        Q = Qn.copy(); configs.append(Q)
        at = Q == G
        prio = np.where(at, prio - np.floor(prio), prio + 1)
    ok = bool((Q == G).all())
    paths, soc = _configs_to_paths(configs, inst)
    return dict(success=ok, soc=soc if ok else -1, time=time.process_time() - t0, steps=len(configs) - 1,
                paths=paths if ok else None, cp=0 if ok else int((Q != G).sum()))

class _Node:
    __slots__ = ('Q', 'parent', 'order', 'prio', 'tree', 'depth')

def run_lacam(inst, time_limit, seed=0):
    """LaCAM: DFS over configurations, lazy low-level constraint generation, PIBT as successor generator.
    Re-inserts already-visited configurations into OPEN (as in LaCAM*), without the cost-refinement phase."""
    t0 = time.process_time(); rng = np.random.default_rng(seed)
    N, C = inst.N, inst.C; G = inst.goals; nbr = inst.nbr; D = inst.D
    occ_now = -np.ones(C, np.int64); occ_next = -np.ones(C, np.int64); Qn = np.empty(N, np.int64)
    cw = np.zeros(N, np.int64); cv = np.zeros(N, np.int64)
    def make(Q, parent):
        n = _Node(); n.Q = Q; n.parent = parent
        if parent is None:
            n.prio = inst.sp / (10.0 * max(1, inst.sp.max())) + rng.random(N) * 1e-4
        else:
            n.prio = np.where(Q == G, parent.prio - np.floor(parent.prio), parent.prio + 1)
        n.order = np.argsort(-n.prio, kind='stable')
        n.tree = deque([((), ())]); n.depth = 0 if parent is None else parent.depth + 1
        return n
    root = make(inst.starts.copy(), None)
    explored = {root.Q.tobytes(): root}
    OPEN = [root]; nodes = 1; gen = 0
    goal_node = None
    while OPEN and time.process_time() - t0 < time_limit:
        node = OPEN[-1]
        if (node.Q == G).all():
            goal_node = node; break
        if not node.tree:
            OPEN.pop(); continue
        who, where = node.tree.popleft()
        k = len(who)
        if k < N:
            i = int(node.order[k])
            cells = nbr[node.Q[i]]; cells = cells[cells >= 0].copy(); rng.shuffle(cells)
            for v in cells:
                node.tree.append((who + (i,), where + (int(v),)))
        for x in range(k):
            cw[x] = who[x]; cv[x] = where[x]
        tie = rng.random((N, 5)) * 0.5
        ok = pibt_step(node.Q, node.order, nbr, D, tie, cw, cv, k, occ_now, occ_next, Qn)
        gen += 1
        if not ok:
            continue
        key = Qn.tobytes()
        if key in explored:
            OPEN.append(explored[key])
            continue
        nn = make(Qn.copy(), node); explored[key] = nn; OPEN.append(nn); nodes += 1
    el = time.process_time() - t0
    if goal_node is None:
        return dict(success=False, soc=-1, time=el, nodes=nodes, cp=-1, paths=None)
    configs = []
    n = goal_node
    while n is not None:
        configs.append(n.Q); n = n.parent
    configs.reverse()
    paths, soc = _configs_to_paths(configs, inst)
    return dict(success=True, soc=soc, time=el, nodes=nodes, steps=len(configs) - 1, paths=paths, cp=0)
