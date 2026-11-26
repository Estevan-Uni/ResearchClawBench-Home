"""MAPF solvers on a shared substrate: soft-PP initial solution, MAPF-LNS2 (ALNS repair with soft PP),
hard PP with random restarts, and the LNS2+RL hybrid (repair operator switch MARL -> PP)."""
import time, numpy as np
from core import *

class Instance:
    def __init__(self, grid, starts, goals, T=None):
        self.grid = grid; H, W = grid.shape; self.H, self.W = H, W
        self.C = H * W
        self.nbr = build_graph(grid)
        self.starts = (starts[:, 0] * W + starts[:, 1]).astype(np.int64)
        self.goals = (goals[:, 0] * W + goals[:, 1]).astype(np.int64)
        self.N = len(starts)
        self.D = bfs_all(self.nbr, self.goals)
        self.DS = bfs_all(self.nbr, self.starts)   # distance from each agent's start (DP pruning)
        self.Z = np.zeros(self.C, np.int32)
        self.sp = self.D[np.arange(self.N), self.starts].astype(np.int64)
        if T is None:
            m = int(self.sp.max())
            T = max(2 * m, m + 2 * (H + W) // 2 + 30)
        self.T = int(min(T, 8000))

class State:
    """Current (possibly colliding) solution + reservation tables + collision graph."""
    def __init__(self, inst):
        self.inst = inst; N, C, T = inst.N, inst.C, inst.T
        self.P = np.zeros((T + 1, N), np.int64)
        self.occ = np.zeros((T + 1, C), np.int16)
        self.occ2 = np.zeros((C, T + 1), np.int16)
        self.eout = np.zeros((T, C, 4), np.int16)
        self.coll = np.zeros((N, N), np.uint8)
        self.deg = np.zeros(N, np.int64)
        self.arr = np.zeros(N, np.int64)
        self.cp = 0
        self.mark = np.zeros(N, np.int64); self.pbuf = np.zeros(N, np.int64)
        self.buf = Buffers(C, T)
        self.path = np.zeros(T + 1, np.int64)
        self.n_astar = 0

    slack = 10**6
    use_sipp = True
    planner = 'dp'
    tie = 0
    def plan(self, a, w, hard=False, maxexp=400000, t0=0, s0=None):
        """plan agent a (must not be in tables) from (s0,t0); prefix P[:t0,a] kept. Returns success."""
        I = self.inst; b = self.buf
        if s0 is None: s0 = I.starts[a] if t0 == 0 else self.P[t0, a]
        self.n_astar += 1
        Tcap = min(I.T, t0 + int(I.D[a, s0]) + self.slack)
        if self.planner == 'dp':
            ds = I.DS[a] if (t0 == 0 and s0 == I.starts[a]) else I.Z
            ta, c = dp_plan(s0, t0, I.goals[a], I.nbr, I.D[a], ds, self.occ, self.eout, I.T, hard,
                            self.vbuf(1)[0], b.hold, self.path, self.tie)
            if ta < 0 and hard: return False
            if ta < 0 and t0 > 0: return False
        elif self.use_sipp:
            ta, c = sipp(s0, t0, I.goals[a], I.nbr, I.D[a], self.occ2, self.eout, I.T, Tcap, w, hard, maxexp,
                         b.gen2, b.cstamp, b.ibeg, b.iend, b.ilo, b.ihi, b.icol, b.icell, b.nta, b.nc, b.npar,
                         b.nclosed, b.hk, b.hs, b.hold, self.path)
        else:
            ta, c = astar(s0, t0, I.goals[a], I.nbr, I.D[a], self.occ, self.eout, I.T, Tcap, w, hard, maxexp,
                          b.gen, b.stamp, b.closed, b.gbest, b.parent, b.hk, b.hs, b.hold, self.path)
        self.last_c = c
        if ta < 0:
            if hard: return False
            # fallback: cheap weighted search
            ta, c = astar(s0, t0, I.goals[a], I.nbr, I.D[a], self.occ, self.eout, I.T, I.T, 1, False, 10**8,
                          b.gen, b.stamp, b.closed, b.gbest, b.parent, b.hk, b.hs, b.hold, self.path)
            if ta < 0:
                if t0 > 0: return False
                raise RuntimeError('soft A* failed (horizon too small?)')
        self.P[t0:, a] = self.path[t0:]
        self.arr[a] = arrival_time(self.P, a, I.goals[a], I.T)
        return True

    def insert(self, a):
        table_update(self.P, a, self.occ, self.eout, self.inst.nbr, self.inst.T, 1, self.occ2)
    def remove(self, a):
        table_update(self.P, a, self.occ, self.eout, self.inst.nbr, self.inst.T, -1, self.occ2)

    def detach(self, S):
        self.cp -= detach(S, self.coll, self.deg)
    def attach(self, S):
        self.cp += attach(S, self.P, self.occ, self.eout, self.inst.nbr, self.inst.T, self.coll, self.deg, self.mark, self.pbuf)

    def vbuf(self, m):
        if not hasattr(self, '_vb') or self._vb.shape[0] < m:
            self._vb = np.zeros((m, self.inst.T + 1, self.inst.C), np.int32)
        return self._vb

    def soc(self):
        return int(self.arr.sum())

    def init_soft_pp(self, rng, w):
        I = self.inst
        order = rng.permutation(I.N)
        for a in order:
            self.plan(a, w); self.insert(a)
        self.attach(np.arange(I.N))

    def paths(self):
        I = self.inst
        out = []
        for a in range(I.N):
            p = self.P[:self.arr[a] + 1, a]
            out.append(np.stack([p // I.W, p % I.W], 1))
        return out


# ---------------- neighbourhood selection -----------------
def nb_collision(st, rng, Nsz):
    """MAPF-LNS2 collision-based neighbourhood: random walk on the collision graph from a random colliding agent;
    if the connected part found is smaller than Nsz, add agents that are spatially close along the paths."""
    cand = np.flatnonzero(st.deg > 0)
    a = int(rng.choice(cand))
    S = [a]; Sset = {a}
    # BFS in collision graph (randomised)
    frontier = [a]
    while frontier and len(S) < Nsz:
        x = frontier.pop(int(rng.integers(len(frontier))))
        nb = np.flatnonzero(st.coll[x])
        rng.shuffle(nb)
        for y in nb:
            if y not in Sset:
                S.append(int(y)); Sset.add(int(y)); frontier.append(int(y))
                if len(S) >= Nsz: break
    tries = 0
    while len(S) < Nsz and tries < 20:
        tries += 1
        x = S[int(rng.integers(len(S)))]
        t = int(rng.integers(0, st.arr[x] + 1))
        v = st.P[t, x]
        # agents near v at time t (within the 5-neighbourhood)
        cells = st.inst.nbr[v]; cells = cells[cells >= 0]
        near = np.flatnonzero(np.isin(st.P[t], cells))
        rng.shuffle(near)
        for y in near:
            if int(y) not in Sset:
                S.append(int(y)); Sset.add(int(y))
                if len(S) >= Nsz: break
    return np.array(S, np.int64)

def nb_random(st, rng, Nsz):
    """MAPF-LNS2 random neighbourhood: N agents sampled w.p. proportional to deg+1"""
    p = (st.deg + 1).astype(np.float64); p /= p.sum()
    return rng.choice(st.inst.N, min(Nsz, st.inst.N), replace=False, p=p).astype(np.int64)

NB_FUNCS = [nb_collision, nb_random]


def repair_pp(st, S, rng, w):
    """standard LNS2 repair: replan S sequentially (random order) with soft A* against everything else"""
    order = S[rng.permutation(len(S))]
    for a in order:
        st.plan(a, w); st.insert(a)


def lns_step(st, rng, repair, Nsz, nb_weights, w, accept_equal=True):
    """one LNS iteration with repair callable(st,S,rng). returns (nb index, delta cp, accepted)"""
    p = nb_weights / nb_weights.sum()
    ni = int(rng.choice(len(p), p=p))
    S = NB_FUNCS[ni](st, rng, Nsz)
    oldP = st.P[:, S].copy(); oldarr = st.arr[S].copy(); oldcp = st.cp; oldsoc = st.arr[S].sum()
    st.detach(S)
    for a in S: st.remove(a)
    repair(st, S, rng)
    st.attach(S)
    newcp = st.cp; newsoc = st.arr[S].sum()
    if newcp < oldcp or (accept_equal and newcp == oldcp and newsoc <= oldsoc):
        return ni, oldcp - newcp, True
    # revert
    st.detach(S)
    for a in S: st.remove(a)
    st.P[:, S] = oldP; st.arr[S] = oldarr
    for a in S: st.insert(a)
    st.attach(S)
    assert st.cp == oldcp
    return ni, 0, False


def run_lns2(inst, time_limit, seed=0, Nsz=8, w=None, log_every=0.5, repair=None, init=None, stats=None):
    """MAPF-LNS2. returns dict with success, cp, soc, time, trace"""
    t0 = time.process_time(); rng = np.random.default_rng(seed)
    if w is None: w = inst.T + 1
    st = State(inst)
    st.init_soft_pp(rng, w) if init is None else init(st, rng)
    trace = [(time.process_time() - t0, st.cp)]
    t_init = time.process_time() - t0
    nbw = np.ones(len(NB_FUNCS)); gamma = 0.1
    it = 0
    rep = repair if repair is not None else (lambda s, S, r: repair_pp(s, S, r, w))
    last_log = 0
    while st.cp > 0 and time.process_time() - t0 < time_limit:
        ni, d, acc = lns_step(st, rng, rep, Nsz, nbw, w)
        nbw[ni] = (1 - gamma) * nbw[ni] + gamma * max(d, 0)
        nbw = np.maximum(nbw, 1e-3)
        it += 1
        el = time.process_time() - t0
        if el - last_log > log_every or st.cp == 0:
            trace.append((el, st.cp)); last_log = el
    el = time.process_time() - t0
    trace.append((el, st.cp))
    return dict(success=st.cp == 0, cp=int(st.cp), soc=st.soc(), time=el, t_init=t_init, iters=it,
                trace=trace, state=st)


def run_pp(inst, time_limit, seed=0, maxexp=None):
    """Prioritised planning (hard constraints) with random restarts."""
    t0 = time.process_time(); rng = np.random.default_rng(seed)
    st = State(inst)
    if maxexp is None: maxexp = 4 * (inst.T + 1) * inst.C
    restarts = 0; best_planned = 0
    while time.process_time() - t0 < time_limit:
        order = rng.permutation(inst.N)
        ok = True; planned = []
        for i, a in enumerate(order):
            if time.process_time() - t0 > time_limit: ok = False; break
            if not st.plan(a, 1, hard=True, maxexp=maxexp):
                ok = False; break
            st.insert(a); planned.append(a)
        best_planned = max(best_planned, len(planned))
        if ok:
            st.attach(np.arange(inst.N))
            return dict(success=st.cp == 0, cp=int(st.cp), soc=st.soc(), time=time.process_time() - t0,
                        restarts=restarts, planned=inst.N, state=st)
        for a in planned: st.remove(a)
        restarts += 1
    return dict(success=False, cp=-1, soc=-1, time=time.process_time() - t0, restarts=restarts,
                planned=best_planned, state=None)


# ---------------- MARL repair & hybrid LNS2+RL -----------------
from marl import marl_rollout, NF

_DUM3 = np.zeros((1, 5, NF)); _DUM2 = np.zeros((1, 5), np.int64); _DUM1 = np.zeros(1); _DUMI = np.zeros(1, np.int64)
_DUMF = np.zeros((1, 1), np.int64)

def marl_window(st, S, policy, Wn, seed, mode=0, temp=1.0):
    I = st.inst
    outP = np.zeros((Wn + 1, len(S)), np.int64)
    marl_rollout(S, I.starts[S], I.goals[S], I.D, I.sp[S], st.occ, st.eout, I.nbr, I.T, Wn,
                 policy.W1, policy.b1, policy.w2, policy.b2, mode, temp, seed, _DUMF, outP,
                 _DUM3, _DUM2, _DUMI, _DUM1, _DUM1, _DUMI, 0, st.vbuf(len(S)))
    return outP

def marl_W(st, S, Wn):
    """Wn<0 means 'full horizon' (2*max shortest path + 20, capped by T)."""
    if Wn >= 0: return min(Wn, st.inst.T // 2)
    return int(min(st.inst.T - st.inst.sp[S].max() - 2, 2 * st.inst.sp[S].max() + 20))

def repair_marl(st, S, rng, policy, Wn, w, mode=0, temp=1.0):
    """MARL plans the first Wn steps jointly for the neighbourhood; agents that are at their goal at the end of
    the window simply stay there, the others are completed by soft A* (as in the PP repair)."""
    I = st.inst
    Wn = marl_W(st, S, Wn)
    done = np.zeros(len(S), bool)
    if Wn > 0:
        outP = marl_window(st, S, policy, Wn, int(rng.integers(1 << 30)), mode, temp)
        st.P[:Wn + 1, S] = outP
        done = outP[Wn] == I.goals[S]
    order = rng.permutation(len(S))
    for i in order:
        a = S[i]
        if done[i]:
            st.P[Wn + 1:, a] = I.goals[a]; st.arr[a] = arrival_time(st.P, a, I.goals[a], I.T)
        elif not st.plan(a, w, t0=Wn):
            st.plan(a, w, t0=0)
    for a in S: st.insert(a)


class Fixed:
    def __init__(self, op): self.op = op; self.switch_time = None
    def choose(self, el): return self.op
    def update(self, op, d, dt, el): pass

class Patience:
    """MARL until k consecutive MARL iterations without CP improvement, then PP forever."""
    def __init__(self, k=10): self.k = k; self.fail = 0; self.switched = False; self.switch_time = None
    def choose(self, el): return 0 if self.switched else 1
    def update(self, op, d, dt, el):
        if self.switched: return
        self.fail = 0 if d > 0 else self.fail + 1
        if self.fail >= self.k: self.switched = True; self.switch_time = el

class RateSwitch:
    """MARL with interleaved PP probe iterations; switch permanently to PP when the EMA improvement rate
    (CP reduction per second) of MARL falls below that of PP."""
    def __init__(self, probe_every=5, beta=0.1, min_probe=5):
        self.pe = probe_every; self.beta = beta; self.min_probe = min_probe
        self.n = 0; self.r = [None, None]; self.np = 0; self.switched = False; self.switch_time = None
    def choose(self, el):
        if self.switched: return 0
        self.n += 1
        return 0 if self.n % self.pe == 0 else 1
    def update(self, op, d, dt, el):
        if self.switched: return
        rate = max(d, 0) / max(dt, 1e-6)
        # EMA over (improvement, time) pairs -> rate = sum improvement / sum time
        if self.r[op] is None: self.r[op] = [max(d, 0), dt]
        else:
            self.r[op][0] = (1 - self.beta) * self.r[op][0] + self.beta * max(d, 0)
            self.r[op][1] = (1 - self.beta) * self.r[op][1] + self.beta * dt
        if op == 0: self.np += 1
        if self.np >= self.min_probe and self.r[1] is not None:
            rm = self.r[1][0] / self.r[1][1]; rp = self.r[0][0] / self.r[0][1]
            if rm < rp: self.switched = True; self.switch_time = el

class Bandit:
    """epsilon-greedy over {PP, MARL} on EMA improvement rate (ablation)."""
    def __init__(self, eps=0.2, beta=0.1, seed=0):
        self.eps = eps; self.beta = beta; self.r = [[1.0, 1e-3], [1.0, 1e-3]]; self.rng = np.random.default_rng(seed)
        self.switch_time = None
    def choose(self, el):
        if self.rng.random() < self.eps: return int(self.rng.integers(2))
        return int(np.argmax([x[0] / x[1] for x in self.r]))
    def update(self, op, d, dt, el):
        self.r[op][0] = (1 - self.beta) * self.r[op][0] + self.beta * max(d, 0)
        self.r[op][1] = (1 - self.beta) * self.r[op][1] + self.beta * dt


def run_lns(inst, time_limit, seed=0, Nsz=8, w=None, policy=None, Wn=8, controller=None, log_every=0.25,
            marl_mode=0, temp=1.0, slack=None):
    """Generic LNS with repair operators {0: PP (soft prioritised planning, LNS2), 1: MARL}.
    controller=Fixed(0) is MAPF-LNS2."""
    t0 = time.process_time(); rng = np.random.default_rng(seed)
    if w is None: w = inst.T + 1
    if controller is None: controller = Fixed(0)
    st = State(inst)
    if slack is not None: st.slack = slack
    st.init_soft_pp(rng, w)
    t_init = time.process_time() - t0
    trace = [(t_init, st.cp, -1)]
    ops = [lambda s, S, r: repair_pp(s, S, r, w),
           lambda s, S, r: repair_marl(s, S, r, policy, Wn, w, marl_mode, temp)]
    nbw = np.ones(len(NB_FUNCS)); gamma = 0.1
    it = 0; last_log = 0; nops = [0, 0]; gains = [0, 0]; tops = [0.0, 0.0]
    while st.cp > 0:
        el = time.process_time() - t0
        if el >= time_limit: break
        op = controller.choose(el)
        ts = time.process_time()
        ni, d, acc = lns_step(st, rng, ops[op], Nsz, nbw, w)
        dt = time.process_time() - ts
        nbw[ni] = max((1 - gamma) * nbw[ni] + gamma * max(d, 0), 1e-3)
        el = time.process_time() - t0
        controller.update(op, d, dt, el)
        nops[op] += 1; gains[op] += d; tops[op] += dt
        it += 1
        if el - last_log > log_every or st.cp == 0:
            trace.append((el, st.cp, op)); last_log = el
    el = time.process_time() - t0
    trace.append((el, st.cp, -1))
    return dict(success=bool(st.cp == 0), cp=int(st.cp), soc=st.soc(), time=el, t_init=t_init, iters=it,
                n_pp=nops[0], n_marl=nops[1], gain_pp=gains[0], gain_marl=gains[1], t_pp=tops[0], t_marl=tops[1],
                switch_time=controller.switch_time, trace=trace, state=st)
