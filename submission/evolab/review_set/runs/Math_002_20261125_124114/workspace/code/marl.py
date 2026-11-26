"""MARL repair operator for LNS.
The neighbourhood agents S act simultaneously (decentralised, parameter-shared policy) for W steps in an
environment where all other agents follow their fixed current paths (dynamic obstacles, read from the
reservation tables).  Within a timestep agents act in a random sequential order and observe the moves already
committed by team-mates in the same step (a minimal form of the intention communication used by SCRIMP).
Observation = per-action feature vectors (local, including several future steps of background occupancy);
the policy is a shared action-scoring MLP; the critic reads the pooled hidden features.
After W steps every agent's path is completed by soft A* (as in the PP repair)."""
import numpy as np
from jit import njit

NF = 26      # per-action feature size
LI = 3       # intention horizon
LOOK = 4     # future background lookahead steps

@njit(cache=True)
def backward_dp(goal, nbr, Drow, occ, eout, T, V):
    """V[t, v] = lexicographic soft cost-to-go key = collisions*4096 + steps, w.r.t. background tables only.
    Terminal: being at goal at t and staying forever costs the future background traffic through goal."""
    C = nbr.shape[0]; INF = 2000000000
    hold = 0
    for v in range(C):
        V[T, v] = INF
    V[T, goal] = 0
    for t in range(T - 1, -1, -1):
        hold += occ[t + 1, goal]
        for v in range(C):
            if nbr[v, 0] < 0 or Drow[v] + t > T:
                V[t, v] = INF; continue
            best = INF
            if v == goal:
                best = hold * 4096
            for k in range(5):
                u = nbr[v, k]
                if u < 0: continue
                nb = V[t + 1, u]
                if nb >= INF: continue
                cost = occ[t + 1, u]
                if k > 0:
                    cost += eout[t, u, k if (k % 2 == 1) else (k - 2)]
                cand = nb + cost * 4096 + 1
                if cand < best: best = cand
            V[t, v] = best

@njit(cache=True)
def _feat(i, k, u, v, t, goal, Drow, sp, occ, eout, nbr, T, claimed, teamat, decided, nxt, pos, m, Wn, out, Vi, qmin, ipath):
    # out: NF vector
    for q in range(NF):
        out[q] = 0.0
    out[0] = 1.0 if k == 0 else 0.0
    du = Drow[u]; dv = Drow[v]
    out[1] = dv - du
    nt = t + 1 if t + 1 <= T else T
    out[2] = min(occ[nt, v], 2)
    if k > 0:
        sl = k if (k % 2 == 1) else (k - 2)
        out[3] = min(eout[t if t < T else T - 1, v, sl], 2)
    if claimed[v] >= 0:
        out[4] = 1.0
    j = teamat[v]
    if j >= 0 and j != i:
        if decided[j] == 1:
            if nxt[j] == u and k > 0:
                out[5] = 1.0
        else:
            out[6] = 1.0
    for l in range(LOOK):
        tt = t + 2 + l
        if tt > T: tt = T
        out[7 + l] = 1.0 if occ[tt, v] > 0 else 0.0
    t2 = t + 2 if t + 2 <= T else T
    fr = 0
    for q in range(5):
        w = nbr[v, q]
        if w >= 0 and occ[t2, w] == 0:
            fr += 1
    out[11] = fr / 5.0
    if v == goal:
        out[12] = 1.0
        hc = 0
        for tt in range(nt + 1, T + 1):
            hc += occ[tt, goal]
            if hc >= 5: break
        out[13] = hc / 5.0
    cg = 0
    for q in range(1, 5):
        w = nbr[v, q]
        if w >= 0 and occ[nt, w] > 0:
            cg += 1
    out[14] = cg / 4.0
    # occupancy along greedy shortest route from v
    x = v; cnt = 0
    for l in range(LOOK):
        best = x; bd = Drow[x]
        for q in range(1, 5):
            w = nbr[x, q]
            if w >= 0 and Drow[w] < bd:
                bd = Drow[w]; best = w
        x = best
        tt = t + 2 + l
        if tt > T: tt = T
        if occ[tt, x] > 0: cnt += 1
    out[15] = cnt / LOOK
    out[16] = du / (sp + 1.0)
    out[17] = 1.0 if u == goal else 0.0
    out[18] = t / (Wn + 1.0)
    nb = 0
    for jj in range(m):
        if jj != i:
            p = pos[jj]
            # manhattan distance via coordinates is not available here; use 2-hop neighbourhood test
            for q in range(5):
                w = nbr[u, q]
                if w >= 0 and (w == p):
                    nb += 1
    out[19] = nb / 4.0
    # planning-informed features from the background cost-to-go
    if t + 1 <= T:
        nbv = Vi[t + 1, v]
        if nbv < 2000000000:
            cost = occ[t + 1, v]
            if k > 0:
                cost += eout[t, v, k if (k % 2 == 1) else (k - 2)]
            q = nbv + cost * 4096 + 1
            ex = q - qmin
            dc = ex // 4096
            dt = (q % 4096) - (qmin % 4096)
            out[20] = min(dc, 3) / 3.0
            out[21] = max(-5, min(dt, 10)) / 10.0
            out[22] = 1.0 if ex == 0 else 0.0
        else:
            out[20] = 1.0; out[21] = 1.0
    out[23] = min(qmin // 4096, 5) / 5.0
    # communicated intentions of team-mates (next LI cells along their own cost-to-go)
    c1 = 0; c2 = 0.0
    for j in range(m):
        if j == i: continue
        if decided[j] == 0 and ipath[j, 1] == v: c1 += 1
        for l in range(2, LI + 1):
            if ipath[j, l] == v: c2 += 1.0 / l
    out[24] = min(c1, 2) / 2.0
    out[25] = min(c2, 1.0)

@njit(cache=True)
def _score(f, W1, b1, w2, b2, h):
    H = b1.shape[0]
    s = b2
    for a in range(H):
        z = b1[a]
        for q in range(NF):
            z += f[q] * W1[q, a]
        z = np.tanh(z)
        h[a] = z
        s += z * w2[a]
    return s

@njit(cache=True)
def marl_rollout(agents, starts, goals, D, sp, occ, eout, nbr, T, Wn, W1, b1, w2, b2, mode, temp, seed,
                 forced, outP, recF, recM, recA, recLP, recR, recAg, nrec_max, Vbuf):
    """Run the MARL policy for agents (m,) for Wn steps from t=0.
    mode: 0 sample, 1 greedy, 2 forced (teacher actions taken from forced[t, i] cell indices -> path).
    outP[t, i] = cell of agent i at time t (t=0..Wn).
    Records (if nrec_max>0) features/mask/action/logp/reward/agent-index per decision.
    Returns number of records, and total team collisions (vertex+swap, bg + team) during the window."""
    np.random.seed(seed)
    m = agents.shape[0]; C = nbr.shape[0]
    pos = np.empty(m, np.int64); nxt = np.empty(m, np.int64); decided = np.zeros(m, np.int64)
    claimed = -np.ones(C, np.int64); teamat = -np.ones(C, np.int64)
    feats = np.zeros((5, NF)); h = np.zeros(b1.shape[0]); logits = np.zeros(5); valid = np.zeros(5, np.int64)
    for i in range(m):
        pos[i] = starts[i]; outP[0, i] = pos[i]
    nrec = 0; totc = 0
    recidx = np.empty(m, np.int64)
    for i in range(m):
        backward_dp(goals[i], nbr, D[agents[i]], occ, eout, T, Vbuf[i])
    ipath = np.zeros((m, LI + 1), np.int64)
    for t in range(Wn):
        # intention broadcast: each agent's greedy path under its own background cost-to-go
        for j in range(m):
            x = pos[j]; ipath[j, 0] = x
            for l in range(1, LI + 1):
                tt = t + l
                if tt > T:
                    ipath[j, l] = x; continue
                bq = 2000000000; bx = x
                for k in range(5):
                    y = nbr[x, k]
                    if y < 0: continue
                    nbv = Vbuf[j, tt, y]
                    if nbv >= 2000000000: continue
                    cost = occ[tt, y]
                    if k > 0:
                        cost += eout[tt - 1, y, k if (k % 2 == 1) else (k - 2)]
                    q = nbv + cost * 4096 + 1
                    if q < bq: bq = q; bx = y
                x = bx; ipath[j, l] = x
        order = np.random.permutation(m)
        for i in range(m):
            teamat[pos[i]] = i; decided[i] = 0
        for oi in range(m):
            i = order[oi]
            a = agents[i]; u = pos[i]; g = goals[i]
            mx = -1e30
            qmin = 2000000000
            for k in range(5):
                v = nbr[u, k]
                if v >= 0 and t + 1 <= T:
                    nbv = Vbuf[i, t + 1, v]
                    if nbv < 2000000000:
                        cost = occ[t + 1, v]
                        if k > 0:
                            cost += eout[t, v, k if (k % 2 == 1) else (k - 2)]
                        q = nbv + cost * 4096 + 1
                        if q < qmin: qmin = q
            for k in range(5):
                v = nbr[u, k]
                if v < 0:
                    valid[k] = 0; logits[k] = -1e30
                    for q in range(NF): feats[k, q] = 0.0
                    continue
                valid[k] = 1
                _feat(i, k, u, v, t, g, D[a], sp[i], occ, eout, nbr, T, claimed, teamat, decided, nxt, pos, m, Wn, feats[k], Vbuf[i], qmin, ipath)
                logits[k] = _score(feats[k], W1, b1, w2, b2, h) / temp
                if logits[k] > mx: mx = logits[k]
            Z = 0.0
            for k in range(5):
                if valid[k]: Z += np.exp(logits[k] - mx)
            if mode == 2:
                tgt = forced[t + 1, i]; act = 0
                for k in range(5):
                    if nbr[u, k] == tgt: act = k
            elif mode == 1:
                act = 0; bv = -1e30
                for k in range(5):
                    if valid[k] and logits[k] > bv:
                        bv = logits[k]; act = k
            else:
                r = np.random.random() * Z; acc = 0.0; act = 0
                for k in range(5):
                    if valid[k]:
                        acc += np.exp(logits[k] - mx); act = k
                        if acc >= r: break
            lp = logits[act] - mx - np.log(Z)
            v = nbr[u, act]
            nxt[i] = v; decided[i] = 1
            claimed[v] = i
            if nrec < nrec_max:
                for k in range(5):
                    recM[nrec, k] = valid[k]
                    for q in range(NF): recF[nrec, k, q] = feats[k, q]
                recA[nrec] = act; recLP[nrec] = lp; recAg[nrec] = i; recR[nrec] = 0.0
                recidx[i] = nrec; nrec += 1
            else:
                recidx[i] = -1
        # apply joint move, compute rewards
        for i in range(m):
            a = agents[i]; u = pos[i]; v = nxt[i]
            c = occ[t + 1, v]
            if u != v:
                for k in range(1, 5):
                    if nbr[u, k] == v:
                        sl = k if (k % 2 == 1) else (k - 2)
                        c += eout[t, v, sl]
            for j in range(m):
                if j != i:
                    if nxt[j] == v: c += 1
                    elif u != v and pos[j] == v and nxt[j] == u: c += 1
            delay = (D[a, v] - D[a, u] + 1)
            if u == goals[i] and v == u: delay = 0
            totc += c
            if recidx[i] >= 0:
                recR[recidx[i]] = -0.3 * c
        for i in range(m):
            teamat[pos[i]] = -1; claimed[nxt[i]] = -1
        alldone = True
        for i in range(m):
            pos[i] = nxt[i]; outP[t + 1, i] = pos[i]
            if pos[i] != goals[i]: alldone = False
        if alldone and t + 1 < Wn:
            for tt in range(t + 2, Wn + 1):
                for i in range(m):
                    outP[tt, i] = pos[i]
            break
    return nrec, totc


class Policy:
    def __init__(self, H=32, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 1 / np.sqrt(NF), (NF, H)); self.b1 = np.zeros(H)
        self.w2 = r.normal(0, 1 / np.sqrt(H), H); self.b2 = 0.0
        self.V1 = r.normal(0, 1 / np.sqrt(H), H); self.vb = 0.0
    def params(self):
        return [self.W1, self.b1, self.w2, np.array([self.b2]), self.V1, np.array([self.vb])]
    def set(self, ps):
        self.W1, self.b1, self.w2, b2, self.V1, vb = [p.copy() for p in ps]
        self.b2 = float(b2[0]); self.vb = float(vb[0])
    def save(self, f):
        np.savez(f, *self.params())
    @staticmethod
    def load(f):
        z = np.load(f); p = Policy(H=z['arr_1'].shape[0]); p.set([z[f'arr_{i}'] for i in range(6)]); return p

    def forward(self, F, M):
        """F (B,5,NF), M (B,5) -> logp (B,5), V (B,), cache"""
        Z = F @ self.W1 + self.b1          # B,5,H
        Hh = np.tanh(Z)
        s = Hh @ self.w2 + self.b2         # B,5
        s = np.where(M > 0, s, -1e9)
        mx = s.max(1, keepdims=True)
        e = np.exp(s - mx) * (M > 0)
        Zs = e.sum(1, keepdims=True)
        logp = s - mx - np.log(Zs)
        pool = (Hh * M[..., None]).sum(1) / np.maximum(M.sum(1, keepdims=True), 1)
        V = pool @ self.V1 + self.vb
        return logp, V, (F, M, Hh, pool, np.exp(logp) * (M > 0))

    def backward(self, cache, dlogp, dV):
        """dlogp: gradient wrt logp (B,5); dV (B,). returns grads list"""
        F, M, Hh, pool, prob = cache
        # logp = s - logsumexp(s) -> ds = dlogp - prob * sum(dlogp)
        ds = (dlogp - prob * dlogp.sum(1, keepdims=True)) * (M > 0)
        gw2 = np.einsum('bk,bkh->h', ds, Hh); gb2 = ds.sum()
        dH = ds[..., None] * self.w2
        gV1 = dV @ pool; gvb = dV.sum()
        dpool = dV[:, None] * self.V1
        dH += dpool[:, None, :] * M[..., None] / np.maximum(M.sum(1), 1)[:, None, None]
        dZ = dH * (1 - Hh ** 2)
        gW1 = np.einsum('bkf,bkh->fh', F, dZ); gb1 = dZ.sum((0, 1))
        return [gW1, gb1, gw2, np.array([gb2]), gV1, np.array([gvb])]


class Adam:
    def __init__(self, ps, lr=3e-3):
        self.m = [np.zeros_like(p) for p in ps]; self.v = [np.zeros_like(p) for p in ps]; self.t = 0; self.lr = lr
    def step(self, ps, gs):
        self.t += 1; out = []
        for i, (p, g) in enumerate(zip(ps, gs)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g; self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t); vh = self.v[i] / (1 - 0.999 ** self.t)
            out.append(p - self.lr * mh / (np.sqrt(vh) + 1e-8))
        return out
