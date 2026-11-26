"""Train the MARL repair policy ONLY on procedurally generated random grid maps (zero-shot for maze/room/
warehouse/empty).  Phase 1: behaviour cloning from soft-PP repairs.  Phase 2: PPO (+BC regulariser).
Checkpoint selection on held-out generated neighbourhoods (mean local CP after repair)."""
import sys, os, time, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from solver import *
from core import arrival_time
from marl import *
from collections import deque

OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs')

def largest_comp(grid):
    H, W = grid.shape; lab = -np.ones((H, W), int); best = None; c = 0
    for r in range(H):
        for q in range(W):
            if grid[r, q] == 0 and lab[r, q] < 0:
                dq = deque([(r, q)]); lab[r, q] = c; cells = [(r, q)]
                while dq:
                    a, b = dq.popleft()
                    for da, db in ((1,0),(-1,0),(0,1),(0,-1)):
                        x, y = a+da, b+db
                        if 0 <= x < H and 0 <= y < W and grid[x, y] == 0 and lab[x, y] < 0:
                            lab[x, y] = c; dq.append((x, y)); cells.append((x, y))
                if best is None or len(cells) > len(best): best = cells
                c += 1
    return np.array(best)

def gen_instance(rng):
    S = int(rng.choice([10, 14, 18, 25]))
    grid = (rng.random((S, S)) < rng.uniform(0.05, 0.25)).astype(np.uint8)
    cells = largest_comp(grid)
    g2 = np.ones_like(grid); g2[cells[:, 0], cells[:, 1]] = 0
    dens = rng.uniform(0.35, 0.8)
    N = max(4, int(dens * len(cells)))
    s = cells[rng.choice(len(cells), N, replace=False)]; g = cells[rng.choice(len(cells), N, replace=False)]
    return Instance(g2, s, g)

def make_state(rng, iters):
    inst = gen_instance(rng); st = State(inst); w = inst.T + 1
    st.init_soft_pp(rng, w)
    nbw = np.ones(len(NB_FUNCS))
    for _ in range(iters):
        if st.cp == 0: break
        lns_step(st, rng, lambda s, S, r: repair_pp(s, S, r, w), 8, nbw, w)
    return st

def episode(st, S, rng, policy, Wn, mode, rec, forced=None, temp=1.0):
    """run one MARL repair on (st,S) and restore state. returns local CP, nrec, per-agent terminal rewards"""
    I = st.inst; w = I.T + 1
    Wn = marl_W(st, S, Wn)
    oldP = st.P[:, S].copy(); oldarr = st.arr[S].copy(); cp0 = st.cp
    st.detach(S)
    for a in S: st.remove(a)
    cprest = st.cp
    m = len(S); outP = np.zeros((Wn + 1, m), np.int64)
    F, M, A, LP, R, AG = rec
    fz = forced if forced is not None else np.zeros((1, 1), np.int64)
    nrec, tc = marl_rollout(S, I.starts[S], I.goals[S], I.D, I.sp[S], st.occ, st.eout, I.nbr, I.T, Wn,
                            policy.W1, policy.b1, policy.w2, policy.b2, mode, temp, int(rng.integers(1 << 30)),
                            fz, outP, F, M, A, LP, R, AG, F.shape[0], st.vbuf(m))
    st.P[:Wn + 1, S] = outP
    done = outP[Wn] == I.goals[S]
    for i in rng.permutation(m):
        if done[i]:
            st.P[Wn + 1:, S[i]] = I.goals[S[i]]; st.arr[S[i]] = arrival_time(st.P, S[i], I.goals[S[i]], I.T)
        elif not st.plan(S[i], w, t0=Wn):
            st.plan(S[i], w, t0=0)
    for a in S: st.insert(a)
    st.attach(S)
    local = st.cp - cprest
    term = -1.0 * st.deg[S] - 0.1 * (st.arr[S] - I.sp[S])
    # restore
    st.detach(S)
    for a in S: st.remove(a)
    st.P[:, S] = oldP; st.arr[S] = oldarr
    for a in S: st.insert(a)
    st.attach(S); assert st.cp == cp0
    return local, nrec, term

def pp_local(st, S, rng, keep_paths=False, Wn=0):
    I = st.inst; w = I.T + 1
    oldP = st.P[:, S].copy(); oldarr = st.arr[S].copy(); cp0 = st.cp
    st.detach(S)
    for a in S: st.remove(a)
    cprest = st.cp
    repair_pp(st, S, rng, w)
    st.attach(S); local = st.cp - cprest
    newP = st.P[:Wn + 1, S].copy() if keep_paths else None
    st.detach(S)
    for a in S: st.remove(a)
    st.P[:, S] = oldP; st.arr[S] = oldarr
    for a in S: st.insert(a)
    st.attach(S); assert st.cp == cp0
    return local, newP

def bc_episode(st, S, rng, policy, Wn, rec):
    """teacher forcing: features computed while agents follow the PP repair paths"""
    I = st.inst
    Wn = marl_W(st, S, Wn)
    _, newP = pp_local(st, S, rng, keep_paths=True, Wn=Wn)
    oldP = st.P[:, S].copy(); oldarr = st.arr[S].copy(); cp0 = st.cp
    st.detach(S)
    for a in S: st.remove(a)
    m = len(S); outP = np.zeros((Wn + 1, m), np.int64)
    F, M, A, LP, R, AG = rec
    nrec, _ = marl_rollout(S, I.starts[S], I.goals[S], I.D, I.sp[S], st.occ, st.eout, I.nbr, I.T, Wn,
                           policy.W1, policy.b1, policy.w2, policy.b2, 2, 1.0, 0, newP, outP,
                           F, M, A, LP, R, AG, F.shape[0], st.vbuf(m))
    for a in S: st.insert(a)
    st.attach(S); assert st.cp == cp0
    return nrec

def alloc(n):
    return (np.zeros((n, 5, NF)), np.zeros((n, 5), np.int64), np.zeros(n, np.int64), np.zeros(n), np.zeros(n),
            np.zeros(n, np.int64))

def returns(R, AG, nrec, m, Wn, term, gamma=0.95):
    # records come in blocks of m per timestep
    G = np.zeros(nrec)
    Wn = (nrec + m - 1) // m
    Rt = np.zeros((Wn, m)); idx = np.zeros((Wn, m), np.int64)
    for r in range(nrec):
        t = r // m; i = AG[r]; Rt[t, i] = R[r]; idx[t, i] = r
    run = term.copy()
    for t in range(Wn - 1, -1, -1):
        run = Rt[t] + gamma * run
        G[idx[t]] = run
    return G

def evaluate(policy, held, Wn, mode=1):
    rng = np.random.default_rng(123); tot = 0; n = 0
    rec = alloc(1)
    for st, Ss in held:
        for S in Ss:
            l, _, _ = episode(st, S, rng, policy, Wn, mode, rec); tot += l; n += 1
    return tot / n

def main(budget_bc=300, budget_ppo=1500, Wn=8, seed=0):
    rng = np.random.default_rng(seed); t0 = time.perf_counter(); log = []
    pol = Policy(H=32, seed=seed)
    # held-out generated states (fixed)
    hr = np.random.default_rng(999); held = []
    for it in [0, 0, 100, 100, 400, 400]:
        st = make_state(hr, it)
        if st.cp == 0: continue
        held.append((st, [nb_collision(st, hr, 8) for _ in range(15)]))
    pp_ref = 0; n = 0; rr = np.random.default_rng(123)
    for st, Ss in held:
        for S in Ss: pp_ref += pp_local(st, S, rr)[0]; n += 1
    pp_ref /= n
    print(f'held-out: {n} neighbourhoods, PP local CP {pp_ref:.2f}', flush=True)
    # ---- phase 1: behaviour cloning ----
    BCF, BCM, BCA = [], [], []
    rec = alloc(300 * 8 + 8)
    while time.perf_counter() - t0 < budget_bc * 0.7:
        st = make_state(rng, int(rng.choice([0, 30, 150, 400])))
        if st.cp == 0: continue
        for _ in range(20):
            if st.cp == 0: break
            S = nb_collision(st, rng, 8) if rng.random() < 0.7 else nb_random(st, rng, 8)
            nr = bc_episode(st, S, rng, pol, Wn, rec)
            BCF.append(rec[0][:nr].copy()); BCM.append(rec[1][:nr].copy()); BCA.append(rec[2][:nr].copy())
    BCF = np.concatenate(BCF); BCM = np.concatenate(BCM); BCA = np.concatenate(BCA)
    print('BC samples', len(BCA), flush=True)
    opt = Adam(pol.params(), lr=3e-3)
    ep = 0
    while time.perf_counter() - t0 < budget_bc:
        perm = rng.permutation(len(BCA))
        for b in range(0, len(perm), 1024):
            ix = perm[b:b + 1024]
            logp, V, cache = pol.forward(BCF[ix], BCM[ix])
            dl = np.zeros_like(logp); dl[np.arange(len(ix)), BCA[ix]] = -1.0 / len(ix)
            g = pol.backward(cache, dl, np.zeros(len(ix)))
            pol.set(opt.step(pol.params(), g))
        ep += 1
        logp, _, _ = pol.forward(BCF[:20000], BCM[:20000])
        acc = (logp.argmax(1) == BCA[:20000]).mean()
    ev = evaluate(pol, held, Wn)
    print(f'BC done epochs {ep} acc {acc:.3f} held-out MARL local CP {ev:.2f} (PP {pp_ref:.2f})', flush=True)
    log.append(dict(phase='bc', t=time.perf_counter() - t0, acc=float(acc), eval=ev, pp=pp_ref))
    best = (ev, [p.copy() for p in pol.params()]); pol.save(os.path.join(OUT, 'policy_bc.npz'))
    # ---- phase 2: PPO with BC regulariser ----
    opt = Adam(pol.params(), lr=1e-3)
    it = 0; tstart = time.perf_counter()
    st = None; st_uses = 0
    rec = alloc(300 * 8 + 8)
    while time.perf_counter() - tstart < budget_ppo:
        Fs, Ms, As, LPs, Gs = [], [], [], [], []; locs = []
        while sum(len(a) for a in As) < 6000:
            if st is None or st_uses > 30 or st.cp == 0:
                st = make_state(rng, int(rng.choice([0, 30, 150, 400]))); st_uses = 0
                if st.cp == 0: continue
            S = nb_collision(st, rng, 8) if rng.random() < 0.7 else nb_random(st, rng, 8)
            l, nr, term = episode(st, S, rng, pol, Wn, 0, rec)
            st_uses += 1
            G = returns(rec[4], rec[5], nr, len(S), 0, term)
            Fs.append(rec[0][:nr].copy()); Ms.append(rec[1][:nr].copy()); As.append(rec[2][:nr].copy())
            LPs.append(rec[3][:nr].copy()); Gs.append(G); locs.append(l)
        F = np.concatenate(Fs); M = np.concatenate(Ms); A = np.concatenate(As); LP = np.concatenate(LPs); G = np.concatenate(Gs)
        _, V0, _ = pol.forward(F, M)
        adv = G - V0; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for epoch in range(4):
            perm = rng.permutation(len(A))
            for b in range(0, len(perm), 1024):
                ix = perm[b:b + 1024]; B = len(ix)
                logp, V, cache = pol.forward(F[ix], M[ix])
                lpa = logp[np.arange(B), A[ix]]
                ratio = np.exp(lpa - LP[ix])
                ad = adv[ix]
                unclipped = ~(((ratio > 1.2) & (ad > 0)) | ((ratio < 0.8) & (ad < 0)))
                dl = np.zeros_like(logp)
                dl[np.arange(B), A[ix]] = -(ad * ratio * unclipped) / B
                p = np.exp(logp) * (M[ix] > 0)
                dl += 0.01 * p * (np.where(M[ix] > 0, logp, 0) + 1) / B   # -entropy gradient
                dV = 0.5 * (V - G[ix]) / B
                g = pol.backward(cache, dl, dV)
                # BC regulariser
                jx = rng.integers(0, len(BCA), 512)
                logp2, _, cache2 = pol.forward(BCF[jx], BCM[jx])
                dl2 = np.zeros_like(logp2); dl2[np.arange(512), BCA[jx]] = -0.1 / 512
                g2 = pol.backward(cache2, dl2, np.zeros(512))
                pol.set(opt.step(pol.params(), [a + b for a, b in zip(g, g2)]))
        it += 1
        if it % 3 == 0:
            ev = evaluate(pol, held, Wn)
            el = time.perf_counter() - t0
            print(f'PPO it {it} t={el:.0f}s train local CP {np.mean(locs):.2f} held-out {ev:.2f} (PP {pp_ref:.2f})', flush=True)
            log.append(dict(phase='ppo', it=it, t=el, train_local=float(np.mean(locs)), eval=ev, pp=pp_ref))
            if ev < best[0]:
                best = (ev, [p.copy() for p in pol.params()])
            json.dump(log, open(os.path.join(OUT, 'train_log.json'), 'w'), indent=1)
    pol.set(best[1]); pol.save(os.path.join(OUT, 'policy.npz'))
    print('best held-out', best[0], flush=True)
    json.dump(log, open(os.path.join(OUT, 'train_log.json'), 'w'), indent=1)

if __name__ == '__main__':
    a = sys.argv
    main(budget_bc=float(a[1]) if len(a) > 1 else 300, budget_ppo=float(a[2]) if len(a) > 2 else 1500,
         Wn=int(a[3]) if len(a) > 3 else 8)
