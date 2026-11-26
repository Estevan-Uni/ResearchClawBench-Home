"""MOT metrics: CLEAR (MOTA, IDSW, MT/ML, Frag), Identity (IDF1) and HOTA, following TrackEval."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from tracker import iou_matrix


def evaluate(gt, pred, log_events=False):
    """gt, pred: lists over frames of (ids array, boxes Nx4 tlbr)."""
    gt_ids_all = np.unique(np.concatenate([g[0] for g in gt]))
    pr_ids_all = np.unique(np.concatenate([p[0] for p in pred] + [np.array([], int)]))
    gmap = {v: i for i, v in enumerate(gt_ids_all)}
    pmap = {v: i for i, v in enumerate(pr_ids_all)}
    G, P = len(gt_ids_all), len(pr_ids_all)
    frames = []
    for (gi, gb), (pi, pb) in zip(gt, pred):
        frames.append((np.array([gmap[x] for x in gi], int), np.array([pmap[x] for x in pi], int),
                       iou_matrix(gb, pb)))
    res = {}
    # ---------------- CLEAR
    TP = FN = FP = IDSW = 0
    events = []  # (frame, gt_id, type) type in {'tp','fn','idsw'}
    inv_g = gt_ids_all
    prev_tr = np.full(G, -1)
    prev_timestep_tr = np.full(G, -1)
    matched_frames = np.zeros(G)
    frag = np.zeros(G)
    gt_count = np.zeros(G)
    for fi, (gi, pi, sim) in enumerate(frames):
        gt_count[gi] += 1
        if len(pi) == 0:
            FN += len(gi); prev_timestep_tr[:] = -1
            if log_events: events += [(fi, int(inv_g[g]), 'fn') for g in gi]
            continue
        if len(gi) == 0:
            FP += len(pi); prev_timestep_tr[:] = -1; continue
        score = sim.copy()
        cont = prev_timestep_tr[gi][:, None] == pi[None, :]
        score = 1000 * cont + score
        score[sim < 0.5 - 1e-9] = 0
        r, c = linear_sum_assignment(-score)
        ok = score[r, c] > 0
        r, c = r[ok], c[ok]
        mg, mp = gi[r], pi[c]
        prev = prev_tr[mg]
        sw = (prev != -1) & (prev != mp)
        IDSW += np.sum(sw)
        if log_events:
            mset = set(mg.tolist())
            events += [(fi, int(inv_g[g]), 'idsw' if s else 'tp') for g, s in zip(mg, sw)]
            events += [(fi, int(inv_g[g]), 'fn') for g in gi if g not in mset]
        # fragmentation: gt was matched before, not matched last frame, matched now
        was_prev_step = prev_timestep_tr[mg] != -1
        frag[mg] += ((matched_frames[mg] > 0) & ~was_prev_step)
        prev_timestep_tr[:] = -1
        prev_timestep_tr[mg] = mp
        prev_tr[mg] = mp
        matched_frames[mg] += 1
        TP += len(r); FN += len(gi) - len(r); FP += len(pi) - len(r)
    ratio = matched_frames / np.maximum(gt_count, 1)
    res.update(MOTA=1 - (FN + FP + IDSW) / max(1, TP + FN), IDSW=int(IDSW), FP=int(FP), FN=int(FN),
               TP=int(TP), MT=int((ratio > 0.8).sum()), ML=int((ratio < 0.2).sum()),
               Frag=int(frag.sum()), Recall=TP / max(1, TP + FN), Precision=TP / max(1, TP + FP))
    # ---------------- Identity
    pot = np.zeros((G, P))
    gcount = np.zeros(G); pcount = np.zeros(P)
    for gi, pi, sim in frames:
        gcount[gi] += 1; pcount[pi] += 1
        mg, mp = np.where(sim >= 0.5)
        np.add.at(pot, (gi[mg], pi[mp]), 1)
    if P:
        r, c = linear_sum_assignment(-pot)
        IDTP = pot[r, c].sum()
    else:
        IDTP = 0
    IDFN, IDFP = gcount.sum() - IDTP, pcount.sum() - IDTP
    res.update(IDF1=2 * IDTP / max(1, 2 * IDTP + IDFN + IDFP), IDP=IDTP / max(1, IDTP + IDFP),
               IDR=IDTP / max(1, IDTP + IDFN))
    # ---------------- HOTA
    alphas = np.arange(0.05, 0.99, 0.05)
    potm = np.zeros((G, P)); gidc = np.zeros((G, 1)); pidc = np.zeros((1, P))
    for gi, pi, sim in frames:
        denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
        s = np.zeros_like(sim); m = denom > 1e-9; s[m] = sim[m] / denom[m]
        potm[np.ix_(gi, pi)] += s
        gidc[gi] += 1; pidc[0, pi] += 1
    gas = potm / np.maximum(gidc + pidc - potm, 1e-9)
    A = len(alphas)
    htp = np.zeros(A); hfn = np.zeros(A); hfp = np.zeros(A); loc = np.zeros(A)
    mc = [np.zeros((G, P)) for _ in alphas]
    for gi, pi, sim in frames:
        if len(gi) == 0 or len(pi) == 0:
            hfn += len(gi); hfp += len(pi); continue
        sc = gas[np.ix_(gi, pi)] * sim
        r, c = linear_sum_assignment(-sc)
        for a, al in enumerate(alphas):
            ok = sim[r, c] >= al - 1e-9
            n = ok.sum()
            htp[a] += n; hfn[a] += len(gi) - n; hfp[a] += len(pi) - n
            if n:
                loc[a] += sim[r[ok], c[ok]].sum()
                mc[a][gi[r[ok]], pi[c[ok]]] += 1
    assa = np.zeros(A)
    for a in range(A):
        m = mc[a]
        assa[a] = (m * (m / np.maximum(1, gidc + pidc - m))).sum() / max(1, htp[a])
    deta = htp / np.maximum(1, htp + hfn + hfp)
    hota = np.sqrt(deta * assa)
    res.update(HOTA=hota.mean(), DetA=deta.mean(), AssA=assa.mean(),
               LocA=(np.maximum(1e-10, loc) / np.maximum(1e-10, htp)).mean(),
               HOTA_alpha=hota.tolist(), num_pred_ids=P, num_gt_ids=G)
    if log_events:
        res['events'] = events
    return res
