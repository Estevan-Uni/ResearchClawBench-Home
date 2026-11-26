"""Tracker-independent diagnostics of pseudo-depth decomposition + per-occlusion error breakdown."""
import json, numpy as np
from common import load, run_eval, gt_of, image_height
from tracker import iou_matrix
from metrics import evaluate

OUT = '/workspace/outputs'
seq = load()
H = image_height(seq)

# ---------- occlusion ratio per (frame, gt) via rasterisation (union of boxes in front, larger y2)
def occlusion_ratios(gtb, res=2.0):
    n = len(gtb); occ = np.zeros(n)
    for i, b in enumerate(gtb):
        front = (gtb[:, 3] > b[3])
        ov = front & (gtb[:, 0] < b[2]) & (gtb[:, 2] > b[0]) & (gtb[:, 1] < b[3]) & (gtb[:, 3] > b[1])
        if not ov.any():
            continue
        xs = np.arange(b[0], b[2], res) + res / 2; ys = np.arange(b[1], b[3], res) + res / 2
        X, Y = np.meshgrid(xs, ys); cov = np.zeros(X.shape, bool)
        for c in gtb[ov]:
            cov |= (X >= c[0]) & (X <= c[2]) & (Y >= c[1]) & (Y <= c[3])
        occ[i] = cov.mean()
    return occ

occ = {}
neigh = {}
for f in seq:
    g = np.array(f['gt_bboxes']); o = occlusion_ratios(g)
    iou = iou_matrix(g, g); np.fill_diagonal(iou, 0)
    for k, gid in enumerate(f['gt_ids']):
        occ[(f['frame'], gid)] = o[k]; neigh[(f['frame'], gid)] = int((iou[k] > 0.3).sum())

# score vs occlusion
sc_occ = [(occ[(f['frame'], d['gt_id'])], d['score']) for f in seq for d in f['detections']]
det_flag = []
for f in seq:
    ds = {d['gt_id'] for d in f['detections']}
    det_flag += [(float(occ[(f['frame'], g)]), bool(g in ds)) for g in f['gt_ids']]

# ---------- pseudo-depth noise and decomposition trade-off
pd_err = []; ambig = {}; cross = {}
Ls = [1, 2, 3, 4, 6, 8, 12, 16]
for t in range(2, len(seq)):
    f, p1, p2 = seq[t], seq[t - 1], seq[t - 2]
    g1 = np.array(p1['gt_bboxes']); g2 = np.array(p2['gt_bboxes'])
    pred = 2 * g1 - g2                         # constant-velocity prediction of every track
    dets = f['detections']; db = np.array([d['bbox'] for d in dets]); did = np.array([d['gt_id'] for d in dets])
    gt = np.array(f['gt_bboxes'])
    pd_err += (db[:, 3] - gt[did, 3]).tolist()
    iou = iou_matrix(pred, db)                 # tracks x dets
    dp, dd = H - pred[:, 3], H - db[:, 3]
    lo, hi = min(dp.min(), dd.min()), max(dp.max(), dd.max())
    for L in Ls:
        lp = np.clip(((dp - lo) / ((hi - lo) / L)).astype(int), 0, L - 1)
        ld = np.clip(((dd - lo) / ((hi - lo) / L)).astype(int), 0, L - 1)
        same = lp[:, None] == ld[None, :]
        cand = (iou >= 0.2)                    # pairs admissible at match_thresh 0.8
        wrong = cand.copy(); wrong[did, np.arange(len(did))] = False
        ambig.setdefault(L, []).append((wrong & same).sum(0).mean())   # competing tracks per det
        cross.setdefault(L, []).append((lp[did] != ld).mean())          # true pair split across levels
    # depth gate trade-off
    h = pred[:, 3] - pred[:, 1]
    for gname, gval in [('gate', g) for g in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 3.0]]:
        keep = np.abs(dp[:, None] - dd[None, :]) <= gval * h[:, None]
        cand = (iou >= 0.2); wrong = cand.copy(); wrong[did, np.arange(len(did))] = False
        ambig.setdefault(f'g{gval}', []).append((wrong & keep).sum(0).mean())
        cross.setdefault(f'g{gval}', []).append((~keep[did, np.arange(len(did))]).mean())

diag = dict(pd_err_std=float(np.std(pd_err)), pd_err_mad=float(np.median(np.abs(pd_err))),
            pd_err=np.array(pd_err).tolist(), depth_range=H,
            ambig={str(k): float(np.mean(v)) for k, v in ambig.items()},
            cross={str(k): float(np.mean(v)) for k, v in cross.items()},
            sc_occ=sc_occ, det_flag=det_flag)

# ---------- error breakdown by occlusion for main methods (single-stage protocol)
P = dict(high_thresh=0.1, new_track_thresh=0.1, match_thresh=0.8)
meth = {'ByteTrack': ('ByteTrack', {}), 'SparseTrack(DCM,L=3)': ('SparseTrack', dict(levels_high=3, levels_low=3)),
        'Depth-gated(g=0.2)': ('SparseTrack', dict(gate=0.2))}
bins = [0, 0.5, 0.8, 0.9, 0.95, 1.0001]
breakdown = {}
preds = {}
for mn, (cls, kw) in meth.items():
    r, pred, trk = run_eval(seq, cls, **P, **kw)
    ev = evaluate(gt_of(seq), pred, log_events=True)['events']
    preds[mn] = [(ids.tolist(), b.tolist()) for ids, b in pred]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        e = [x[2] for x in ev if lo <= occ[(x[0], x[1])] < hi]
        n = len(e)
        rows.append(dict(lo=lo, hi=hi, n=n, idsw_rate=e.count('idsw') / max(1, n),
                         fn_rate=e.count('fn') / max(1, n)))
    nb = [0, 1, 2, 3, 4, 5, 100]
    rows2 = []
    for lo, hi in zip(nb[:-1], nb[1:]):
        e = [x[2] for x in ev if lo <= neigh[(x[0], x[1])] < hi]
        rows2.append(dict(lo=lo, hi=hi, n=len(e), idsw_rate=e.count('idsw') / max(1, len(e))))
    breakdown[mn] = dict(occ=rows, neigh=rows2)
    # per-level sparsity log from DCM (nonzero admissible fraction)
    if hasattr(trk, 'assoc_log') and trk.assoc_log:
        diag.setdefault('nnz', {})[mn] = trk.assoc_log[:5]
diag['breakdown'] = breakdown
json.dump(diag, open(f'{OUT}/diagnostics.json', 'w'), default=lambda o: o.item() if hasattr(o, 'item') else str(o))
json.dump(preds, open(f'{OUT}/tracks_single_protocol.json', 'w'))
print('pd err std', diag['pd_err_std'], 'mad', diag['pd_err_mad'], 'H', H)
print('ambig', diag['ambig']); print('cross', diag['cross'])
for mn, b in breakdown.items():
    print(mn, [(r['lo'], r['n'], round(r['idsw_rate'], 4)) for r in b['occ']])
    print('  neigh', [(r['lo'], r['n'], round(r['idsw_rate'], 4)) for r in b['neigh']])
