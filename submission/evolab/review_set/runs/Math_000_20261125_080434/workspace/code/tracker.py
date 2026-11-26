"""ByteTrack and SparseTrack (pseudo-depth + Depth Cascade Matching) re-implementation.

Pure numpy/scipy. Follows the structure of the official ByteTrack / SparseTrack code
(Kalman filter in xyah space, two-stage high/low score association, unconfirmed track
handling, lost-track buffer) without GMC, which is irrelevant for a static simulated camera.
"""
import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment


# ----------------------------------------------------------------------------- Kalman
class KalmanFilter:
    def __init__(self):
        ndim, dt = 4, 1.0
        self.F = np.eye(2 * ndim)
        for i in range(ndim):
            self.F[i, ndim + i] = dt
        self.H = np.eye(ndim, 2 * ndim)
        self.wp, self.wv = 1.0 / 20, 1.0 / 160

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self.wp * h, 2 * self.wp * h, 1e-2, 2 * self.wp * h,
               10 * self.wv * h, 10 * self.wv * h, 1e-5, 10 * self.wv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self.wp * h, self.wp * h, 1e-2, self.wp * h,
               self.wv * h, self.wv * h, 1e-5, self.wv * h]
        Q = np.diag(np.square(std))
        return self.F @ mean, self.F @ cov @ self.F.T + Q

    def update(self, mean, cov, z):
        h = mean[3]
        R = np.diag(np.square([self.wp * h, self.wp * h, 1e-1, self.wp * h]))
        pm, pc = self.H @ mean, self.H @ cov @ self.H.T + R
        cho = scipy.linalg.cho_factor(pc, lower=True, check_finite=False)
        K = scipy.linalg.cho_solve(cho, (cov @ self.H.T).T, check_finite=False).T
        mean = mean + (z - pm) @ K.T
        cov = cov - K @ pc @ K.T
        return mean, cov


def tlbr_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / h, h])


def xyah_to_tlbr(x):
    w = x[2] * x[3]
    h = x[3]
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


# ----------------------------------------------------------------------------- tracks
TRACKED, LOST, REMOVED = 0, 1, 2


class Track:
    _next_id = 1
    kf = KalmanFilter()

    def __init__(self, tlbr, score):
        self.det_tlbr = np.asarray(tlbr, float)
        self.score = score
        self.mean = self.cov = None
        self.is_activated = False
        self.state = TRACKED
        self.track_id = 0
        self.frame_id = self.start_frame = 0

    @property
    def tlbr(self):
        if self.mean is None:
            return self.det_tlbr.copy()
        return xyah_to_tlbr(self.mean[:4])

    def predict(self):
        m = self.mean.copy()
        if self.state != TRACKED:
            m[7] = 0
        self.mean, self.cov = self.kf.predict(m, self.cov)

    def activate(self, frame_id):
        self.track_id = Track._next_id
        Track._next_id += 1
        self.mean, self.cov = self.kf.initiate(tlbr_to_xyah(self.det_tlbr))
        self.state = TRACKED
        self.is_activated = frame_id == 1
        self.frame_id = self.start_frame = frame_id

    def update(self, det, frame_id):
        self.frame_id = frame_id
        self.mean, self.cov = self.kf.update(self.mean, self.cov, tlbr_to_xyah(det.det_tlbr))
        self.state = TRACKED
        self.is_activated = True
        self.score = det.score

    def re_activate(self, det, frame_id):
        self.update(det, frame_id)


# ----------------------------------------------------------------------------- matching utils
def iou_matrix(a, b):
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def iou_distance(tracks, dets):
    return 1 - iou_matrix([t.tlbr for t in tracks], [d.tlbr for d in dets])


def fuse_score(cost, dets):
    if cost.size == 0:
        return cost
    s = np.array([d.score for d in dets])[None, :]
    return 1 - (1 - cost) * s


def linear_assignment(cost, thresh):
    if cost.size == 0:
        return np.empty((0, 2), int), list(range(cost.shape[0])), list(range(cost.shape[1]))
    c = cost.copy()
    c[c > thresh] = thresh + 1e-4
    r, k = linear_sum_assignment(c)
    m = [(i, j) for i, j in zip(r, k) if cost[i, j] <= thresh]
    mi = {i for i, _ in m}
    mj = {j for _, j in m}
    return (np.array(m, int).reshape(-1, 2),
            [i for i in range(cost.shape[0]) if i not in mi],
            [j for j in range(cost.shape[1]) if j not in mj])


# ----------------------------------------------------------------------------- trackers
class ByteTrack:
    """ByteTrack (Zhang et al., 2022). Optionally with SparseTrack's DCM (subclass)."""

    def __init__(self, img_h=640, high_thresh=0.6, low_thresh=0.1, new_track_thresh=None,
                 match_thresh=0.8, second_thresh=0.5, unconfirmed_thresh=0.7,
                 track_buffer=30, fuse=False, init_from_low=False, low_init_thresh=0.1):
        self.img_h = img_h
        self.high, self.low = high_thresh, low_thresh
        self.new_thresh = high_thresh + 0.1 if new_track_thresh is None else new_track_thresh
        self.m1, self.m2, self.m3 = match_thresh, second_thresh, unconfirmed_thresh
        self.buffer = track_buffer
        self.fuse = fuse
        self.init_from_low, self.low_init = init_from_low, low_init_thresh
        self.tracked, self.lost, self.removed = [], [], []
        self.frame_id = 0
        Track._next_id = 1
        self.assoc_log = []  # per-frame diagnostics

    # first and second association steps; overridden by SparseTrack
    def associate(self, tracks, dets, thresh, fuse, stage):
        cost = iou_distance(tracks, dets)
        if fuse:
            cost = fuse_score(cost, dets)
        return linear_assignment(cost, thresh)

    def update(self, boxes, scores):
        self.frame_id += 1
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        scores = np.asarray(scores, float)
        hi = scores >= self.high
        lo = (scores >= self.low) & ~hi
        dets_hi = [Track(b, s) for b, s in zip(boxes[hi], scores[hi])]
        dets_lo = [Track(b, s) for b, s in zip(boxes[lo], scores[lo])]

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        confirmed = [t for t in self.tracked if t.is_activated]
        pool = confirmed + [t for t in self.lost if t not in confirmed]
        for t in pool:
            t.predict()
        for t in unconfirmed:
            t.predict()

        activated, refind, lost_now, removed_now = [], [], [], []

        # 1st association: high score detections
        m, ut, ud = self.associate(pool, dets_hi, self.m1, self.fuse, 1)
        for i, j in m:
            t = pool[i]
            if t.state == TRACKED:
                t.update(dets_hi[j], self.frame_id); activated.append(t)
            else:
                t.re_activate(dets_hi[j], self.frame_id); refind.append(t)
        # 2nd association: low score detections with remaining tracked tracks
        if self.init_from_low:   # occlusion-aware variant: lost tracks may be recovered by low dets
            r_tracks = [pool[i] for i in ut]
        else:
            r_tracks = [pool[i] for i in ut if pool[i].state == TRACKED]
        m2, ut2, ud_lo = self.associate(r_tracks, dets_lo, self.m2, False, 2)
        for i, j in m2:
            t = r_tracks[i]
            if t.state == TRACKED:
                t.update(dets_lo[j], self.frame_id); activated.append(t)
            else:
                t.re_activate(dets_lo[j], self.frame_id); refind.append(t)
        for i in ut2:
            t = r_tracks[i]
            if t.state == TRACKED:
                t.state = LOST; lost_now.append(t)
        # unconfirmed tracks with remaining high detections (+ low ones in occlusion-aware variant)
        dets_left = [dets_hi[j] for j in ud]
        if self.init_from_low:
            dets_left = dets_left + [dets_lo[j] for j in ud_lo]
        cost = iou_distance(unconfirmed, dets_left)
        if self.fuse:
            cost = fuse_score(cost, dets_left)
        m3, uu, ud3 = linear_assignment(cost, self.m3)
        for i, j in m3:
            unconfirmed[i].update(dets_left[j], self.frame_id); activated.append(unconfirmed[i])
        for i in uu:
            unconfirmed[i].state = REMOVED; removed_now.append(unconfirmed[i])
        # new tracks
        for j in ud3:
            d = dets_left[j]
            if d.score < (self.low_init if self.init_from_low else self.new_thresh):
                continue
            d.activate(self.frame_id); activated.append(d)
        # (adaptation) unmatched low-score dets may start tracks: in this data low score == occluded,
        # not unreliable, so without this the occluded majority can never be initialised.
        if False:
            left_lo = [dets_lo[j] for j in ud_lo]
            if left_lo and activated:
                ov = iou_matrix([d.tlbr for d in left_lo], [t.tlbr for t in activated]).max(1)
            else:
                ov = np.zeros(len(left_lo))
            for d, o in zip(left_lo, ov):
                if d.score >= self.low_init and o < 0.7:
                    d.activate(self.frame_id); activated.append(d)
        # remove old lost tracks
        for t in self.lost:
            if self.frame_id - t.frame_id > self.buffer:
                t.state = REMOVED; removed_now.append(t)

        self.tracked = [t for t in self.tracked if t.state == TRACKED]
        self.tracked = _join(self.tracked, activated)
        self.tracked = _join(self.tracked, refind)
        self.lost = _sub(self.lost, self.tracked)
        self.lost.extend(lost_now)
        self.lost = _sub(self.lost, removed_now)
        self.removed.extend(removed_now)
        self.tracked, self.lost = _remove_duplicates(self.tracked, self.lost)
        out = [t for t in self.tracked if t.is_activated]
        return [(t.track_id, t.tlbr, t.score) for t in out]


class SparseTrack(ByteTrack):
    """SparseTrack (Liu et al., 2023): pseudo-depth + Depth Cascade Matching (DCM).

    Pseudo-depth of a box = distance from its bottom edge to the image bottom
    (img_h - y2): small = close to camera.  Detections and tracks (KF-predicted box)
    are split into `levels` depth intervals spanning [min, max] pseudo-depth; association
    proceeds from near to far, and unmatched tracks/detections of a shallower level are
    carried to the next level.
    depth_mode: 'pseudo' (SparseTrack), 'random' (control: random partition),
                'reverse' (far-to-near cascade).
    """

    def __init__(self, levels_high=3, levels_low=3, depth_mode='pseudo', seed=0,
                 margin=0.0, gate=None, **kw):
        super().__init__(**kw)
        self.margin = margin  # soft-boundary DCM: overlap between adjacent levels (fraction of level width)
        self.gate = gate      # joint depth-gated assignment: max |pseudo-depth diff| / track height
        self.lv = {1: levels_high, 2: levels_low}
        self.mode = depth_mode
        self.rng = np.random.default_rng(seed)

    def pseudo_depth(self, objs):
        return np.array([self.img_h - o.tlbr[3] for o in objs])

    def level_index(self, depth, lo, hi, levels):
        if hi - lo < 1e-9 or levels <= 1:
            return np.zeros(len(depth), int)
        step = (hi - lo) / levels
        return np.clip(((depth - lo) / step).astype(int), 0, levels - 1)

    def associate(self, tracks, dets, thresh, fuse, stage):
        levels = self.lv[stage]
        if self.gate is not None and len(tracks) and len(dets):
            return self.gated_associate(tracks, dets, thresh, fuse)
        if levels <= 1 or len(tracks) == 0 or len(dets) == 0:
            return super().associate(tracks, dets, thresh, fuse, stage)
        dt, dd = self.pseudo_depth(tracks), self.pseudo_depth(dets)
        lo, hi = min(dt.min(), dd.min()), max(dt.max(), dd.max())
        lt = self.level_index(dt, lo, hi, levels)
        ld = self.level_index(dd, lo, hi, levels)
        if self.mode == 'random':
            lt = self.rng.integers(0, levels, len(tracks))
            ld = self.rng.integers(0, levels, len(dets))
        order = range(levels) if self.mode != 'reverse' else range(levels - 1, -1, -1)
        matches, carry_t, carry_d = [], [], []
        nnz = []
        width = (hi - lo) / levels if hi > lo else 1.0
        used_d = set()
        for k in order:
            ti = carry_t + [i for i in range(len(tracks)) if lt[i] == k]
            if self.margin > 0 and self.mode == 'pseudo':
                a, b = lo + (k - self.margin) * width, lo + (k + 1 + self.margin) * width
                di = [j for j in range(len(dets)) if j not in used_d and a <= dd[j] <= b]
                di = sorted(set(di) | set(carry_d), key=lambda j: dd[j])
            else:
                di = carry_d + [j for j in range(len(dets)) if ld[j] == k]
            if len(ti) == 0 or len(di) == 0:
                carry_t, carry_d = ti, di
                continue
            sub_t = [tracks[i] for i in ti]
            sub_d = [dets[j] for j in di]
            cost = iou_distance(sub_t, sub_d)
            if fuse:
                cost = fuse_score(cost, sub_d)
            nnz.append(((cost < 1).sum(), cost.size))
            m, ut, ud = linear_assignment(cost, thresh)
            matches += [(ti[a], di[b]) for a, b in m]
            carry_t = [ti[a] for a in ut]
            used_d |= {di[b] for _, b in m}
            # with overlapping levels, only dets of this (non-expanded) level are carried on
            carry_d = [di[b] for b in ud if (self.margin == 0 or ld[di[b]] <= k)]
        if self.margin > 0:  # any detection not yet used must still be returned unmatched
            pass
        mt = {i for i, _ in matches}
        md = {j for _, j in matches}
        self.assoc_log.append((stage, nnz))
        return (np.array(matches, int).reshape(-1, 2),
                [i for i in range(len(tracks)) if i not in mt],
                [j for j in range(len(dets)) if j not in md])


def _gated(self, tracks, dets, thresh, fuse):
    cost = iou_distance(tracks, dets)
    if fuse:
        cost = fuse_score(cost, dets)
    dt, dd = self.pseudo_depth(tracks), self.pseudo_depth(dets)
    h = np.array([t.tlbr[3] - t.tlbr[1] for t in tracks])
    cost[np.abs(dt[:, None] - dd[None, :]) > self.gate * h[:, None]] = 1.0
    return linear_assignment(cost, thresh)


SparseTrack.gated_associate = _gated


def _join(a, b):
    ids = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in ids]


def _sub(a, b):
    ids = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids]


def _remove_duplicates(a, b):
    if not a or not b:
        return a, b
    d = iou_distance(a, b)
    pa, pb = set(), set()
    for p, q in zip(*np.where(d < 0.15)):
        ta = a[p].frame_id - a[p].start_frame
        tb = b[q].frame_id - b[q].start_frame
        if ta > tb:
            pb.add(q)
        else:
            pa.add(p)
    return [t for i, t in enumerate(a) if i not in pa], [t for i, t in enumerate(b) if i not in pb]


def run_tracker(tracker, seq, keep_ids=None):
    """Run tracker over sequence. Returns list (per frame) of (ids, boxes)."""
    results = []
    for f in seq:
        dets = f['detections']
        if keep_ids is not None:
            dets = [x for x in dets if x['gt_id'] in keep_ids]
        b = np.array([x['bbox'] for x in dets], float).reshape(-1, 4)
        s = np.array([x['score'] for x in dets], float)
        out = tracker.update(b, s)
        results.append((np.array([o[0] for o in out], int),
                        np.array([o[1] for o in out], float).reshape(-1, 4)))
    return results
