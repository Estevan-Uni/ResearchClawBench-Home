import json, numpy as np
from tracker import ByteTrack, SparseTrack, run_tracker
from metrics import evaluate

DATA = '/workspace/data/simulated_sequence.json'


def load():
    return json.load(open(DATA))


def image_height(seq):
    return float(np.ceil(max(max(b[3] for b in f['gt_bboxes']) for f in seq)))


def gt_of(seq, keep_ids=None):
    out = []
    for f in seq:
        ids = np.array(f['gt_ids']); b = np.array(f['gt_bboxes'], float)
        if keep_ids is not None:
            m = np.isin(ids, list(keep_ids)); ids, b = ids[m], b[m]
        out.append((ids, b))
    return out


def make(name, img_h, **kw):
    if name == 'ByteTrack':
        kw = {k: v for k, v in kw.items() if k not in ('levels_high', 'levels_low', 'depth_mode', 'seed')}
        return ByteTrack(img_h=img_h, **kw)
    return SparseTrack(img_h=img_h, **kw)


def run_eval(seq, name, keep_ids=None, **kw):
    t = make(name, image_height(seq), **kw)
    pred = run_tracker(t, seq, keep_ids)
    r = evaluate(gt_of(seq, keep_ids), pred)
    return r, pred, t
