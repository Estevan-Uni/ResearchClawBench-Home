"""All quantitative experiments. Results -> /workspace/outputs/*.json"""
import json, itertools, numpy as np, os
from multiprocessing import Pool
from common import load, run_eval

OUT = '/workspace/outputs'
SEQ = load()
KEYS = ['HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1', 'IDSW', 'FP', 'FN', 'Frag', 'MT', 'ML', 'Recall',
        'Precision', 'num_pred_ids', 'num_gt_ids']

PROTOCOLS = {
    'default': dict(high_thresh=0.6, low_thresh=0.1, match_thresh=0.8, second_thresh=0.5),
    'single': dict(high_thresh=0.1, new_track_thresh=0.1, match_thresh=0.8),
    'occ2stage': dict(high_thresh=0.5, init_from_low=True, match_thresh=0.8, second_thresh=0.5),
}
METHODS = {
    'ByteTrack': ('ByteTrack', {}),
    'SparseTrack(DCM,L=3)': ('SparseTrack', dict(levels_high=3, levels_low=3)),
    'SparseTrack(DCM,L=2)': ('SparseTrack', dict(levels_high=2, levels_low=2)),
    'DCM-reverse(L=3)': ('SparseTrack', dict(levels_high=3, levels_low=3, depth_mode='reverse')),
    'DCM-random(L=3)': ('SparseTrack', dict(levels_high=3, levels_low=3, depth_mode='random')),
    'Depth-gated(g=0.2)': ('SparseTrack', dict(gate=0.2)),
}


def job(args):
    tag, name, kw, keep = args
    r, _, _ = run_eval(SEQ, name, keep_ids=keep, **kw)
    d = {k: float(r[k]) for k in KEYS}
    d['HOTA_alpha'] = r['HOTA_alpha']
    return tag, d


def run(jobs, fname):
    with Pool(min(16, os.cpu_count())) as p:
        res = p.map(job, jobs, chunksize=1)
    out = [dict(**t, **d) for t, d in res]
    json.dump(out, open(f'{OUT}/{fname}', 'w'), indent=1)
    print(fname, len(out))
    return out


if __name__ == '__main__':
    # E1: main comparison, full sequence (200 objects), 3 protocols x methods
    jobs = []
    for pn, pk in PROTOCOLS.items():
        for mn, (cls, mk) in METHODS.items():
            seeds = range(5) if 'random' in mn else [0]
            for s in seeds:
                kw = dict(pk, **mk)
                if 'random' in mn: kw['seed'] = s
                jobs.append((dict(protocol=pn, method=mn, seed=s), cls, kw, None))
    run(jobs, 'E1_main.json')

    # E2: grid over first-stage IoU threshold x depth levels (both protocols)
    jobs = []
    for pn in ['single', 'occ2stage']:
        for m1 in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
            for L in [1, 2, 3, 4, 6, 8, 12]:
                kw = dict(PROTOCOLS[pn], match_thresh=m1, levels_high=L, levels_low=L)
                jobs.append((dict(protocol=pn, match_thresh=m1, levels=L), 'SparseTrack', kw, None))
            for g in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 3.0]:
                kw = dict(PROTOCOLS[pn], match_thresh=m1, gate=g)
                jobs.append((dict(protocol=pn, match_thresh=m1, gate=g), 'SparseTrack', kw, None))
    run(jobs, 'E2_grid.json')

    # E3: SparseTrack-style split (levels_high=1, levels_low=L) in the occlusion-aware 2-stage protocol
    jobs = []
    for L in [1, 2, 3, 4, 6, 8, 12]:
        for m2 in [0.6, 0.5, 0.4]:
            kw = dict(PROTOCOLS['occ2stage'], second_thresh=m2, levels_high=1, levels_low=L)
            jobs.append((dict(levels_low=L, second_thresh=m2), 'SparseTrack', kw, None))
    run(jobs, 'E3_lowlevels.json')

    # E4: density study: random subsets of N objects, 10 subsets each
    rng = np.random.default_rng(2024)
    jobs = []
    for N in [10, 20, 50, 100, 150, 200]:
        for s in range(10 if N < 200 else 1):
            keep = set(rng.choice(200, N, replace=False).tolist()) if N < 200 else None
            for pn in ['single', 'occ2stage']:
                for mn in ['ByteTrack', 'SparseTrack(DCM,L=3)', 'SparseTrack(DCM,L=2)', 'DCM-random(L=3)',
                           'Depth-gated(g=0.2)']:
                    cls, mk = METHODS[mn]
                    jobs.append((dict(N=N, subset=s, protocol=pn, method=mn), cls,
                                 dict(PROTOCOLS[pn], **mk), keep))
    run(jobs, 'E4_density.json')
