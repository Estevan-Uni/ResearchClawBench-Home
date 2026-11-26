"""Offline ablation on the saved maps (no model calls): would box selection on the RAW attention map A(x,q) (same
layer band as rel-att, no generic-prompt normalisation) localise the evidence as well as rel-att / grad-att?
Also measures how often the map peak falls on a recurring 'attention-sink' token."""
import json, os, sys, glob, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_textvqa import grounding, contains_center, load
from vicrop_common import select_bbox

ROOT = '/workspace'
gt = grounding()
BANDS = {'llava': (list(range(14, 19)), list(range(16, 21)), 336),
         'qwen': (list(range(27, 32)), list(range(29, 34)), 448)}
out = {}
for model, (rb, gb, bs) in BANDS.items():
    recs = load(model)
    res = collections.defaultdict(list); peaks = collections.defaultdict(list)
    for f in glob.glob(os.path.join(ROOT, f'outputs/textvqa_{model}/eval/maps/*.npz')):
        qid = int(os.path.basename(f)[:-4])
        if qid not in recs or qid not in gt or not gt[qid].get('box'): continue
        z = np.load(f); grid = tuple(z['grid']); valid = z['valid']; off = z['off']; region = tuple(z['region'])
        W, H = recs[qid]['W'], recs[qid]['H']
        maps = {'raw': z['Aq'][rb].mean(0).reshape(grid),
                'rel': (z['Aq'][rb] / (z['Ag'][rb] + 1e-8)).mean(0).reshape(grid),
                'grad': z['G'][gb].sum(0).reshape(grid)}
        if z['PG'].size > 1: maps['pure'] = z['PG'].reshape(grid)
        for k, m in maps.items():
            b, _ = select_bbox(m, region, bs, valid=valid)
            b = (b[0] - off[0], b[1] - off[1], b[2] - off[0], b[3] - off[1])
            res[k].append(contains_center(b, gt[qid]['box']))
            mm = m * valid
            peaks[k].append(int(np.argmax(mm)))
    o = {}
    for k in res:
        c = collections.Counter(peaks[k]).most_common(1)[0]
        o[k] = dict(hit_rate=float(np.mean(res[k])), n=len(res[k]), top_peak_token=c[0],
                    top_peak_share=c[1] / len(peaks[k]))
    out[model] = o
    print(model, json.dumps(o))
json.dump(out, open(os.path.join(ROOT, 'outputs/analysis/offline_box_ablation.json'), 'w'), indent=1)
