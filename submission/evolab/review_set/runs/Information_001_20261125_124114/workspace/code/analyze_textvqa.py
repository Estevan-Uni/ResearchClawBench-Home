"""Aggregate TextVQA results: accuracy + 95% bootstrap CIs, paired tests vs baseline, crop statistics,
pairwise crop IoU / answer agreement, hit rates vs pseudo-GT boxes, layer-wise attention ratios.
usage: python analyze_textvqa.py"""
import json, os, glob, sys, numpy as np, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vicrop_common import read_jsonl, box_iou, normalize_answer
from scipy.stats import binomtest

ROOT = '/workspace'
OUT = os.path.join(ROOT, 'outputs/analysis'); os.makedirs(OUT, exist_ok=True)
METHODS = ['baseline', 'rel', 'grad', 'pure', 'random']
NAMES = {'baseline': 'Baseline (no crop)', 'rel': 'ViCrop rel-att', 'grad': 'ViCrop grad-att',
         'pure': 'ViCrop pure-grad', 'random': 'Random crop (size-matched)'}
rng = np.random.default_rng(0)


def load(model, split='eval', tag=''):
    recs = {}
    for f in glob.glob(os.path.join(ROOT, f'outputs/textvqa_{model}{tag}/{split}/shard*.jsonl')):
        for r in read_jsonl(f):
            if 'acc' in r: recs[r['question_id']] = r
    return recs


def grounding():
    g = {}
    for f in glob.glob(os.path.join(ROOT, 'outputs/grounding/*.jsonl')):
        for r in read_jsonl(f):
            if r.get('answer') in ('unanswerable', 'yes', 'no'): r['box'] = None
            g[r['question_id']] = r
    return g


def boot_ci(x, B=10000):
    x = np.asarray(x); idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b, B=10000):
    a, b = np.asarray(a), np.asarray(b); d = b - a
    idx = rng.integers(0, len(d), (B, len(d))); md = d[idx].mean(1)
    ca, cb = a > 0, b > 0
    wins, losses = int((cb & ~ca).sum()), int((ca & ~cb).sum())
    p = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses > 0 else 1.0
    return dict(delta=float(d.mean()), ci=(float(np.percentile(md, 2.5)), float(np.percentile(md, 97.5))),
                p_boot_two_sided=float(min(1, 2 * min((md <= 0).mean(), (md >= 0).mean()))),
                wins=wins, losses=losses, mcnemar_p=float(p))


def token_cov(box_img, grid, off, region):
    """fraction of each token cell (in region coords) covered by box (image coords)."""
    gh, gw = grid; RW, RH = region
    x1, y1, x2, y2 = box_img[0] + off[0], box_img[1] + off[1], box_img[2] + off[0], box_img[3] + off[1]
    xs = np.arange(gw + 1) * RW / gw; ys = np.arange(gh + 1) * RH / gh
    cx = np.clip(np.minimum(xs[1:], x2) - np.maximum(xs[:-1], x1), 0, None) / (RW / gw)
    cy = np.clip(np.minimum(ys[1:], y2) - np.maximum(ys[:-1], y1), 0, None) / (RH / gh)
    return cy[:, None] * cx[None, :]


def contains_center(box, gt):
    cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def layer_ratios(model, split, gt):
    """mean over questions of (mean attention inside GT / mean attention over valid tokens), per layer,
    for raw A(x,q) and relative A(x,q)/A(x,q') and grad-att (if all layers stored)."""
    R = collections.defaultdict(list)
    for f in glob.glob(os.path.join(ROOT, f'outputs/textvqa_{model}/{split}/maps/*.npz')):
        qid = int(os.path.basename(f)[:-4])
        if qid not in gt or gt[qid].get('box') is None: continue
        z = np.load(f); grid = tuple(z['grid']); valid = z['valid'].astype(bool)
        cov = token_cov(gt[qid]['box'], grid, z['off'], z['region'])
        if cov.sum() < 1e-6: continue
        def ratio(a):
            a = a.reshape(grid)
            return float((a * cov).sum() / cov.sum() / a[valid].mean())
        Aq, Ag = z['Aq'], z['Ag']
        R['raw'].append([ratio(Aq[l]) for l in range(len(Aq))])
        R['rel'].append([ratio(Aq[l] / (Ag[l] + 1e-8)) for l in range(len(Aq))])
        if len(z['gband']) == len(Aq):
            R['grad'].append([ratio(z['G'][l] + 1e-12) for l in range(len(Aq))])
        if z['PG'].size > 1: R['pure'].append([ratio(z['PG'])] * len(Aq))
    return {k: np.array(v) for k, v in R.items()}


if __name__ == '__main__':
    gt = grounding()
    summary = {}
    ev_order = [q['question_id'] for q in json.load(open(os.path.join(ROOT, 'outputs/textvqa_ids.json')))['eval']]
    first102 = set(ev_order[:102])
    for model in ['llava', 'llava_ext', 'qwen']:
        recs = load(model.replace('_ext', ''))
        if model == 'llava': recs = {k: v for k, v in recs.items() if k in first102}
        if model == 'llava_ext' and len(recs) <= 102: continue
        if not recs: continue
        qids = sorted(recs)
        ms = [m for m in METHODS if all(m in recs[q]['acc'] for q in qids)]
        S = dict(n=len(qids), methods={})
        for m in ms:
            acc = [recs[q]['acc'][m] for q in qids]
            S['methods'][m] = dict(acc=float(np.mean(acc)), ci=boot_ci(acc))
            if m != 'baseline':
                S['methods'][m]['vs_baseline'] = paired([recs[q]['acc']['baseline'] for q in qids], acc)
        for a, b in [('random', 'rel'), ('random', 'grad'), ('random', 'pure'), ('grad', 'rel')]:
            if a in ms and b in ms:
                S['methods'][b].setdefault('vs', {})[a] = paired([recs[q]['acc'][a] for q in qids], [recs[q]['acc'][b] for q in qids])
        # crop statistics
        crops = {}
        for m in [x for x in ms if x != 'baseline']:
            rel_side = [np.sqrt((recs[q]['boxes'][m][2] - recs[q]['boxes'][m][0]) * (recs[q]['boxes'][m][3] - recs[q]['boxes'][m][1]) / (recs[q]['W'] * recs[q]['H'])) for q in qids]
            hits = [contains_center(recs[q]['boxes'][m], gt[q]['box']) for q in qids if q in gt and gt[q].get('box')]
            crops[m] = dict(median_rel_side=float(np.median(rel_side)), mean_rel_side=float(np.mean(rel_side)),
                            hit_rate=float(np.mean(hits)) if hits else None, n_hit=len(hits))
        S['crops'] = crops
        # pairwise IoU / answer agreement
        pm = [m for m in ['rel', 'grad', 'pure', 'random'] if m in ms]
        S['pair_iou'] = {f'{a}-{b}': float(np.mean([box_iou(recs[q]['boxes'][a], recs[q]['boxes'][b]) for q in qids]))
                         for i, a in enumerate(pm) for b in pm[i + 1:]}
        S['pair_agree'] = {f'{a}-{b}': float(np.mean([normalize_answer(recs[q]['pred'][a]) == normalize_answer(recs[q]['pred'][b]) for q in qids]))
                           for i, a in enumerate(['baseline'] + pm) for b in (['baseline'] + pm)[i + 1:]}
        # accuracy by pseudo-GT box size (small vs large evidence)
        sz = {q: np.sqrt((gt[q]['box'][2] - gt[q]['box'][0]) * (gt[q]['box'][3] - gt[q]['box'][1]) / (recs[q]['W'] * recs[q]['H']))
              for q in qids if q in gt and gt[q].get('box')}
        if sz:
            med = np.median(list(sz.values()))
            S['by_size'] = {'median_rel_side_gt': float(med)}
            for name, sel in [('small', [q for q in sz if sz[q] <= med]), ('large', [q for q in sz if sz[q] > med])]:
                S['by_size'][name] = {m: float(np.mean([recs[q]['acc'][m] for q in sel])) for m in ms}
                S['by_size'][name]['n'] = len(sel)
        T = collections.defaultdict(list)
        for q in qids:
            for k, v in recs[q]['timing'].items(): T[k].append(v)
        S['timing_mean_s'] = {k: float(np.mean(v)) for k, v in T.items()}
        summary[model] = S
        if model == 'llava_ext': summary[model] = S; continue
        lr = layer_ratios(model, 'eval', gt)
        np.savez(os.path.join(OUT, f'layer_ratios_{model}_eval.npz'), **lr)
        lrc = layer_ratios(model, 'calib', gt)
        if lrc: np.savez(os.path.join(OUT, f'layer_ratios_{model}_calib.npz'), **lrc)
    json.dump(summary, open(os.path.join(OUT, 'textvqa_summary.json'), 'w'), indent=1)
    for model, S in summary.items():
        print(f'== {model}  n={S["n"]}')
        for m, v in S['methods'].items():
            extra = ''
            if 'vs_baseline' in v:
                d = v['vs_baseline']; extra = f"  d={d['delta']*100:+.1f} [{d['ci'][0]*100:+.1f},{d['ci'][1]*100:+.1f}] W/L={d['wins']}/{d['losses']} p_mcn={d['mcnemar_p']:.3f}"
            print(f"  {m:9s} {v['acc']*100:5.1f} [{v['ci'][0]*100:.1f},{v['ci'][1]*100:.1f}]{extra}")
        for m, v in S['methods'].items():
            for a, d in v.get('vs', {}).items():
                print(f"  {m} vs {a}: d={d['delta']*100:+.1f} [{d['ci'][0]*100:+.1f},{d['ci'][1]*100:+.1f}] W/L={d['wins']}/{d['losses']} p_mcn={d['mcnemar_p']:.3f}")
        print('  crops', S['crops']); print('  iou', S['pair_iou']); print('  by_size', S.get('by_size'))
