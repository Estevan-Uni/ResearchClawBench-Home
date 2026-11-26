"""Figures for Experiment 1 (demo MCQs): per (question, model) one PNG with one row per localisation method:
(a) original image + selected crop box (thick cyan) + target box (dashed white), (b) localisation heatmap (viridis,
token grid, no overlay / no darkening), (c) the zoomed crop that is fed to the MLLM, with baseline -> ViCrop answers."""
import json, os, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mp
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vicrop_common import read_jsonl

ROOT = '/workspace'; IMG = os.path.join(ROOT, 'report/images'); os.makedirs(IMG, exist_ok=True)
QS = {q['id']: q for q in json.load(open(os.path.join(ROOT, 'code/demo_questions.json')))}
MNAME = {'llava': 'LLaVA-1.5-7B', 'qwen': 'Qwen2.5-VL-3B'}
METH = [('rel', 'rel-att'), ('grad', 'grad-att'), ('pure', 'pure-grad'), ('rel_hr', 'rel-att (hi-res tiles)'), ('grad_hr', 'grad-att (hi-res tiles)'), ('pure_hr', 'pure-grad (hi-res tiles)')]
LET = ['A', 'B', 'C', 'D']


def valid_crop(m, valid):
    rows = np.where(valid.any(1))[0]; cols = np.where(valid.any(0))[0]
    return m[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def ans_str(q, letter):
    return f"({letter}) {q['options'][LET.index(letter)]}"


rows_summary = []
for model, SUF in [('llava', ''), ('qwen', ''), ('llava', '_free'), ('qwen', '_free')]:
    f = os.path.join(ROOT, f'outputs/demo/demo_{model}{SUF}.jsonl')
    if not os.path.exists(f): continue
    for r in read_jsonl(f):
        q = QS[r['id']]; img = Image.open(os.path.join(ROOT, q['image'])).convert('RGB')
        z = np.load(os.path.join(ROOT, f"outputs/demo/maps_{r['id']}_{model}{SUF}.npz"))
        if SUF:  # free-form: answers are the generated strings
            GTF = {'demo1_date': '02/20/2012', 'demo2_pose': 'squatting'}[r['id']]
            for v in r['results'].values(): v['letter'] = v['text']
        meths = [(k, n) for k, n in METH if k in r['boxes'] and f'map_{k}' in z.files] + [('oracle', 'oracle (target) crop')]
        W, H = img.size
        fig, axes = plt.subplots(len(meths), 3, figsize=(17, 4.6 * len(meths) * max(0.75, H / W)),
                                 gridspec_kw=dict(width_ratios=[1.4, 1.4, 1]))
        base = r['results']['baseline']['letter']
        ok = (lambda L: 'correct' if L == q['answer'] else 'wrong') if not SUF else (lambda L: 'correct' if GTF in L.lower().replace('crouch', 'squat') else 'wrong')
        AS = (lambda L: ans_str(q, L)) if not SUF else (lambda L: f'"{L}"')
        for i, (k, n) in enumerate(meths):
            ax = axes[i, 0]; ax.imshow(img); ax.set_axis_off()
            b = r['boxes'][k]; g = q['gt_box']
            ax.add_patch(mp.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec='#00e5ff', lw=4))
            ax.add_patch(mp.Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False, ec='#ff2d55', lw=2.5))
            ax.set_title(f'(a) {n}: selected crop (cyan), target (red)', fontsize=12)
            ax = axes[i, 1]
            if k == 'oracle':
                ax.set_axis_off(); ax.text(0.5, 0.5, 'oracle: crop centred on the\nannotated target (upper bound)',
                                           ha='center', va='center', fontsize=13, transform=ax.transAxes)
            else:
                m = z[f'map_{k}']
                ax.imshow(m, cmap='viridis', interpolation='nearest', extent=(0, W, H, 0)); ax.set_axis_off()
                ax.add_patch(mp.Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False, ec='#ff2d55', lw=1.5))
                ax.set_title(f'(b) {n} map ({m.shape[0]}x{m.shape[1]} tokens; target in red)', fontsize=12)
            ax = axes[i, 2]; ax.imshow(img.crop(tuple(b))); ax.set_axis_off()
            vc = r['results'][k]['letter']; pv = max(r['results'][k]['probs'])
            ax.set_title(f"(c) zoomed crop {b[2]-b[0]}x{b[3]-b[1]} px\nbaseline: {AS(base)} [{ok(base)}]\n"
                         f"-> ViCrop: {AS(vc)} [{ok(vc)}]" + ('' if SUF else f", p={pv:.2f}"), fontsize=12, fontweight='bold')
            rows_summary.append(dict(model=model, q=r['id'], method=k, baseline=base, vicrop=vc, gt=q['answer']))
        opts = '  '.join(f'({l}) {o}' for l, o in zip(LET, q['options'])) if not SUF else '(open-ended, single word or phrase)'
        fig.suptitle(f"{MNAME[model]} | Q: {q['question']}  {opts}   [GT: {q['answer'] if not SUF else GTF}]", fontsize=13)
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        out = os.path.join(IMG, f"demo{SUF}_{r['id']}_{model}.png"); plt.savefig(out, dpi=72); plt.close()
        print('saved', out)
json.dump(rows_summary, open(os.path.join(ROOT, 'outputs/demo/fig_summary.json'), 'w'), indent=1)
