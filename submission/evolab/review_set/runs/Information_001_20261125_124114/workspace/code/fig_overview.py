"""Data overview figure: demo images with target boxes, TextVQA subset examples with pseudo-GT evidence boxes,
and distribution of the evidence size relative to the image."""
import json, os, sys, glob, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mp
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_textvqa import grounding

ROOT = '/workspace'; IMG = os.path.join(ROOT, 'report/images')
meta = json.load(open(os.path.join(ROOT, 'outputs/textvqa_ids.json')))
gt = grounding(); QS = json.load(open(os.path.join(ROOT, 'code/demo_questions.json')))
ev = [q for q in meta['eval'] if q['question_id'] in gt and gt[q['question_id']].get('box')]
rel = [np.sqrt((gt[q['question_id']]['box'][2] - gt[q['question_id']]['box'][0]) * (gt[q['question_id']]['box'][3] - gt[q['question_id']]['box'][1]) / (q['W'] * q['H'])) for q in ev]

fig = plt.figure(figsize=(16, 8.2))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])
for i, (path, qids) in enumerate([('data/demo_imgs/demo1.png', ['demo1_date', 'demo1_helmet']), ('data/demo_imgs/demo2.png', ['demo2_pose'])]):
    ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2]); im = Image.open(os.path.join(ROOT, path)); ax.imshow(im); ax.set_axis_off()
    for q in QS:
        if q['id'] in qids:
            b = q['gt_box']; ax.add_patch(mp.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec='#00e5ff', lw=3))
            ax.text(b[0], b[1] - 8, q['id'].split('_')[1], color='#00e5ff', fontsize=11, fontweight='bold')
    ax.set_title(f"{os.path.basename(path)}  ({im.size[0]}x{im.size[1]} px) — demo MCQ targets", fontsize=12)
rng = np.random.default_rng(3); pick = rng.choice(len(ev), 3, replace=False)
for j, k in enumerate(pick):
    q = ev[k]; ax = fig.add_subplot(gs[1, j]); im = Image.open(os.path.join(ROOT, 'outputs/ext/textvqa_imgs', q['image_id'] + '.jpg'))
    ax.imshow(im); ax.set_axis_off(); b = gt[q['question_id']]['box']
    ax.add_patch(mp.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec='#ff2d55', lw=3))
    ax.set_title(f"Q: {q['question'][:42]}\nA: {gt[q['question_id']]['answer'][:30]}", fontsize=10)
ax = fig.add_subplot(gs[1, 3])
ax.hist(rel, bins=np.linspace(0, 0.6, 25), color='#2a78d6')
ax.axvline(np.median(rel), color='#0b0b0b', ls='--', lw=1.5); ax.text(np.median(rel) + 0.01, ax.get_ylim()[1] * 0.9, f'median {np.median(rel):.2f}', fontsize=10)
ax.set_xlabel('√(evidence box area / image area)'); ax.set_ylabel('# TextVQA eval questions')
ax.set_title(f'Evidence size (pseudo-GT, n={len(rel)}/{len(meta["eval"])})', fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(IMG, 'data_overview.png'), dpi=110); plt.close(); print('ok', len(rel))
