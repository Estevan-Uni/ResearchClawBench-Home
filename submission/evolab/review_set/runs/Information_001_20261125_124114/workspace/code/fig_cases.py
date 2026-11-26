"""TextVQA qualitative cases (deterministic selection): for each model the first 3 questions (by question id) where the
baseline scores 0 and rel-att >= 0.6, plus the first regression (baseline >= 0.6, rel-att 0). Columns: image + rel-att
box (cyan) + pseudo-GT (red) | rel-att map | zoomed crop with answers."""
import os, sys, json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mp
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_textvqa import load, grounding

ROOT = '/workspace'; gt = grounding()
FIRST102 = set(q['question_id'] for q in json.load(open(os.path.join(ROOT, 'outputs/textvqa_ids.json')))['eval'][:102])
BANDS = {'llava': list(range(14, 19)), 'qwen': list(range(27, 32))}
MN = {'llava': 'LLaVA-1.5-7B', 'qwen': 'Qwen2.5-VL-3B'}
for model in ['llava', 'qwen']:
    R = load(model)
    if model == 'llava': R = {k: v for k, v in R.items() if k in FIRST102}
    qs = sorted(R)
    good = [q for q in qs if R[q]['acc']['baseline'] == 0 and R[q]['acc']['rel'] >= 0.6][:3]
    bad = [q for q in qs if R[q]['acc']['baseline'] >= 0.6 and R[q]['acc']['rel'] == 0][:1]
    sel = good + bad
    fig, axes = plt.subplots(len(sel), 3, figsize=(15, 4.2 * len(sel)), gridspec_kw=dict(width_ratios=[1.3, 1.3, 1]))
    for i, q in enumerate(sel):
        r = R[q]; img = Image.open(os.path.join(ROOT, 'outputs/ext/textvqa_imgs', r['image_id'] + '.jpg')).convert('RGB')
        W, H = img.size
        z = np.load(os.path.join(ROOT, f'outputs/textvqa_{model}/eval/maps/{q}.npz'))
        grid = tuple(z['grid']); rb = BANDS[model]
        m = (z['Aq'][rb] / (z['Ag'][rb] + 1e-8)).mean(0).reshape(grid)
        if model == 'llava':
            v = z['valid'].astype(bool); rows = np.where(v.any(1))[0]; cols = np.where(v.any(0))[0]
            m = m[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        b = r['boxes']['rel']
        ax = axes[i, 0]; ax.imshow(img); ax.set_axis_off()
        ax.add_patch(mp.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec='#00e5ff', lw=4))
        if q in gt and gt[q].get('box'):
            g = gt[q]['box']; ax.add_patch(mp.Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False, ec='#ff2d55', lw=2.5))
        tag = 'CORRECTED' if q in good else 'REGRESSION'
        ax.set_title(f"[{tag}] Q: {r['question'][:60]}", fontsize=11)
        ax = axes[i, 1]; ax.imshow(m, cmap='viridis', interpolation='nearest', extent=(0, W, H, 0)); ax.set_axis_off()
        ax.set_title(f'rel-att map ({m.shape[0]}x{m.shape[1]} tokens)', fontsize=11)
        ax = axes[i, 2]; ax.imshow(img.crop(tuple(b))); ax.set_axis_off()
        ax.set_title(f"baseline: \"{r['pred']['baseline'][:22]}\" ({r['acc']['baseline']:.1f})\nrel-att ViCrop: \"{r['pred']['rel'][:22]}\" ({r['acc']['rel']:.1f})\nGT: {r['answers'][0][:25]}",
                     fontsize=11, fontweight='bold')
    fig.suptitle(f'{MN[model]} — TextVQA cases (cyan = rel-att crop, red = pseudo-GT evidence)', fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.97)); plt.savefig(os.path.join(ROOT, f'report/images/textvqa_cases_{model}.png'), dpi=75); plt.close()
    print(model, sel)
