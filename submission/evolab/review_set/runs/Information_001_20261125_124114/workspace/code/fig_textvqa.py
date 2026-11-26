"""Figures for the TextVQA experiments (reads outputs/analysis/* produced by analyze_textvqa.py)."""
import json, os, sys, glob, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mp
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_textvqa import load, grounding
from vicrop_common import read_jsonl

ROOT = '/workspace'; IMG = os.path.join(ROOT, 'report/images'); A = os.path.join(ROOT, 'outputs/analysis')
S = json.load(open(os.path.join(A, 'textvqa_summary.json')))
COL = {'baseline': '#8a8984', 'rel': '#2a78d6', 'grad': '#eb6834', 'pure': '#1baf7a', 'random': '#eda100'}
LAB = {'baseline': 'baseline', 'rel': 'rel-att', 'grad': 'grad-att', 'pure': 'pure-grad', 'random': 'random crop'}
MN = {'llava': 'LLaVA-1.5-7B', 'qwen': 'Qwen2.5-VL-3B (448² budget)'}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                     'ytick.color': '#52514e'})
models = [m for m in ['llava', 'qwen'] if m in S]
from analyze_textvqa import json as _j
FIRST102 = set(q['question_id'] for q in json.load(open(os.path.join(ROOT, 'outputs/textvqa_ids.json')))['eval'][:102])

# 1. main accuracy chart
fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.2), squeeze=False)
for ax, m in zip(axes[0], models):
    ms = list(S[m]['methods'])
    for i, k in enumerate(ms):
        v = S[m]['methods'][k]; a = v['acc'] * 100; lo, hi = v['ci'][0] * 100, v['ci'][1] * 100
        ax.barh(i, a, color=COL[k], height=0.6)
        ax.errorbar(a, i, xerr=[[a - lo], [hi - a]], color='#0b0b0b', capsize=4, lw=1.5)
        ax.text(hi + 0.8, i, f'{a:.1f}', va='center', fontsize=11, color='#0b0b0b')
    ax.set_yticks(range(len(ms))); ax.set_yticklabels([LAB[k] for k in ms]); ax.invert_yaxis()
    ax.set_xlabel('TextVQA accuracy (%)  [95% bootstrap CI]')
    ax.set_title(f"{MN[m]}  (n={S[m]['n']})", fontsize=12)
    ax.set_xlim(0, max(v['ci'][1] for v in S[m]['methods'].values()) * 100 + 9)
    ax.grid(axis='x', color='#e5e4e0', lw=0.8); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(os.path.join(IMG, 'textvqa_accuracy.png'), dpi=130); plt.close()

# 2. paired deltas vs baseline
fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 3.6), squeeze=False)
for ax, m in zip(axes[0], models):
    ms = [k for k in S[m]['methods'] if k != 'baseline']
    for i, k in enumerate(ms):
        d = S[m]['methods'][k]['vs_baseline']; x = d['delta'] * 100
        ax.errorbar(x, i, xerr=[[x - d['ci'][0] * 100], [d['ci'][1] * 100 - x]], fmt='o', color=COL[k],
                    ms=9, capsize=4, lw=2)
        ax.text(d['ci'][1] * 100 + 0.4, i, f"{x:+.1f}  (W/L {d['wins']}/{d['losses']}, p={d["mcnemar_p"]:.3f})".replace("p=0.000", "p<0.001"),
                va='center', fontsize=10, color='#0b0b0b')
    ax.axvline(0, color='#52514e', lw=1)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels([LAB[k] for k in ms]); ax.invert_yaxis()
    ax.set_xlabel('Δ accuracy vs baseline (pp), paired bootstrap 95% CI'); ax.set_title(MN[m], fontsize=12)
    lo = min(S[m]['methods'][k]['vs_baseline']['ci'][0] for k in ms) * 100
    hi = max(S[m]['methods'][k]['vs_baseline']['ci'][1] for k in ms) * 100
    ax.set_xlim(min(lo - 1, -1), hi + 12)
plt.tight_layout(); plt.savefig(os.path.join(IMG, 'textvqa_paired_delta.png'), dpi=130); plt.close()

# 3. layer-wise attention ratio inside pseudo-GT evidence box
fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 3.8), squeeze=False)
for ax, m in zip(axes[0], models):
    for split, ls in [('calib', '--'), ('eval', '-')]:
        f = os.path.join(A, f'layer_ratios_{m}_{split}.npz')
        if not os.path.exists(f): continue
        z = np.load(f)
        for k, c in [('raw', '#52514e'), ('rel', COL['rel']), ('grad', COL['grad'])]:
            if k not in z.files: continue
            g = np.exp(np.nanmean(np.log(np.clip(z[k], 1e-6, None)), 0))
            ax.plot(np.arange(len(g)), g, ls=ls, color=c, lw=2,
                    label=f"{ {'raw': 'raw A(x,q)', 'rel': 'rel-att', 'grad': 'grad-att'}[k]} ({split}, n={len(z[k])})")
    ax.axhline(1, color='#a3a29d', lw=1, ls=':')
    cal = os.path.join(A, f'calib_layers_{m}.json')
    if os.path.exists(cal):
        c = json.load(open(cal))
        for k in ['rel', 'grad']:
            b = c[k]['band']; ax.axvspan(b[0] - 0.5, b[1] - 0.5, color=COL[k], alpha=0.08)
    ax.set_yscale('log'); ax.set_xlabel('LLM layer'); ax.set_ylabel('attention ratio in evidence box\n(geo-mean; 1 = uniform)')
    ax.set_title(MN[m], fontsize=12); ax.legend(fontsize=8, frameon=False, ncol=1)
plt.tight_layout(); plt.savefig(os.path.join(IMG, 'layer_attention_ratio.png'), dpi=130); plt.close()

# 4. crop size distribution
fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 3.4), squeeze=False)
gt = grounding()
for ax, m in zip(axes[0], models):
    R = load(m)
    if m == 'llava': R = {k: v for k, v in R.items() if k in FIRST102}
    q = sorted(R)
    ks = [k for k in ['rel', 'grad', 'pure'] if k in R[q[0]]['boxes']]
    data = [[np.sqrt((R[i]['boxes'][k][2] - R[i]['boxes'][k][0]) * (R[i]['boxes'][k][3] - R[i]['boxes'][k][1]) / (R[i]['W'] * R[i]['H'])) for i in q] for k in ks]
    gts = [np.sqrt((gt[i]['box'][2] - gt[i]['box'][0]) * (gt[i]['box'][3] - gt[i]['box'][1]) / (R[i]['W'] * R[i]['H'])) for i in q if i in gt and gt[i].get('box')]
    bins = np.linspace(0, 1, 26)
    if gts: ax.hist(gts, bins=bins, color='#52514e', alpha=0.35, label=f'pseudo-GT evidence (n={len(gts)})')
    for k, d in zip(ks, data):
        ax.hist(d, bins=bins, histtype='step', lw=2.2, color=COL[k], label=f"{LAB[k]} crop (median {np.median(d):.2f})")
    ax.set_xlabel('√(box area / image area)'); ax.set_ylabel('# questions'); ax.set_title(MN[m], fontsize=12)
    ax.legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(IMG, 'crop_size_distribution.png'), dpi=130); plt.close()

# 5. hit rate + pairwise IoU
fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 3.4), squeeze=False)
for ax, m in zip(axes[0], models):
    cr = S[m]['crops']; ks = [k for k in ['rel', 'grad', 'pure', 'random'] if k in cr and cr[k]['hit_rate'] is not None]
    for i, k in enumerate(ks):
        ax.bar(i, cr[k]['hit_rate'] * 100, color=COL[k], width=0.6)
        ax.text(i, cr[k]['hit_rate'] * 100 + 1.5, f"{cr[k]['hit_rate']*100:.0f}%", ha='center', fontsize=11)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([LAB[k] for k in ks]); ax.set_ylim(0, 105)
    ax.set_ylabel('crop contains evidence centre (%)'); ax.set_title(f"{MN[m]}  (n={cr[ks[0]]['n_hit']} with pseudo-GT)", fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(IMG, 'crop_hit_rate.png'), dpi=130); plt.close()

# 6. accuracy by evidence size
fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 3.6), squeeze=False)
for ax, m in zip(axes[0], models):
    bs = S[m].get('by_size')
    if not bs: continue
    ms = list(S[m]['methods']); w = 0.8 / len(ms)
    for j, k in enumerate(ms):
        vals = [bs['small'][k] * 100, bs['large'][k] * 100]
        ax.bar(np.arange(2) + (j - (len(ms) - 1) / 2) * w, vals, width=w * 0.92, color=COL[k], label=LAB[k])
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"small evidence (n={bs['small']['n']})", f"large evidence (n={bs['large']['n']})"])
    ax.set_ylabel('accuracy (%)'); ax.set_title(f"{MN[m]} — split at median √area {bs['median_rel_side_gt']:.2f}", fontsize=11)
    ax.legend(fontsize=8, frameon=False, ncol=3)
plt.tight_layout(); plt.savefig(os.path.join(IMG, 'accuracy_by_evidence_size.png'), dpi=130); plt.close()
print('ok')
