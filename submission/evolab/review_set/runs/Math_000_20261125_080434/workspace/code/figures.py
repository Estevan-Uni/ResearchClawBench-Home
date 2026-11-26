import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import load, gt_of

OUT, IMG = '/workspace/outputs', '/workspace/report/images'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e3', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                     'ytick.color': '#52514e', 'legend.frameon': False, 'figure.dpi': 130, 'savefig.dpi': 160})
C = {'ByteTrack': '#2a78d6', 'SparseTrack(DCM,L=3)': '#eb6834', 'SparseTrack(DCM,L=2)': '#eda100',
     'Depth-gated(g=0.2)': '#1baf7a', 'DCM-reverse(L=3)': '#6250d6', 'DCM-random(L=3)': '#8a8984'}
LBL = {'ByteTrack': 'ByteTrack', 'SparseTrack(DCM,L=3)': 'SparseTrack DCM (L=3)', 'SparseTrack(DCM,L=2)': 'SparseTrack DCM (L=2)',
       'Depth-gated(g=0.2)': 'Depth-gated joint (g=0.2)', 'DCM-reverse(L=3)': 'DCM far→near (L=3)',
       'DCM-random(L=3)': 'Random partition (L=3)'}
PROT = {'single': 'Single-stage protocol (τ_high = 0.1)', 'occ2stage': 'Occlusion-aware two-stage (τ_high = 0.5)'}
seq = load()
diag = json.load(open(f'{OUT}/diagnostics.json'))

# ------------------------------------------------------------------ Fig 1 data overview
fig, ax = plt.subplots(1, 4, figsize=(16, 4.3), gridspec_kw=dict(width_ratios=[1.0, 1.1, 1.1, 1.1]))
f0 = seq[50]; g = np.array(f0['gt_bboxes'])
from diagnostics_helpers import occ_frame
o = occ_frame(g)
cmap = plt.get_cmap('Blues')
order = np.argsort(g[:, 3])
for k in order:
    b = g[k]
    ax[0].add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=True, fc=cmap(0.15 + 0.8 * o[k]),
                              ec='white', lw=0.6, alpha=0.95))
for d in f0['detections']:
    if d['score'] > 0.5:
        b = d['bbox']; ax[0].add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec='#eb6834', lw=1.4))
ax[0].set_xlim(-5, 490); ax[0].set_ylim(650, -5); ax[0].set_aspect('equal'); ax[0].grid(False)
ax[0].set_title('(a) Frame 50: 200 GT boxes\n(fill = occlusion ratio; orange = score-0.9 dets)', fontsize=9)
sc = np.array(diag['sc_occ'])
ax[1].hist(sc[sc[:, 1] < 0.5, 1], bins=np.linspace(0, 1, 41), color='#2a78d6', label='occluded (score 0.1–0.4)')
ax[1].hist(sc[sc[:, 1] > 0.5, 1], bins=np.linspace(0, 1, 41), color='#eb6834', label='unoccluded (score 0.9)')
ax[1].set_yscale('log'); ax[1].set_xlabel('detection confidence'); ax[1].set_ylabel('count (log)')
ax[1].set_title('(b) Score distribution (n=%d dets)' % len(sc), fontsize=9); ax[1].legend(fontsize=8)
df = np.array(diag['det_flag'], float)
ax[2].hist(df[:, 0], bins=np.linspace(0, 1, 41), color='#2a78d6')
ax[2].set_xlabel('occlusion ratio (area covered by boxes in front)'); ax[2].set_ylabel('object-frames')
ax2 = ax[2].twinx(); bins = np.linspace(0, 1, 11); idx = np.digitize(df[:, 0], bins) - 1
rate = [df[idx == i, 1].mean() if (idx == i).sum() > 20 else np.nan for i in range(10)]
ax2.plot((bins[:-1] + bins[1:]) / 2, rate, 'o-', color='#0b0b0b', ms=5, lw=1.5); ax2.set_ylim(0, 1)
ax2.set_ylabel('detection rate', color='#0b0b0b'); ax2.grid(False); ax2.spines['right'].set_visible(True)
ax[2].set_title('(c) Occlusion is pervasive; recall ≈0.79 irrespective', fontsize=9)
# neighbours
import tracker
nn = []
for f in seq:
    gg = np.array(f['gt_bboxes']); I = tracker.iou_matrix(gg, gg); np.fill_diagonal(I, 0)
    nn += (I > 0.3).sum(1).tolist()
ax[3].hist(nn, bins=np.arange(0, max(nn) + 2) - 0.5, color='#2a78d6')
ax[3].set_xlabel('# other GT objects with IoU > 0.3'); ax[3].set_ylabel('object-frames')
ax[3].set_title('(d) Crowding: mean %.1f neighbours (IoU>0.3)' % np.mean(nn), fontsize=9)
plt.tight_layout(); plt.savefig(f'{IMG}/fig1_data_overview.png'); plt.close()

# ------------------------------------------------------------------ Fig 2 main comparison
e1 = pd.DataFrame(json.load(open(f'{OUT}/E1_main.json')))
agg = e1.groupby(['protocol', 'method'])[['HOTA', 'AssA', 'DetA', 'MOTA', 'IDF1']].agg(['mean', 'std'])
meths = ['ByteTrack', 'SparseTrack(DCM,L=2)', 'SparseTrack(DCM,L=3)', 'DCM-reverse(L=3)', 'DCM-random(L=3)', 'Depth-gated(g=0.2)']
mets = ['HOTA', 'AssA', 'DetA', 'MOTA', 'IDF1']
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
w = 0.13
for a, p in zip(ax, ['single', 'occ2stage']):
    for i, m in enumerate(meths):
        mu = [agg.loc[(p, m), (k, 'mean')] for k in mets]; sd = [agg.loc[(p, m), (k, 'std')] for k in mets]
        sd = [0 if np.isnan(s) else s for s in sd]
        x = np.arange(len(mets)) + (i - 2.5) * w
        a.bar(x, mu, w * 0.9, color=C[m], label=LBL[m], yerr=[s if s > 0 else None for s in sd] if any(s > 0 for s in sd) else None, capsize=2, error_kw=dict(lw=0.8))
        if m in ('ByteTrack', 'Depth-gated(g=0.2)', 'SparseTrack(DCM,L=3)'):
            for xx, v in zip(x, mu):
                a.text(xx, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=6, rotation=90, color='#52514e')
    a.set_xticks(range(len(mets))); a.set_xticklabels(mets); a.set_title(PROT[p], fontsize=10); a.set_ylim(0, 1)
ax[0].set_ylabel('score (200 objects, 100 frames)')
ax[1].legend(ncol=2, fontsize=8, loc='upper left')
plt.tight_layout(); plt.savefig(f'{IMG}/fig2_main_comparison.png'); plt.close()

# ------------------------------------------------------------------ Fig 3 level / gate sweeps
e2 = pd.DataFrame(json.load(open(f'{OUT}/E2_grid.json')))
fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
blues = plt.get_cmap('Blues')
mts = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
for row, p in enumerate(['single', 'occ2stage']):
    sub = e2[(e2.protocol == p) & e2.levels.notna()]
    for k, m1 in enumerate(mts):
        s = sub[sub.match_thresh == m1].sort_values('levels')
        ax[row, 0].plot(s.levels, s.HOTA, 'o-', color=blues(0.3 + 0.7 * k / 5), lw=2, ms=5, label=f'IoU-cost thr {m1}')
    ax[row, 0].set_xscale('log', base=2); ax[row, 0].set_xticks([1, 2, 3, 4, 6, 8, 12]); ax[row, 0].set_xticklabels([1, 2, 3, 4, 6, 8, 12])
    ax[row, 0].set_xlabel('# depth levels L in DCM (L=1 ≡ ByteTrack)'); ax[row, 0].set_ylabel('HOTA')
    ax[row, 0].set_title(f'DCM depth levels — {PROT[p]}', fontsize=9)
    sub = e2[(e2.protocol == p) & e2.gate.notna()]
    for k, m1 in enumerate(mts):
        s = sub[sub.match_thresh == m1].sort_values('gate')
        base = e2[(e2.protocol == p) & (e2.levels == 1) & (e2.match_thresh == m1)].HOTA.values[0]
        ax[row, 1].plot(s.gate, s.HOTA, 'o-', color=blues(0.3 + 0.7 * k / 5), lw=2, ms=5, label=f'IoU-cost thr {m1}')
        ax[row, 1].axhline(base, color=blues(0.3 + 0.7 * k / 5), lw=0.8, ls=':')
    ax[row, 1].set_xscale('log'); ax[row, 1].set_xlabel('depth gate g  (max |Δ pseudo-depth| / box height)')
    ax[row, 1].set_ylabel('HOTA'); ax[row, 1].set_title(f'Joint depth-gated assignment — {PROT[p]}\n(dotted = ByteTrack at same threshold)', fontsize=9)
ax[0, 0].legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig(f'{IMG}/fig3_level_gate_sweeps.png'); plt.close()

# ------------------------------------------------------------------ Fig 4 density
e4 = pd.DataFrame(json.load(open(f'{OUT}/E4_density.json')))
dm = ['ByteTrack', 'SparseTrack(DCM,L=2)', 'SparseTrack(DCM,L=3)', 'Depth-gated(g=0.2)', 'DCM-random(L=3)']
fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
for row, p in enumerate(['single', 'occ2stage']):
    for col, met in enumerate(['HOTA', 'IDF1', 'IDSW']):
        a = ax[row, col]
        for m in dm:
            s = e4[(e4.protocol == p) & (e4.method == m)].copy()
            if met == 'IDSW':
                s['v'] = s.IDSW / s.N
            else:
                s['v'] = s[met]
            gg = s.groupby('N').v.agg(['mean', 'sem']).reset_index(); e = gg['sem'].fillna(0).values
            lo = np.minimum(e, 0.9 * gg['mean'].values)
            a.errorbar(gg.N, gg['mean'], yerr=[lo, e], color=C[m], marker='o', ms=5, lw=2, capsize=2, label=LBL[m])
        a.set_xlabel('# objects in scene (random subsets of the 200)')
        a.set_ylabel(met if met != 'IDSW' else 'ID switches per object')
        if met == 'IDSW': a.set_yscale('log')
        a.set_title(f'{met} — {PROT[p]}  (mean ± s.e.m., 10 subsets)', fontsize=9)
ax[0, 0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{IMG}/fig4_density.png'); plt.close()

# ------------------------------------------------------------------ Fig 5 mechanism
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
pe = np.array(diag['pd_err'])
ax[0].hist(pe, bins=80, color='#2a78d6')
ax[0].set_xlabel('detection y2 − GT y2  (px) = pseudo-depth error'); ax[0].set_ylabel('count')
ax[0].set_title(f'(a) Pseudo-depth noise: σ = {pe.std():.1f} px\n(depth range {diag["depth_range"]:.0f} px; L=4 level width = {diag["depth_range"]/4:.0f} px)', fontsize=9)
Ls = [1, 2, 3, 4, 6, 8, 12, 16]; gs = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 3.0]
am = diag['ambig']; cr = diag['cross']
xL = [am[str(L)] for L in Ls]; yL = [100 * cr[str(L)] for L in Ls]
xg = [am[f'g{g}'] for g in gs]; yg = [100 * cr[f'g{g}'] for g in gs]
ax[1].plot(xL, yL, 'o-', color=C['SparseTrack(DCM,L=3)'], lw=2, ms=6, label='hard depth levels (DCM)')
for L, x, y in zip(Ls, xL, yL): ax[1].annotate(f'L={L}', (x, y), textcoords='offset points', xytext=(5, 4), fontsize=7, color='#52514e')
ax[1].plot(xg, yg, 's-', color=C['Depth-gated(g=0.2)'], lw=2, ms=6, label='relative depth gate')
for g_, x, y in zip(gs, xg, yg):
    if g_ in (0.05, 0.1, 0.2, 0.5): ax[1].annotate(f'g={g_}', (x, y), textcoords='offset points', xytext=(5, -10), fontsize=7, color='#52514e')
ax[1].set_xlabel('wrong admissible candidates per detection (IoU ≥ 0.2, same subset)')
ax[1].set_ylabel('% true pairs separated by decomposition')
ax[1].set_title('(b) Sparsification vs. separation trade-off\n(constant-velocity predictions, all 200 objects)', fontsize=9)
ax[1].set_yscale('symlog', linthresh=0.1); ax[1].legend(fontsize=8)
# per-level sparsity: how many candidates remain as function of L, normalized
ax[2].plot(Ls, xL, 'o-', color=C['SparseTrack(DCM,L=3)'], lw=2, label='wrong candidates / det')
ax[2].set_xscale('log', base=2); ax[2].set_xticks(Ls); ax[2].set_xticklabels(Ls)
ax[2].set_xlabel('# depth levels L'); ax[2].set_ylabel('wrong candidates per detection')
a2 = ax[2].twinx(); a2.plot(Ls, yL, 's--', color='#52514e', lw=1.5, label='% true pairs split'); a2.set_ylabel('% true pairs split across levels')
a2.grid(False); a2.spines['right'].set_visible(True)
ax[2].set_title('(c) Hard partition: ambiguity ∝ 1/L, split rate ∝ L', fontsize=9)
h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax[2].legend(h1 + h2, l1 + l2, fontsize=8, loc='upper center')
plt.tight_layout(); plt.savefig(f'{IMG}/fig5_mechanism.png'); plt.close()

# ------------------------------------------------------------------ Fig 6 breakdown
br = diag['breakdown']
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
bm = ['ByteTrack', 'SparseTrack(DCM,L=3)', 'Depth-gated(g=0.2)']
for i, m in enumerate(bm):
    r = br[m]['occ']; x = np.arange(len(r)) + (i - 1) * 0.27
    ax[0].bar(x, [100 * q['idsw_rate'] for q in r], 0.25, color=C[m], label=LBL[m])
    r2 = br[m]['neigh']; x2 = np.arange(len(r2)) + (i - 1) * 0.27
    ax[1].bar(x2, [100 * q['idsw_rate'] for q in r2], 0.25, color=C[m], label=LBL[m])
r = br['ByteTrack']['occ']
ax[0].set_xticks(range(len(r))); ax[0].set_xticklabels([f"{q['lo']:.2f}–{min(q['hi'],1):.2f}\n(n={q['n']})" for q in r], fontsize=8)
ax[0].set_xlabel('GT occlusion ratio'); ax[0].set_ylabel('ID switches per 100 matched object-frames')
ax[0].set_title('(a) ID-switch rate by occlusion level (single-stage, 200 objects)', fontsize=9); ax[0].legend(fontsize=8)
r2 = br['ByteTrack']['neigh']
ax[1].set_xticks(range(len(r2))); ax[1].set_xticklabels([f"{q['lo']}" + ('+' if q['hi'] > 50 else '') + f"\n(n={q['n']})" for q in r2], fontsize=8)
ax[1].set_xlabel('# GT neighbours with IoU > 0.3'); ax[1].set_ylabel('ID switches per 100 matched object-frames')
ax[1].set_title('(b) ID-switch rate by local crowding', fontsize=9)
plt.tight_layout(); plt.savefig(f'{IMG}/fig6_occlusion_breakdown.png'); plt.close()

# ------------------------------------------------------------------ Fig 7 identity timelines
tr = json.load(open(f'{OUT}/tracks_single_protocol.json'))
from tracker import iou_matrix
from scipy.optimize import linear_sum_assignment
gts = gt_of(seq)
def assign(pred):
    A = np.full((200, len(seq)), -1)
    for t, ((gi, gb), (pi, pb)) in enumerate(zip(gts, pred)):
        if len(pi) == 0: continue
        I = iou_matrix(gb, np.array(pb)); r, c = linear_sum_assignment(-I)
        for a, b in zip(r, c):
            if I[a, b] >= 0.5: A[gi[a], t] = pi[b]
    return A
rng = np.random.default_rng(1); sel = np.sort(rng.choice(200, 40, replace=False))
fig, ax = plt.subplots(1, 3, figsize=(16, 6.5), sharey=True)
for a, m in zip(ax, bm):
    A = assign(tr[m])
    nsw = 0
    for row, gidx in enumerate(sel):
        seg = A[gidx]; prev = None; col = 0
        for t in range(len(seq)):
            if seg[t] < 0:
                continue
            if prev is not None and seg[t] != prev:
                col ^= 1; a.plot([t, t], [row - 0.45, row + 0.45], color='#0b0b0b', lw=1.2); nsw += 1
            a.add_patch(Rectangle((t, row - 0.35), 1, 0.7, color=C[m], alpha=0.95 if col == 0 else 0.45, lw=0))
            prev = seg[t]
    a.set_xlim(0, len(seq)); a.set_ylim(-1, len(sel)); a.grid(False)
    a.set_title(f'{LBL[m]}\n({nsw} identity changes among 40 sampled objects)', fontsize=9)
    a.set_xlabel('frame')
ax[0].set_ylabel('sampled GT object'); ax[0].set_yticks(range(0, 40, 5)); ax[0].set_yticklabels(sel[::5])
plt.tight_layout(); plt.savefig(f'{IMG}/fig7_identity_timelines.png'); plt.close()

# ------------------------------------------------------------------ Fig 8 HOTA vs alpha
fig, a = plt.subplots(figsize=(6, 4))
alphas = np.arange(0.05, 0.99, 0.05)
for m in ['ByteTrack', 'SparseTrack(DCM,L=3)', 'Depth-gated(g=0.2)']:
    v = e1[(e1.protocol == 'single') & (e1.method == m)].HOTA_alpha.values[0]
    a.plot(alphas, v, 'o-', ms=4, color=C[m], lw=2, label=LBL[m])
a.set_xlabel('localisation threshold α'); a.set_ylabel('HOTA_α'); a.legend(fontsize=8)
a.set_title('HOTA across localisation thresholds (single-stage, 200 objects)', fontsize=9)
plt.tight_layout(); plt.savefig(f'{IMG}/fig8_hota_alpha.png'); plt.close()
print('done')
