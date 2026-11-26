"""Aggregate run jsons -> tables (outputs/tables/*.csv, outputs/tables.md) and figures."""
import sys, json, glob, os, re
import numpy as np, pandas as pd
sys.path.insert(0, '/workspace/code')
from common import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
style()
R = {}
for f in glob.glob(f'{OUT}/runs/*.json'):
    r = json.load(open(f)); R[os.path.basename(f)[:-5]] = r
os.makedirs(f'{OUT}/tables', exist_ok=True)
md = []


def runs(prefix, method, tag=''):
    pat = re.compile(rf'^{prefix}_{method}_s(\d){tag}$')
    return [R[k] for k in sorted(R) if pat.match(k)]


def ms(v, d=3, scale=100):
    v = np.asarray(v, float) * scale
    if len(v) == 0: return 'n/a'
    return f'{v.mean():.{max(d-2,1)}f} ± {v.std():.{max(d-2,1)}f}' if len(v) > 1 else f'{v.mean():.{max(d-2,1)}f}'


# =============== known protocol
KM = [('RF', 'rf'), ('MLP', 'mlp'), ('E-GraphSAGE', 'egs'), ('TGN', 'tgn'), ('3D-IDS', '3dids'), ('DIDS-MFL', 'dids_mfl')]
rows = []; per_class = {}
for lab, m in KM:
    rr = runs('known', m)
    if not rr: continue
    T = [r['test'] for r in rr]
    pc = np.array([t['per_class_f1'] for t in T]); per_class[lab] = pc
    rows.append({'Method': lab, 'seeds': len(rr), 'Binary F1': ms([t['bin_f1'] for t in T]),
                 'Binary AUC': ms([t['bin_auc'] for t in T]), 'Macro-F1 (10 cls)': ms([t['macro_f1'] for t in T]),
                 'Attack bal. acc': ms([t['attack_bal_acc'] for t in T]),
                 'min class F1': ms(pc.min(1)), 'std of class F1': ms(pc.std(1)),
                 'infer ms/flow': f"{np.mean([r.get('infer_ms_per_flow', np.nan) for r in rr]):.4f}"})
known_df = pd.DataFrame(rows); known_df.to_csv(f'{OUT}/tables/known.csv', index=False)
md += ['## Known attacks (test period, chronological split)', known_df.to_markdown(index=False), '']
pcrows = []
for lab, pc in per_class.items():
    pcrows.append({'Method': lab, **{CLASSES[c]: f'{100*pc[:, c].mean():.1f}' for c in range(10)}})
pc_df = pd.DataFrame(pcrows); pc_df.to_csv(f'{OUT}/tables/known_per_class.csv', index=False)
md += ['### Per-class F1 (known, mean over seeds)', pc_df.to_markdown(index=False), '']
if per_class:
    fig, ax = plt.subplots(figsize=(12, 3.6))
    labs = list(per_class); w = 0.8 / len(labs)
    for j, lab in enumerate(labs):
        pc = per_class[lab]
        ax.bar(np.arange(10) + (j - len(labs) / 2 + 0.5) * w, pc.mean(0), w * 0.9, yerr=pc.std(0) if len(pc) > 1 else None,
               color=PAL[j], label=lab, error_kw=dict(lw=0.8, ecolor=INK2))
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=20); ax.set_ylabel('F1 (test)'); ax.set_ylim(0, 1.05)
    ax.legend(ncol=6, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.0)); ax.grid(axis='x', visible=False)
    plt.tight_layout(); plt.savefig(f'{IMG}/known_per_class_f1.png'); plt.close()

# long-budget (20 epochs, patience 6) known runs
rows = []
for lab, m in [('TGN', 'tgn'), ('3D-IDS', '3dids'), ('DIDS-MFL', 'dids_mfl'), ('DIDS-MFL w/o diffusion', 'mfl_noDiff')]:
    rr = runs('known', m, tag='_long')
    if not rr: continue
    T = [r['test'] for r in rr]; pc = np.array([t['per_class_f1'] for t in T])
    rows.append({'Method (20 epochs)': lab, 'seeds': len(rr), 'Binary F1': ms([t['bin_f1'] for t in T]),
                 'Macro-F1': ms([t['macro_f1'] for t in T]), 'Attack bal. acc': ms([t['attack_bal_acc'] for t in T]),
                 'min class F1': ms(pc.min(1)), 'best epoch': ', '.join(str(r['best_epoch'] + 1) for r in rr)})
long_df = pd.DataFrame(rows); long_df.to_csv(f'{OUT}/tables/known_long.csv', index=False)
md += ['## Known attacks with a longer training budget (20 epochs, patience 6)', long_df.to_markdown(index=False), '']
rr = runs('fewshot', 'dids_mfl', tag='_long') + runs('fewshot', 'mfl_noDiff', tag='_long')
if rr:
    rows = [{'run': r['name'], 'best epoch': r['best_epoch'] + 1, 'base-class test macro-F1': f"{100*r['test_base']['macro_f1']:.1f}",
             'novel 4-way 5-shot F1 (MFL head)': f"{100*r['fewshot']['mfl_head']['all_4way_K5']['f1']:.1f}",
             'novel 4-way 5-shot F1 (proto)': f"{100*r['fewshot']['euclid']['all_4way_K5']['f1']:.1f}"} for r in rr]
    md += ['## Few-shot protocol with a longer budget (seed 0)', pd.DataFrame(rows).to_markdown(index=False), '']

# =============== few-shot episodic protocol
FS = [('Raw features + ProtoNet head', 'raw', 'euclid'), ('Raw features + 1-NN', 'raw', '1nn'),
      ('Raw features + RF on support', 'raw', 'rf_support'),
      ('ProtoNet', 'protonet', 'euclid'), ('RelationNet', 'relationnet', 'learned'), ('Baseline++ (cosine)', 'baselinepp', 'cosine'),
      ('BSNet-style (bi-similarity)', 'bsnet', 'learned'), ('MLP pretrain + proto', 'mlp', 'euclid'),
      ('MLP pretrain + fine-tune LR', 'mlp', 'finetune_lr'), ('E-GraphSAGE + proto', 'egs', 'euclid'),
      ('TGN + proto', 'tgn', 'euclid'), ('3D-IDS + proto', '3dids', 'euclid'),
      ('DIDS-MFL (embedding + proto)', 'dids_mfl', 'euclid'), ('DIDS-MFL (bi-similarity MFL head)', 'dids_mfl', 'mfl_head')]
FS_PH = [('DIDS-MFL + fine-tune LR (post-hoc)', 'dids_mfl', 'lr')]


def fs_table(pool, setting):
    rows = []
    for lab, m, h in FS:
        rr = runs('fewshot', m)
        rr = [r for r in rr if 'fewshot' in r and h in r['fewshot']]
        if not rr: continue
        row = {'Method': lab, 'seeds': len(rr)}
        for K in (1, 5, 10):
            k = f'{pool}_{setting}_K{K}'
            row[f'F1 K={K}'] = ms([r['fewshot'][h][k]['f1'] for r in rr])
            row[f'NMI K={K}'] = ms([r['fewshot'][h][k]['nmi'] for r in rr])
        rows.append(row)
    return pd.DataFrame(rows)


for pool in ['all', 'dedup']:
    for st in ['4way', '5way']:
        df = fs_table(pool, st); df.to_csv(f'{OUT}/tables/fewshot_{pool}_{st}.csv', index=False)
        md += [f'## Few-shot episodic ({st}, pool={pool}; 500 episodes, Q=15; mean ± std over seeds)', df.to_markdown(index=False), '']

# post-hoc heads on saved embeddings
phs = sorted(glob.glob(f'{OUT}/posthoc/*.json'))
if phs:
    P = {}
    order = ['raw', 'mlp', 'bsnet', 'egs', 'tgn', '3dids', 'dids_mfl', 'mfl_noSD', 'mfl_noDiff', 'mfl_noEp']
    for o in order:
        if os.path.exists(f'{OUT}/posthoc/{o}.json'): P.update(json.load(open(f'{OUT}/posthoc/{o}.json')))
    rows = []
    for meth, d in P.items():
        row = {'Embedding + head': meth, 'seeds': len(d['all_K5']['f1'])}
        for K in (1, 5, 10):
            row[f'F1 K={K}'] = ms(d[f'all_K{K}']['f1']); row[f'NMI K={K}'] = ms(d[f'all_K{K}']['nmi'])
        row['dedup F1 K=5'] = ms(d['dedup_K5']['f1']); row['dedup NMI K=5'] = ms(d['dedup_K5']['nmi'])
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(f'{OUT}/tables/fewshot_posthoc.csv', index=False)
    md += ['## Few-shot (4-way, pool=all): common heads applied post-hoc to saved embeddings', df.to_markdown(index=False), '']

# K-curve figure
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
show = [('Raw features + ProtoNet head', 'raw', 'euclid'), ('ProtoNet', 'protonet', 'euclid'), ('BSNet-style (bi-similarity)', 'bsnet', 'learned'),
        ('MLP pretrain + fine-tune LR', 'mlp', 'finetune_lr'), ('E-GraphSAGE + proto', 'egs', 'euclid'), ('TGN + proto', 'tgn', 'euclid'),
        ('3D-IDS + proto', '3dids', 'euclid'), ('DIDS-MFL (bi-similarity MFL head)', 'dids_mfl', 'mfl_head')]
for ax, metric in zip(axs, ['f1', 'nmi']):
    for j, (lab, m, h) in enumerate(show):
        rr = [r for r in runs('fewshot', m) if 'fewshot' in r and h in r['fewshot']]
        if not rr: continue
        v = np.array([[r['fewshot'][h][f'all_4way_K{K}'][metric] for K in (1, 5, 10)] for r in rr])
        col = PAL[j] if j < 8 else INK
        ax.errorbar([1, 5, 10], v.mean(0), yerr=v.std(0) if len(v) > 1 else None, color=col, marker='o', ms=5, lw=2 if 'DIDS' in lab else 1.4,
                    label=lab, capsize=2)
    ax.set_xlabel('K (shots per novel class)'); ax.set_ylabel({'f1': 'macro-F1', 'nmi': 'NMI'}[metric]); ax.set_xticks([1, 5, 10])
    ax.set_title(f'4-way novel-class episodes: {metric.upper()}')
axs[1].legend(fontsize=7, loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(); plt.savefig(f'{IMG}/fewshot_k_curve.png'); plt.close()

# =============== unknown protocol
UM = [('RF', 'rf'), ('MLP', 'mlp'), ('E-GraphSAGE', 'egs'), ('TGN', 'tgn'), ('3D-IDS', '3dids'), ('DIDS-MFL', 'dids_mfl')]
rows = []; unk = {}
for lab, m in UM:
    row = {'Method': lab}; recs = []
    for h in UNKNOWN_TYPES:
        rr = runs(f'unknown_h{h}', m)
        if not rr: row[f'{CLASSES[h]} recall'] = 'n/a'; continue
        rec = [r['test']['recall_unknown'] for r in rr]; f1 = [r['test']['f1_unknown'] for r in rr]
        auc = [r['test']['auc_unknown'] for r in rr]
        row[f'{CLASSES[h]} recall'] = ms(rec); row[f'{CLASSES[h]} F1'] = ms(f1); row[f'{CLASSES[h]} AUC'] = ms(auc); recs.append(np.mean(rec))
        row.setdefault('_f1', []).append(np.mean(f1))
        unk.setdefault(lab, {})[CLASSES[h]] = (np.mean(rec), np.std(rec), np.mean(f1))
    if recs: row['mean recall'] = f'{100*np.mean(recs):.1f}'; row['min recall'] = f'{100*np.min(recs):.1f}'; row['mean F1'] = f"{100*np.mean(row.pop('_f1')):.1f}"
    rows.append(row)
unk_df = pd.DataFrame(rows); unk_df.to_csv(f'{OUT}/tables/unknown.csv', index=False)
md += ['## Unknown attacks (leave-one-attack-out; detection on benign ∪ unseen type in test)', unk_df.to_markdown(index=False), '']
if unk:
    fig, ax = plt.subplots(figsize=(9, 3.4)); labs = list(unk); w = 0.8 / len(labs)
    for j, lab in enumerate(labs):
        v = [unk[lab].get(CLASSES[h], (np.nan, 0, 0))[0] for h in UNKNOWN_TYPES]
        e = [unk[lab].get(CLASSES[h], (np.nan, 0, 0))[1] for h in UNKNOWN_TYPES]
        ax.bar(np.arange(4) + (j - len(labs) / 2 + 0.5) * w, v, w * 0.9, yerr=e, color=PAL[j], label=lab, error_kw=dict(lw=0.8))
    ax.set_xticks(range(4)); ax.set_xticklabels([CLASSES[h] for h in UNKNOWN_TYPES]); ax.set_ylabel('detection recall of unseen type')
    ax.set_ylim(0, 1.05); ax.legend(ncol=6, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.0)); ax.grid(axis='x', visible=False)
    plt.tight_layout(); plt.savefig(f'{IMG}/unknown_recall.png'); plt.close()

# =============== ablations
AB = [('DIDS-MFL (full)', 'dids_mfl'), ('w/o SD (F only)', 'mfl_noSD'), ('h replaces F', 'mfl_hrep'), ('w/o RD', 'mfl_noRD'),
      ('w/o RD and w/o Dis-reg', 'mfl_noRDDis'), ('RD/Dis β=1', 'mfl_rd1'),
      ('w/o diffusion', 'mfl_noDiff'), ('diffusion→temporal attention', 'mfl_attn'), ('w/o multi-scale transform', 'mfl_noMST'),
      ('w/o Dis-reg', 'mfl_noDis'), ('single (cosine) similarity', 'mfl_cos'), ('w/o episodes', 'mfl_noEp'),
      ('3D-IDS (no MFL)', '3dids'), ('TGN', 'tgn')]
rows = []; abl = {}
for lab, m in AB:
    rk = [r for r in runs('known', m) if 'test' in r]; rf = [r for r in runs('fewshot', m) if 'fewshot' in r]
    if not rk and not rf: continue
    head = 'mfl_head' if (rf and 'mfl_head' in rf[0]['fewshot']) else 'euclid'
    row = {'Variant': lab, 'seeds (known/fs)': f'{len(rk)}/{len(rf)}'}
    if rk:
        row['Known macro-F1'] = ms([r['test']['macro_f1'] for r in rk]); row['Known bin-F1'] = ms([r['test']['bin_f1'] for r in rk])
        row['|corr| z (known)'] = ms([r['offdiag_corr_z_test'] for r in rk], scale=1, d=5)
        row['|corr| z_m (known)'] = ms([r['offdiag_corr_zm_test'] for r in rk], scale=1, d=5)
    if rf:
        for K in (1, 5):
            row[f'FS F1 K={K}'] = ms([r['fewshot'][head][f'all_4way_K{K}']['f1'] for r in rf])
        row['FS NMI K=5'] = ms([r['fewshot'][head][f'all_4way_K5']['nmi'] for r in rf])
        row['FS head'] = head
    abl[lab] = (rk, rf, head); rows.append(row)
abl_df = pd.DataFrame(rows); abl_df.to_csv(f'{OUT}/tables/ablation.csv', index=False)
md += ['## Ablations (known: test macro-F1; few-shot: 4-way pool=all)', abl_df.to_markdown(index=False), '']
if abl:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    labs = [l for l in abl]
    for ax, which in zip(axs, ['known', 'fs']):
        vals, errs = [], []
        for l in labs:
            rk, rf, head = abl[l]
            v = [r['test']['macro_f1'] for r in rk] if which == 'known' else [r['fewshot'][head]['all_4way_K5']['f1'] for r in rf]
            vals.append(np.mean(v) if v else np.nan); errs.append(np.std(v) if len(v) > 1 else 0)
        cols = [PAL[0] if l == 'DIDS-MFL (full)' else (GRAY if l in ('3D-IDS (no MFL)', 'TGN') else PAL[2]) for l in labs]
        ax.barh(range(len(labs)), vals, xerr=errs, color=cols, error_kw=dict(lw=0.8))
        ref = vals[0]; ax.axvline(ref, color=INK2, ls='--', lw=0.8)
        lo = np.nanmin(vals) - 0.03; ax.set_xlim(max(lo, 0), min(np.nanmax(np.array(vals) + np.array(errs)) + 0.04, 1))
        for i, v in enumerate(vals):
            if not np.isnan(v): ax.text(v + errs[i] + 0.003, i, f'{v:.3f}', va='center', fontsize=7, color=INK2)
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs); ax.grid(axis='y', visible=False)
        if which == 'known': ax.invert_yaxis()
        ax.set_title('Known: test macro-F1' if which == 'known' else 'Few-shot: 4-way 5-shot F1 (novel classes)')
    plt.tight_layout(); plt.savefig(f'{IMG}/ablation.png'); plt.close()

# =============== latency
rows = []
for lab, m in KM:
    rr = runs('known', m)
    if rr: rows.append({'Method': lab, 'ms/flow': np.mean([r.get('infer_ms_per_flow', np.nan) for r in rr]),
                        'train s/epoch': np.mean([r.get('train_s_per_epoch', np.nan) for r in rr])})
lat = pd.DataFrame(rows)
if os.path.exists(f'{OUT}/llm/llm_results.json'):
    L = json.load(open(f'{OUT}/llm/llm_results.json'))
    lat = pd.concat([lat, pd.DataFrame([{'Method': 'LLM (Qwen2.5-1.5B, in-context)', 'ms/flow': 1000 * L['per_flow_s_mean'], 'train s/epoch': np.nan}])])
lat.to_csv(f'{OUT}/tables/latency.csv', index=False)
md += ['## Inference latency (CPU, shared host)', lat.to_markdown(index=False, floatfmt='.4f'), '']
open(f'{OUT}/tables.md', 'w').write('\n'.join(md))
print('\n'.join(md))

# =============== figure: common-head comparison of embeddings (post-hoc)
if phs:
    embs = [('Raw [F||h]', 'Raw [F||h]'), ('MLP', 'MLP (supervised pretrain)'), ('BSNet-style', 'BSNet-style'), ('E-GraphSAGE', 'E-GraphSAGE'),
            ('TGN', 'TGN'), ('3D-IDS', '3D-IDS'), ('DIDS-MFL', 'DIDS-MFL')]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, key in zip(axs, ['all_K1', 'all_K5', 'all_K10']):
        for j, (hn, col) in enumerate([('proto', PAL[0]), ('LR', PAL[1])]):
            v = [np.mean(P[f'{e} + {hn}'][key]['f1']) if f'{e} + {hn}' in P else np.nan for _, e in embs]
            s_ = [np.std(P[f'{e} + {hn}'][key]['f1']) if f'{e} + {hn}' in P else 0 for _, e in embs]
            ax.bar(np.arange(len(embs)) + (j - 0.5) * 0.38, v, 0.36, yerr=s_, color=col, label={'proto': 'prototype (Euclidean) head', 'LR': 'logistic-regression head'}[hn],
                   error_kw=dict(lw=0.8))
        ax.set_xticks(range(len(embs))); ax.set_xticklabels([e for e, _ in embs], rotation=30, ha='right')
        ax.set_title(f'4-way, K={key.split("K")[1]}'); ax.set_ylim(0.4, 0.9); ax.grid(axis='x', visible=False)
    axs[0].set_ylabel('episode macro-F1 (novel classes)'); axs[0].legend(fontsize=7, loc='upper left')
    plt.tight_layout(); plt.savefig(f'{IMG}/fewshot_common_heads.png'); plt.close()

# =============== figure: accuracy vs latency incl. LLM
if os.path.exists(f'{OUT}/llm/llm_results.json'):
    L = json.load(open(f'{OUT}/llm/llm_results.json'))
    pts = [('Qwen2.5-1.5B (in-context, 20 shots)', L['per_flow_s_mean'] * 1000, L['macro_f1'], L['bin_f1']),
           ('RF', None, L['rf_same_flows']['macro_f1'], L['rf_same_flows']['bin_f1']),
           ('DIDS-MFL', None, L['dids_mfl_same_flows']['macro_f1'], L['dids_mfl_same_flows']['bin_f1']),
           ('RF on the 20 prompt flows', None, L['rf_on_prompt_shots']['macro_f1'], L['rf_on_prompt_shots']['bin_f1'])]
    lat_map = dict(zip(lat['Method'], lat['ms/flow']))
    pts[1] = (pts[1][0], lat_map.get('RF'), pts[1][2], pts[1][3]); pts[2] = (pts[2][0], lat_map.get('DIDS-MFL'), pts[2][2], pts[2][3])
    pts[3] = (pts[3][0], lat_map.get('RF'), pts[3][2], pts[3][3])
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    labs = [p[0] for p in pts]
    for k, (ax, idx, ttl) in enumerate(zip(axs, [2, 3], ['10-class macro-F1', 'binary F1'])):
        ax.barh(range(len(pts)), [p[idx] for p in pts], color=[PAL[1], PAL[2], PAL[0], GRAY])
        for i, p in enumerate(pts): ax.text(p[idx] + 0.01, i, f'{p[idx]:.2f}', va='center', fontsize=8, color=INK2)
        ax.set_yticks(range(len(pts))); ax.set_yticklabels(labs if k == 0 else []); ax.set_xlim(0, 1.1); ax.invert_yaxis()
        ax.set_title(f'{ttl} on the same 100 test flows (10/class)'); ax.grid(axis='y', visible=False)
    plt.tight_layout(); plt.savefig(f'{IMG}/llm_comparison.png'); plt.close()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    L2 = [(r['Method'], r['ms/flow']) for _, r in lat.iterrows()]
    ax.barh(range(len(L2)), [v for _, v in L2], color=[PAL[1] if 'LLM' in m else (PAL[0] if 'DIDS' in m else GRAY) for m, _ in L2])
    ax.set_xscale('log'); ax.set_yticks(range(len(L2))); ax.set_yticklabels([m for m, _ in L2]); ax.invert_yaxis()
    for i, (_, v) in enumerate(L2): ax.text(v * 1.15, i, f'{v:.3g} ms', va='center', fontsize=7, color=INK2)
    ax.set_xlabel('inference time per flow (ms, log scale, CPU)'); ax.grid(axis='y', visible=False)
    ax.set_xlim(right=max(v for _, v in L2) * 8)
    plt.tight_layout(); plt.savefig(f'{IMG}/latency.png'); plt.close()
