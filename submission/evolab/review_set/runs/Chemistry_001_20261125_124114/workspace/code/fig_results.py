"""Figures 2-5 + summary tables from training logs and evaluation JSONs."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from fig_style import style, C, SERIES, plt
from biomol import load_2l3r, parse_sdf, ligand_reference_conformer
from metrics import ligand_dock_baseline

style()
R = '/workspace/outputs/runs'
O = '/workspace/outputs'
IMG = '/workspace/report/images'


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def read_log(run):
    lg = load(f'{R}/{run}/log.json')
    if lg is None:  # fall back to the printed log (every 25 steps)
        p = f'{O}/train_{run}.log'
        lg = [json.loads(l) for l in open(p) if l.startswith('{')] if os.path.exists(p) else None
    return lg


def smooth(y, k=25):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    c = np.convolve(y, np.ones(k) / k, mode='valid')
    return np.concatenate([np.full(k - 1, np.nan), c])


# ------------------------------------------------------------- Fig 2 training curves
fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
series = [('full', 'augmentation-only (AF3-style)', C['blue']), ('canon', 'PCA frame (stopped at 210)', C['red']),
          ('seqframe', 'sequence frame', C['orange']), ('seq_no_trunk', 'seq. frame, no trunk', C['aqua']),
          ('seq_no_ref', 'seq. frame, no ref. conformer', C['yellow']), ('seq_no_bonds', 'seq. frame, no ligand bonds', C['violet'])]
for key, lab, col in series:
    lg = read_log(key)
    if lg is None:
        continue
    st = np.array([r['step'] for r in lg])
    k_s = 25 if len(lg) > 100 else 1
    for a, k, ttl in zip(ax, ['mse', 'lddt', 'bond', 'disto'],
                         ['weighted MSE (EDM-scaled)', 'smooth-lDDT loss', 'ligand bond loss (Å²)', 'distogram cross-entropy']):
        a.plot(st, smooth([r[k] for r in lg], k_s), color=col, label=lab, lw=1.6)
        a.set_title(ttl); a.set_xlabel('training step')
ax[0].set_yscale('log'); ax[2].set_yscale('log')
fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(f'{IMG}/fig2_training_curves.png', dpi=150); plt.close()

# ------------------------------------------------------------- Fig 3 denoising error vs sigma
aug = [('full', 'model_400', 400), ('full', 'model_800', 800), ('full', 'model', 1200)]
seq = [('seqframe', 'model_200', 200), ('seqframe', 'model_400', 400), ('seqframe', 'model_600', 600), ('seqframe', 'model', 800)]
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.3))
for a, group, cmap, ttl in [(ax[0], aug, plt.get_cmap('Blues'), 'augmentation-only (AF3-style)'),
                            (ax[1], seq, plt.get_cmap('Oranges'), 'sequence-anchored frame')]:
    for i, (r, t, s) in enumerate(group):
        e = load(f'{R}/{r}/eval_{t}.json')
        if not e or 'sigma_curve' not in e:
            continue
        cv = e['sigma_curve']; sg = np.array([c['sigma'] for c in cv])
        a.plot(sg, [c['rmsd_all'] for c in cv], color=cmap(0.4 + 0.6 * i / max(len(group) - 1, 1)), marker='o', ms=4, label=f'step {s}')
    a.plot(sg, sg * np.sqrt(3), color=C['muted'], ls='--', lw=1.2, label='input noise RMS (σ√3)')
    a.set_title(ttl); a.legend(fontsize=8)
abl = [('seqframe', 'model_200', 'sequence frame', C['orange']), ('canon', 'model_200', 'PCA frame', C['red']),
       ('seq_no_trunk', 'model_200', 'seq. frame − trunk', C['aqua']), ('seq_no_ref', 'model_200', 'seq. frame − ref. conformer', C['yellow']),
       ('seq_no_bonds', 'model_200', 'seq. frame − ligand bonds', C['violet'])]
for r, t, lab, col in abl:
    e = load(f'{R}/{r}/eval_{t}.json')
    if not e or 'sigma_curve' not in e:
        continue
    cv = e['sigma_curve']
    ax[2].plot([c['sigma'] for c in cv], [c['rmsd_all'] for c in cv], color=col, marker='o', ms=4, label=lab)
ax[2].plot(sg, sg * np.sqrt(3), color=C['muted'], ls='--', lw=1.2, label='input noise RMS (σ√3)')
ax[2].set_title('matched budget: 200 training steps'); ax[2].legend(fontsize=8)
for a in ax:
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('noise level σ (Å)')
ax[0].set_ylabel('all-atom RMSD of one-step denoised output (Å)')
plt.tight_layout(); plt.savefig(f'{IMG}/fig3_denoising_vs_sigma.png', dpi=150); plt.close()

# ------------------------------------------------------------- baseline
f = load_2l3r(ref_seed=0)
sdf = parse_sdf('/workspace/data/sample/2l3r/2l3r_ligand.sdf')
pos, _ = ligand_reference_conformer(sdf, seed=0)
base = ligand_dock_baseline(f, pos, n=200)
json.dump(dict(random_placement_lig_rmsd=base), open(f'{O}/baseline_random_placement.json', 'w'))

# ------------------------------------------------------------- Fig 4 sample metrics
metrics = [('prot_ca_rmsd', 'protein Cα RMSD (Å)', 'log'), ('lig_rmsd_pocketaligned', 'peptide RMSD, pocket-aligned (Å)', 'log'),
           ('lddt_all', 'all-atom lDDT', None), ('lddt_interface', 'interface lDDT (protein–peptide)', None),
           ('fnat', 'fraction of native contacts', None), ('lig_bond_mae', 'peptide bond-length MAE (Å)', 'log'),
           ('n_clashes', 'steric clashes (<1.5 Å)', 'log'), ('ca_chirality_ok', 'Cα chirality correct', None)]
bl, ob = plt.get_cmap('Blues'), plt.get_cmap('Oranges')
conds = [('full', 'untrained', 'untrained', C['muted'])]
conds += [(r, t, f'AF3-aug @{s}', bl(0.4 + 0.6 * i / 2)) for i, (r, t, s) in enumerate(aug)]
conds += [('canon', 'model_200', 'PCA frame @200', C['red'])]
conds += [(r, t, f'seq frame @{s}', ob(0.4 + 0.6 * i / 3)) for i, (r, t, s) in enumerate(seq)]
conds += [('seq_no_trunk', 'model_200', '− trunk @200', C['aqua']), ('seq_no_ref', 'model_200', '− ref. conf. @200', C['yellow']),
          ('seq_no_bonds', 'model_200', '− lig. bonds @200', C['violet'])]
rows = []
for r, t, lab, col in conds:
    e = load(f'{R}/{r}/eval_{t}.json')
    if e is not None:
        rows.append((lab, col, e['main']['metrics']))
fig, ax = plt.subplots(2, 4, figsize=(19, 8.4))
rng = np.random.default_rng(0)
summary = {}
for a, (k, ttl, sc) in zip(ax.flat, metrics):
    for i, (lab, col, m) in enumerate(rows):
        v = np.array([x[k] for x in m], float)
        a.bar(i, np.median(v), color=col, width=0.72, alpha=0.45, edgecolor=col)
        a.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=10, color=col, edgecolor=C['ink'], lw=0.3, zorder=3)
        summary.setdefault(lab, {})[k] = dict(mean=float(v.mean()), median=float(np.median(v)), std=float(v.std()),
                                              min=float(v.min()), max=float(v.max()))
    if k == 'lig_rmsd_pocketaligned':
        a.axhline(np.median(base), color=C['red'], ls='--', lw=1.2)
        a.text(-0.4, np.median(base) * 1.06, f'random rigid placement (median {np.median(base):.1f} Å)', fontsize=7, color=C['red'])
    a.set_xticks(range(len(rows))); a.set_xticklabels([r[0] for r in rows], rotation=50, ha='right', fontsize=7.5)
    a.set_title(ttl, fontsize=10)
    if sc:
        a.set_yscale(sc)
plt.tight_layout(); plt.savefig(f'{IMG}/fig4_metrics.png', dpi=150); plt.close()
json.dump(summary, open(f'{O}/metrics_summary.json', 'w'), indent=1)

# ------------------------------------------------------------- Fig 5 sampler sweeps
ev = load(f'{R}/seqframe/eval_model.json')
if ev and 'sweeps' in ev:
    sw = ev['sweeps']
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
    st = [s for s in sw if s['kind'] == 'steps']
    xs = [s['n_steps'] for s in st]
    for a, k, ttl in zip(ax[:2], ['lig_rmsd_pocketaligned', 'lddt_all'], ['peptide RMSD, pocket-aligned (Å)', 'all-atom lDDT']):
        med = [np.median([m[k] for m in s['metrics']]) for s in st]
        lo = [np.min([m[k] for m in s['metrics']]) for s in st]; hi = [np.max([m[k] for m in s['metrics']]) for s in st]
        a.fill_between(xs, lo, hi, color=C['orange'], alpha=0.18, lw=0)
        a.plot(xs, med, color=C['orange'], marker='o', ms=6, label='η = 1.5 (median, range over 5 samples)')
        others = sorted([s for s in sw if s['kind'] == 'step_scale'], key=lambda s: s['step_scale'])
        for s, col in zip(others, [C['blue'], C['aqua'], C['violet']]):
            a.scatter([50], [np.median([m[k] for m in s['metrics']])], color=col, marker='D', s=45, zorder=4,
                      edgecolor=C['surface'], label=f"η = {s['step_scale']} (50 steps)")
        a.set_xscale('log'); a.set_xlabel('number of sampling steps'); a.set_title(ttl)
    ax[0].legend(fontsize=8)
    ax[2].plot(xs, [s['time_s'] / 5 for s in st], color=C['ink2'], marker='o', ms=6)
    ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlabel('number of sampling steps')
    ax[2].set_ylabel('seconds'); ax[2].set_title('CPU time per sample (shared host)')
    plt.tight_layout(); plt.savefig(f'{IMG}/fig5_sampler_sweeps.png', dpi=150); plt.close()

print(json.dumps({k: {m: round(v[m]['median'], 3) for m in v} for k, v in summary.items()}, indent=1))
print('baseline random placement median', np.median(base), 'min', np.min(base))
