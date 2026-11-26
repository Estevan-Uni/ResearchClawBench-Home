"""Analyse growth runs in ../outputs/growth -> ../outputs/growth_summary.json + figures."""
import json, glob, os, re, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import icoshell as ic

COL = {'MC': '#1f77b4', 'BG': '#ff7f0e', 'Ch1': '#2ca02c', 'Ch2': '#d62728', 'Ch3': '#9467bd', 'none': '#999999'}
IMG = '../report/images/'


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0, 0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(c - h, 0), min(c + h, 1))


def parse(name):
    return dict(p.split('=') for p in name.split('_')[1:])


def excess(cov, hk):
    """coverage above the 12 shared-vertex baseline, scaled to [0,1]"""
    N = ic.N_of(*hk)
    b = 12 / N
    return max(0.0, (cov - b) / (1 - b))


def tup(s):
    return tuple(int(x) for x in re.findall(r'-?\d+', s))


def dominant(idr, seed_cls, thr=0.3):
    if 'f1' in idr:          # final structures: F1 of recall and precision
        best = max(idr['f1'], key=idr['f1'].get)
        bv = idr['f1'][best]
        if bv < 0.6:
            return seed_cls, tup(best), bv
        return ic.shell_class(*tup(best)), tup(best), bv
    best, bv = None, -1
    for c, v in idr['cov'].items():
        e = excess(v, tup(c))
        if e > bv:
            best, bv = tup(c), e
    if bv < thr:
        return seed_cls, best, bv
    return ic.shell_class(*best), best, bv


runs = {}
for fn in sorted(glob.glob('../outputs/growth/*_s*.json')):
    if os.path.basename(fn).startswith('reid_'):
        continue
    r = json.load(open(fn))
    rf = fn.replace('/growth/', '/growth/reid_')
    if os.path.exists(rf):
        rd = json.load(open(rf))
        r['id_grown'], r['id_anneal'] = rd['X_grown'], rd['X_anneal']
    runs.setdefault(r['name'], []).append(r)

SEEDCLS = {'E1': 'MC', 'E3a': 'MC', 'E3b': 'BG', 'E2': 'Ch1', 'E2b': 'Ch1', 'E3c': 'MC'}
summary = {}
for name, rs in runs.items():
    fam = name.split('_')[0]
    recs = []
    for r in rs:
        rec = dict(seed=r['seed'], E_grown=r['E_grown'], E_anneal=r.get('E_anneal'), wall=r['wall'])
        for stage in ['id_grown', 'id_anneal']:
            idr = r[stage]
            cls, best, ex = dominant(idr, SEEDCLS[fam])
            rec[stage] = dict(cov=idr['cov'], best=best, excess=ex, cls=cls, n_bound=idr.get('n_bound'),
                              f1=idr.get('f1'), prec=idr.get('prec'), resolved=bool(ex >= (0.6 if 'f1' in idr else 0.3)),
                              chi={k: v for k, v in idr.items() if k.startswith('chi_') and 'n_' not in k},
                              frame_rms=idr['frame_rms'])
            if not rec[stage]['resolved']:
                rec[stage]['cls'] = 'unresolved'
        rec['ideal'] = r['ideal']
        rec['hist'] = [dict(n=h['n'], cov=h['cov'], Eq=h['Eq'],
                            chi={k: v for k, v in h.items() if k.startswith('chi_') and 'n_' not in k})
                       for h in r['hist']]
        recs.append(rec)
    summary[name] = dict(runs=recs, n=len(recs))

# ------------------------------------------------ E1 statistics
def complete(rec, stage, shell, thr=0.9):
    return rec[stage]['cov'].get(str(shell), 0) >= thr


E1 = {}
for name, s in summary.items():
    if not (name.startswith('E1') or name.startswith('E3a') or name.startswith('E3b') or name.startswith('E2')
            or name.startswith('E3c')):
        continue
    fam = name.split('_')[0]
    target = {'E1': (0, 2), 'E3a': (1, 1), 'E3b': (1, 2), 'E2': None, 'E2b': None, 'E3c': (0, 2)}[fam]
    st = {}
    for stage in ['id_grown', 'id_anneal']:
        n = s['n']
        if target is not None:
            if fam == 'E3b':
                k = sum(1 for r in s['runs'] if max(r[stage]['cov'].get('(1, 2)', 0), r[stage]['cov'].get('(2, 1)', 0)) >= 0.75)
            else:
                k = sum(1 for r in s['runs'] if complete(r, stage, target, 0.9 if fam != 'E3a' else 0.8))
            st[stage] = dict(k=k, n=n, ci=wilson(k, n))
        classes = [r[stage]['cls'] for r in s['runs']]
        st[stage + '_classes'] = {c: classes.count(c) for c in set(classes)}
        st[stage + '_best'] = [str(r[stage]['best']) for r in s['runs']]
    # energies vs ideal (E1: complete Mackay has exactly 13+42 atoms)
    if fam == 'E1':
        Eid = s['runs'][0]['ideal']['(0, 2)']['E']
        st['dE_grown'] = [r['E_grown'] - Eid for r in s['runs']]
        st['dE_anneal'] = [r['E_anneal'] - Eid for r in s['runs']]
        st['E_ideal'] = Eid
    s['stats'] = st
    E1[name] = st

json.dump(summary, open('../outputs/growth_summary.json', 'w'), indent=1, default=str)

# ------------------------------------------------ Figure: E1 conservative preference
e1names = [n for n in summary if n.startswith('E1')]
if e1names:
    order = sorted(e1names, key=lambda n: (float(parse(n)['d']), float(parse(n)['T']), float(parse(n)['tau'])))
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
    x = np.arange(len(order))
    for j, stage in enumerate(['id_grown', 'id_anneal']):
        p = [summary[n]['stats'][stage]['ci'] for n in order]
        y = [q[0] for q in p]; lo = [max(q[0] - q[1], 0) for q in p]; hi = [max(q[2] - q[0], 0) for q in p]
        axs[0].bar(x + (j - 0.5) * 0.38, y, 0.38, yerr=[lo, hi], capsize=3,
                   color=['#9ecae1', '#1f77b4'][j], label=['as grown (quench)', 'after anneal'][j])
    labs = ['$\\delta$=%s\nT*=%s\n$\\tau_{dep}$=%s\nn=%d' % (parse(n)['d'], parse(n)['T'], parse(n)['tau'], summary[n]['n'])
            for n in order]
    axs[0].set_xticks(x); axs[0].set_xticklabels(labs, fontsize=8)
    axs[0].set_ylabel('fraction of runs with Mackay (0,2) coverage $\\geq$0.9'); axs[0].set_ylim(0, 1.05)
    axs[0].legend(fontsize=8, loc='upper left'); axs[0].set_title('(a) conservative MC$\\to$MC step on A$_{13}$ (95% Wilson CI)', fontsize=10)
    for i, n in enumerate(order):
        for stage, mk, off in [('id_grown', 'o', -0.19), ('id_anneal', 's', 0.19)]:
            cv = [r[stage]['cov'].get('(0, 2)', 0) for r in summary[n]['runs']]
            ca = [r[stage]['cov'].get('(1, 1)', 0) for r in summary[n]['runs']]
            axs[1].scatter(np.full(len(cv), i + off) + np.random.default_rng(i).uniform(-0.06, 0.06, len(cv)), cv,
                           marker=mk, c='#1f77b4', s=22, alpha=0.8)
            axs[1].scatter(np.full(len(ca), i + off) + np.random.default_rng(i + 7).uniform(-0.06, 0.06, len(ca)), ca,
                           marker=mk, c='#ff7f0e', s=22, alpha=0.8)
        dE = summary[n]['stats']['dE_anneal']
        axs[2].scatter(np.full(len(dE), i) + np.random.default_rng(i).uniform(-0.1, 0.1, len(dE)), dE, c='k', s=20)
        axs[2].errorbar(i + 0.25, np.mean(dE), yerr=np.std(dE) / np.sqrt(len(dE)), fmt='D', c='r', capsize=3)
    axs[1].axhline(12 / 42, ls=':', c='#1f77b4', lw=1); axs[1].axhline(12 / 32, ls=':', c='#ff7f0e', lw=1)
    axs[1].scatter([], [], c='#1f77b4', label='(0,2) Mackay coverage'); axs[1].scatter([], [], c='#ff7f0e', label='(1,1) anti-Mackay coverage')
    axs[1].scatter([], [], marker='o', c='gray', label='as grown'); axs[1].scatter([], [], marker='s', c='gray', label='annealed')
    axs[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=4); axs[1].set_xticks(x); axs[1].set_xticklabels(labs, fontsize=8)
    axs[1].set_ylabel('coverage'); axs[1].set_title('(b) per-run coverages (dotted: 12-vertex baseline)', fontsize=10)
    axs[2].set_xticks(x); axs[2].set_xticklabels(labs, fontsize=8)
    axs[2].set_ylabel('E$_{anneal}$ - E(ideal Mackay 55)  [$\\epsilon$]')
    axs[2].set_title('(c) energy above the ideal conservative structure', fontsize=10)
    plt.tight_layout(); plt.savefig(IMG + 'fig5_growth_E1_conservative.png', dpi=140); plt.close()

# ------------------------------------------------ Figure: E2/E2b/E3b coverage trajectories
fams = [n for n in summary if n.split('_')[0] in ('E2', 'E2b', 'E3b', 'E3a')]
if fams:
    fams = sorted(fams)
    nc = 4; nr = int(np.ceil(len(fams) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.8 * nc, 3.9 * nr), squeeze=False)
    for a in axs.ravel()[len(fams):]:
        a.axis('off')
    for a, n in zip(axs.ravel(), fams):
        for ri, r in enumerate(summary[n]['runs']):
            ns = [h['n'] for h in r['hist']]
            keys = list(r['hist'][0]['cov'].keys())
            for c in keys:
                hk = tup(c)
                col = COL.get(ic.shell_class(*hk), 'k')
                ls = '--' if hk[0] > hk[1] and hk[1] > 0 else '-'
                a.plot(ns, [h['cov'][c] for h in r['hist']], ls, color=col, alpha=[0.95, 0.55, 0.35][ri % 3],
                       label='%s %s' % (c, ic.shell_class(*hk)) if ri == 0 else None, lw=1.6)
        a.set_xlabel('deposited atoms'); a.set_ylabel('coverage (recall)')
        p = parse(n)
        a.set_title('%s $\\delta$=%s T*=%s $\\tau$=%s (%d runs)' % (n.split('_')[0], p['d'], p['T'], p['tau'], summary[n]['n']),
                    fontsize=9)
        a.set_ylim(0, 1.05); a.legend(fontsize=6.5, loc='upper left')
    fig.suptitle('colour = class of template, dashed = opposite hand (h>k), opacity = run index', fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.savefig(IMG + 'fig6_growth_coverage_traj.png', dpi=130); plt.close()

json.dump(E1, open('../outputs/growth_stats.json', 'w'), indent=1, default=str)
for k, v in E1.items():
    print(k, {kk: vv for kk, vv in v.items() if not kk.startswith('dE')})
