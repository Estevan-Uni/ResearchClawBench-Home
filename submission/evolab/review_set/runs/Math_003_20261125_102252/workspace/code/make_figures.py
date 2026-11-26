import json, os, pickle, sys
import numpy as np
from figstyle import *
OUT = '/workspace/report/images/'
O = '../outputs/'


def fig_ablation():
    d = json.load(open(O + 'ddar_ablation.json'))
    names = list(d)
    vals = [[len(v) for v in d[n].values()] for n in names]
    m = [np.mean(v) for v in vals]
    fig, ax = plt.subplots(figsize=(12, 4.3))
    x = np.arange(len(names))
    cols = [SERIES[0] if n == 'full' else (SERIES[1] if n.startswith('no_AR') else SERIES[2]) for n in names]
    ax.bar(x, m, 0.6, color=cols)
    for i, v in enumerate(vals):
        ax.plot([i, i], [min(v), max(v)], color=INK, lw=1.5)
        ax.text(i, max(v) + 0.4, '%.1f' % m[i], ha='center', fontsize=9, color=INK2)
    lab = {'full': 'full DD+AR', 'no_AR (DD only)': 'DD only\n(no AR)'}
    ax.set_xticks(x); ax.set_xticklabels([lab.get(n, 'w/o ' + n[1:]) for n in names], fontsize=9, rotation=25, ha='right')
    ax.set_ylabel('problems solved (of 30)'); ax.set_ylim(0, 20)
    ax.set_title('DD+AR without auxiliary points: 5 random diagrams per problem (bar = mean, line = min-max)')
    plt.savefig(OUT + 'fig2_ddar_ablation.png'); plt.close()


def fig_budget():
    r = json.load(open(O + 'search_results.json'))
    B = r['budgets']
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1.4]))
    for i, (k, lab) in enumerate((('random', 'random order (expected)'), ('heuristic', 'hand-designed heuristic'),
                                  ('learned', 'learned prior (synthetic data only)'))):
        ax[0].plot(B, r['curves'][k], marker='o', ms=6, color=SERIES[i], label=lab)
    ax[0].axhline(15, color=MUTED, ls=':', lw=1)
    ax[0].text(1.1, 15.2, 'DD+AR alone (15)', color=INK2, fontsize=8)
    ax[0].set_xscale('log'); ax[0].set_xlabel('candidate budget k (depth-1 aux points tried)')
    ax[0].set_ylabel('problems solved (of 30)'); ax[0].set_ylim(14, 26)
    ax[0].legend(loc='upper left'); ax[0].set_title('(a) Solved vs. search budget')
    rows = r['rows']
    nms = [n for n, v in rows.items() if v['successes'] > 0]
    x = np.arange(len(nms))
    for i, (k, lab) in enumerate((('rank_random_expected', 'random (expected)'), ('rank_heuristic', 'heuristic'),
                                  ('rank_learned', 'learned'))):
        ax[1].bar(x + (i - 1) * 0.27, [rows[n][k] - 0.5 for n in nms], 0.25, color=SERIES[i], label=lab, bottom=0.5)
    ax[1].set_yscale('log'); ax[1].set_xticks(x)
    ax[1].set_xticklabels([n.replace('imo_', '').replace('_', ' ') + '\n(%d/%d)' % (rows[n]['successes'], rows[n]['pool'])
                           for n in nms], fontsize=8)
    ax[1].set_ylabel('candidates tried until first proof'); ax[1].legend(loc='upper left', ncol=3)
    ax[1].set_ylim(0.5, 2e4)
    for xi, n in zip(x, nms):
        ax[1].text(xi + 0.27, rows[n]['rank_learned'] * 1.15, str(rows[n]['rank_learned']), ha='center', fontsize=8, color=INK2)
    ax[1].set_title('(b) Rank of first successful aux point (label: successes/pool)')
    plt.tight_layout(); plt.savefig(OUT + 'fig3_search_budget.png'); plt.close()


def fig_prior():
    r = json.load(open(O + 'prior_results.json'))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    tags = [t for t in ('selfplay', 'traceback', 'combined') if t in r['cv']]
    x = np.arange(len(tags))
    for i, (k, lab) in enumerate((('mrr_random', 'random'), ('mrr_heuristic', 'heuristic'), ('mrr_learned', 'learned'))):
        v = [r['cv'][t][k] for t in tags]
        ax[0].bar(x + (i - 1) * 0.27, v, 0.25, color=SERIES[i], label=lab)
    ax[0].set_xticks(x); ax[0].set_xticklabels(['%s\n(%d groups)' % (t, r['cv'][t]['n_groups']) for t in tags])
    ax[0].set_ylabel('MRR of first positive (grouped 5-fold CV)'); ax[0].legend(loc='upper left')
    ax[0].set_title('(a) Held-out synthetic problems'); ax[0].set_ylim(0, 1)
    lc = r['learning_curve']
    ns = sorted(set(e['n_train_groups'] for e in lc))
    mm = [np.mean([e['mrr'] for e in lc if e['n_train_groups'] == n]) for n in ns]
    ax[1].plot(ns, mm, marker='o', ms=6, color=SERIES[2], label='learned')
    ax[1].scatter([e['n_train_groups'] for e in lc], [e['mrr'] for e in lc], s=12, color=SERIES[2], alpha=0.4)
    ax[1].axhline(r['test_mrr_random'], color=SERIES[0], ls='--', lw=1.5, label='random')
    ax[1].axhline(r['test_mrr_heuristic'], color=SERIES[1], ls='--', lw=1.5, label='heuristic')
    ax[1].set_xscale('log'); ax[1].set_xlabel('training problems (synthetic groups)'); ax[1].set_ylabel('test MRR')
    ax[1].legend(loc='lower right'); ax[1].set_title('(b) Learning curve (fixed held-out groups)')
    md = [np.median([e['imo_median_rank'] for e in lc if e['n_train_groups'] == n and e['imo_median_rank']]) for n in ns]
    ax[2].plot(ns, md, marker='o', ms=6, color=SERIES[2], label='learned')
    ax[2].axhline(r['imo_heuristic_median'], color=SERIES[1], ls='--', lw=1.5, label='heuristic')
    ax[2].axhline(r['imo_random_median'], color=SERIES[0], ls='--', lw=1.5, label='random (expected)')
    ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlabel('training problems (synthetic groups)')
    ax[2].set_ylabel('median candidates to first IMO proof'); ax[2].legend(loc='lower left'); ax[2].set_ylim(15, 2000)
    ax[2].set_title('(c) Transfer to IMO-AG-30 (8 problems with a depth-1 proof)')
    plt.tight_layout(); plt.savefig(OUT + 'fig5_prior.png'); plt.close()


def fig_validation():
    m = json.load(open(O + 'mutation_all.json'))
    t = json.load(open(O + 'transfer.json'))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ks = list(m['mutants'])
    v = [100 * m['mutants'][k]['rejected'] / m['mutants'][k]['n'] for k in ks]
    ax[0].barh(ks[::-1], v[::-1], color=SERIES[0])
    for i, (k, vv) in enumerate(zip(ks[::-1], v[::-1])):
        ax[0].text(vv - 1, i, '%.1f%%  (%d/%d)' % (vv, m['mutants'][k]['rejected'], m['mutants'][k]['n']), va='center',
                   ha='right', color='white', fontsize=9)
    ax[0].set_xlim(0, 100); ax[0].set_xlabel('mutants rejected by the checker (%)')
    ax[0].set_title('(a) Mutation controls on %d accepted proofs' % m['base'])
    nms = sorted(t)
    ok = [sum(r['results'].count('ok') for r in t[n]) for n in nms]
    tot = [sum(len(r['results']) for r in t[n]) for n in nms]
    x = np.arange(len(nms))
    ax[1].bar(x, tot, 0.6, color=GRID, label='fresh diagrams tested')
    ax[1].bar(x, ok, 0.6, color=SERIES[2], label='proved + checker-accepted')
    ax[1].set_xticks(x); ax[1].set_xticklabels([n.replace('imo_', '').replace('_', ' ') for n in nms], rotation=45, fontsize=8)
    ax[1].legend(loc='upper left'); ax[1].set_title('(b) Transfer of found aux constructions to 4 new diagrams each')
    ax[1].set_ylabel('diagrams'); ax[1].set_ylim(0, max(tot) * 1.3)
    plt.tight_layout(); plt.savefig(OUT + 'fig8_validation.png'); plt.close()


def fig_premises():
    d = json.load(open(O + 'premise_usage.json'))['table']
    nms = list(d)
    cls_names = ['independent', 'redundant (implied by the other predicates of its construction)', 'point irrelevant to the goal']
    used = [d[n]['min_used'] for n in nms]
    cnt = {c: [sum(1 for v in d[n]['unused_in_best_class'].values() if v == c) for n in nms] for c in cls_names}
    x = np.arange(len(nms))
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.bar(x, used, 0.65, color=SERIES[0], label='used by the proof')
    b = np.array(used, float)
    labs = {'independent': 'unused: independent premise', cls_names[1]: 'unused: redundant predicate',
            cls_names[2]: 'unused: point irrelevant to goal'}
    for i, c in enumerate(cls_names):
        ax.bar(x, cnt[c], 0.65, bottom=b, color=[SERIES[1], GRID, SERIES[3]][i], label=labs[c])
        b += np.array(cnt[c])
    ax.set_xticks(x); ax.set_xticklabels([n.replace('imo_', '').replace('_', ' ') for n in nms], rotation=45, fontsize=8)
    ax.set_ylabel('statement premise predicates'); ax.legend(loc='upper right', ncol=2); ax.set_ylim(0, max(b) * 1.35)
    ax.set_title('Traceback premise audit (proof citing the fewest premises, per solved problem)')
    plt.tight_layout(); plt.savefig(OUT + 'fig9_premise_audit.png'); plt.close()


if __name__ == '__main__':
    for f in sys.argv[1:]:
        globals()['fig_' + f]()
