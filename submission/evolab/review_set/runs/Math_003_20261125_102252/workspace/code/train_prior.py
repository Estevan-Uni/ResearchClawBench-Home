"""Train the learned prior on synthetic data only; grouped CV, learning curve,
IMO evaluation (IMO labels are never used for training)."""
import json
import pickle
import sys

import numpy as np

from prior import LogReg, load_synth, mrr_groups, mrr_random, heuristic_from_features, first_rank
from eval_search import load_labels, evaluate
from geo.aux import FEAT_NAMES


def grouped_folds(g, k=5, seed=0):
    ug = np.unique(g)
    rng = np.random.RandomState(seed)
    rng.shuffle(ug)
    return [set(x) for x in np.array_split(ug, k)]


def cv(F, y, g, l2=1.0, k=5):
    folds = grouped_folds(g, k)
    sc = np.zeros(len(y))
    for fo in folds:
        te = np.isin(g, list(fo))
        m = LogReg(l2).fit(F[~te], y[~te])
        sc[te] = m.score(F[te])
    return sc


def imo_median_rank(L, model):
    rows, curves = evaluate(L, model)
    rk = [r['rank_learned'] for r in rows.values() if r['successes'] > 0]
    return float(np.median(rk)) if rk else None, rows, curves


def main():
    data = {}
    for tag, path in (('selfplay', '../outputs/synth_B'), ('traceback', '../outputs/synth_A_aux')):
        d = load_synth([(path, tag)])
        if d is not None:
            data[tag] = d
            F, y, g, _ = d
            print(tag, 'records', len(y), 'positives', int(y.sum()), 'groups', len(np.unique(g)),
                  'groups with positive', len(np.unique(g[y > 0])))
    # combined
    Fs = np.vstack([d[0] for d in data.values()])
    ys = np.concatenate([d[1] for d in data.values()])
    gs = np.concatenate([d[2] + 10 ** 8 * i for i, d in enumerate(data.values())])
    data['combined'] = (Fs, ys, gs, None)
    L = load_labels()
    out = {'cv': {}, 'learning_curve': {}, 'imo': {}}
    for tag, (F, y, g, _) in data.items():
        sc = cv(F, y, g)
        m_l, ng = mrr_groups(sc, y, g)
        m_h, _ = mrr_groups(heuristic_from_features(F), y, g)
        m_r = mrr_random(y, g)
        out['cv'][tag] = dict(mrr_learned=m_l, mrr_heuristic=m_h, mrr_random=m_r, n_groups=ng, n=len(y),
                              n_pos=int(y.sum()))
        print('CV', tag, out['cv'][tag])
        model = LogReg().fit(F, y)
        med, rows, curves = imo_median_rank(L, model)
        out['imo'][tag] = dict(median_rank=med, curves=curves,
                               ranks={k: v['rank_learned'] for k, v in rows.items()})
        print('IMO', tag, med, curves)
        pickle.dump(model, open('../outputs/prior_model_%s.pkl' % tag, 'wb'))
    # learning curve on the combined data: fixed held-out groups, growing training sets
    F, y, g, _ = data['combined']
    ug_pos = np.unique(g[y > 0])
    rng = np.random.RandomState(1)
    rng.shuffle(ug_pos)
    allg = np.unique(g)
    test_g = set(ug_pos[: max(10, len(ug_pos) // 5)])
    train_g = np.array([x for x in allg if x not in test_g])
    rng.shuffle(train_g)
    te = np.isin(g, list(test_g))
    sizes = [s for s in [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120] if s < len(train_g)] + [len(train_g)]
    lc = []
    for s in sizes:
        for rep in range(3 if s < len(train_g) else 1):
            sub = set(np.random.RandomState(rep).permutation(train_g)[:s])
            tr = np.isin(g, list(sub))
            if y[tr].sum() == 0:
                continue
            m = LogReg().fit(F[tr], y[tr])
            mrr, _ = mrr_groups(m.score(F[te]), y[te], g[te])
            med, rows, curves = imo_median_rank(L, m)
            lc.append(dict(n_train_groups=s, rep=rep, n_train_pos=int(y[tr].sum()), mrr=mrr, imo_median_rank=med,
                           imo_solved_at_100=curves['learned'][BUD.index(100)], imo_solved_at_10=curves['learned'][BUD.index(10)]))
            print('LC', lc[-1], flush=True)
    out['learning_curve'] = lc
    out['test_mrr_random'] = mrr_random(y[te], g[te])
    out['test_mrr_heuristic'] = mrr_groups(heuristic_from_features(F[te]), y[te], g[te])[0]
    final = LogReg().fit(F, y)
    pickle.dump(final, open('../outputs/prior_model.pkl', 'wb'))
    rows, curves = evaluate(L, None)
    rr = [r['rank_random_expected'] for r in rows.values() if r['successes'] > 0]
    rh = [r['rank_heuristic'] for r in rows.values() if r['successes'] > 0]
    out['imo_random_median'] = float(np.median(rr)); out['imo_heuristic_median'] = float(np.median(rh))
    out['coef'] = dict(zip(FEAT_NAMES + ['bias'], final.w.tolist()))
    json.dump(out, open('../outputs/prior_results.json', 'w'), indent=1)


from eval_search import BUDGETS as BUD

if __name__ == '__main__':
    main()
