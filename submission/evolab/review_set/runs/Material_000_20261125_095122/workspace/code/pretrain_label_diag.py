"""DIAGNOSTIC ONLY: are the provided labels learnable from graphs at all? Uses the 5000 balanced pretrain labels
(never used for screening). Also pre-registered test of the atom-count lead found in pretrain+finetune."""
import sys; sys.path.insert(0, '/workspace/code')
import numpy as np, pandas as pd, torch, json
torch.set_num_threads(1)
import screen
from common import *
D = load_cache(); gs = D['pretrain']; y = np.array([g['y'] for g in gs]); items = screen.featurize(gs)
folds = screen.strat_folds(y, 5, 7); Xd = screen.comp_desc_features(gs); W = screen.wl_features(items, depth=3)
rows = []
oof = {m: np.zeros(len(y)) for m in ['gbdt', 'wl_lr', 'scratch_gnn']}
for f in range(5):
    tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
    oof['gbdt'][te] = screen.fit_classical('gbdt', Xd[tr], y[tr], [Xd[te]])[0]
    oof['wl_lr'][te] = screen.fit_classical('wl_lr', [W[i] for i in tr], y[tr], [[W[i] for i in te]])[0]
    if f < 3:   # GNN on 3 folds (time budget)
        inner = screen.strat_folds(y[tr], 5, f); itr, iva = tr[inner != 0], tr[inner == 0]
        m, log, be = screen.train_gnn('scratch', [items[i] for i in itr], y[itr], [items[i] for i in iva], y[iva], epochs=30, seed=f, bs=128, select='loss', patience=10)
        oof['scratch_gnn'][te] = screen.predict(m, screen.collate([items[i] for i in te])).numpy()
    else: oof['scratch_gnn'][te] = np.nan
for m, o in oof.items():
    ok = ~np.isnan(o); ci = bootstrap_ci(y[ok], o[ok], auc, 500)
    rows.append(dict(method=m, n=int(ok.sum()), cv_auc=auc(y[ok], o[ok]), lo=ci[0], hi=ci[1]))
# pre-registered size lead: score = +1 for 14 atoms, -1 for 24 atoms (derived from pretrain/finetune), tested on candidate
cd = D['candidate']; cy = np.array([g['y'] for g in cd]); n = np.array([len(g['x']) for g in cd])
sc = (n == 14).astype(float) - (n == 24).astype(float)
from scipy.stats import fisher_exact
t14 = [[int(((n == 14) & (cy == 1)).sum()), int(((n == 14) & (cy == 0)).sum())], [int(((n != 14) & (cy == 1)).sum()), int(((n != 14) & (cy == 0)).sum())]]
rows.append(dict(method='size lead on candidates (14:+1, 24:-1)', n=len(cy), cv_auc=auc(cy, sc), lo=np.nan, hi=np.nan,
                 note=f'14-atom positives {t14[0][0]}/{t14[0][0]+t14[0][1]} vs others {t14[1][0]}/{t14[1][0]+t14[1][1]}; one-sided Fisher p={fisher_exact(t14, alternative="greater")[1]:.3f}'))
R = pd.DataFrame(rows); print(R.to_string()); R.to_csv('outputs/pretrain_label_diagnostic.csv', index=False)
print('DONE')
