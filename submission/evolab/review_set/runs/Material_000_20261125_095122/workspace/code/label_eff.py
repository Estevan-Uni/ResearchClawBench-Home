"""Label-efficiency on the positive-control task: n_pos in {10,25,50,100}, several seeds, fixed epoch budget."""
import sys, json, time; sys.path.insert(0, '/workspace/code')
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
import screen
from common import *
from control_labels import control_subset
methods = sys.argv[1].split(','); seeds = [int(s) for s in sys.argv[2].split(',')]; out = sys.argv[3]
EPOCHS = 40
D = load_cache()
kc, cy, _ = control_subset(D['candidate'], 50, 1); cd = [D['candidate'][i] for i in kc]; cd_items = screen.featurize(cd); cb = screen.collate(cd_items)
ft_items_all = screen.featurize(D['finetune']); from control_labels import has_motif
mot = np.array([has_motif(it) for it in ft_items_all]); neg = np.where(~mot)[0]; pos_pool = np.where(mot)[0]
Xd_all = screen.comp_desc_features(D['finetune']); Xd_cd = screen.comp_desc_features(cd)
rows = []
for seed in seeds:
    for n_pos in (10, 25, 50, 100):
        pos = np.random.default_rng(1000 + seed).choice(pos_pool, n_pos, replace=False)
        idx = np.concatenate([neg, pos]); y = mot[idx].astype(int)
        for m in methods:
            t0 = time.time()
            if m == 'gbdt':
                s = screen.fit_classical('gbdt', Xd_all[idx], y, [Xd_cd], seed)[0]
            else:
                it = [ft_items_all[i] for i in idx]
                model, log, _ = screen.train_gnn(m, it, y, it[:200], y[:200], epochs=EPOCHS, seed=seed, select='last', patience=10**6)
                s = screen.predict(model, cb).numpy()
            rows.append(dict(method=m, n_pos=n_pos, seed=seed, auc=auc(cy, s), ap=ap(cy, s), hits50=hits_at(cy, s, 50)))
            print(rows[-1], f'{time.time()-t0:.0f}s', flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
print('DONE')
