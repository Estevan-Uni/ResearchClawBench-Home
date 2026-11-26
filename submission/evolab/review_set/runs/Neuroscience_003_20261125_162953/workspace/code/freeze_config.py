import json, datetime
from core import DEFAULT_CFG
cfg = {'frozen_at': datetime.datetime.now().isoformat(),
 'note': 'Frozen before any label-based evaluate() call on RPE or simulation. Labels (age, phase, batch, true DE, true t) are used only in evaluation and DPT root orientation.',
 'primary': 'DELVE-faithful',
 'variants': {'DELVE-faithful': dict(DEFAULT_CFG),
              '+noise-filter': dict(DEFAULT_CFG, noise_filter=True),
              '+dedup': dict(DEFAULT_CFG, dedup=True),
              '+whiten': dict(DEFAULT_CFG, whiten=True)},
 'sim_overrides': {'n_anchors': 300, 'n_perm': 300, 'n_seeds': 3},
 'eval': {'headline_k': 30, 'k_grid': [5, 10, 20, 30, 50], 'knn_acc_k': 5, 'folds': 5,
          'nmi_clusters_rpe': 4, 'nmi_seeds': 5, 'dpt_knn': 15, 'panel_npc': 10,
          'root': 'medoid (in panel PCA space) of the 2% youngest non-G0 cells (sim: 1% smallest true t)'}}
json.dump(cfg, open('/workspace/outputs/config_frozen.json', 'w'), indent=1)
print(cfg['frozen_at'])
