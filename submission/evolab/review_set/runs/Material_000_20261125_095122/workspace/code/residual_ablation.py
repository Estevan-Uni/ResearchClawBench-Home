"""Residual on/off at 6 layers (from scratch) on the positive-control task: training-loss stability & candidate AUC."""
import sys, json; sys.path.insert(0, '/workspace/code')
import numpy as np, torch
torch.set_num_threads(1)
import screen
from common import *
from control_labels import control_subset
D = load_cache()
kf, fy, _ = control_subset(D['finetune'], 100, 0); kc, cy, _ = control_subset(D['candidate'], 50, 1)
ft = screen.featurize([D['finetune'][i] for i in kf]); cd = screen.featurize([D['candidate'][i] for i in kc])
out = {}
for layers in (3, 6):
    for res in (True, False):
        for seed in (0, 1):
            m, log, _ = screen.train_gnn('scratch', ft, fy, cd, cy, epochs=40, seed=seed, layers=layers, residual=res, select='last', patience=10**6)
            out[f'L{layers}_res{int(res)}_s{seed}'] = [dict(epoch=r['epoch'], train_loss=r['train_loss_running'], cand_auc=r['val_auc'], cand_ap=r['val_ap']) for r in log]
            print(layers, res, seed, log[-1]['train_loss_running'], log[-1]['val_auc'], flush=True)
            json.dump(out, open('/workspace/outputs/residual_ablation.json', 'w'))
print('DONE')
