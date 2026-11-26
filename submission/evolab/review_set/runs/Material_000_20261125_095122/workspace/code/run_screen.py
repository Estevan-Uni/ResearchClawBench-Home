"""Driver: python run_screen.py real|control"""
import sys; sys.path.insert(0, '/workspace/code')
import numpy as np, torch
torch.set_num_threads(1)
import screen
from common import load_cache
from control_labels import control_subset
which = sys.argv[1]; methods = sys.argv[2].split(',') if len(sys.argv) > 2 else ['random', 'gbdt', 'wl_lr', 'scratch', 'pre_frozen', 'pre_ft']
D = load_cache()
if which == 'real':
    ft, cd = D['finetune'], D['candidate']; fy = [g['y'] for g in ft]; cy = [g['y'] for g in cd]
    np.save('/workspace/outputs/idx_real_cand.npy', np.arange(len(cd)))
else:
    kf, fy, _ = control_subset(D['finetune'], 100, 0); kc, cy, _ = control_subset(D['candidate'], 50, 1)
    ft = [D['finetune'][i] for i in kf]; cd = [D['candidate'][i] for i in kc]
    np.save('/workspace/outputs/idx_control_cand.npy', kc); np.save('/workspace/outputs/idx_control_ft.npy', kf)
print(which, len(ft), sum(fy), len(cd), sum(cy), flush=True)
screen.screen(ft, fy, cd, cy, methods, which + ('' if len(sys.argv) <= 3 else sys.argv[3]))
print('DONE')
