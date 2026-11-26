"""Stability of canonical frames under Gaussian noise (no network involved).
Compares the PCA/third-moment frame with sequence-anchored frames built from Fourier modes of the
token position. Writes outputs/frame_stability.json and report/images/fig_frames.png."""
import sys, os, math, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from biomol import load_2l3r
from model import AF3Lite
from common import to_torch
from fig_style import style, C, plt

torch.manual_seed(0)
f = to_torch(load_2l3r())
gt = f['coords'] - f['coords'].mean(0)
T = f['restype'].shape[0]
t = (f['atom_to_token'].float() + 0.5) / T
modes = {'t': t - t.mean(), 'cos1': torch.cos(2 * math.pi * t), 'sin1': torch.sin(2 * math.pi * t),
         'cos2': torch.cos(4 * math.pi * t)}


def seq_frame(x, a, b):
    xc = x - x.mean(1, keepdim=True)
    u1 = torch.einsum('n,snj->sj', modes[a], xc); u2 = torch.einsum('n,snj->sj', modes[b], xc)
    e1 = u1 / u1.norm(dim=-1, keepdim=True); u2 = u2 - (u2 * e1).sum(-1, keepdim=True) * e1
    e2 = u2 / u2.norm(dim=-1, keepdim=True)
    return torch.stack([e1, e2, torch.cross(e1, e2, dim=-1)], -1)


frames = {'PCA + 3rd-moment signs': lambda x: AF3Lite.canonical_frame(x)[1],
          'sequence (t, cos1) [used]': lambda x: seq_frame(x, 't', 'cos1'),
          'sequence (t, sin1)': lambda x: seq_frame(x, 't', 'sin1'),
          'sequence (cos1, sin1)': lambda x: seq_frame(x, 'cos1', 'sin1'),
          'sequence (t, cos2)': lambda x: seq_frame(x, 't', 'cos2')}
sigmas = [1, 2, 4, 8, 16, 32, 64, 128]
res = {}
for name, fn in frames.items():
    V0 = fn(gt[None])[0]
    med, q90 = [], []
    for s in sigmas:
        x = gt[None] + s * torch.randn(300, *gt.shape)
        V = fn(x)
        Rrel = torch.einsum('ij,sik->sjk', V0, V)
        ang = torch.rad2deg(torch.arccos(((Rrel.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(-1, 1)))
        med.append(float(ang.median())); q90.append(float(ang.quantile(0.9)))
    res[name] = dict(sigma=sigmas, median_deg=med, q90_deg=q90)
    print(name, np.round(med, 1))
json.dump(res, open('/workspace/outputs/frame_stability.json', 'w'), indent=1)

style()
fig, ax = plt.subplots(figsize=(6.2, 4.2))
cols = [C['red'], C['blue'], C['aqua'], C['orange'], C['violet']]
items = list(zip(res.items(), cols))
items = [it for it in items if 'used' not in it[0][0]] + [it for it in items if 'used' in it[0][0]]
for (name, r), col in items:
    ax.plot(r['sigma'], r['median_deg'], marker='o', ms=6 if 'used' in name else 4, color=col, label=name,
            lw=2.6 if 'used' in name else 1.3, zorder=5 if 'used' in name else 3)
ax.set_xscale('log'); ax.set_xlabel('noise level σ (Å)'); ax.set_ylabel('median frame error (degrees)')
ax.axhline(126.5, color=C['muted'], ls=':', lw=1); ax.text(1.05, 129, 'uniformly random rotation (median 126.5°)', fontsize=7.5, color=C['ink2'])
ax.set_title('Stability of canonical frames under noise'); ax.legend(fontsize=7.5, loc='center left', bbox_to_anchor=(0, 0.55))
plt.tight_layout(); plt.savefig('/workspace/report/images/fig_frames.png', dpi=150)
