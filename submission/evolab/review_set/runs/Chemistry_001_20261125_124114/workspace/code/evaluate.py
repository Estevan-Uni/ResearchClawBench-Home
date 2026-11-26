"""Evaluate a trained checkpoint: sample structures with the AF3 EDM sampler and score them.
Usage: python evaluate.py --run full [--ckpt model.pt] [--n_samples 10] [--n_steps 100] [--sweeps]"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from biomol import load_2l3r
from model import AF3Lite
from common import to_torch, sample, weighted_kabsch, random_rotation
from metrics import evaluate, kabsch, apply, rmsd

p = argparse.ArgumentParser()
p.add_argument('--run', default='full')
p.add_argument('--ckpt', default='model.pt')
p.add_argument('--tag', default='')
p.add_argument('--n_samples', type=int, default=10)
p.add_argument('--n_steps', type=int, default=100)
p.add_argument('--step_scale', type=float, default=1.5)
p.add_argument('--seed', type=int, default=123)
p.add_argument('--sweeps', action='store_true')
p.add_argument('--untrained', action='store_true')
p.add_argument('--sigma_curve', action='store_true')
args = p.parse_args()
torch.set_grad_enabled(False)

fnp = load_2l3r(ref_seed=0)
f = to_torch(fnp)
rd = f'/workspace/outputs/runs/{args.run}'
if args.untrained:
    model = AF3Lite(n_pairformer=2); tag = 'untrained'
else:
    ck = torch.load(f'{rd}/{args.ckpt}', weights_only=False)
    a = ck['args']
    model = AF3Lite(n_pairformer=a.get('n_pairformer', 2), canon=a.get('canon', False), canon_mode=a.get('canon_mode', 'pca'), use_trunk=not a['no_trunk'],
                    use_bonds=not a['no_bonds'], use_ref=not a['no_ref'])
    model.load_state_dict(ck['model'])
    tag = args.tag or args.ckpt.replace('.pt', '')
model.eval()
os.makedirs(rd, exist_ok=True)
tr = model.trunk(f)


def write_pdb(X, path):
    names = fnp['atom_names']; a2t = fnp['atom_to_token']; mol = fnp['atom_moltype']
    from biomol import RESTYPES
    with open(path, 'w') as fh:
        for i, (x, y, z) in enumerate(X):
            t = a2t[i]
            rn = RESTYPES[fnp['restype'][t]] if mol[i] == 0 else 'LIG'
            ch = 'A' if mol[i] == 0 else 'B'
            rs = fnp['residue_index'][t] + 1 if mol[i] == 0 else 1
            el = names[i][0] if mol[i] == 0 else ''.join(c for c in names[i] if c.isalpha())
            fh.write('HETATM' if mol[i] else 'ATOM  ')
            fh.write(f'{i+1:5d} {names[i][:4]:<4s} {rn:>3s} {ch}{rs:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}\n')
        fh.write('END\n')


def run_samples(n, steps, step_scale, seed):
    t0 = time.time()
    X = sample(model, f, tr, n_samples=n, n_steps=steps, step_scale=step_scale, seed=seed)
    dt = time.time() - t0
    mets = [evaluate(x.numpy(), fnp) for x in X]
    return X.numpy(), mets, dt


res = {}
X, mets, dt = run_samples(args.n_samples, args.n_steps, args.step_scale, args.seed)
res['main'] = dict(n_steps=args.n_steps, step_scale=args.step_scale, time_s=dt, metrics=mets)
np.save(f'{rd}/samples_{tag}.npy', X)
best = int(np.argmin([m['lig_rmsd_pocketaligned'] for m in mets]))
R, t = kabsch(X[best][fnp['token_center_atom'][fnp['moltype'] == 0]], fnp['coords'][fnp['token_center_atom'][fnp['moltype'] == 0]])
write_pdb(apply(R, t, X[best]), f'{rd}/best_sample_{tag}.pdb')
print(tag, 'sampling time', dt, flush=True)
for k in mets[0]:
    v = np.array([m[k] for m in mets], float)
    print(f'  {k:24s} mean {v.mean():8.3f}  median {np.median(v):8.3f}  min {v.min():8.3f}  max {v.max():8.3f}')

if args.sigma_curve:
    # single-step denoising accuracy as a function of noise level (held-out noise draws)
    gt = f['coords'] - f['coords'].mean(0)
    sig = torch.tensor([0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 160.0])
    g = torch.Generator().manual_seed(7)
    curve = []
    for s in sig:
        S = 6
        R = random_rotation(S, g)
        xg = torch.einsum('sij,nj->sni', R, gt)
        xn = xg + s * torch.randn(xg.shape, generator=g)
        xd = model.denoise(f, tr, xn, torch.full((S,), float(s)))
        al = weighted_kabsch(xg, xd, torch.ones(gt.shape[0]))
        e = ((xd - al) ** 2).sum(-1)
        lig = f['atom_moltype'] == 3
        curve.append(dict(sigma=float(s), rmsd_all=float(e.mean(1).sqrt().mean()),
                          rmsd_lig=float(e[:, lig].mean(1).sqrt().mean()),
                          rmsd_prot=float(e[:, ~lig].mean(1).sqrt().mean())))
        print('sigma', float(s), curve[-1], flush=True)
    res['sigma_curve'] = curve

if args.sweeps:
    sw = []
    for steps in [10, 25, 50, 100, 200]:
        _, m, dt = run_samples(5, steps, 1.5, args.seed + 1)
        sw.append(dict(kind='steps', n_steps=steps, step_scale=1.5, time_s=dt, metrics=m))
        print('steps', steps, dt, np.mean([x['lig_rmsd_pocketaligned'] for x in m]), np.mean([x['lddt_all'] for x in m]), flush=True)
    for ss in [1.0, 1.25, 2.0]:
        _, m, dt = run_samples(5, 50, ss, args.seed + 1)
        sw.append(dict(kind='step_scale', n_steps=50, step_scale=ss, time_s=dt, metrics=m))
        print('step_scale', ss, np.mean([x['lig_rmsd_pocketaligned'] for x in m]), np.mean([x['lddt_all'] for x in m]), flush=True)
    res['sweeps'] = sw
json.dump(res, open(f'{rd}/eval_{tag}.json', 'w'))
