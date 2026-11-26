"""Train the AF3-lite model on the 2L3R complex (UHRF1 TTD + H3K9me3 peptide).
Usage: python train.py --name full --steps 3000 [--no_trunk] [--no_bonds] [--no_ref]"""
import argparse, json, time, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from biomol import load_2l3r
from model import AF3Lite
from common import to_torch, random_rotation, weighted_kabsch, smooth_lddt_loss, distogram_loss, lddt_pairs

p = argparse.ArgumentParser()
p.add_argument('--name', default='full')
p.add_argument('--steps', type=int, default=3000)
p.add_argument('--diff_batch', type=int, default=8)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--no_trunk', action='store_true')
p.add_argument('--no_bonds', action='store_true')
p.add_argument('--no_ref', action='store_true')
p.add_argument('--no_lddt', action='store_true')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--n_pairformer', type=int, default=2)
p.add_argument('--save_at', type=str, default='500,1000,1500,2000')
p.add_argument('--p_mean', type=float, default=-1.2)
p.add_argument('--p_std', type=float, default=1.5)
p.add_argument('--canon', action='store_true')
p.add_argument('--canon_mode', default='pca')
p.add_argument('--init_from', type=str, default=None)
p.add_argument('--warmup', type=int, default=200)
p.add_argument('--threads', type=int, default=2)
p.add_argument('--stop_at', type=int, default=None)
p.add_argument('--max_minutes', type=float, default=1e9)
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.set_num_threads(args.threads)
out = f'/workspace/outputs/runs/{args.name}'
os.makedirs(out, exist_ok=True)
f = to_torch(load_2l3r(ref_seed=0))
gt = f['coords'] - f['coords'].mean(0)
w = f['atom_weight']
pairs = lddt_pairs(gt, f)
print('lddt pairs', pairs[0].numel())
model = AF3Lite(n_pairformer=args.n_pairformer, canon=args.canon, canon_mode=args.canon_mode, use_trunk=not args.no_trunk, use_bonds=not args.no_bonds, use_ref=not args.no_ref)
if args.init_from:
    model.load_state_dict(torch.load(args.init_from, weights_only=False)['model'])
    print('initialised from', args.init_from, flush=True)
n_par = sum(p.numel() for p in model.parameters())
print('params', n_par, 'atoms', gt.shape[0], 'tokens', f['restype'].shape[0], flush=True)
opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
sd = model.sd
warm = args.warmup
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + np.cos(np.pi * min(s, args.steps) / args.steps)) * 0.95 + 0.05))
lb = f['lig_bonds']
log = []
t0 = time.time()
for step in range(args.steps):
    S = args.diff_batch
    tr = model.trunk(f)
    R = random_rotation(S)
    x_gt = torch.einsum('sij,nj->sni', R, gt) + torch.randn(S, 1, 3)
    sigma = sd * torch.exp(args.p_mean + args.p_std * torch.randn(S))
    x_noisy = x_gt + sigma[:, None, None] * torch.randn_like(x_gt)
    x_den = model.denoise(f, tr, x_noisy, sigma)
    with torch.no_grad():
        x_gt_al = weighted_kabsch(x_gt, x_den, w)
    lw = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2
    mse = ((x_den - x_gt_al) ** 2).sum(-1) * w[None]
    l_mse = (lw * mse.mean(1) / 3.0).mean()
    dpb = (x_den[:, lb[:, 0]] - x_den[:, lb[:, 1]]).norm(dim=-1)
    dgb = (x_gt[:, lb[:, 0]] - x_gt[:, lb[:, 1]]).norm(dim=-1)
    l_bond = ((dpb - dgb) ** 2).mean()
    l_lddt = smooth_lddt_loss(x_den, pairs) if not args.no_lddt else torch.zeros(())
    l_dist = distogram_loss(tr['disto'], gt, f)
    a_bond = 1.0 if (step >= int(0.3 * args.steps) or args.init_from) else 0.0  # AF3: bond loss only in later stage
    loss = l_mse + a_bond * l_bond + l_lddt + 0.1 * l_dist
    opt.zero_grad()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step(); sched.step()
    rec = dict(step=step, loss=loss.item(), mse=l_mse.item(), bond=l_bond.item(), lddt=l_lddt.item(),
               disto=l_dist.item(), gn=gn.item(), t=time.time() - t0)
    log.append(rec)
    if step % 25 == 0:
        print(json.dumps(rec), flush=True)
    if (step + 1) % 250 == 0 or step == args.steps - 1:
        torch.save(dict(model=model.state_dict(), args=vars(args), step=step), f'{out}/model.pt')
        json.dump(log, open(f'{out}/log.json', 'w'))
    if str(step + 1) in args.save_at.split(','):
        torch.save(dict(model=model.state_dict(), args=vars(args), step=step), f'{out}/model_{step+1}.pt')
    if args.stop_at is not None and step + 1 >= args.stop_at:
        print('stop_at reached', step + 1, flush=True)
        break
    if (time.time() - t0) / 60 > args.max_minutes:
        print('time budget reached at step', step, flush=True)
        break
torch.save(dict(model=model.state_dict(), args=vars(args), step=step), f'{out}/model.pt')
json.dump(log, open(f'{out}/log.json', 'w'))
print('done', time.time() - t0)
