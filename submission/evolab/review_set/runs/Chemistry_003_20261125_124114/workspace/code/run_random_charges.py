"""Random point charges benchmark: SR vs LES; charge recovery; cutoff dependence; species-blind ablation."""
import sys, json, time, argparse
import numpy as np, torch
from common import load_random_charges, split
from les import LESPotential, train, evaluate, born_effective_charges

p = argparse.ArgumentParser()
p.add_argument('--rc', type=float, default=5.0)
p.add_argument('--lr_on', type=int, default=1)
p.add_argument('--blind', type=int, default=0)
p.add_argument('--epochs', type=int, default=300)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--bec', type=int, default=0)
p.add_argument('--tag', type=str, default='')
p.add_argument("--lr", type=float, default=5e-3)
p.add_argument("--ntrain", type=int, default=80)
a = p.parse_args()
torch.set_num_threads(1)
torch.manual_seed(a.seed)
frames = load_random_charges(species_blind=bool(a.blind))
tr, te = split(len(frames), 20, seed=0)
tr = tr[:a.ntrain]
ns = 1 if a.blind else 2
model = LESPotential(ns, rc=a.rc, n_chan=8, lmax=1, hidden=64, use_lr=bool(a.lr_on), pair_prior=True, sigma=1.0, dl=2.0)
name = a.tag or f"rc_rc{a.rc}_{'les' if a.lr_on else 'sr'}{'_blind' if a.blind else ''}{'' if a.ntrain == 80 else f'_n{a.ntrain}'}_s{a.seed}"
log = open(f'../outputs/logs/{name}.log', 'w')
t0 = time.time()
hist = train(model, [frames[i] for i in tr], [frames[i] for i in te], epochs=a.epochs, lr=a.lr, batch=4,
             wE=1.0, wF=1.0, seed=a.seed, verbose=10, log=log)
ttrain = time.time() - t0
torch.save(model.state_dict(), f'../outputs/models/{name}.pt')
res_te = evaluate(model, [frames[i] for i in te])
res_tr = evaluate(model, [frames[i] for i in tr])
out = dict(name=name, rc=a.rc, les=a.lr_on, blind=a.blind, train_time=ttrain,
           test=dict((k, float(v)) for k, v in res_te.items() if np.isscalar(v)),
           train=dict((k, float(v)) for k, v in res_tr.items() if np.isscalar(v)), hist=hist)
np.savez(f'../outputs/preds/{name}.npz', Ep=res_te['Ep'], Et=res_te['Et'], Fp=res_te['Fp'], Ft=res_te['Ft'],
         q=np.array([q.numpy() for q in res_te['q']]) if res_te['q'][0] is not None else np.zeros(1),
         qtrue=np.array([frames[i].true_q.numpy() for i in te]))
if a.bec and a.lr_on:
    Zs = []
    for i in te[:3]:
        Z, q = born_effective_charges(model, frames[i])
        Zs.append(Z.numpy())
    np.save(f'../outputs/preds/{name}_bec.npy', np.array(Zs))
json.dump(out, open(f'../outputs/results/{name}.json', 'w'), indent=1)
print(json.dumps(out['test'], indent=1))
