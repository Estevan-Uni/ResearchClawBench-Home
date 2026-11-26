"""Charged dimer (CH3+ ... CH3-) benchmark: SR vs LES binding curves, latent charges, dipoles, BECs."""
import json, time, argparse
import numpy as np, torch
from les import LESPotential, Frame, train, evaluate, born_effective_charges, KE
from io_utils import read_extxyz
from common import split

# exact charges of the generating model (identified by fitting inter-molecular forces, residual 1e-8 eV/A; k_e = 1)
Q_TRUE = np.array([0.5, 1/6, 1/6, 1/6, -0.5, -1/6, -1/6, -1/6])
K_DATA = 1.0

p = argparse.ArgumentParser()
p.add_argument('--rc', type=float, default=4.0)
p.add_argument('--lr_on', type=int, default=1)
p.add_argument('--tagged', type=int, default=0)
p.add_argument('--epochs', type=int, default=1500)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--labels', default='corrected')  # raw | corrected | energy
p.add_argument('--wE', type=float, default=10.0)
a = p.parse_args()
torch.set_num_threads(1)


def species_of(sym, tagged):
    base = np.array([0 if s == 'C' else 1 for s in sym])
    if tagged:  # cation atoms (first molecule) and anion atoms carry distinct chemical labels
        base[4:] += 2
    return base


raw = read_extxyz('../data/charged_dimer.xyz')
ns = 4 if a.tagged else 2
Fcorr = np.load('../outputs/charged_dimer_corrected_forces.npz')['forces']
# training labels; evaluation of forces is always against the conservative (corrected) forces
frames = [Frame(r['pos'], species_of(r['symbols'], a.tagged), r['energy'], r['forces'] if a.labels == 'raw' else Fcorr[k])
          for k, r in enumerate(raw)]
eval_frames = [Frame(r['pos'], species_of(r['symbols'], a.tagged), r['energy'], Fcorr[k]) for k, r in enumerate(raw)]
tr, te = split(len(frames), 12, seed=a.seed)
name = f"dimer_{'les' if a.lr_on else 'sr'}_{'tagged' if a.tagged else 'plain'}_{a.labels}_rc{a.rc}_s{a.seed}"
torch.manual_seed(a.seed)
model = LESPotential(ns, rc=a.rc, n_rad=8, n_chan=12, hidden=64, use_lr=bool(a.lr_on), sigma=1.0)
log = open(f'../outputs/logs/{name}.log', 'w')
t0 = time.time()
hist = train(model, [frames[i] for i in tr], [eval_frames[i] for i in te], epochs=a.epochs, lr=3e-3, batch=4,
             wE=a.wE, wF=0.0 if a.labels == 'energy' else 1.0, seed=a.seed, verbose=100, log=log)
ttrain = time.time() - t0
torch.save(model.state_dict(), f'../outputs/models/{name}.pt')
rte, rtr = evaluate(model, [eval_frames[i] for i in te]), evaluate(model, [eval_frames[i] for i in tr])

# ---- binding curve with rigid ideal CH3 fragments (same orientation as in the data)
def ch3(center):
    h = 1.5
    return np.array([[0, 0, 0], [h, 0, 0], [-h / 2, h * np.sqrt(3) / 2, 0], [-h / 2, -h * np.sqrt(3) / 2, 0]]) + center
sp_full = species_of(['C', 'H', 'H', 'H'] * 2, a.tagged)
model.eval()
eA = model(Frame(ch3(np.zeros(3)), sp_full[:4]), need_forces=False)['E'].item()
eB = model(Frame(ch3(np.zeros(3)), sp_full[4:]), need_forces=False)['E'].item()
Rs = np.linspace(2.8, 16.0, 67)
Eml, Eref = [], []
for R in Rs:
    pos = np.vstack([ch3(np.zeros(3)), ch3(np.array([R, 0, 0]))])
    Eml.append(model(Frame(pos, sp_full), need_forces=False)['E'].item() - eA - eB)
    d = pos[:4, None] - pos[None, 4:]
    Eref.append(K_DATA * (Q_TRUE[:4, None] * Q_TRUE[None, 4:] / np.linalg.norm(d, axis=-1)).sum())

# ---- latent charges, dipoles and BECs on test frames
qs, mus_ml, mus_true, Zs = [], [], [], []
if a.lr_on:
    for i in te:
        f = frames[i]
        Z, q = born_effective_charges(model, f)
        qs.append(q.numpy()); Zs.append(Z.numpy())
        com = f.pos.numpy().mean(0)
        mus_ml.append((q.numpy()[:, None] * (f.pos.numpy() - com)).sum(0))
        mus_true.append((Q_TRUE[:, None] * (f.pos.numpy() - com)).sum(0))
np.savez(f'../outputs/preds/{name}.npz', Ep=rte['Ep'], Et=rte['Et'], Fp=rte['Fp'], Ft=rte['Ft'], Rs=Rs, Eml=Eml, Eref=Eref,
         q=np.array(qs), Z=np.array(Zs), mu_ml=np.array(mus_ml), mu_true=np.array(mus_true),
         Rte=np.array([np.linalg.norm(frames[i].pos[0] - frames[i].pos[4]).item() for i in te]))
out = dict(name=name, train_time=ttrain, test=dict((k, float(v)) for k, v in rte.items() if np.isscalar(v)),
           train=dict((k, float(v)) for k, v in rtr.items() if np.isscalar(v)), hist=hist)
json.dump(out, open(f'../outputs/results/{name}.json', 'w'), indent=1)
print(json.dumps(out['test'], indent=1))
