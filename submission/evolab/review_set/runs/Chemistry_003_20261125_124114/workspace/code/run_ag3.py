"""Ag3 in two charge states (+1/-1): need for a global charge embedding.
dataset 'orig' : data/ag3_chargestates.xyz as provided (NB: both charge states have identical E and F)
dataset 'mod'  : controlled variant — Ag3(+) unchanged, Ag3(-) relabelled with a different (synthetic) Morse PES
"""
import json, time, argparse
import numpy as np, torch
from les import LESPotential, Frame, train, evaluate
from io_utils import read_extxyz

p = argparse.ArgumentParser()
p.add_argument('--data', default='orig')
p.add_argument('--model', default='sr')  # sr | sr_q | les | les_q | sep
p.add_argument('--epochs', type=int, default=1500)
p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
torch.set_num_threads(1)

# generator identified by least squares (max residual 1e-7): E = sum_pairs Morse(D=.5,a=1.5,r0=2.5) + 1.6/r - 1.5
ANION_PES = dict(D=0.35, a=1.3, r0=2.75)


def morse_coul(pos, D, a_, r0, k=1.6, c=None):
    pos = torch.tensor(pos, requires_grad=True)
    e = -3 * D if c is None else c
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        r = (pos[i] - pos[j]).norm()
        e = e + D * (1 - torch.exp(-a_ * (r - r0))) ** 2 + k / r
    f = -torch.autograd.grad(e, pos)[0]
    return e.item(), f.numpy()


raw = read_extxyz('../data/ag3_chargestates.xyz')
frames = []
for r in raw:
    E, F = r['energy'], r['forces']
    if a.data == 'mod' and r['total_charge'] < 0:
        E, F = morse_coul(r['pos'], ANION_PES['D'], ANION_PES['a'], ANION_PES['r0'])
    frames.append(Frame(r['pos'], np.zeros(3, int), E, F, total_charge=r['total_charge']))
ngeo = 30
idx = np.random.default_rng(a.seed).permutation(ngeo)
te_geo, tr_geo = idx[:8], idx[8:]
tr = list(tr_geo) + list(tr_geo + ngeo)
te = list(te_geo) + list(te_geo + ngeo)
name = f"ag3_{a.data}_{a.model}_s{a.seed}"
log = open(f'../outputs/logs/{name}.log', 'w')
t0 = time.time()


def make(model_type):
    return LESPotential(1, rc=6.0, n_rad=8, n_chan=8, hidden=32, use_lr=model_type.startswith('les'),
                        sigma=1.0, charge_embedding=model_type.endswith('_q'))


torch.manual_seed(a.seed)
results = {}
if a.model == 'sep':
    preds = {}
    for Q, sl_tr, sl_te in [(1, tr_geo, te_geo), (-1, tr_geo + ngeo, te_geo + ngeo)]:
        m = make('sr')
        train(m, [frames[i] for i in sl_tr], [frames[i] for i in sl_te], epochs=a.epochs, lr=3e-3, batch=4,
              seed=a.seed, verbose=250, log=log)
        preds[Q] = (evaluate(m, [frames[i] for i in sl_te]), m)
    res = {Q: preds[Q][0] for Q in preds}
    models = {Q: preds[Q][1] for Q in preds}
else:
    m = make(a.model)
    train(m, [frames[i] for i in tr], [frames[i] for i in te], epochs=a.epochs, lr=3e-3, batch=4,
          seed=a.seed, verbose=250, log=log)
    res = {1: evaluate(m, [frames[i] for i in te_geo]), -1: evaluate(m, [frames[i] for i in te_geo + ngeo])}
    models = {1: m, -1: m}

# energy along symmetric stretch of equilateral triangle
ds = np.linspace(2.0, 4.0, 81)
scan = {}
for Q in (1, -1):
    Es = []
    for d in ds:
        pos = np.array([[0, 0, 0], [d, 0, 0], [d / 2, d * np.sqrt(3) / 2, 0]])
        Es.append(models[Q](Frame(pos, np.zeros(3, int), total_charge=Q), need_forces=False)['E'].item())
    ref = []
    for d in ds:
        pos = np.array([[0, 0, 0], [d, 0, 0], [d / 2, d * np.sqrt(3) / 2, 0]])
        if a.data == 'mod' and Q < 0:
            ref.append(morse_coul(pos, **{'D': ANION_PES['D'], 'a_': ANION_PES['a'], 'r0': ANION_PES['r0']})[0])
        else:
            ref.append(morse_coul(pos, 0.5, 1.5, 2.5)[0])
    scan[Q] = (np.array(Es), np.array(ref))
out = dict(name=name, train_time=time.time() - t0)
for Q in (1, -1):
    out[f'Q{Q:+d}'] = dict(E_mae_atom=float(res[Q]['E_mae_atom']), F_mae=float(res[Q]['F_mae']),
                          E_rmse_atom=float(res[Q]['E_rmse_atom']), F_rmse=float(res[Q]['F_rmse']))
np.savez(f'../outputs/preds/{name}.npz', ds=ds, scan_p=scan[1][0], ref_p=scan[1][1], scan_m=scan[-1][0], ref_m=scan[-1][1],
         Ep_p=res[1]['Ep'], Et_p=res[1]['Et'], Ep_m=res[-1]['Ep'], Et_m=res[-1]['Et'])
json.dump(out, open(f'../outputs/results/{name}.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
