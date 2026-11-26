"""Identify the generating model of data/charged_dimer.xyz and test energy/force consistency."""
import numpy as np, torch, json
from scipy.optimize import least_squares
from io_utils import read_extxyz
torch.set_default_dtype(torch.float64)
d = read_extxyz('../data/charged_dimer.xyz')
P = torch.tensor(np.array([f['pos'] for f in d])); F = np.array([f['forces'] for f in d]); E = np.array([f['energy'] for f in d])


def terms(pos, qC=0.5):
    q = torch.tensor([qC] + [(1 - qC) / 3] * 3 + [-qC] + [-(1 - qC) / 3] * 3)
    bond = []
    for m in (0, 4):
        for h in (1, 2, 3):
            bond.append((pos[:, m + h] - pos[:, m]).norm(dim=-1))
    bond = torch.stack(bond, -1)
    dd = pos[:, :, None] - pos[:, None]; r = dd.norm(dim=-1) + torch.eye(8)
    mask = torch.ones(8, 8); mask[:4, :4] = 0; mask[4:, 4:] = 0; mask = torch.triu(mask, 1)
    ecoul = ((q[:, None] * q[None]) * mask / r).sum((1, 2))
    return bond, ecoul


def energy_parts(p):
    kb, r0, c = p
    pos = P.clone().requires_grad_(True)
    bond, ecoul = terms(pos)
    eb = kb * ((bond - r0) ** 2).sum(-1)
    fb = -torch.autograd.grad(eb.sum(), pos, retain_graph=True)[0]
    fc = -torch.autograd.grad(ecoul.sum(), pos)[0]
    return (eb + ecoul + c).detach().numpy(), fb.numpy(), fc.numpy()


r = least_squares(lambda p: energy_parts(p)[0] - E, [5., 1.5, 0.])
e, fb, fc = energy_parts(r.x)
out = dict(params_energy_fit=dict(k_bond=r.x[0], r0=r.x[1], const=r.x[2]), E_resid_max=float(np.abs(e - E).max()))
for name, Fm in [('-grad(E)', fb + fc), ('+grad(E_bond) - grad(E_coul)', -fb + fc)]:
    out[name] = dict(F_mae=float(np.abs(Fm - F).mean()), F_max=float(np.abs(Fm - F).max()))
print(json.dumps(out, indent=1))
json.dump(out, open('../outputs/results/dimer_generator_analysis.json', 'w'), indent=1)

# corrected (conservative) forces: flip the sign of the bond-stretch contribution
F_corr = F + 2 * fb
np.savez('../outputs/charged_dimer_corrected_forces.npz', forces=F_corr, fb=fb, fc=fc)
print('corrected-force consistency vs -grad(E_fit): max', np.abs(F_corr - (fb + fc)).max())
