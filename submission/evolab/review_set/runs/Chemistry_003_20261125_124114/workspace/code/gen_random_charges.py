"""Generate reference energies/forces for data/random_charges.xyz.
Reference potential (stated assumptions; the file only provides positions and charges):
  * cubic periodic box L = 15 A (all coordinates lie in [0, 15))
  * point-charge Coulomb, Ewald summation, k_e = 14.399645 eV A / e^2
  * repulsive Lennard-Jones term  E_rep = sum_{i<j} 4 eps (sig/r)^12, eps = 0.25 eV, sig = 1.0 A (minimum image)
"""
import numpy as np, torch
from les import point_ewald, KE
from io_utils import read_extxyz

L, EPS, SIG = 15.0, 0.25, 1.0


def lj_rep(pos, L=L, eps=EPS, sig=SIG):
    n = pos.shape[0]
    d = pos[None] - pos[:, None]
    d = d - L * torch.round(d / L)
    iu = torch.triu_indices(n, n, 1)
    r = d[iu[0], iu[1]].norm(dim=-1)
    return (4 * eps * (sig / r) ** 12).sum()


if __name__ == '__main__':
    frames = read_extxyz('../data/random_charges.xyz')
    P, E, F, Ec, Er, Q = [], [], [], [], [], []
    for fr in frames:
        pos = torch.tensor(fr['pos'], requires_grad=True)
        q = torch.tensor(fr['true_charges'])
        ec = point_ewald(q, pos, L, alpha=0.6, nmax=14)
        er = lj_rep(pos)
        e = ec + er
        f = -torch.autograd.grad(e, pos)[0]
        P.append(fr['pos']); E.append(e.item()); F.append(f.numpy()); Ec.append(ec.item()); Er.append(er.item()); Q.append(fr['true_charges'])
    np.savez('../outputs/random_charges_labeled.npz', pos=np.array(P), energy=np.array(E), forces=np.array(F),
             e_coul=np.array(Ec), e_rep=np.array(Er), q=np.array(Q), L=L)
    E, Ec, Er, F = map(np.array, (E, Ec, Er, F))
    print('E_coul range', Ec.min(), Ec.max(), 'E_rep range', Er.min(), Er.max())
    print('|F| percentiles', np.percentile(np.linalg.norm(F, axis=-1), [50, 90, 99, 100]))
