import numpy as np, torch
from les import Frame, KE
from io_utils import read_extxyz
from gen_random_charges import lj_rep, L


def load_random_charges(species_blind=False):
    """Learning target = Coulomb part (the fixed LJ repulsion is treated as a known analytic baseline)."""
    d = np.load('../outputs/random_charges_labeled.npz')
    frames = []
    for p, q in zip(d['pos'], d['q']):
        pos = torch.tensor(p, requires_grad=True)
        er = lj_rep(pos)
        frl = -torch.autograd.grad(er, pos)[0]
        frames.append((p, q, er.item(), frl.numpy()))
    out = []
    for k, (p, q, er, frl) in enumerate(frames):
        sp = np.zeros(len(q), int) if species_blind else (q < 0).astype(int)
        out.append(Frame(p, sp, d['energy'][k] - er, d['forces'][k] - frl, cell=float(d['L']), true_q=q))
    return out


def split(n, ntest, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return idx[ntest:], idx[:ntest]
