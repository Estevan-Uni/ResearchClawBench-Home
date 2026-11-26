"""Shared utilities: feature conversion, alignment, EDM noise schedule and sampler, losses."""
import math
import numpy as np
import torch
import torch.nn.functional as F

INT_KEYS = {'restype', 'moltype', 'residue_index', 'token_index', 'asym_id', 'entity_id', 'sym_id',
            'atom_to_token', 'ref_element', 'ref_atom_name_chars', 'ref_space_uid', 'atom_moltype',
            'token_center_atom', 'token_rep_atom', 'lig_bonds'}


def to_torch(f):
    out = {}
    for k, v in f.items():
        if k == 'atom_names':
            continue
        out[k] = torch.as_tensor(v, dtype=torch.long if k in INT_KEYS else torch.float32)
    return out


def random_rotation(n, gen=None):
    q = torch.randn(n, 4, generator=gen)
    q = q / q.norm(dim=-1, keepdim=True)
    a, b, c, d = q.unbind(-1)
    R = torch.stack([a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c),
                     2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b),
                     2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d], -1)
    return R.reshape(n, 3, 3)


def weighted_kabsch(mobile, target, w):
    """Align mobile onto target (batched (S,N,3)); w (N,). Returns aligned mobile."""
    w = w / w.sum()
    mu_m = (mobile * w[None, :, None]).sum(1, keepdim=True)
    mu_t = (target * w[None, :, None]).sum(1, keepdim=True)
    A = mobile - mu_m; B = target - mu_t
    H = torch.einsum('sni,snj->sij', A * w[None, :, None], B)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(torch.einsum('sij,sjk->sik', Vt.transpose(1, 2), U.transpose(1, 2))))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    return torch.einsum('sij,snj->sni', R, A) + mu_t


def noise_schedule(n_steps, sd=16.0, smax=160.0, smin=4e-4, rho=7.0):
    t = torch.linspace(0, 1, n_steps + 1)
    return sd * (smax ** (1 / rho) + t * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho


@torch.no_grad()
def sample(model, f, tr, n_samples=5, n_steps=200, gamma0=0.8, gamma_min=1.0, noise_scale=1.003,
           step_scale=1.5, seed=0, return_traj=False):
    """AF3 Algorithm 18 (EDM stochastic sampler with centring/random augmentation)."""
    g = torch.Generator().manual_seed(seed)
    N = f['ref_pos'].shape[0]
    c = noise_schedule(n_steps, model.sd)
    x = c[0] * torch.randn(n_samples, N, 3, generator=g)
    traj = []
    for i in range(1, n_steps + 1):
        x = x - x.mean(1, keepdim=True)
        x = torch.einsum('sij,snj->sni', random_rotation(n_samples, g), x) + torch.randn(n_samples, 1, 3, generator=g)
        gamma = gamma0 if c[i] > gamma_min else 0.0
        t_hat = c[i - 1] * (gamma + 1)
        xi = noise_scale * math.sqrt(max(t_hat ** 2 - c[i - 1] ** 2, 0.0)) * torch.randn(x.shape, generator=g)
        x_noisy = x + xi
        x_den = model.denoise(f, tr, x_noisy, torch.full((n_samples,), float(t_hat)))
        delta = (x_noisy - x_den) / t_hat
        x = x_noisy + step_scale * (c[i] - t_hat) * delta
        if return_traj:
            traj.append(x.clone())
    return (x, traj) if return_traj else x


def lddt_pairs(x, f, cutoff_prot=15.0, cutoff_na=30.0):
    """Pairs (i<j) within the AF3 smooth-lDDT inclusion radius in the ground truth x (N,3)."""
    dg = torch.cdist(x, x)
    na = (f['atom_moltype'] == 1) | (f['atom_moltype'] == 2)
    cut = torch.where(na[None, :], cutoff_na, cutoff_prot)
    m = torch.triu(dg < cut, 1)
    i, j = torch.where(m)
    return i, j, dg[i, j]


def smooth_lddt_loss(xp, pairs):
    """AF3 Algorithm 27 on a precomputed sparse pair list. xp (S,N,3)."""
    i, j, dg = pairs
    dp = (xp[:, i] - xp[:, j]).norm(dim=-1)
    dd = (dp - dg[None]).abs()
    eps = 0.25 * (torch.sigmoid(0.5 - dd) + torch.sigmoid(1 - dd) + torch.sigmoid(2 - dd) + torch.sigmoid(4 - dd))
    return 1 - eps.mean()


def distogram_loss(logits, x, f, n_bins=64, dmin=2.0, dmax=22.0):
    rep = x[f['token_rep_atom']]
    d = torch.cdist(rep, rep)
    edges = torch.linspace(dmin, dmax, n_bins - 1)
    tgt = torch.bucketize(d, edges)
    return F.cross_entropy(logits.reshape(-1, n_bins), tgt.reshape(-1))
