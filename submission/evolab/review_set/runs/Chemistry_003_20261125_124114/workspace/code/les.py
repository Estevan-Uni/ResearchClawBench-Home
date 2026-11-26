"""
Latent Ewald Summation (LES) machine-learning interatomic potential — compact PyTorch implementation.

E_total = E_sr + E_lr (+ optional fixed analytic baseline)
  E_sr = sum_i  MLP_E(B_i)                (local, cutoff r_c)
  q_i  = MLP_q(B_i)                       (latent charges, same local descriptor)
  E_lr = periodic : (2 pi k_e / V) sum_{0<|k|<k_c} exp(-sigma^2 k^2 / 2) / k^2 |S(k)|^2,  S(k)=sum_i q_i e^{i k.r_i}
         isolated : (k_e/2) sum_{i!=j} q_i q_j erf(r_ij / (sqrt(2) sigma)) / r_ij
Forces are -dE/dr via autograd.  Latent charges are never fitted to reference charges.

Descriptor B_i: a small, learnable ACE/SOAP-like invariant (l = 0,1,2) built from
species-dependent radial channels, so the model is many-body and rotationally invariant.
"""
import math
import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
KE = 14.399645  # eV * Angstrom / e^2


# ----------------------------------------------------------------------------- data
class Frame:
    def __init__(self, pos, species, energy=None, forces=None, cell=None, total_charge=0.0, true_q=None):
        self.pos = torch.as_tensor(pos, dtype=torch.float64)
        self.species = torch.as_tensor(species, dtype=torch.long)
        self.energy = None if energy is None else float(energy)
        self.forces = None if forces is None else torch.as_tensor(forces, dtype=torch.float64)
        self.cell = cell  # cubic box length (float) or None
        self.Q = float(total_charge)
        self.true_q = None if true_q is None else torch.as_tensor(true_q, dtype=torch.float64)
        self.n = len(species)


def pair_list(pos, cell, rc):
    """All pairs i!=j within rc (minimum image for cubic boxes, rc <= L/2)."""
    n = pos.shape[0]
    d = pos[None, :, :] - pos[:, None, :]  # d[i,j] = r_j - r_i
    if cell is not None:
        d = d - cell * torch.round(d / cell)
    r = d.norm(dim=-1)
    mask = (r < rc) & (~torch.eye(n, dtype=torch.bool))
    i, j = mask.nonzero(as_tuple=True)
    return i, j


def cutoff_fn(r, rc):
    return 0.5 * (torch.cos(math.pi * r / rc) + 1.0) * (r < rc)


# ----------------------------------------------------------------------------- long range
def ewald_reciprocal(q, pos, L, sigma=1.0, dl=1.5, ke=KE):
    """LES reciprocal-space energy for Gaussian-smeared charges in a cubic box (k = -k pairs merged)."""
    nmax = int(math.ceil(L / dl))
    rng = torch.arange(-nmax, nmax + 1, dtype=torch.float64)
    g = torch.stack(torch.meshgrid(rng, rng, rng, indexing='ij'), -1).reshape(-1, 3)
    # half space: keep vectors with first nonzero component > 0
    keep = (g[:, 0] > 0) | ((g[:, 0] == 0) & (g[:, 1] > 0)) | ((g[:, 0] == 0) & (g[:, 1] == 0) & (g[:, 2] > 0))
    k = 2 * math.pi / L * g[keep]
    k2 = (k ** 2).sum(-1)
    kc = 2 * math.pi / dl
    sel = k2 < kc ** 2
    k, k2 = k[sel], k2[sel]
    phase = pos @ k.T  # (n, nk)
    Sr = (q[:, None] * torch.cos(phase)).sum(0)
    Si = (q[:, None] * torch.sin(phase)).sum(0)
    V = L ** 3
    # factor 2 for the merged +-k pairs
    return 2.0 * (2 * math.pi * ke / V) * (torch.exp(-0.5 * sigma ** 2 * k2) / k2 * (Sr ** 2 + Si ** 2)).sum()


def direct_lr(q, pos, sigma=1.0, ke=KE):
    d = pos[None] - pos[:, None]
    n = pos.shape[0]
    iu = torch.triu_indices(n, n, 1)
    r = d[iu[0], iu[1]].norm(dim=-1)
    return ke * (q[iu[0]] * q[iu[1]] * torch.erf(r / (math.sqrt(2) * sigma)) / r).sum()


def point_ewald(q, pos, L, alpha=0.45, rcut=None, nmax=10, ke=KE):
    """Exact (converged) Ewald energy for point charges in a cubic box — used to generate reference data."""
    n = pos.shape[0]
    rcut = L / 2 if rcut is None else rcut
    d = pos[None] - pos[:, None]
    d = d - L * torch.round(d / L)
    iu = torch.triu_indices(n, n, 1)
    r = d[iu[0], iu[1]].norm(dim=-1)
    m = r < rcut
    e_real = ke * (q[iu[0]] * q[iu[1]] * torch.erfc(alpha * r) / r * m).sum()
    rng = torch.arange(-nmax, nmax + 1, dtype=torch.float64)
    g = torch.stack(torch.meshgrid(rng, rng, rng, indexing='ij'), -1).reshape(-1, 3)
    g = g[(g ** 2).sum(-1) > 0]
    k = 2 * math.pi / L * g
    k2 = (k ** 2).sum(-1)
    ph = pos @ k.T
    S2 = (q[:, None] * torch.cos(ph)).sum(0) ** 2 + (q[:, None] * torch.sin(ph)).sum(0) ** 2
    e_rec = (2 * math.pi * ke / L ** 3) * (torch.exp(-k2 / (4 * alpha ** 2)) / k2 * S2).sum()
    e_self = -ke * alpha / math.sqrt(math.pi) * (q ** 2).sum()
    return e_real + e_rec + e_self


# ----------------------------------------------------------------------------- model
def mlp(nin, nh, nout, nlayer=2):
    layers, d = [], nin
    for _ in range(nlayer):
        layers += [nn.Linear(d, nh), nn.SiLU()]
        d = nh
    layers += [nn.Linear(d, nout)]
    return nn.Sequential(*layers)


class LESPotential(nn.Module):
    def __init__(self, n_species, rc=5.0, n_rad=8, n_chan=12, hidden=64,
                 use_lr=True, sigma=1.0, dl=1.5, ke=KE, charge_embedding=False,
                 pair_prior=False, baseline=None, lmax=2):
        super().__init__()
        self.ns, self.rc, self.nrad, self.K = n_species, rc, n_rad, n_chan
        self.use_lr, self.sigma, self.dl, self.ke = use_lr, sigma, dl, ke
        self.charge_embedding = charge_embedding
        self.lmax = lmax
        self.register_buffer('mu', torch.linspace(0.5, rc, n_rad))
        self.gamma = (n_rad / rc) ** 2
        # species-pair dependent mixing of radial basis into K channels: W[s_i, s_j]
        self.W = nn.Parameter(torch.randn(n_species, n_species, n_rad, n_chan) / math.sqrt(n_rad))
        nfeat = n_chan + lmax * n_chan * (n_chan + 1) // 2 + n_species + (1 if charge_embedding else 0)
        self.nfeat = nfeat
        self.register_buffer('fmean', torch.zeros(nfeat))
        self.register_buffer('fstd', torch.ones(nfeat))
        self.E_net = mlp(nfeat, hidden, 1)
        self.e0 = nn.Parameter(torch.zeros(n_species))
        if use_lr:
            self.q_net = mlp(nfeat, hidden, 1)
        self.pair_prior = pair_prior
        if pair_prior:  # learnable B_ab / r * fc(r): lets the local model represent short-range point-charge singularities
            self.B = nn.Parameter(torch.zeros(n_species, n_species))
        self.baseline = baseline  # fixed analytic function(frame, pos) -> energy (not learned)
        iu = torch.triu_indices(n_chan, n_chan)
        self.register_buffer('iu0', iu[0])
        self.register_buffer('iu1', iu[1])

    # -- descriptor (operates on a batch of frames merged into one disjoint graph)
    def _pairs(self, fr):
        key = ('pairs', self.rc)
        if not hasattr(fr, '_cache'):
            fr._cache = {}
        if key not in fr._cache or fr._cache.get('pos_ref') is not fr.pos:
            fr._cache[key] = pair_list(fr.pos.detach(), fr.cell, self.rc)
            fr._cache['pos_ref'] = fr.pos
        return fr._cache[key]

    def features(self, frames, pos, use_cache=True):
        """frames: list of Frame; pos: (N_tot, 3) concatenated positions."""
        if isinstance(frames, Frame):
            frames = [frames]
        ii, jj, cells, off = [], [], [], 0
        for f in frames:
            if use_cache:
                i, j = self._pairs(f)
            else:
                i, j = pair_list(pos[off:off + f.n].detach(), f.cell, self.rc)
            ii.append(i + off); jj.append(j + off)
            cells.append(torch.full((len(i),), f.cell if f.cell is not None else 0.0))
            off += f.n
        i, j, cell = torch.cat(ii), torch.cat(jj), torch.cat(cells)
        species = torch.cat([f.species for f in frames])
        d = pos[j] - pos[i]
        per = cell > 0
        if per.any():
            c = cell[:, None]
            d = torch.where(per[:, None], d - c * torch.round(d.detach() / torch.where(per, cell, 1.0)[:, None]), d)
        r = d.norm(dim=-1)
        u = d / r[:, None]
        fb = torch.exp(-self.gamma * (r[:, None] - self.mu) ** 2) * cutoff_fn(r, self.rc)[:, None]  # (P, nrad)
        Wp = self.W[species[i], species[j]]  # (P, nrad, K)
        R = torch.einsum('pn,pnk->pk', fb, Wp)  # (P, K)
        n = pos.shape[0]
        A0 = torch.zeros(n, self.K).index_add(0, i, R)
        feats = [A0]
        if self.lmax >= 1:
            A1 = torch.zeros(n, self.K, 3).index_add(0, i, R[:, :, None] * u[:, None, :])
            G1 = torch.bmm(A1, A1.transpose(1, 2))
            feats.append(G1[:, self.iu0, self.iu1])
        if self.lmax >= 2:
            uu = (u[:, :, None] * u[:, None, :] - torch.eye(3) / 3.0).reshape(-1, 9)
            A2 = torch.zeros(n, self.K, 9).index_add(0, i, R[:, :, None] * uu[:, None, :])
            G2 = torch.bmm(A2, A2.transpose(1, 2))
            feats.append(G2[:, self.iu0, self.iu1])
        feats.append(torch.nn.functional.one_hot(species, self.ns).double())
        if self.charge_embedding:
            feats.append(torch.cat([torch.full((f.n, 1), f.Q) for f in frames]))
        X = torch.cat(feats, -1)
        return X, (i, j, r, species)

    def set_normalization(self, frames):
        with torch.no_grad():
            X = torch.cat([self.features([f], f.pos)[0] for f in frames])
            self.fmean.copy_(X.mean(0))
            s = X.std(0)
            s[s < 1e-6] = 1.0
            self.fstd.copy_(s)

    def forward(self, frames, pos=None, need_forces=True, use_cache=None):
        single = isinstance(frames, Frame)
        if single:
            frames = [frames]
        if use_cache is None:
            use_cache = pos is None
        if pos is None:
            pos = torch.cat([f.pos for f in frames]).clone()
        if need_forces and not pos.requires_grad:
            pos.requires_grad_(True)
        nf = len(frames)
        batch = torch.cat([torch.full((f.n,), k, dtype=torch.long) for k, f in enumerate(frames)])
        X, (i, j, r, species) = self.features(frames, pos, use_cache=use_cache)
        Xn = (X - self.fmean) / self.fstd
        e_atom = self.E_net(Xn).squeeze(-1) + self.e0[species]
        E_sr = torch.zeros(nf).index_add(0, batch, e_atom)
        if self.pair_prior:
            ep = 0.5 * self.B[species[i], species[j]] / r * cutoff_fn(r, self.rc)
            E_sr = E_sr + torch.zeros(nf).index_add(0, batch[i], ep)
        out = {'E_sr': E_sr}
        E = E_sr
        if self.use_lr:
            q = self.q_net(Xn).squeeze(-1)
            elr, off = [], 0
            for f in frames:
                qf, pf = q[off:off + f.n], pos[off:off + f.n]
                if f.cell is not None:
                    elr.append(ewald_reciprocal(qf, pf, f.cell, self.sigma, self.dl, self.ke))
                else:
                    elr.append(direct_lr(qf, pf, self.sigma, self.ke))
                off += f.n
            E_lr = torch.stack(elr)
            E = E + E_lr
            out['q'] = q
            out['E_lr'] = E_lr
        if self.baseline is not None:
            E = E + torch.stack([self.baseline(f, pos) for f in frames])
        out['E'] = E
        if need_forces:
            g, = torch.autograd.grad(E.sum(), pos, create_graph=self.training)
            out['F'] = -g
        out['pos'] = pos
        if single:
            out['E'] = E[0]
            out['E_sr'] = out['E_sr'][0]
            if 'E_lr' in out:
                out['E_lr'] = out['E_lr'][0]
        return out


# ----------------------------------------------------------------------------- training
def evaluate(model, frames):
    model.eval()
    Ep, Et, Fp, Ft, Q = [], [], [], [], []
    for f in frames:
        o = model(f)
        Ep.append(o['E'].item()); Et.append(f.energy)
        Fp.append(o['F'].detach()); Ft.append(f.forces)
        Q.append(o['q'].detach() if 'q' in o else None)
    n = np.array([f.n for f in frames])
    Ep, Et = np.array(Ep), np.array(Et)
    Fp, Ft = torch.cat(Fp).numpy(), torch.cat(Ft).numpy()
    res = dict(E_mae_atom=np.mean(np.abs(Ep - Et) / n), E_rmse_atom=np.sqrt(np.mean(((Ep - Et) / n) ** 2)),
               E_mae=np.mean(np.abs(Ep - Et)), F_mae=np.mean(np.abs(Fp - Ft)), F_rmse=np.sqrt(np.mean((Fp - Ft) ** 2)),
               Ep=Ep, Et=Et, Fp=Fp, Ft=Ft, q=Q)
    return res


def train(model, train_frames, val_frames=None, epochs=1000, lr=3e-3, wE=1.0, wF=1.0, batch=8,
          seed=0, verbose=100, log=None, fit_e0=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.set_normalization(train_frames)
    if fit_e0:  # least-squares per-species reference energies
        ns = model.ns
        A = np.array([np.bincount(f.species.numpy(), minlength=ns) for f in train_frames], float)
        y = np.array([f.energy for f in train_frames])
        e0 = np.linalg.lstsq(A, y, rcond=None)[0]
        with torch.no_grad():
            model.e0.copy_(torch.tensor(e0))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr * 0.02)
    hist = []
    for ep in range(epochs):
        model.train()
        idx = rng.permutation(len(train_frames))
        tot = 0.0
        for b in range(0, len(idx), batch):
            opt.zero_grad()
            fb = [train_frames[k] for k in idx[b:b + batch]]
            o = model(fb)
            Et = torch.tensor([f.energy for f in fb])
            nat = torch.tensor([float(f.n) for f in fb])
            Ft = torch.cat([f.forces for f in fb])
            lE = (((o['E'] - Et) / nat) ** 2).mean()
            lF = ((o['F'] - Ft) ** 2).mean()
            loss = wE * lE + wF * lF
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        if verbose and (ep % verbose == 0 or ep == epochs - 1):
            msg = f'ep {ep:5d} loss {tot:.4e}'
            if val_frames is not None:
                r = evaluate(model, val_frames)
                msg += f"  val E_mae/atom {r['E_mae_atom']*1e3:.3f} meV  F_mae {r['F_mae']*1e3:.2f} meV/A"
                hist.append((ep, tot, r['E_mae_atom'], r['F_mae']))
            print(msg, flush=True)
            if log is not None:
                print(msg, file=log, flush=True)
    return hist


# ----------------------------------------------------------------------------- derived quantities
def born_effective_charges(model, fr):
    """Z*_{i,ab} = dP_a/dr_{i,b},  P = sum_j q_j (r_j - r_ref) (relative MIC coordinates for periodic cells).
    Uses translational invariance (sum_j dq_j/dr_i = 0) so that the result is origin independent."""
    model.eval()
    pos = fr.pos.clone().requires_grad_(True)
    o = model(fr, pos, need_forces=False)
    q = o['q']
    n = fr.n
    Z = torch.zeros(n, 3, 3)
    J = torch.zeros(n, n, 3)  # J[j, i, b] = dq_j / dr_ib
    for jj in range(n):
        g, = torch.autograd.grad(q[jj], pos, retain_graph=True)
        J[jj] = g
    for ii in range(n):
        d = fr.pos - fr.pos[ii]
        if fr.cell is not None:
            d = d - fr.cell * torch.round(d / fr.cell)
        Z[ii] = q[ii].detach() * torch.eye(3) + torch.einsum('ja,jb->ab', d, J[:, ii, :])
    return Z.detach(), q.detach()
