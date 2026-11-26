"""Helpers: load trained dual VAE, encode/decode monomers, torch GP surrogate in latent space."""
import sys, pickle, numpy as np, torch, math
sys.path.insert(0, 'code')
from vae_lib import *

torch.set_num_threads(2)


def load(tag='b01'):
    D = pickle.load(open('outputs/vae_data.pkl', 'rb'))
    m = DualVAE(len(D['va'].itos), len(D['ve'].itos))
    m.load_state_dict(torch.load(f'outputs/vae_{tag}.pt')); m.eval()
    return m, D


@torch.no_grad()
def encode(model, smiles, kind, bs=512):
    enc = model.enc_a if kind == 'acid' else model.enc_e
    mus, lvs = [], []
    for b in range(0, len(smiles), bs):
        g = batch_graphs([mol_graph(s) for s in smiles[b:b + bs]])
        mu, lv = enc(*g); mus.append(mu); lvs.append(lv)
    return torch.cat(mus).numpy(), torch.cat(lvs).numpy()


@torch.no_grad()
def decode(model, D, z, kind, temp=0.0):
    dec = model.dec_a if kind == 'acid' else model.dec_e
    voc = D['va'] if kind == 'acid' else D['ve']
    toks = dec.generate(torch.as_tensor(z, dtype=torch.float32), temp=temp)
    return [selfies_to_smiles(voc.decode(t)) for t in toks]


@torch.no_grad()
def head(model, Z):
    return model.head(torch.as_tensor(Z, dtype=torch.float32))[:, 0].numpy() * 33.7 + 424.0


class TorchGP:
    """Exact GP, isotropic-per-block RBF (separate length-scales for z_acid / z_epox) + const mean."""

    def __init__(self, X, y, iters=80, init=None):
        self.X = torch.as_tensor(X, dtype=torch.float64); self.yraw = torch.as_tensor(y, dtype=torch.float64)
        self.ym = self.yraw.mean(); self.ys = self.yraw.std()
        self.y = (self.yraw - self.ym) / self.ys
        p = torch.tensor(init if init is not None else [math.log(4.0), math.log(4.0), 0.0, math.log(0.1)], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([p], lr=0.08)
        for _ in range(iters):
            opt.zero_grad(); l = self._nlml(p); l.backward(); opt.step()
            with torch.no_grad():
                p[3].clamp_(math.log(1e-3), math.log(1.0)); p[0].clamp_(math.log(0.5), math.log(50)); p[1].clamp_(math.log(0.5), math.log(50)); p[2].clamp_(math.log(0.05), math.log(20.0))
        self.p = p.detach(); self._fit()

    def _k(self, A, B, p):
        d = A.shape[1] // 2
        la, le = torch.exp(p[0]), torch.exp(p[1])
        da = torch.cdist(A[:, :d] / la, B[:, :d] / la) ** 2; de = torch.cdist(A[:, d:] / le, B[:, d:] / le) ** 2
        return torch.exp(p[2]) * torch.exp(-0.5 * (da + de))

    def _nlml(self, p):
        K = self._k(self.X, self.X, p) + (torch.exp(p[3]) + 1e-6) * torch.eye(len(self.X), dtype=torch.float64)
        L = torch.linalg.cholesky(K)
        a = torch.cholesky_solve(self.y[:, None], L)
        return 0.5 * (self.y[:, None] * a).sum() + torch.log(torch.diag(L)).sum()

    def _fit(self):
        K = self._k(self.X, self.X, self.p) + (torch.exp(self.p[3]) + 1e-6) * torch.eye(len(self.X), dtype=torch.float64)
        self.L = torch.linalg.cholesky(K)
        self.alpha = torch.cholesky_solve(self.y[:, None], self.L)

    def predict_t(self, Z):
        """Differentiable torch posterior (mean, std) in original units."""
        Ks = self._k(Z, self.X, self.p)
        mu = (Ks @ self.alpha)[:, 0]
        v = torch.cholesky_solve(Ks.T, self.L)
        var = torch.clamp(torch.exp(self.p[2]) - (Ks * v.T).sum(1), min=1e-9)
        return self.ym + self.ys * mu, self.ys * torch.sqrt(var)

    def predict(self, Z, bs=20000, noise=False):
        mus, sds = [], []
        with torch.no_grad():
            for b in range(0, len(Z), bs):
                m, s = self.predict_t(torch.as_tensor(Z[b:b + bs], dtype=torch.float64))
                if noise:
                    s = torch.sqrt(s ** 2 + (self.ys ** 2) * torch.exp(self.p[3]))
                mus.append(m.numpy()); sds.append(s.numpy())
        return np.concatenate(mus), np.concatenate(sds)
