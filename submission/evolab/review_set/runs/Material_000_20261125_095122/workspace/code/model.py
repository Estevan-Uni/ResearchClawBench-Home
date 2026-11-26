"""Pure-PyTorch gated-residual CGCNN encoder, masked-node decoder and classifier."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
N_ELEM, MASK = 28, 28

def featurize(gs):
    """Return list of (x, src, dst, bond_value, bond_type) with symmetrised, de-duplicated edges."""
    out = []
    for g in gs:
        ei = g['ei'].numpy(); ea = g['ea'].numpy()
        seen = set(); s_, d_, v_, t_ = [], [], [], []
        for (a, b), (v, t) in zip(ei.T, ea):
            for (p, q) in ((a, b), (b, a)):
                key = (p, q, round(float(v), 6), int(t))
                if key in seen or p == q: continue
                seen.add(key); s_.append(p); d_.append(q); v_.append(v); t_.append(int(t))
        out.append((g['x'].long(), torch.tensor(s_, dtype=torch.long), torch.tensor(d_, dtype=torch.long),
                    torch.tensor(v_, dtype=torch.float), torch.tensor(t_, dtype=torch.long)))
    return out

def collate(items):
    xs, ss, ds, vs, ts, bs = [], [], [], [], [], []; off = 0
    for k, (x, s, d, v, t) in enumerate(items):
        xs.append(x); ss.append(s + off); ds.append(d + off); vs.append(v); ts.append(t)
        bs.append(torch.full((len(x),), k, dtype=torch.long)); off += len(x)
    return dict(x=torch.cat(xs), src=torch.cat(ss), dst=torch.cat(ds), v=torch.cat(vs), t=torch.cat(ts),
                batch=torch.cat(bs), n_graphs=len(items))

class RBF(nn.Module):
    def __init__(self, n=16, lo=0., hi=1.):
        super().__init__(); self.register_buffer('mu', torch.linspace(lo, hi, n)); self.gamma = 1.0 / ((hi - lo) / (n - 1)) ** 2
    def forward(self, v): return torch.exp(-self.gamma * (v[:, None] - self.mu) ** 2)

class GatedCGConv(nn.Module):
    """m_ij = sigmoid(W_f z_ij) * softplus(W_s z_ij), z_ij=[h_i,h_j,e_ij];  h_i <- act(h_i + LN(sum_j m_ij))."""
    def __init__(self, h, e, residual=True):
        super().__init__(); self.lin = nn.Linear(2 * h + e, 2 * h); self.ln = nn.LayerNorm(h); self.residual = residual
    def forward(self, h, src, dst, e):
        z = torch.cat([h[dst], h[src], e], 1)            # message from src (j) to dst (i)
        gate, core = self.lin(z).chunk(2, 1)
        m = torch.sigmoid(gate) * F.softplus(core)
        agg = torch.zeros_like(h).index_add_(0, dst, m)
        return F.softplus(h + self.ln(agg)) if self.residual else F.softplus(self.ln(agg))

class Encoder(nn.Module):
    def __init__(self, h=64, e=32, layers=3, residual=True, n_rbf=16):
        super().__init__()
        self.emb = nn.Embedding(N_ELEM + 1, h); self.rbf = RBF(n_rbf); self.temb = nn.Embedding(3, 8)
        self.edge_mlp = nn.Sequential(nn.Linear(n_rbf + 8, e), nn.Softplus(), nn.Linear(e, e))
        self.convs = nn.ModuleList([GatedCGConv(h, e, residual) for _ in range(layers)]); self.h = h
    def forward(self, b, x=None):
        x = b['x'] if x is None else x
        h = self.emb(x); e = self.edge_mlp(torch.cat([self.rbf(b['v']), self.temb(b['t'])], 1))
        for c in self.convs: h = c(h, b['src'], b['dst'], e)
        return h

def pool(h, b, mode='meanmax'):
    n = b['n_graphs']; cnt = torch.zeros(n).index_add_(0, b['batch'], torch.ones(len(h))).clamp(min=1)
    mean = torch.zeros(n, h.shape[1]).index_add_(0, b['batch'], h) / cnt[:, None]
    if mode == 'mean': return mean
    mx = torch.full((n, h.shape[1]), -1e9).scatter_reduce(0, b['batch'][:, None].expand_as(h), h, 'amax')
    return torch.cat([mean, mx], 1)

class MaskedNodeModel(nn.Module):
    """Pre-training: replace 15% of atoms by MASK token, reconstruct element one-hot with an MLP decoder (CE)."""
    def __init__(self, enc):
        super().__init__(); self.enc = enc; h = enc.h
        self.dec = nn.Sequential(nn.Linear(h, h), nn.Softplus(), nn.Linear(h, N_ELEM))
    def forward(self, b, rate=0.15, gen=None):
        n = len(b['x']); m = torch.rand(n, generator=gen) < rate
        if m.sum() == 0: m[torch.randint(0, n, (1,), generator=gen)] = True
        xm = b['x'].clone(); xm[m] = MASK
        logits = self.dec(self.enc(b, xm)[m])
        return F.cross_entropy(logits, b['x'][m]), (logits.argmax(1) == b['x'][m]).float().mean(), m, logits

class Classifier(nn.Module):
    """Encoder -> global mean||max pooling -> FC -> ReLU -> Dropout -> FC -> logit."""
    def __init__(self, enc, hidden=64, dropout=0.3, pool_mode='meanmax'):
        super().__init__(); self.enc = enc; self.pool_mode = pool_mode
        d = enc.h * (2 if pool_mode == 'meanmax' else 1)
        self.head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
    def embed(self, b): return pool(self.enc(b), b, self.pool_mode)
    def forward(self, b): return self.head(self.embed(b)).squeeze(1)
