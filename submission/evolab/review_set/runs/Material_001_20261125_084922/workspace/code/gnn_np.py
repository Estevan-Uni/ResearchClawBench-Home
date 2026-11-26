"""Numpy CGCNN-style message-passing network with manual backprop.

Graphs have a fixed number of nodes N (here 5, fully connected = K5 template).
Inputs: X (B,N,F) node features, E (B,N,N,K) edge features (Gaussian RBF of distance),
mask (N,N) adjacency without self loops.
Conv (Xie & Grossman 2018, Eq. 5 without batch-norm):
    z_ij = h_i Wa + h_j Wb + e_ij We + b ;  m_ij = sigmoid(z^f_ij) * softplus(z^s_ij)
    h_i <- softplus(h_i + sum_j A_ij m_ij)
Readout: mean pool -> softplus(W1) -> W2 -> scalar.
"""
import numpy as np

sig = lambda x: 0.5 * (1 + np.tanh(0.5 * x))
def softplus(x):
    return np.logaddexp(0, x)


def init_params(F, K, H=32, L=3, Hfc=32, seed=0):
    rng = np.random.default_rng(seed)
    g = lambda i, o: rng.normal(0, np.sqrt(1.0 / i), size=(i, o))
    p = {'Wemb': g(F, H), 'bemb': np.zeros(H)}
    for l in range(L):
        p[f'Wa{l}'] = g(H, 2 * H) * 0.5; p[f'Wb{l}'] = g(H, 2 * H) * 0.5
        p[f'We{l}'] = g(K, 2 * H) * 0.5 if K > 0 else np.zeros((0, 2 * H)); p[f'bc{l}'] = np.zeros(2 * H)
    p['W1'] = g(H, Hfc); p['b1'] = np.zeros(Hfc); p['W2'] = g(Hfc, 1) * 0.1; p['b2'] = np.zeros(1)
    return p


def n_layers(p):
    return sum(1 for k in p if k.startswith('Wa'))


def forward(p, X, E, mask, cache=False):
    L = n_layers(p); H = p['bemb'].size
    h = X @ p['Wemb'] + p['bemb']
    cs = []
    for l in range(L):
        pre = (h @ p[f'Wa{l}'])[:, :, None, :] + (h @ p[f'Wb{l}'])[:, None, :, :] + p[f'bc{l}']
        if E.shape[-1] > 0:
            pre = pre + E @ p[f'We{l}']
        f = sig(pre[..., :H]); s = softplus(pre[..., H:])
        m = f * s * mask[None, :, :, None]
        u = h + m.sum(2)
        cs.append((h, pre, f, s, u))
        h = softplus(u)
    g = h.mean(1)                       # (B,H) crystal/cluster embedding
    a1 = g @ p['W1'] + p['b1']; z1 = softplus(a1)
    out = (z1 @ p['W2'] + p['b2'])[:, 0]
    if cache:
        return out, dict(cs=cs, h=h, g=g, a1=a1, z1=z1, X=X, E=E, mask=mask)
    return out


def backward(p, c, dout):
    L = n_layers(p); H = p['bemb'].size; gr = {}
    dz1 = dout[:, None] @ p['W2'].T
    gr['W2'] = c['z1'].T @ dout[:, None]; gr['b2'] = np.array([dout.sum()])
    da1 = dz1 * sig(c['a1'])
    gr['W1'] = c['g'].T @ da1; gr['b1'] = da1.sum(0)
    dg = da1 @ p['W1'].T
    N = c['h'].shape[1]
    dh = np.repeat(dg[:, None, :] / N, N, axis=1)
    E, mask = c['E'], c['mask']
    for l in reversed(range(L)):
        h, pre, f, s, u = c['cs'][l]
        du = dh * sig(u)
        dhp = du.copy()
        dm = du[:, :, None, :] * mask[None, :, :, None]
        dpre = np.concatenate([dm * s * f * (1 - f), dm * f * sig(pre[..., H:])], axis=-1)
        Si, Sj = dpre.sum(2), dpre.sum(1)
        gr[f'Wa{l}'] = h.reshape(-1, H).T @ Si.reshape(-1, 2 * H)
        gr[f'Wb{l}'] = h.reshape(-1, H).T @ Sj.reshape(-1, 2 * H)
        gr[f'We{l}'] = E.reshape(-1, E.shape[-1]).T @ dpre.reshape(-1, 2 * H) if E.shape[-1] > 0 else np.zeros_like(p[f'We{l}'])
        gr[f'bc{l}'] = dpre.sum((0, 1, 2))
        dh = dhp + Si @ p[f'Wa{l}'].T + Sj @ p[f'Wb{l}'].T
    gr['Wemb'] = c['X'].reshape(-1, c['X'].shape[-1]).T @ dh.reshape(-1, H); gr['bemb'] = dh.sum((0, 1))
    return gr


def mse_loss_grad(p, X, E, mask, y):
    out, c = forward(p, X, E, mask, cache=True)
    r = out - y; loss = np.mean(r ** 2)
    return loss, backward(p, c, 2 * r / len(y))


def embed(p, X, E, mask):
    _, c = forward(p, X, E, mask, cache=True)
    return c['g']


def train(p, data, epochs=300, bs=64, lr=3e-3, wd=0.0, seed=0, verbose=50):
    """data: dict with Xtr,Etr,ytr,Xva,Eva,yva,mask. Returns best params and history (MSE in standardised units)."""
    from common import Adam
    rng = np.random.default_rng(seed)
    opt = Adam(p, lr=lr, wd=wd)
    hist = {'train': [], 'val': []}; best = (np.inf, None, 0)
    ntr = len(data['ytr'])
    for ep in range(epochs):
        opt.lr = lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))  # cosine schedule
        perm = rng.permutation(ntr)
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            _, gr = mse_loss_grad(p, data['Xtr'][idx], data['Etr'][idx], data['mask'], data['ytr'][idx])
            opt.step(gr)
        sub = slice(0, min(ntr, 1000))  # train loss on a fixed 1000-graph subset (speed)
        tr = np.mean((forward(p, data['Xtr'][sub], data['Etr'][sub], data['mask']) - data['ytr'][sub]) ** 2)
        va = np.mean((forward(p, data['Xva'], data['Eva'], data['mask']) - data['yva']) ** 2)
        hist['train'].append(tr); hist['val'].append(va)
        if va < best[0]:
            best = (va, {k: v.copy() for k, v in p.items()}, ep)
        if verbose and (ep % verbose == 0 or ep == epochs - 1):
            print(f'  ep {ep:4d}  train {tr:.4f}  val {va:.4f}', flush=True)
    return best[1], hist, best[2]
