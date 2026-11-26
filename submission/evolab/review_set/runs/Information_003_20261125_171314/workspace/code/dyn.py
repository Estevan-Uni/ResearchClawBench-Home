"""Dynamic (TGN-style) intrusion detection models: TGN baseline, 3D-IDS, DIDS-MFL + ablations.

Stream processing (chronological, batch size bs). For batch b:
  1. messages of batch b-1 update memory (GRU) *with gradient* inside a local memory view
  2. event embedding from three scales:
       z_e  edge / statistical scale  (MLP on [F || h_SD] + dt + layer marks)
       z_m  memory scale              (projection of src/dst memory)
       z_d  diffusion scale           (Perona-Malik diffusion, RK4, on the star graph of the
                                       K most recent temporal neighbours; TGN: temporal attention)
  3. fusion: MFL = scale-attention + multi-scale transform (residual MLP on concatenated scales);
     non-MFL = MLP on concatenation
  4. heads: two-step (binary + attack type, Eq.18) and, for MFL, bi-similarity prototype heads
     (cosine + learned relation) trained on class-balanced episodes from a per-class buffer.
Losses: L_int + a*L_smooth + b*L_RD (decorrelation of memory-scale rep) + g*L_dis (decorrelation of fused z)
        + l*L_episode.
Protocols: known (10 classes), fewshot (train labels only on base classes; novel classes evaluated
episodically), unknown (one attack type removed from train+val).
"""
import sys, os, time, argparse, json, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, '/workspace/code')
from common import *
from fewshot_eval import evaluate as fs_evaluate, head_euclid, head_cosine

VARIANTS = {
    'dids_mfl':   dict(sd='cat', rd=0.1, emb='pm', mfl=True, mst=True, dis=0.1, ep=1.0, sim='bi'),
    '3dids':      dict(sd='cat', rd=0.1, emb='pm', mfl=False, mst=False, dis=0.0, ep=0.0, sim='none'),
    'tgn':        dict(sd='none', rd=0.0, emb='attn', mfl=False, mst=False, dis=0.0, ep=0.0, sim='none'),
    'mfl_noSD':   dict(sd='none', rd=0.1, emb='pm', mfl=True, mst=True, dis=0.1, ep=1.0, sim='bi'),
    'mfl_hrep':   dict(sd='rep', rd=0.1, emb='pm', mfl=True, mst=True, dis=0.1, ep=1.0, sim='bi'),
    'mfl_noRD':   dict(sd='cat', rd=0.0, emb='pm', mfl=True, mst=True, dis=0.1, ep=1.0, sim='bi'),
    'mfl_noDiff': dict(sd='cat', rd=0.1, emb='none', mfl=True, mst=True, dis=0.1, ep=1.0, sim='bi'),
    'mfl_attn':   dict(sd='cat', rd=0.1, emb='attn', mfl=True, mst=True, dis=0.1, ep=1.0, sim='bi'),
    'mfl_noMST':  dict(sd='cat', rd=0.1, emb='pm', mfl=True, mst=False, dis=0.1, ep=1.0, sim='bi'),
    'mfl_noDis':  dict(sd='cat', rd=0.1, emb='pm', mfl=True, mst=True, dis=0.0, ep=1.0, sim='bi'),
    'mfl_cos':    dict(sd='cat', rd=0.1, emb='pm', mfl=True, mst=True, dis=0.1, ep=1.0, sim='cos'),
    'mfl_noEp':   dict(sd='cat', rd=0.1, emb='pm', mfl=True, mst=True, dis=0.1, ep=0.0, sim='cos'),
    'mfl_noRDDis': dict(sd='cat', rd=0.0, emb='pm', mfl=True, mst=True, dis=0.0, ep=1.0, sim='bi'),
    'mfl_rd1':    dict(sd='cat', rd=1.0, emb='pm', mfl=True, mst=True, dis=1.0, ep=1.0, sim='bi'),
}


# ------------------------------------------------------------------ data
def prepare(protocol, holdout, bs, K):
    D = load(); y_all = D['attack']; n_all = len(y_all)
    tr, va, te = chrono_split(n_all)
    period = np.zeros(n_all, np.int64); period[va] = 1; period[te] = 2
    keep = np.ones(n_all, bool)
    if protocol == 'unknown':
        keep &= ~((y_all == holdout) & (period < 2))
    gid = np.where(keep)[0]
    H = np.load(f'{OUT}/sd_h.npy')
    fit_rows = np.where(keep & (period == 0))[0]
    Fq = quantile_fit(D['msg'], fit_rows); Hq = quantile_fit(H, fit_rows)
    E = dict(gid=gid, src=D['src_i'][gid], dst=D['dst_i'][gid], t=D['t'][gid].astype(np.float32),
             y=y_all[gid], period=period[gid], ls=D['layer_src'][gid].astype(np.float32),
             ld=D['layer_dst'][gid].astype(np.float32), dt=D['dt'][gid].astype(np.float32),
             Fq=Fq[gid], Hq=Hq[gid], n_nodes=D['n_nodes'])
    if protocol == 'known':
        att = [c for c in range(10) if c != BENIGN]; lab_cls = list(range(10))
    elif protocol == 'fewshot':
        att = [c for c in BASE if c != BENIGN]; lab_cls = BASE
    else:
        att = [c for c in range(10) if c not in (BENIGN, holdout)]; lab_cls = [c for c in range(10) if c != holdout]
    E['att_classes'] = att; E['lab_classes'] = lab_cls
    E['labeled'] = np.isin(E['y'], lab_cls)
    # neighbours from strictly earlier batches + previous-seen time per endpoint
    n = len(gid); src, dst, t = E['src'], E['dst'], E['t']
    nb = -np.ones((n, 2, K), np.int64); prev_t = np.zeros((n, 2), np.float32)
    hist = {}; last = {}
    for b in range(0, n, bs):
        rng_ = range(b, min(b + bs, n))
        for i in rng_:
            for ep, v in ((0, src[i]), (1, dst[i])):
                h = hist.get(v)
                if h:
                    lst = h[-K:][::-1]; nb[i, ep, :len(lst)] = lst
                prev_t[i, ep] = t[i] - last.get(v, t[i])
                last[v] = t[i]
        for i in rng_:
            hist.setdefault(src[i], []).append(i); hist.setdefault(dst[i], []).append(i)
            if len(hist[src[i]]) > 4 * K: hist[src[i]] = hist[src[i]][-K:]
            if len(hist[dst[i]]) > 4 * K: hist[dst[i]] = hist[dst[i]][-K:]
    other = np.where(nb >= 0, 0, -1)
    nbv = np.where(nb >= 0, nb, 0)
    node_self = np.stack([src, dst], 1)[:, :, None]
    nb_node = np.where(src[nbv] == node_self, dst[nbv], src[nbv])
    E['nb'] = nb; E['nb_node'] = np.where(nb >= 0, nb_node, 0); E['prev_t'] = prev_t
    return E


# ------------------------------------------------------------------ model
class TimeEnc(nn.Module):
    def __init__(s, d=16):
        super().__init__()
        s.w = nn.Parameter(torch.from_numpy(1 / 10 ** np.linspace(0, 5, d)).float())
        s.b = nn.Parameter(torch.zeros(d))
    def forward(s, dt):
        return torch.cos(dt[..., None] * s.w + s.b)


def mlp(i, o, h=None):
    h = h or max(o, 64)
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class PMDiffusion(nn.Module):
    """Perona-Malik graph diffusion on the star graph (centre + K temporal neighbours):
    dX/dt = -M^T sigma(M X K^T) S (M X K^T) K, sigma(x)=exp(-|x|), solved with RK4."""
    def __init__(s, d, dk=32, steps=2, h=0.5):
        super().__init__()
        s.K = nn.Parameter(torch.randn(dk, d) * (1.0 / np.sqrt(d)))
        s.steps, s.h = steps, h
    def f(s, X, S, mask):
        # X: [B, K+1, d]; edge k connects centre(0) and leaf k
        g = (X[:, 1:] - X[:, :1]) @ s.K.T                    # M X K^T   [B,K,dk]
        flux = torch.exp(-g.abs()) * g * (S * mask)[..., None]
        fl = flux @ s.K                                       # [B,K,d]
        dc = fl.sum(1, keepdim=True)                          # -M^T: centre gets +sum
        return torch.cat([dc, -fl], 1)
    def forward(s, X, S, mask):
        for _ in range(s.steps):
            k1 = s.f(X, S, mask); k2 = s.f(X + 0.5 * s.h * k1, S, mask)
            k3 = s.f(X + 0.5 * s.h * k2, S, mask); k4 = s.f(X + s.h * k3, S, mask)
            X = X + s.h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return X[:, 0]


class Net(nn.Module):
    def __init__(s, din, n_att, cfg, dm=64, dz=64, dtim=16):
        super().__init__()
        s.cfg = cfg; s.dm, s.dz = dm, dz
        s.time = TimeEnc(dtim)
        s.edge = mlp(din + 3, dz)
        s.msg = mlp(2 * dm + dtim + 2 + dz, dm)
        s.gru = nn.GRUCell(dm, dm)
        s.mem_proj = mlp(2 * dm, dz)
        emb = cfg['emb']
        if emb != 'none':
            s.leaf = nn.Linear(dm + dz, dm); s.centre = nn.Linear(dm, dm)
            if emb == 'pm':
                s.coef = mlp(2 + dtim, 1, 32); s.diff = PMDiffusion(dm)
            else:
                s.q = nn.Linear(dm + dtim, dm); s.k = nn.Linear(dm + dtim, dm); s.v = nn.Linear(dm + dtim, dm)
                s.att_out = mlp(2 * dm, dm)
            s.diff_proj = mlp(2 * dm, dz)
        ns = 3 if emb != 'none' else 2
        if cfg['mfl']:
            s.sc_W = nn.Linear(dz, 32); s.sc_v = nn.Linear(32, 1, bias=False)
            if cfg['mst']:
                s.mst = mlp(ns * dz, dz)
        else:
            s.comb = mlp(ns * dz, dz)
        s.bin = nn.Linear(dz, 1); s.cls = nn.Linear(dz, n_att)
        if cfg['sim'] == 'bi':
            s.rel = nn.Sequential(nn.Linear(2 * dz, 64), nn.ReLU(), nn.Linear(64, 1))
        s.tau = nn.Parameter(torch.tensor(10.0))

    def relation(s, zq, P):  # zq [n,d], P [C,d] -> [n,C] in (0,1)
        n, C = zq.shape[0], P.shape[0]
        pair = torch.cat([zq[:, None].expand(n, C, -1), P[None].expand(n, C, -1)], -1)
        return torch.sigmoid(s.rel(pair).squeeze(-1))

    def cos_logits(s, zq, P):
        return s.tau * Fn.normalize(zq, dim=-1) @ Fn.normalize(P, dim=-1).T


class Stream:
    """Holds memory state and runs the model over the event stream."""
    def __init__(s, E, net, x_e, bs):
        s.E, s.net, s.bs = E, net, bs
        s.x = torch.from_numpy(x_e); s.t = torch.from_numpy(E['t'])
        s.ls = torch.from_numpy(E['ls']); s.ld = torch.from_numpy(E['ld'])
        s.src = torch.from_numpy(E['src']); s.dst = torch.from_numpy(E['dst'])
        s.nb = torch.from_numpy(E['nb']); s.nbn = torch.from_numpy(E['nb_node'])
        s.prev_t = torch.from_numpy(E['prev_t']); s.E_dt = torch.from_numpy(E['dt'])
        s.reset()

    def reset(s):
        s.M = torch.zeros(s.E['n_nodes'], s.net.dm); s.pending = None

    def step(s, idx):
        net, cfg = s.net, s.net.cfg
        src, dst = s.src[idx], s.dst[idx]
        nbv = s.nb[idx]; mask = (nbv >= 0).float(); nbe = nbv.clamp(min=0); nbn = s.nbn[idx]
        parts = [src, dst, nbn.reshape(-1)]
        P = s.pending
        if P is not None:
            parts += [s.src[P], s.dst[P]]
        nodes, inv = torch.unique(torch.cat(parts), return_inverse=True)
        mem = s.M[nodes]
        off = [0]
        for p in parts: off.append(off[-1] + len(p))
        pos = lambda k: inv[off[k]:off[k + 1]]
        upd = None
        if P is not None:
            ps, pd = pos(3), pos(4)
            ee = net.edge(s._edge_in(P))
            lsP, ldP = s.ls[P], s.ld[P]
            ms, md = mem[ps], mem[pd]
            c_s = net.msg(torch.cat([ms, md, net.time(s.prev_t[P, 0]), lsP[:, None], ldP[:, None], ee], 1))
            c_d = net.msg(torch.cat([md, ms, net.time(s.prev_t[P, 1]), ldP[:, None], lsP[:, None], ee], 1))
            un, uinv = torch.unique(torch.cat([ps, pd]), return_inverse=True)
            agg = torch.zeros(len(un), net.dm).index_add_(0, uinv, torch.cat([c_s, c_d]))
            cnt = torch.zeros(len(un)).index_add_(0, uinv, torch.ones(len(uinv)))
            old = mem[un]
            new = net.gru(agg / cnt[:, None], old)
            mem = mem.index_put((un,), new)
            upd = (nodes[un], new, old)
        # ---- event embeddings
        z_e = net.edge(s._edge_in(idx))
        m_s, m_d = mem[pos(0)], mem[pos(1)]
        z_m = net.mem_proj(torch.cat([m_s, m_d], 1))
        scales = [z_e, z_m]
        if cfg['emb'] != 'none':
            B, Kn = nbv.shape[0], nbv.shape[2]
            mnb = mem[pos(2)].reshape(B, 2, Kn, -1)
            ee_nb = net.edge(s._edge_in(nbe.reshape(-1))).reshape(B, 2, Kn, -1)
            leaf = net.leaf(torch.cat([mnb, ee_nb], -1))
            dtn = (s.t[idx][:, None, None] - s.t[nbe]) * mask
            outs = []
            for ep, mc in ((0, m_s), (1, m_d)):
                cen = net.centre(mc)
                tdt = net.time(dtn[:, ep])
                if cfg['emb'] == 'pm':
                    lsrc = (s.ls if ep == 0 else s.ld)[idx][:, None].expand(-1, Kn)
                    lnb = torch.where(s.src[nbe[:, ep]] == (src if ep == 0 else dst)[:, None],
                                      s.ls[nbe[:, ep]], s.ld[nbe[:, ep]])
                    S = torch.sigmoid(net.coef(torch.cat([lsrc[..., None], lnb[..., None], tdt], -1))).squeeze(-1)
                    X = torch.cat([cen[:, None], leaf[:, ep]], 1)
                    outs.append(net.diff(X, S, mask[:, ep]))
                else:
                    q = net.q(torch.cat([mc, net.time(torch.zeros(len(idx)))], 1))
                    kv_in = torch.cat([leaf[:, ep], tdt], -1)
                    k, v = net.k(kv_in), net.v(kv_in)
                    a = (k @ q[:, :, None]).squeeze(-1) / np.sqrt(net.dm)
                    a = a.masked_fill(mask[:, ep] == 0, -1e9)
                    w = torch.softmax(a, 1) * (mask[:, ep].sum(1, keepdim=True) > 0)
                    outs.append(net.att_out(torch.cat([cen, (w[..., None] * v).sum(1)], 1)))
            z_d = net.diff_proj(torch.cat(outs, 1))
            scales.append(z_d)
        Zs = torch.stack(scales, 1)
        if cfg['mfl']:
            a = torch.softmax(net.sc_v(torch.tanh(net.sc_W(Zs))).squeeze(-1), 1)
            z = (a[..., None] * Zs).sum(1)
            if cfg['mst']:
                z = z + net.mst(torch.cat(scales, 1))
        else:
            a = None; z = net.comb(torch.cat(scales, 1))
        return dict(z=z, z_m=z_m, bin=net.bin(z).squeeze(-1), cls=net.cls(z), att=a, upd=upd)

    def _edge_in(s, idx):
        return torch.cat([s.x[idx], s.E_dt[idx][:, None], s.ls[idx][:, None], s.ld[idx][:, None]], 1)

    def commit(s, out, idx):
        if out['upd'] is not None:
            nodes, new, _ = out['upd']; s.M[nodes] = new.detach()
        s.pending = idx


def decor(Z):
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-5)
    C = Z.T @ Z / len(Z)
    return ((C - torch.eye(Z.shape[1])) ** 2).sum() / Z.shape[1] ** 2


def offdiag_abs_corr(Z):
    Z = np.asarray(Z, np.float64); Z = Z[:, Z.std(0) > 1e-8]
    C = np.corrcoef(Z.T); return float(np.abs(C[~np.eye(len(C), dtype=bool)]).mean())


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', default='known'); ap.add_argument('--holdout', type=int, default=-1)
    ap.add_argument('--variant', default='dids_mfl'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--bs', type=int, default=200); ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--tag', default='')
    ap.add_argument('--max_train_events', type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(2); torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = VARIANTS[args.variant]
    name = f'{args.protocol}{"_h" + str(args.holdout) if args.protocol == "unknown" else ""}_{args.variant}_s{args.seed}{args.tag}'
    out_dir = f'{OUT}/runs'; os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(f'{out_dir}/{name}.json') or os.path.exists(f'{out_dir}/{name}.lock'):
        print('exists', name); return
    open(f'{out_dir}/{name}.lock', 'w').close()
    t0 = time.time()
    E = prepare(args.protocol, args.holdout, args.bs, args.K)
    x_e = {'cat': np.c_[E['Fq'], E['Hq']], 'none': E['Fq'], 'rep': E['Hq']}[cfg['sd']].astype(np.float32)
    att_cls = E['att_classes']; lab_cls = E['lab_classes']
    amap = -np.ones(10, np.int64); amap[att_cls] = np.arange(len(att_cls))
    y = E['y']; period = E['period']; labeled = E['labeled']
    tr_ev = np.where(period == 0)[0]; va_ev = np.where(period == 1)[0]
    if args.max_train_events:
        tr_ev = tr_ev[:args.max_train_events]
    # class weights (inverse sqrt frequency) over attack classes
    cnt = np.bincount(y[tr_ev][labeled[tr_ev]], minlength=10)
    cw = np.array([cnt[c] ** -0.5 if cnt[c] > 0 else 0 for c in att_cls]); cw = cw / cw.mean()
    cw_t = torch.tensor(cw, dtype=torch.float32)
    net = Net(x_e.shape[1], len(att_cls), cfg)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    st = Stream(E, net, x_e, args.bs)
    yt = torch.from_numpy(y); amap_t = torch.from_numpy(amap); lab_t = torch.from_numpy(labeled)
    ALPHA = 1e-3
    buf = {c: [] for c in lab_cls}; BUF = 64
    rng = np.random.default_rng(args.seed)

    def predict_probs(Z, protos):
        """Z: [n,d] tensor; returns p(class over 10) numpy"""
        with torch.no_grad():
            p_att = torch.sigmoid(net.bin(Z)).squeeze(-1); q = torch.softmax(net.cls(Z), 1)
            if cfg['mfl'] and cfg['ep'] > 0 and protos is not None:
                P = protos[att_cls]
                qb = torch.softmax(net.cos_logits(Z, P), 1)
                if cfg['sim'] == 'bi':
                    r = net.relation(Z, P); r = r / (r.sum(1, keepdim=True) + 1e-9)
                    qb = 0.5 * qb + 0.5 * r
                q = 0.5 * q + 0.5 * qb
            pr = torch.zeros(len(Z), 10)
            pr[:, BENIGN] = 1 - p_att
            pr[:, att_cls] = (p_att.reshape(-1, 1)) * q
        return pr.numpy()

    def run_eval(events, collect_train_protos=False):
        """no-grad pass over events (continuing current stream state)."""
        Zs, Zm, A = [], [], []
        with torch.no_grad():
            for b in range(0, len(events), args.bs):
                idx = torch.from_numpy(events[b:b + args.bs])
                o = st.step(idx); st.commit(o, idx)
                Zs.append(o['z']); Zm.append(o['z_m'])
                if o['att'] is not None: A.append(o['att'])
        return torch.cat(Zs), torch.cat(Zm), (torch.cat(A) if A else None)

    def protos_from(Z, events):
        P = torch.zeros(10, net.dz)
        for c in lab_cls:
            m = (y[events] == c) & labeled[events]
            if m.any(): P[c] = Z[torch.from_numpy(np.where(m)[0])].mean(0)
        return P

    hist = []; best = (-1, None, -1); bad = 0; ep_times = []
    for epoch in range(args.epochs):
        net.train(); st.reset(); te0 = time.time()
        tot = {'int': 0, 'rd': 0, 'dis': 0, 'ep': 0, 'n': 0}
        Ztr = []
        for b in range(0, len(tr_ev), args.bs):
            idx_np = tr_ev[b:b + args.bs]; idx = torch.from_numpy(idx_np)
            o = st.step(idx)
            yb = yt[idx]; lm = lab_t[idx]
            is_att = (yb != BENIGN).float()
            l_bin = Fn.binary_cross_entropy_with_logits(o['bin'][lm], is_att[lm]) if lm.any() else o['bin'].sum() * 0
            am = lm & (yb != BENIGN)
            l_cls = Fn.cross_entropy(o['cls'][am], amap_t[yb[am]], weight=cw_t) if am.any() else 0 * o['cls'].sum()
            loss = l_bin + l_cls
            if o['upd'] is not None:
                loss = loss + ALPHA * ((o['upd'][1] - o['upd'][2]) ** 2).sum(1).mean()
            if cfg['rd'] > 0:
                l_rd = decor(o['z_m']); loss = loss + cfg['rd'] * l_rd; tot['rd'] += l_rd.item()
            if cfg['dis'] > 0:
                l_dis = decor(o['z']); loss = loss + cfg['dis'] * l_dis; tot['dis'] += l_dis.item()
            # ---- class-balanced episode from per-class buffer
            if cfg['ep'] > 0:
                cls_ok = [c for c in lab_cls if len(buf[c]) >= 5 and ((yb == c) & lm).any()]
                if len(cls_ok) >= 2:
                    P = torch.stack([torch.stack([buf[c][i] for i in rng.choice(len(buf[c]), 5, replace=False)]).mean(0) for c in cls_ok])
                    qi, ql = [], []
                    for j, c in enumerate(cls_ok):
                        w = torch.where((yb == c) & lm)[0]
                        if len(w) > 16: w = w[torch.from_numpy(rng.choice(len(w), 16, replace=False))]
                        qi.append(w); ql.append(torch.full((len(w),), j))
                    qi = torch.cat(qi); ql = torch.cat(ql); zq = o['z'][qi]
                    l_ep = Fn.cross_entropy(net.cos_logits(zq, P), ql)
                    if cfg['sim'] == 'bi':
                        r = net.relation(zq, P)
                        l_ep = l_ep + Fn.binary_cross_entropy(r, Fn.one_hot(ql, len(cls_ok)).float())
                    loss = loss + cfg['ep'] * l_ep; tot['ep'] += l_ep.item()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            st.commit(o, idx)
            zd = o['z'].detach(); Ztr.append(zd)
            for c in lab_cls:
                w = torch.where((yb == c) & lm)[0]
                if len(w):
                    if c == BENIGN: w = w[:8]
                    buf[c].extend(list(zd[w])); buf[c] = buf[c][-BUF:]
            tot['int'] += (l_bin + l_cls).item(); tot['n'] += 1
        ttrain = time.time() - te0
        # ---- validation (continue stream)
        net.eval()
        Zv, _, _ = run_eval(va_ev)
        protos = protos_from(torch.cat(Ztr), tr_ev)
        pv = predict_probs(Zv, protos).argmax(1)
        vm = va_ev[labeled[va_ev]]
        vpos = np.where(labeled[va_ev])[0]
        vf1 = f1_score(y[vm], pv[vpos], labels=lab_cls, average='macro', zero_division=0)
        ep_times.append(ttrain)
        rec = {k: v / max(tot['n'], 1) for k, v in tot.items() if k != 'n'}
        rec.update(epoch=epoch, val_macro_f1=float(vf1), train_s=ttrain, total_s=time.time() - te0)
        hist.append(rec); print(name, rec, flush=True)
        if vf1 > best[0]:
            best = (vf1, copy.deepcopy(net.state_dict()), epoch); bad = 0
        else:
            bad += 1
            if bad >= args.patience: break
    # ---------------- final full pass with best weights
    net.load_state_dict(best[1]); net.eval(); st.reset()
    all_ev = np.arange(len(y))
    tinf = time.time()
    Z, Zm, A = run_eval(all_ev)
    t_inf = time.time() - tinf
    protos = protos_from(Z[torch.from_numpy(tr_ev)], tr_ev)
    PR = predict_probs(Z, protos)
    te_ev = np.where(period == 2)[0]
    pred = PR.argmax(1)
    res = dict(name=name, args=vars(args), cfg=cfg, history=hist, best_epoch=best[2], best_val=best[0],
               train_s_per_epoch=float(np.mean(ep_times)), infer_total_s=t_inf,
               infer_ms_per_flow=1000 * t_inf / len(all_ev), n_events=len(all_ev), wall_s=time.time() - t0)
    yte = y[te_ev]; pte = pred[te_ev]; patt = 1 - PR[te_ev, BENIGN]
    np.savez_compressed(f'{out_dir}/{name}_testpred.npz', gid=E['gid'][te_ev], pred=pte.astype(np.int8), patt=patt.astype(np.float32))
    if args.protocol == 'known':
        res['test'] = known_metrics(yte, pte, patt)
    elif args.protocol == 'unknown':
        h = args.holdout; m = (yte == h) | (yte == BENIGN)
        yb = (yte[m] == h).astype(int); pb = (pte[m] != BENIGN).astype(int)
        res['test'] = dict(holdout=CLASSES[h], recall_unknown=float(pb[yb == 1].mean()),
                           f1_unknown=float(f1_score(yb, pb)), auc_unknown=float(roc_auc_score(yb, patt[m])),
                           overall=known_metrics(yte, pte, patt, classes=lab_cls))
    else:
        # base-class known performance on the test period + episodic novel-class evaluation
        mb = np.isin(yte, lab_cls)
        res['test_base'] = known_metrics(yte[mb], pte[mb], patt[mb], classes=lab_cls)
        Znp = Z.numpy(); gpos = {g: i for i, g in enumerate(E['gid'])}
        emb_fn = lambda gi: Znp[[gpos[g] for g in gi]]
        res['fewshot'] = {'euclid': fs_evaluate(emb_fn, head_euclid), 'cosine': fs_evaluate(emb_fn, head_cosine)}
        if cfg['mfl'] and cfg['ep'] > 0:
            def head_mfl(S, Qe):
                with torch.no_grad():
                    P = torch.from_numpy(S.mean(1)).float(); q = torch.from_numpy(Qe).float()
                    pc = torch.softmax(net.cos_logits(q, P), 1)
                    if cfg['sim'] == 'bi':
                        r = net.relation(q, P); r = r / (r.sum(1, keepdim=True) + 1e-9); pc = 0.5 * pc + 0.5 * r
                return pc.numpy()
            res['fewshot']['mfl_head'] = fs_evaluate(emb_fn, head_mfl)
        # save novel+benign embeddings for visualisation
        sel = np.where(np.isin(y, NOVEL) | ((y == BENIGN) & (period == 2)))[0]
        sel = np.r_[sel[y[sel] != BENIGN], rng.choice(sel[y[sel] == BENIGN], 1500, replace=False)]
        np.savez_compressed(f'{out_dir}/{name}_emb.npz', z=Znp[sel], zm=Zm.numpy()[sel], y=y[sel], gid=E['gid'][sel])
    # representation diagnostics on test period
    Zt = Z[torch.from_numpy(te_ev)].numpy(); Zmt = Zm[torch.from_numpy(te_ev)].numpy()
    res['offdiag_corr_z_test'] = offdiag_abs_corr(Zt); res['offdiag_corr_zm_test'] = offdiag_abs_corr(Zmt)
    if A is not None:
        At = A[torch.from_numpy(te_ev)].numpy()
        res['scale_attention_by_class'] = {CLASSES[c]: At[yte == c].mean(0).tolist() for c in range(10) if (yte == c).any()}
    if args.protocol == 'known':
        sel = np.concatenate([rng.choice(te_ev[yte == c], min(300, (yte == c).sum()), replace=False) for c in range(10) if (yte == c).any()])
        np.savez_compressed(f'{out_dir}/{name}_emb.npz', z=Z.numpy()[sel], zm=Zm.numpy()[sel], y=y[sel],
                            pred=pred[sel], gid=E['gid'][sel])
    save_json(res, f'{out_dir}/{name}.json')
    print('DONE', name, 'wall', time.time() - t0, flush=True)


if __name__ == '__main__':
    main()
