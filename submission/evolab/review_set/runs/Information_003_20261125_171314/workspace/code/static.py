"""Static baselines: RF, MLP, E-GraphSAGE (known / unknown / few-shot) and dedicated
few-shot learners (ProtoNet, RelationNet, Baseline++, BSNet-style, fine-tune-linear, raw-feature
prototype / 1-NN) evaluated on the shared episodic harness.
usage: python static.py --method rf|mlp|egs|protonet|relationnet|baselinepp|bsnet|finetune|raw
                        --protocol known|unknown|fewshot --holdout c --seed s
"""
import sys, os, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn
import scipy.sparse as sp
sys.path.insert(0, '/workspace/code')
from common import *
from fewshot_eval import evaluate as fs_evaluate, head_euclid, head_cosine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser()
ap.add_argument('--method'); ap.add_argument('--protocol', default='known'); ap.add_argument('--holdout', type=int, default=-1)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--feat', default='cat')
args = ap.parse_args()
torch.set_num_threads(2); torch.manual_seed(args.seed); np.random.seed(args.seed)
name = f'{args.protocol}{"_h" + str(args.holdout) if args.protocol == "unknown" else ""}_{args.method}{"" if args.feat == "cat" else "_" + args.feat}_s{args.seed}'
out_dir = f'{OUT}/runs'; os.makedirs(out_dir, exist_ok=True)
if os.path.exists(f'{out_dir}/{name}.json'):
    print('exists'); sys.exit()
t0 = time.time()
D = load(); y = D['attack']; n = len(y)
tr, va, te = chrono_split(n)
period = np.zeros(n, int); period[va] = 1; period[te] = 2
keep = np.ones(n, bool)
if args.protocol == 'unknown':
    keep &= ~((y == args.holdout) & (period < 2))
H = np.load(f'{OUT}/sd_h.npy')
fit_rows = np.where(keep & (period == 0))[0]
Fq = quantile_fit(D['msg'], fit_rows); Hq = quantile_fit(H, fit_rows)
Xq = {'cat': np.c_[Fq, Hq], 'F': Fq, 'h': Hq}[args.feat].astype(np.float32)
Xraw = {'cat': np.c_[D['msg'], H], 'F': D['msg'], 'h': H}[args.feat]
if args.protocol == 'known':
    lab_cls = list(range(10))
elif args.protocol == 'fewshot':
    lab_cls = BASE
else:
    lab_cls = [c for c in range(10) if c != args.holdout]
TR = np.where(keep & (period == 0) & np.isin(y, lab_cls))[0]
VA = np.where(keep & (period == 1) & np.isin(y, lab_cls))[0]
TE = np.where(period == 2)[0]
cmap = -np.ones(10, int); cmap[lab_cls] = np.arange(len(lab_cls))
rng = np.random.default_rng(args.seed)
res = dict(name=name, args=vars(args))

# ---------- E-GraphSAGE aggregated inputs (transductive static graph over kept flows)
def egs_inputs(X):
    ids = np.where(keep)[0]
    s, d = D['src_i'][ids], D['dst_i'][ids]; nn_ = D['n_nodes']
    rows = np.r_[s, d]; cols = np.r_[ids, ids]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nn_, n))
    deg = np.asarray(A.sum(1)).ravel(); A = sp.diags(1 / np.maximum(deg, 1)) @ A
    agg = (A @ X).astype(np.float32)                  # mean of incident edge features per node
    # 2-hop: mean over neighbouring nodes' aggregated features via edges
    Nn = sp.csr_matrix((np.ones(2 * len(ids)), (np.r_[s, d], np.r_[d, s])), shape=(nn_, nn_))
    dn = np.asarray(Nn.sum(1)).ravel(); Nn = sp.diags(1 / np.maximum(dn, 1)) @ Nn
    agg2 = (Nn @ agg).astype(np.float32)
    return agg, agg2


class EGS(nn.Module):
    """E-GraphSAGE (Lo et al.): node emb from mean-aggregated incident edge features (2 layers),
    edge emb = [z_u || z_v], classifier on edge emb."""
    def __init__(s, din, nc, d=64):
        super().__init__()
        s.l1 = nn.Linear(1 + din, d); s.l2 = nn.Linear(d + d, d); s.p2 = nn.Linear(din, d)
        s.out = nn.Linear(2 * d, nc); s.d = d
    def node(s, a1, a2):
        h1 = torch.relu(s.l1(torch.cat([torch.ones(len(a1), 1), a1], 1)))
        hn = torch.relu(s.p2(a2))
        return torch.relu(s.l2(torch.cat([h1, hn], 1)))
    def emb(s, i, A1, A2):
        su, dv = D['src_i'][i], D['dst_i'][i]
        return torch.cat([s.node(A1[su], A2[su]), s.node(A1[dv], A2[dv])], 1)
    def forward(s, i, A1, A2):
        return s.out(s.emb(i, A1, A2))


class Enc(nn.Module):
    def __init__(s, din, d=64):
        super().__init__(); s.f = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Linear(128, d))
    def forward(s, x): return s.f(x)


def train_sup(model_fwd, params, Xidx, epochs=15, bs=512, cosine=False):
    """supervised softmax (class-weighted) with early stopping on val macro-F1."""
    cnt = np.bincount(cmap[y[TR]], minlength=len(lab_cls)); w = torch.tensor(cnt ** -0.5, dtype=torch.float32); w = w / w.mean()
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
    best, state, bad, hist = -1, None, 0, []
    for ep in range(epochs):
        perm = rng.permutation(TR)
        for b in range(0, len(perm), bs):
            i = perm[b:b + bs]
            loss = Fn.cross_entropy(model_fwd(i), torch.from_numpy(cmap[y[i]]), weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pv = model_fwd(VA).argmax(1).numpy()
        f = f1_score(cmap[y[VA]], pv, average='macro'); hist.append(float(f))
        if f > best: best, bad = f, 0; state = [p.detach().clone() for p in params]
        else:
            bad += 1
            if bad >= 3: break
    with torch.no_grad():
        for p, s_ in zip(params, state): p.copy_(s_)
    return hist


def finish_known_unknown(prob_te):
    pred = np.array(lab_cls)[prob_te.argmax(1)]
    patt = 1 - prob_te[:, cmap[BENIGN]]
    yte = y[TE]
    np.savez_compressed(f'{out_dir}/{name}_testpred.npz', gid=TE, pred=pred.astype(np.int8), patt=patt.astype(np.float32))
    if args.protocol == 'known':
        res['test'] = known_metrics(yte, pred, patt)
    elif args.protocol == 'unknown':
        h = args.holdout; m = (yte == h) | (yte == BENIGN)
        yb = (yte[m] == h).astype(int); pb = (pred[m] != BENIGN).astype(int)
        res['test'] = dict(holdout=CLASSES[h], recall_unknown=float(pb[yb == 1].mean()), f1_unknown=float(f1_score(yb, pb)),
                           auc_unknown=float(roc_auc_score(yb, patt[m])), overall=known_metrics(yte, pred, patt, classes=lab_cls))
    else:
        mb = np.isin(yte, lab_cls)
        res['test_base'] = known_metrics(yte[mb], pred[mb], patt[mb], classes=lab_cls)


def episodic_train(enc, head, n_iter=3000, N=4, K=5, Q=15, extra=()):
    pools = {c: TR[y[TR] == c] for c in lab_cls}
    opt = torch.optim.Adam(list(enc.parameters()) + list(extra), lr=1e-3)
    X = torch.from_numpy(Xq)
    for it in range(n_iter):
        cls = rng.choice(lab_cls, N, replace=False)
        S, Qi = [], []
        for c in cls:
            p = rng.choice(pools[c], K + Q, replace=False); S.append(p[:K]); Qi.append(p[K:])
        zs = enc(X[np.concatenate(S)]).reshape(N, K, -1); zq = enc(X[np.concatenate(Qi)])
        yq = torch.arange(N).repeat_interleave(Q)
        loss = head(zs, zq, yq)
        opt.zero_grad(); loss.backward(); opt.step()


m = args.method
X_t = torch.from_numpy(Xq)
if m == 'rf':
    rf = RandomForestClassifier(200, n_jobs=2, random_state=args.seed, class_weight='balanced_subsample', min_samples_leaf=1)
    rf.fit(Xraw[TR], cmap[y[TR]])
    ti = time.time(); P = rf.predict_proba(Xraw[TE]); res['infer_ms_per_flow'] = 1000 * (time.time() - ti) / len(TE)
    finish_known_unknown(P)
    if args.protocol == 'fewshot':   # RF leaf-embedding is not meaningful; use raw-feature episodic heads instead
        pass
elif m in ('mlp', 'finetune'):
    enc = Enc(Xq.shape[1]); lin = nn.Linear(64, len(lab_cls))
    fwd = lambda i: lin(torch.relu(enc(X_t[i])))
    res['val_hist'] = train_sup(fwd, list(enc.parameters()) + list(lin.parameters()), TR)
    with torch.no_grad():
        ti = time.time(); P = torch.softmax(fwd(TE), 1).numpy(); res['infer_ms_per_flow'] = 1000 * (time.time() - ti) / len(TE)
    finish_known_unknown(P)
    if args.protocol == 'fewshot':
        emb = lambda gi: torch.relu(enc(X_t[gi])).detach().numpy()
        res['fewshot'] = {'euclid': fs_evaluate(emb, head_euclid), 'cosine': fs_evaluate(emb, head_cosine)}
        def head_lr(S, Qe):
            N, K, d = S.shape
            if K == 1:
                return head_cosine(S, Qe)
            lr = LogisticRegression(C=1.0, max_iter=200).fit(S.reshape(N * K, d), np.repeat(np.arange(N), K))
            return lr.predict_proba(Qe)
        res['fewshot']['finetune_lr'] = fs_evaluate(emb, head_lr)
        sel = np.where(np.isin(y, NOVEL) | ((y == BENIGN) & (period == 2)))[0]
        sel = np.r_[sel[y[sel] != BENIGN], rng.choice(sel[y[sel] == BENIGN], 1500, replace=False)]
        np.savez_compressed(f'{out_dir}/{name}_emb.npz', z=emb(sel), y=y[sel], gid=sel)
elif m == 'egs':
    A1, A2 = egs_inputs(Xq); A1 = torch.from_numpy(A1); A2 = torch.from_numpy(A2)
    net = EGS(Xq.shape[1], len(lab_cls))
    res['val_hist'] = train_sup(lambda i: net(i, A1, A2), list(net.parameters()), TR)
    with torch.no_grad():
        ti = time.time(); P = torch.softmax(net(TE, A1, A2), 1).numpy(); res['infer_ms_per_flow'] = 1000 * (time.time() - ti) / len(TE)
        Zte = net.emb(TE, A1, A2).numpy()
    finish_known_unknown(P)
    from dyn import offdiag_abs_corr
    res['offdiag_corr_z_test'] = offdiag_abs_corr(Zte)
    if args.protocol == 'known':
        yte = y[TE]
        sel = np.concatenate([rng.choice(np.where(yte == c)[0], min(300, (yte == c).sum()), replace=False) for c in range(10)])
        np.savez_compressed(f'{out_dir}/{name}_emb.npz', z=Zte[sel], y=yte[sel], gid=TE[sel])
    if args.protocol == 'fewshot':
        emb = lambda gi: net.emb(np.asarray(gi), A1, A2).detach().numpy()
        res['fewshot'] = {'euclid': fs_evaluate(emb, head_euclid), 'cosine': fs_evaluate(emb, head_cosine)}
        sel = np.where(np.isin(y, NOVEL) | ((y == BENIGN) & (period == 2)))[0]
        sel = np.r_[sel[y[sel] != BENIGN], rng.choice(sel[y[sel] == BENIGN], 1500, replace=False)]
        np.savez_compressed(f'{out_dir}/{name}_emb.npz', z=emb(sel), y=y[sel], gid=sel)
elif m == 'raw':
    emb = lambda gi: Xq[gi]
    res['fewshot'] = {'euclid': fs_evaluate(emb, head_euclid), 'cosine': fs_evaluate(emb, head_cosine)}
    def head_1nn(S, Qe):
        N, K, d = S.shape; Sf = S.reshape(N * K, d)
        dist = ((Qe[:, None] - Sf[None]) ** 2).sum(-1)
        sc = np.full((len(Qe), N), -np.inf)
        for c in range(N): sc[:, c] = -dist[:, c * K:(c + 1) * K].min(1)
        return sc
    res['fewshot']['1nn'] = fs_evaluate(emb, head_1nn)
    def head_rf(S, Qe):
        N, K, d = S.shape
        rf = RandomForestClassifier(100, random_state=0, n_jobs=1).fit(S.reshape(N * K, d), np.repeat(np.arange(N), K))
        return rf.predict_proba(Qe)
    res['fewshot']['rf_support'] = fs_evaluate(emb, head_rf, n_ep=200)
else:
    # dedicated episodic few-shot learners trained on base classes
    enc = Enc(Xq.shape[1]); extra = []
    tau = nn.Parameter(torch.tensor(10.0))
    rel = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def relscore(zq, P):
        n_, C = zq.shape[0], P.shape[0]
        return torch.sigmoid(rel(torch.cat([zq[:, None].expand(n_, C, -1), P[None].expand(n_, C, -1)], -1)).squeeze(-1))
    if m == 'protonet':
        head = lambda zs, zq, yq: Fn.cross_entropy(-torch.cdist(zq, zs.mean(1)) ** 2, yq)
    elif m == 'relationnet':
        extra = list(rel.parameters())
        head = lambda zs, zq, yq: Fn.mse_loss(relscore(zq, zs.mean(1)), Fn.one_hot(yq, zs.shape[0]).float())
    elif m == 'bsnet':
        extra = list(rel.parameters()) + [tau]
        head = lambda zs, zq, yq: Fn.cross_entropy(tau * Fn.normalize(zq, dim=-1) @ Fn.normalize(zs.mean(1), dim=-1).T, yq) + \
            Fn.mse_loss(relscore(zq, zs.mean(1)), Fn.one_hot(yq, zs.shape[0]).float())
    if m == 'baselinepp':
        W = nn.Parameter(torch.randn(len(lab_cls), 64) * 0.1)
        fwd = lambda i: tau * Fn.normalize(enc(X_t[i]), dim=-1) @ Fn.normalize(W, dim=-1).T
        res['val_hist'] = train_sup(fwd, list(enc.parameters()) + [W, tau], TR)
    else:
        episodic_train(enc, head, extra=extra)
    emb = lambda gi: enc(X_t[gi]).detach().numpy()
    res['fewshot'] = {'euclid': fs_evaluate(emb, head_euclid), 'cosine': fs_evaluate(emb, head_cosine)}
    if m in ('relationnet', 'bsnet'):
        def head_learned(S, Qe):
            with torch.no_grad():
                P = torch.from_numpy(S.mean(1)).float(); q = torch.from_numpy(Qe).float(); r = relscore(q, P)
                if m == 'bsnet':
                    r = 0.5 * r / r.sum(1, keepdim=True) + 0.5 * torch.softmax(tau * Fn.normalize(q, dim=-1) @ Fn.normalize(P, dim=-1).T, 1)
            return r.numpy()
        res['fewshot']['learned'] = fs_evaluate(emb, head_learned)
    if m == 'bsnet':
        sel = np.where(np.isin(y, NOVEL) | ((y == BENIGN) & (period == 2)))[0]
        sel = np.r_[sel[y[sel] != BENIGN], rng.choice(sel[y[sel] == BENIGN], 1500, replace=False)]
        np.savez_compressed(f'{out_dir}/{name}_emb.npz', z=emb(sel), y=y[sel], gid=sel)
res['wall_s'] = time.time() - t0
save_json(res, f'{out_dir}/{name}.json')
print('DONE', name, res['wall_s'])
