"""Fine-tuning / screening library: nested CV, per-epoch logging, baselines, calibration."""
import sys, copy, json, time; sys.path.insert(0, '/workspace/code')
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
from common import *
from model import *

def strat_folds(y, k=5, seed=0):
    rng = np.random.default_rng(seed); f = np.zeros(len(y), int)
    for c in (0, 1):
        idx = rng.permutation(np.where(y == c)[0]); f[idx] = np.arange(len(idx)) % k
    return f

def metrics(y, p, thr=0.5):
    y = np.asarray(y); pred = (p >= thr).astype(int)
    tpr = pred[y == 1].mean() if (y == 1).any() else np.nan; tnr = 1 - pred[y == 0].mean() if (y == 0).any() else np.nan
    return dict(acc=(pred == y).mean(), bacc=(tpr + tnr) / 2, auc=auc(y, p), ap=ap(y, p))

# ---------------------------------------------------------------- GNN
def make_model(kind, enc_path, seed, layers=3, residual=True, dropout=0.3):
    torch.manual_seed(seed)
    enc = Encoder(layers=layers, residual=residual)
    if kind in ('pre_ft', 'pre_frozen'):
        enc.load_state_dict(torch.load(enc_path))
    return Classifier(enc, dropout=dropout)

def predict(model, cb):
    model.eval()
    with torch.no_grad(): return model(cb)

def train_gnn(kind, tr_items, tr_y, va_items, va_y, extra_eval=None, enc_path='/workspace/outputs/encoder_main.pt',
              epochs=60, seed=0, bs=64, lr=1e-3, enc_lr_mult=0.3, layers=3, residual=True, select='ap', patience=25):
    """Train with class-weighted BCE; select the epoch with best inner-validation metric. Returns best model + per-epoch log."""
    model = make_model(kind, enc_path, seed, layers, residual)
    if kind == 'pre_frozen':
        for p in model.enc.parameters(): p.requires_grad = False
        params = [{'params': model.head.parameters(), 'lr': lr}]
    elif kind == 'pre_ft':
        params = [{'params': model.enc.parameters(), 'lr': lr * enc_lr_mult}, {'params': model.head.parameters(), 'lr': lr}]
    else:
        params = [{'params': model.parameters(), 'lr': lr}]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    tr_y = np.asarray(tr_y); pw = torch.tensor((tr_y == 0).sum() / max(1, (tr_y == 1).sum()), dtype=torch.float)
    cb_tr = collate(tr_items); cb_va = collate(va_items); yt = torch.tensor(tr_y, dtype=torch.float)
    ext = {k: (collate(it), np.asarray(yy)) for k, (it, yy) in (extra_eval or {}).items()}
    rng = np.random.default_rng(seed); best = (-np.inf, None, 0); log = []
    for ep in range(epochs):
        model.train()
        if kind == 'pre_frozen': model.enc.eval()
        perm = rng.permutation(len(tr_items)); L = []
        for k in range(0, len(perm), bs):
            idx = perm[k:k + bs]; b = collate([tr_items[i] for i in idx])
            loss = F.binary_cross_entropy_with_logits(model(b), yt[idx], pos_weight=pw)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); L.append(loss.item() * len(idx))
        row = dict(epoch=ep + 1, train_loss_running=np.sum(L) / len(perm))
        for name, cb, yy in [('train', cb_tr, tr_y), ('val', cb_va, np.asarray(va_y))] + [(k, c, y_) for k, (c, y_) in ext.items()]:
            lo = predict(model, cb); pr = torch.sigmoid(lo).numpy()
            row[f'{name}_loss'] = F.binary_cross_entropy_with_logits(lo, torch.tensor(yy, dtype=torch.float), pos_weight=pw).item()
            for m, v in metrics(yy, pr).items(): row[f'{name}_{m}'] = v
        log.append(row)
        score = row['val_ap'] if select == 'ap' else (ep if select == 'last' else -row['val_loss'])
        if score > best[0]: best = (score, copy.deepcopy(model.state_dict()), ep + 1)
        if ep + 1 - best[2] >= patience: break
    model.load_state_dict(best[1])
    return model, log, best[2]

# ---------------------------------------------------------------- classical baselines
def comp_desc_features(gs):
    d = pd.DataFrame([descriptors(g) for g in gs]).values.astype(float)
    comp = np.array([np.bincount(g['x'].numpy(), minlength=28) for g in gs], float)
    return np.hstack([d, comp])

def wl_features(items, depth=2, nbins=5):
    docs = []
    for x, s, d, v, t in items:
        lab = [str(int(a)) for a in x]; feats = Counter(lab)
        nb = [[] for _ in range(len(x))]
        for a, b, vv, tt in zip(s.tolist(), d.tolist(), v.tolist(), t.tolist()): nb[b].append((a, tt, min(int(vv * nbins), nbins - 1)))
        for it in range(depth):
            new = [lab[i] + '|' + ','.join(sorted(f'{lab[a]}:{tt}:{vb}' for a, tt, vb in nb[i])) for i in range(len(x))]
            lab = [str(hash(z) % (1 << 30)) for z in new]; feats.update(f'd{it}_{z}' for z in lab)
        docs.append(feats)
    return docs

def fit_classical(kind, Xtr, ytr, Xte_list, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction import DictVectorizer
    if kind == 'gbdt':
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, class_weight='balanced', random_state=seed)
        m.fit(Xtr, ytr); return [m.predict_proba(X)[:, 1] for X in Xte_list]
    if kind == 'wl_lr':
        dv = DictVectorizer(); A = dv.fit_transform(Xtr)
        m = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000); m.fit(A, ytr)
        return [m.predict_proba(dv.transform(X))[:, 1] for X in Xte_list]
    raise ValueError(kind)

# ---------------------------------------------------------------- full nested-CV screen
def screen(ft_gs, ft_y, cd_gs, cd_y, methods, tag, k=5, seed=0, epochs=60, log_curves=True):
    ft_items = featurize(ft_gs); cd_items = featurize(cd_gs); ft_y = np.asarray(ft_y); cd_y = np.asarray(cd_y)
    folds = strat_folds(ft_y, k, seed)
    Xd_ft, Xd_cd = comp_desc_features(ft_gs), comp_desc_features(cd_gs)
    W_ft, W_cd = wl_features(ft_items), wl_features(cd_items)
    oof = {m: np.zeros(len(ft_y)) for m in methods}; cand = {m: np.zeros(len(cd_y)) for m in methods}
    curves = {}; best_ep = {}
    for m in methods:
        t0 = time.time()
        for f in range(k):
            tr = np.where(folds != f)[0]; te = np.where(folds == f)[0]
            if m == 'random':
                r = np.random.default_rng(seed + f); oof[m][te] = r.random(len(te)); cand[m] += r.random(len(cd_y)) / k; continue
            if m in ('gbdt', 'wl_lr'):
                Xa, Xc = (Xd_ft, Xd_cd) if m == 'gbdt' else (W_ft, W_cd)
                Xtr = Xa[tr] if m == 'gbdt' else [Xa[i] for i in tr]; Xte = Xa[te] if m == 'gbdt' else [Xa[i] for i in te]
                pte, pc = fit_classical(m, Xtr, ft_y[tr], [Xte, Xc], seed)
                oof[m][te] = pte; cand[m] += pc / k; continue
            # GNN: inner split of the outer-training part for early stopping
            inner = strat_folds(ft_y[tr], 5, seed + 100 + f); itr = tr[inner != 0]; iva = tr[inner == 0]
            extra = {'test': ([ft_items[i] for i in te], ft_y[te])} if log_curves else None
            model, log, be = train_gnn(m, [ft_items[i] for i in itr], ft_y[itr], [ft_items[i] for i in iva], ft_y[iva],
                                       extra_eval=extra, epochs=epochs, seed=seed + f)
            curves[f'{m}_fold{f}'] = log; best_ep[f'{m}_fold{f}'] = be
            # store logits (averaged over folds for candidates), convert later
            oof[m][te] = predict(model, collate([ft_items[i] for i in te])).numpy()
            cand[m] += predict(model, collate(cd_items)).numpy() / k
        if m not in ('random', 'gbdt', 'wl_lr'):   # convert logits -> probabilities
            oof[m] = 1 / (1 + np.exp(-oof[m])); cand[m] = 1 / (1 + np.exp(-cand[m]))
        print(f'[{tag}] {m}: OOF AUC {auc(ft_y, oof[m]):.3f} AP {ap(ft_y, oof[m]):.3f} | cand AUC {auc(cd_y, cand[m]):.3f} AP {ap(cd_y, cand[m]):.3f} hits@50 {hits_at(cd_y, cand[m], 50)} ({time.time()-t0:.0f}s)', flush=True)
    np.savez(f'/workspace/outputs/scores_{tag}.npz', ft_y=ft_y, cd_y=cd_y, folds=folds, **{f'oof_{m}': oof[m] for m in methods}, **{f'cand_{m}': cand[m] for m in methods})
    json.dump(dict(curves=curves, best_epoch=best_ep), open(f'/workspace/outputs/curves_{tag}.json', 'w'))
    return oof, cand

def evaluate(tag, methods, B=1000):
    from sklearn.linear_model import LogisticRegression
    z = np.load(f'/workspace/outputs/scores_{tag}.npz'); ft_y, cd_y = z['ft_y'], z['cd_y']; rows = []; hc = []
    for m in methods:
        o, c = z[f'oof_{m}'], z[f'cand_{m}']
        pa, null = perm_p(cd_y, c, ap, B); pu, _ = perm_p(cd_y, c, auc, B, seed=1)
        ci_auc = bootstrap_ci(cd_y, c, auc, B); ci_ap = bootstrap_ci(cd_y, c, ap, B); ci_h = bootstrap_ci(cd_y, c, lambda a, b: hits_at(a, b, 50), B)
        rows.append(dict(method=m, oof_auc=auc(ft_y, o), oof_ap=ap(ft_y, o), cand_auc=auc(cd_y, c), cand_auc_lo=ci_auc[0], cand_auc_hi=ci_auc[1],
                         cand_ap=ap(cd_y, c), cand_ap_lo=ci_ap[0], cand_ap_hi=ci_ap[1], prevalence=cd_y.mean(),
                         hits50=hits_at(cd_y, c, 50), hits50_lo=ci_h[0], hits50_hi=ci_h[1], expected_hits50=50 * cd_y.mean(),
                         ef5=ef_at(cd_y, c, 0.05), perm_p_ap=pa, perm_p_auc=pu, **{f'cand_{k}': v for k, v in metrics(cd_y, c).items() if k in ('acc', 'bacc')}))
        # Platt calibration on OOF scores (logit), applied to candidate scores
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        cal = LogisticRegression(C=1e6).fit(lg(o)[:, None], ft_y); cc = cal.predict_proba(lg(c)[:, None])[:, 1]
        np.save(f'/workspace/outputs/calibrated_{tag}_{m}.npy', cc)
        for kind, sc in (('raw', c), ('platt', cc)):
            for thr in (0.5, 0.8, 0.9):
                sel = sc >= thr; k_ = int(cd_y[sel].sum()); n_ = int(sel.sum()); p_, lo, hi = wilson(k_, n_)
                hc.append(dict(method=m, scores=kind, threshold=thr, n_selected=n_, true_pos=k_, precision=p_, prec_lo=lo, prec_hi=hi,
                               base_rate=cd_y.mean(), recall=k_ / cd_y.sum()))
    R = pd.DataFrame(rows); H = pd.DataFrame(hc)
    R.to_csv(f'/workspace/outputs/metrics_{tag}.csv', index=False); H.to_csv(f'/workspace/outputs/highconf_{tag}.csv', index=False)
    return R, H
