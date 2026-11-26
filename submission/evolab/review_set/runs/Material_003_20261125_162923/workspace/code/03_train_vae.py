"""Step 3: train the dual-encoder graph VAE (acid encoder/decoder, epoxide encoder/decoder, Tg head).
usage: python code/03_train_vae.py TAG BETA EPOCHS MAXMIN
Logs per-epoch metrics to outputs/vae_{TAG}_log.csv and checkpoints to outputs/vae_{TAG}.pt"""
import sys, time, pickle, numpy as np, pandas as pd, torch
sys.path.insert(0, 'code')
from vae_lib import *

torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
TAG = sys.argv[1]; BETA = float(sys.argv[2]); EPOCHS = int(sys.argv[3]); MAXMIN = float(sys.argv[4])
LAM = 40.0; BS = 128; MAXLEN = 110

prep_path = 'outputs/vae_data.pkl'
try:
    D = pickle.load(open(prep_path, 'rb'))
except FileNotFoundError:
    vit = pd.read_csv('data/tg_vitrimer_MD.csv')
    ta = [to_selfies_tokens(s) for s in vit.acid]; te = [to_selfies_tokens(s) for s in vit.epoxide]
    ok = np.array([a is not None and e is not None and len(a) + 2 <= MAXLEN and len(e) + 2 <= MAXLEN for a, e in zip(ta, te)])
    print('usable pairs', ok.sum(), '/', len(vit))
    va = Vocab([t for t, o in zip(ta, ok) if o]); ve = Vocab([t for t, o in zip(te, ok) if o])
    idx = np.where(ok)[0]
    D = dict(idx=idx, va=va, ve=ve,
             ga=[mol_graph(vit.acid[i]) for i in idx], ge=[mol_graph(vit.epoxide[i]) for i in idx],
             sa=[va.encode(ta[i], MAXLEN) for i in idx], se=[ve.encode(te[i], MAXLEN) for i in idx],
             acid=[Chem.MolToSmiles(Chem.MolFromSmiles(vit.acid[i])) for i in idx],
             epox=[Chem.MolToSmiles(Chem.MolFromSmiles(vit.epoxide[i])) for i in idx],
             tg=vit.tg.values[idx].astype(np.float32))
    rng = np.random.default_rng(0); perm = rng.permutation(len(idx))
    D['val'] = np.sort(perm[:int(0.05 * len(idx))]); D['train'] = np.sort(perm[int(0.05 * len(idx)):])
    pickle.dump(D, open(prep_path, 'wb'))
print('vocab acid %d epox %d' % (len(D['va'].itos), len(D['ve'].itos)))
TGM, TGS = 424.0, 33.7
y = (D['tg'] - TGM) / TGS


def pad(seqs):
    T = max(len(s) for s in seqs)
    out = np.zeros((len(seqs), T), np.int64)
    for i, s in enumerate(seqs): out[i, :len(s)] = s
    return torch.from_numpy(out)


model = DualVAE(len(D['va'].itos), len(D['ve'].itos))
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-4)
print('params', sum(p.numel() for p in model.parameters()))


def run_batch(ids, train=True, beta=BETA):
    ga = batch_graphs([D['ga'][i] for i in ids]); ge = batch_graphs([D['ge'][i] for i in ids])
    (mua, lva), (mue, lve) = model.encode(ga, ge)
    if train:
        za = mua + torch.randn_like(mua) * torch.exp(0.5 * lva); ze = mue + torch.randn_like(mue) * torch.exp(0.5 * lve)
    else:
        za, ze = mua, mue
    ra = model.dec_a(za, pad([D['sa'][i] for i in ids])); re = model.dec_e(ze, pad([D['se'][i] for i in ids]))
    kl = (-0.5 * (1 + lva - mua ** 2 - lva.exp()).sum(1)) + (-0.5 * (1 + lve - mue ** 2 - lve.exp()).sum(1))
    pred = model.head(torch.cat([za, ze], 1))[:, 0]
    mse = (pred - torch.from_numpy(y[ids])) ** 2
    loss = (ra + re + beta * kl + LAM * mse).mean()
    return loss, ra.mean().item(), re.mean().item(), kl.mean().item(), mse.mean().item(), pred.detach().numpy(), (mua, mue)


@torch.no_grad()
def evaluate(nrec=300, nprior=200):
    model.eval()
    val = D['val']; preds = []; RA = RE = KL = 0; mus_a, mus_e = [], []
    for b in range(0, len(val), 128):
        ids = val[b:b + 128]
        _, ra, re, kl, _, p, (ma, me) = run_batch(ids, train=False)
        RA += ra * len(ids); RE += re * len(ids); KL += kl * len(ids); preds.append(p); mus_a.append(ma); mus_e.append(me)
    preds = np.concatenate(preds); ma = torch.cat(mus_a); me = torch.cat(mus_e)
    yt = y[val]; r2 = 1 - ((preds - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    k = min(nrec, len(val))
    ga = model.dec_a.generate(ma[:k]); ge = model.dec_e.generate(me[:k])
    sa = [selfies_to_smiles(D['va'].decode(g)) for g in ga]; se = [selfies_to_smiles(D['ve'].decode(g)) for g in ge]
    exa = np.mean([s == D['acid'][i] for s, i in zip(sa, val[:k])]); exe = np.mean([s == D['epox'][i] for s, i in zip(se, val[:k])])
    fga_rec = np.mean([fg_check(s, 'acid') for s in sa]); fge_rec = np.mean([fg_check(s, 'epox') for s in se])
    # near-data perturbation (sigma=0.5) and prior samples
    zp = torch.randn(nprior, model.zdim)
    pa = [selfies_to_smiles(D['va'].decode(g)) for g in model.dec_a.generate(zp)]
    pe = [selfies_to_smiles(D['ve'].decode(g)) for g in model.dec_e.generate(torch.randn(nprior, model.zdim))]
    na = [selfies_to_smiles(D['va'].decode(g)) for g in model.dec_a.generate(ma[:nprior] + 0.5 * torch.randn(min(nprior, len(ma)), model.zdim))]
    ne = [selfies_to_smiles(D['ve'].decode(g)) for g in model.dec_e.generate(me[:nprior] + 0.5 * torch.randn(min(nprior, len(me)), model.zdim))]
    model.train()
    n = len(val)
    return dict(val_rec_acid=RA / n, val_rec_epox=RE / n, val_kl=KL / n, head_r2=r2,
                head_mae_K=float(np.abs(preds - yt).mean() * TGS),
                exact_acid=exa, exact_epox=exe, fg_rec_acid=fga_rec, fg_rec_epox=fge_rec,
                fg_prior_acid=np.mean([fg_check(s, 'acid') for s in pa]), fg_prior_epox=np.mean([fg_check(s, 'epox') for s in pe]),
                fg_near_acid=np.mean([fg_check(s, 'acid') for s in na]), fg_near_epox=np.mean([fg_check(s, 'epox') for s in ne]))


log = []; t0 = time.time()
tr = D['train']
for ep in range(EPOCHS):
    beta = BETA * min(1.0, (ep + 1) / 4)
    perm = np.random.permutation(tr)
    stats = []
    for b in range(0, len(perm), BS):
        ids = perm[b:b + BS]
        loss, ra, re, kl, mse, _, _ = run_batch(ids, beta=beta)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        stats.append((loss.item(), ra, re, kl, mse))
    sched.step()
    s = np.mean(stats, 0)
    ev = evaluate()
    row = dict(epoch=ep + 1, beta=beta, minutes=(time.time() - t0) / 60, train_loss=s[0], train_rec_acid=s[1],
               train_rec_epox=s[2], train_kl=s[3], train_mse=s[4], **ev)
    log.append(row); pd.DataFrame(log).to_csv(f'outputs/vae_{TAG}_log.csv', index=False)
    torch.save(model.state_dict(), f'outputs/vae_{TAG}.pt')
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    if (time.time() - t0) / 60 > MAXMIN:
        print('time budget reached'); break
print('done')
