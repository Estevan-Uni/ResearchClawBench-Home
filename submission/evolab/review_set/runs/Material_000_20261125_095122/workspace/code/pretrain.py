"""Self-supervised masked-node pre-training on pretrain_data.pt ONLY (labels never used)."""
import sys, json, time; sys.path.insert(0, '/workspace/code')
import numpy as np, torch
from common import load_cache
from model import *
torch.manual_seed(0); torch.set_num_threads(2)
tag = sys.argv[1] if len(sys.argv) > 1 else 'main'
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
D = load_cache()
sets = ['pretrain'] if tag != 'transductive' else ['pretrain', 'finetune', 'candidate']
items = sum([featurize(D[s]) for s in sets], [])
val_idx = np.random.default_rng(0).permutation(len(items))[:500]; vs = set(val_idx)
tr = [it for i, it in enumerate(items) if i not in vs]; va = [items[i] for i in val_idx]
enc = Encoder(); model = MaskedNodeModel(enc)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
# baselines: uniform (1/28) and majority element frequency (training atoms)
allx = torch.cat([it[0] for it in tr]); freq = torch.bincount(allx, minlength=28).float(); freq /= freq.sum()
base = dict(uniform_acc=1 / 28, majority_acc=float(freq.max()), uniform_ce=float(np.log(28)),
            marginal_ce=float(-(freq * freq.clamp(min=1e-12).log()).sum()))
# "composition-aware" baseline: predict masked atom from the unmasked atoms of the same graph? (computed below in val loop)
log = []; g = torch.Generator().manual_seed(1)
for ep in range(EPOCHS):
    model.train(); perm = np.random.default_rng(ep).permutation(len(tr)); L = []; A = []; t0 = time.time()
    for k in range(0, len(tr), 64):
        b = collate([tr[i] for i in perm[k:k + 64]])
        loss, acc, _, _ = model(b, gen=g); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); L.append(loss.item()); A.append(acc.item())
    sched.step(); model.eval(); vl = []; vacc = []; vg = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for k in range(0, len(va), 250):
            b = collate(va[k:k + 250]); l, a, m, lg = model(b, gen=vg); vl.append(l.item()); vacc.append(a.item())
    log.append(dict(epoch=ep + 1, train_loss=np.mean(L), train_acc=np.mean(A), val_loss=np.mean(vl), val_acc=np.mean(vacc), sec=time.time() - t0))
    print(log[-1], flush=True)
torch.save(enc.state_dict(), f'/workspace/outputs/encoder_{tag}.pt')
json.dump(dict(log=log, baselines=base), open(f'/workspace/outputs/pretrain_log_{tag}.json', 'w'), indent=1)
print('DONE')
