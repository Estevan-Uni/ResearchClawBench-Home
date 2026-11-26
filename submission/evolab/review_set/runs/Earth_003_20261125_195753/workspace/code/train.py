"""Training of the FuXi-style cascade on ERA5 (WB2 64x32).

Stages
  pretrain : one-step (6 h) training on ERA5 pairs
  short    : autoregressive fine-tuning from pretrain (curriculum of K rollout steps), leads 0-5 d
  medium   : initialised from short; inputs are SHORT's own day-5 forecast states (steps 19,20),
             targets ERA5 at steps 21..20+K                                          (leads 5-10 d)
  long     : initialised from medium; inputs are the cascade's day-10 states (steps 39,40)   (10-15 d)
  shortplus: compute-matched control: continue training SHORT on ERA5 initial states with the same
             number of extra iterations as medium+long received.
"""
import argparse, os, time, json, glob
import numpy as np, torch
import torch.nn.functional as F
from common import *
from model import UTransformer

torch.set_num_threads(int(os.environ.get('NT', 2)))
CK = f'{ROOT}/ckpt'; os.makedirs(CK, exist_ok=True)
TRAIN_YEARS = list(range(2000, 2018))
MODEL_KW = dict(dim=int(os.environ.get('DIM', 192)), depth=int(os.environ.get('DEPTH', 8)), heads=6)


class Pool:
    """All available training years in normalised float32 + forcing, as contiguous segments."""
    def __init__(self, years=None, holdout_days=0):
        nm = Normaliser(); self.nm = nm
        years = [y for y in (years or TRAIN_YEARS) if os.path.exists(f'{ERA}/{y}.npy')]
        Xs, Ds = [], []
        for y in sorted(years):
            X, d = load_year_physical(y); Xs.append(nm.norm(X)); Ds.append(d)
        self.years = sorted(years)
        self.X = np.concatenate(Xs); self.D = np.concatenate(Ds)
        self.Fo = forcing(self.D)
        gap = np.diff(self.D.astype(np.int64)) != 6
        self.seg = np.concatenate([[0], np.cumsum(gap)])           # segment id per index
        n = len(self.D)
        self.seg_end = np.zeros(n, np.int64)
        for s in np.unique(self.seg):
            ii = np.where(self.seg == s)[0]; self.seg_end[ii] = ii[-1]
        # validation hold-out: last `holdout_days` of the latest year
        self.val_start = n - holdout_days * 4 if holdout_days else n
        print('pool years', self.years, self.X.shape, flush=True)

    def valid_starts(self, need, val=False):
        """indices i (current time t) such that i-1 >= seg start and i+need <= seg end."""
        i = np.arange(1, len(self.D))
        ok = (self.seg[i - 1] == self.seg[i]) & (i + need <= self.seg_end[i])
        if val: ok &= (i >= self.val_start)
        else: ok &= (i + need < self.val_start)
        return i[ok]


def lat_l1(pred, tgt):
    w = torch.from_numpy(WLAT.astype(np.float32)).view(1, 1, -1, 1)
    return ((pred - tgt).abs() * w).mean()


def step(model, x0, x1, f, dstd):
    return x1 + dstd * model(x0, x1, f)


def make_batch(pool, idx, K, offset=0, gen=None, gi=None):
    """Return x0,x1 (inputs), forcing list, targets list as torch tensors.
    idx: current-time indices of the *ERA5 initial time*; offset: number of steps already taken
    (inputs at idx+offset-1, idx+offset); gen: pre-generated states (N,2,70,H,W) used as inputs."""
    if gen is None:
        x0 = pool.X[idx + offset - 1]; x1 = pool.X[idx + offset]
    else:
        x0 = gen[gi, 0].astype(np.float32); x1 = gen[gi, 1].astype(np.float32)
    fs = [torch.from_numpy(pool.Fo[idx + offset + k]) for k in range(K)]
    ts = [torch.from_numpy(pool.X[idx + offset + k + 1]) for k in range(K)]
    return torch.from_numpy(np.ascontiguousarray(x0)), torch.from_numpy(np.ascontiguousarray(x1)), fs, ts


def rollout(model, x0, x1, fs, dstd):
    outs = []
    for f in fs:
        x2 = step(model, x0, x1, f, dstd); outs.append(x2); x0, x1 = x1, x2
    return outs


def evaluate(model, pool, dstd, K, offset, gen=None, gidx=None, gstarts=None, n=64, seed=0):
    """Mean lat-weighted RMSE (normalised units) at the last rollout step on held-out samples."""
    model.eval(); rng = np.random.default_rng(seed)
    with torch.no_grad():
        if gen is None:
            st = pool.valid_starts(offset + K + 1, val=True); idx = rng.choice(st, min(n, len(st)), replace=False); gi = None
        else:
            gi = gidx; idx = gstarts[gi]
        x0, x1, fs, ts = make_batch(pool, idx, K, offset, gen, gi)
        outs = rollout(model, x0, x1, fs, dstd)
        w = torch.from_numpy(WLAT.astype(np.float32)).view(1, 1, -1, 1)
        r = torch.sqrt((((outs[-1] - ts[-1]) ** 2) * w).mean((0, 2, 3))).mean().item()
    model.train(); return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    ap.add_argument('--init', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--curriculum', default='1:1000')     # "K:iters,K:iters"
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--gen', default=None)                # npz with states + starts
    ap.add_argument('--offset', type=int, default=0)      # steps already taken by the inputs
    ap.add_argument('--minutes', type=float, default=1e9) # wall-clock cap
    ap.add_argument('--warmup', type=int, default=200)
    args = ap.parse_args()
    pool = Pool(holdout_days=45)
    dstd = torch.from_numpy(Normaliser().dsd).view(1, -1, 1, 1)
    model = UTransformer(**MODEL_KW)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location='cpu')['model'])
    print('params %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    cur = [(int(a), int(b)) for a, b in (c.split(':') for c in args.curriculum.split(','))]
    total = sum(b for _, b in cur)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1, (it + 1) / args.warmup) * 0.5 * (1 + np.cos(np.pi * min(it, total) / total)))
    gen = gstarts = None
    if args.gen:
        g = np.load(args.gen); gen = g['states']
        # map ERA5 dates of the starts to pool indices
        dd = {int(d): i for i, d in enumerate(pool.D.astype(np.int64))}
        gstarts = np.array([dd[int(d)] for d in g['start_dates']])
        nval = max(32, len(gstarts) // 20)
        gval = np.arange(len(gstarts) - nval, len(gstarts)); gtr = np.arange(len(gstarts) - nval)
        print('gen states', gen.shape, 'train', len(gtr), 'val', len(gval), flush=True)
    rng = np.random.default_rng(1)
    log = []; it = 0; t0 = time.time()
    for K, n_it in cur:
        starts = pool.valid_starts(args.offset + K + 1)
        for _ in range(n_it):
            if gen is None and it % 500 == 0 and it > 0:
                have = [y for y in TRAIN_YEARS if os.path.exists(f'{ERA}/{y}.npy')]
                if have != pool.years:            # include newly downloaded years
                    pool = Pool(holdout_days=45); starts = pool.valid_starts(args.offset + K + 1)
            if gen is None:
                idx = rng.choice(starts, args.bs); gi = None
            else:
                gi = rng.choice(gtr, args.bs); idx = gstarts[gi]
            x0, x1, fs, ts = make_batch(pool, idx, K, args.offset, gen, gi)
            outs = rollout(model, x0, x1, fs, dstd)
            loss = sum(lat_l1(o, t) for o, t in zip(outs, ts)) / K
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); it += 1
            if it % 50 == 0:
                el = (time.time() - t0) / 60
                rec = dict(it=it, K=K, loss=loss.item(), lr=sched.get_last_lr()[0], min=el)
                if it % 250 == 0:
                    rec['val_rmse'] = evaluate(model, pool, dstd, K, args.offset, gen, gval if gen is not None else None, gstarts)
                log.append(rec); print(json.dumps(rec), flush=True)
                if it % 500 == 0:
                    torch.save({'model': model.state_dict(), 'kw': MODEL_KW}, args.out)
                if el > args.minutes: break
        if (time.time() - t0) / 60 > args.minutes: break
    torch.save({'model': model.state_dict(), 'kw': MODEL_KW, 'iters': it}, args.out)
    json.dump(log, open(args.out.replace('.pt', '_log.json'), 'w'))
    print('done', it, (time.time() - t0) / 60, flush=True)


if __name__ == '__main__':
    main()
