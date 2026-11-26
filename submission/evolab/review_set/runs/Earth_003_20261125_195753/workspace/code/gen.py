"""Generate the cascade's own forecast states used to fine-tune the next stage.
  python gen.py short  N      -> outputs/gen/short_day5.npz   (states at steps 19,20 from Short)
  python gen.py medium        -> outputs/gen/medium_day10.npz (Medium continues the Short states 20 more steps)
"""
import sys, os, time, numpy as np, torch
from common import *
from model import UTransformer
from train import Pool, step, CK

torch.set_num_threads(int(os.environ.get('NT', 2)))
G = f'{ROOT}/gen'; os.makedirs(G, exist_ok=True)


def load(name):
    ck = torch.load(f'{CK}/{name}.pt', map_location='cpu')
    m = UTransformer(**ck['kw']); m.load_state_dict(ck['model']); m.eval(); return m


def run(model, pool, idx, x0, x1, offset, nsteps, dstd, bs=16):
    out0 = np.zeros((len(idx), 70, 32, 64), np.float32); out1 = np.zeros_like(out0)
    with torch.no_grad():
        for b in range(0, len(idx), bs):
            sl = slice(b, b + bs); i = idx[sl]
            a, c = torch.from_numpy(x0[sl]), torch.from_numpy(x1[sl])
            for k in range(nsteps):
                f = torch.from_numpy(pool.Fo[i + offset + k])
                a, c = c, step(model, a, c, f, dstd)
            out0[sl], out1[sl] = a.numpy(), c.numpy()
    return out0, out1


def main():
    which = sys.argv[1]
    pool = Pool(holdout_days=45)
    dstd = torch.from_numpy(Normaliser().dsd).view(1, -1, 1, 1)
    t = time.time()
    if which == 'short':
        N = int(sys.argv[2])
        rng = np.random.default_rng(7)
        st = pool.valid_starts(40 + 8 + 1)
        idx = np.sort(rng.choice(st, N, replace=False))
        a, c = run(load('short'), pool, idx, pool.X[idx - 1], pool.X[idx], 0, 20, dstd)
        off = 20; fn = f'{G}/short_day5.npz'
    else:
        g = np.load(f'{G}/short_day5.npz')
        dd = {int(d): i for i, d in enumerate(pool.D.astype(np.int64))}
        idx = np.array([dd[int(d)] for d in g['start_dates']])
        a, c = run(load('medium'), pool, idx, g['states'][:, 0].astype(np.float32), g['states'][:, 1].astype(np.float32), 20, 20, dstd)
        off = 40; fn = f'{G}/medium_day10.npz'
    truth = pool.X[idx + off]
    mae = float(np.abs(c - truth).mean()); clim_mae = float(np.abs(truth).mean())
    print(which, 'MAE(normalised) vs ERA5 at step', off, mae, ' |anomaly| scale', clim_mae, f'{time.time()-t:.0f}s', flush=True)
    np.savez(fn, states=np.stack([a, c], 1).astype(np.float16), start_dates=pool.D[idx].astype(np.int64), mae=mae)


if __name__ == '__main__':
    main()
