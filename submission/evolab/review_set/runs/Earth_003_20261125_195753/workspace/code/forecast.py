"""Run 15-day (60 x 6 h) forecasts from ERA5 initial conditions for the 2020 test inits.
Systems: cascade (Short 1-20, Medium 21-40, Long 41-60), short_only (Short for all 60 steps, the single
autoregressive ML baseline), shortplus (compute-matched single model, if trained), and optionally
medium_only/long_only diagnostics.  Saves the 11 verification variables in physical units:
outputs/fc/{system}.npy  (n_init, 60, 11, 32, 64)."""
import os, sys, time, numpy as np, torch
from common import *
from gen import load
from train import step

torch.set_num_threads(int(os.environ.get('NT', 2)))
FC = f'{ROOT}/fc'; os.makedirs(FC, exist_ok=True)


def test_inputs():
    nm = Normaliser()
    X, d = load_year_physical(2020)
    if os.path.exists(f'{ERA}/2019.npy'):
        X0, d0 = load_year_physical(2019); X = np.concatenate([X0[-4:], X]); d = np.concatenate([d0[-4:], d])
    inits = np.load(f'{ROOT}/bench/inits.npy').astype('datetime64[h]')
    pos = {int(t): i for i, t in enumerate(d.astype(np.int64))}
    ti = inits.astype(np.int64)
    ok = np.array([(int(t) in pos) and (int(t) - 6 in pos) for t in ti])
    ii = np.array([pos[int(t)] for t in ti[ok]])
    Z = nm.norm(X)
    return nm, Z, X, d, inits, ok, ii


def run_system(models, Z, d, ii, nm, bs=16):
    """models: list of (model, n_steps). Returns physical verification vars (n, 60, 11, 32, 64)."""
    dstd = torch.from_numpy(nm.dsd).view(1, -1, 1, 1)
    out = np.zeros((len(ii), 60, 11, 32, 64), np.float32)
    with torch.no_grad():
        for b in range(0, len(ii), bs):
            i = ii[b:b + bs]
            a, c = torch.from_numpy(Z[i - 1]), torch.from_numpy(Z[i])
            k = 0
            for m, n in models:
                for _ in range(n):
                    f = torch.from_numpy(forcing(d[i] + np.timedelta64(6 * k, 'h')))
                    a, c = c, step(m, a, c, f, dstd)
                    out[b:b + bs, k] = nm.denorm(c.numpy())[:, VER_IDX]
                    k += 1
    return out


if __name__ == '__main__':
    systems = sys.argv[1:] or ['cascade', 'short_only']
    nm, Z, X, d, inits, ok, ii = test_inputs()
    np.save(f'{FC}/init_ok.npy', ok)
    print('test inits', ok.sum(), '/', len(ok), flush=True)
    for s in systems:
        if os.path.exists(f'{FC}/{s}.npy'): continue
        t = time.time()
        if s == 'cascade': ms = [(load('short'), 20), (load('medium'), 20), (load('long'), 20)]
        elif s == 'short_only': ms = [(load('short'), 60)]
        elif s == 'shortplus': ms = [(load('shortplus'), 60)]
        elif s == 'pretrain_only': ms = [(load('pre'), 60)]
        out = run_system(ms, Z, d, ii, nm)
        full = np.full((len(ok),) + out.shape[1:], np.nan, np.float32); full[ok] = out
        np.save(f'{FC}/{s}.npy', full); print(s, f'{time.time()-t:.0f}s', flush=True)
