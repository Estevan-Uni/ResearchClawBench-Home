"""Step 6: Bayesian optimisation in the dual-VAE latent space with virtual-MD + GP-calibration oracle.
Objectives: maximise Tg, minimise Tg, hit 373 K exactly, hit near-room-temperature 300 K.
Each round: fit latent GP -> optimise acquisition (EI or P(|Tg-T*|<5K)) from many seeds with a latent
trust-region penalty -> decode (greedy + stochastic) -> functional-group filter -> re-encode -> oracle -> append."""
import sys, numpy as np, pandas as pd, pickle, time, torch, math
sys.path.insert(0, 'code')
from latent_lib import *
from oracle_lib import Oracle
from torch.distributions import Normal

TAG = sys.argv[1] if len(sys.argv) > 1 else 'b01'
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
Q = 16; NSAMP = 4; TEMP = 0.7; NSEED = 96; STEPS = 60
torch.manual_seed(0); rng = np.random.default_rng(0)
model, D = load(TAG)
C = np.load('outputs/chemspace.npz'); meta = pickle.load(open('outputs/chemspace_meta.pkl', 'rb'))
Zlib, ycal = C['Zlib'], C['ycal']
lib_acids, lib_epox = set(D['acid']), set(D['epox'])
LIBMAX, LIBMIN = ycal.max(), ycal.min()
_sub = Zlib[np.random.default_rng(5).choice(len(Zlib), 1000, replace=False)]
_dd = np.linalg.norm(_sub[:, None, :Zlib.shape[1] // 2] - _sub[None, :, :Zlib.shape[1] // 2], axis=-1); np.fill_diagonal(_dd, np.inf)
DELTA = float(np.median(_dd.min(1)))  # median nearest-neighbour distance between library acid latents
print('trust-region radius per monomer block: %.2f' % DELTA)
orc = Oracle()
d = Zlib.shape[1] // 2
rad_a = np.quantile(np.linalg.norm(Zlib[:, :d], axis=1), 0.99); rad_e = np.quantile(np.linalg.norm(Zlib[:, d:], axis=1), 0.99)
N01 = Normal(0., 1.)

OBJ = {'max': dict(kind='max'), 'min': dict(kind='min'), 'target373': dict(kind='target', T=373.0),
       'target300': dict(kind='target', T=300.0)}
if len(sys.argv) > 3:
    OBJ = {k: OBJ[k] for k in sys.argv[3].split(',')}
OUT = sys.argv[4] if len(sys.argv) > 4 else 'outputs/bo_designs.csv'
MODE = sys.argv[5] if len(sys.argv) > 5 else 'bo'   # 'bo' or 'rand' (random trust-region perturbation baseline, no acquisition)


def score(y, o):
    if o['kind'] == 'max': return y
    if o['kind'] == 'min': return -y
    return -np.abs(y - o['T'])


def acq(mu, sd, o, best):
    if o['kind'] in ('max', 'min'):
        f = mu if o['kind'] == 'max' else -mu
        zz = (f - best) / sd
        return sd * (zz * N01.cdf(zz) + torch.exp(N01.log_prob(zz)))
    T = o['T']
    return N01.cdf((T + 5 - mu) / sd) - N01.cdf((T - 5 - mu) / sd) + 1e-6 * torch.exp(-((mu - T) / 50) ** 2)


def penalty(z):
    na = torch.linalg.norm(z[:, :d], dim=1); ne = torch.linalg.norm(z[:, d:], dim=1)
    return 5.0 * (torch.relu(na - rad_a) ** 2 + torch.relu(ne - rad_e) ** 2)


all_rows = []
base_X = np.r_[Zlib[C['trn']], Zlib[np.argsort(ycal)[:100]], Zlib[np.argsort(ycal)[-100:]]]
base_y = np.r_[ycal[C['trn']], np.sort(ycal)[:100], np.sort(ycal)[-100:]]
for oname, o in OBJ.items():
    X, Y = base_X.copy(), base_y.copy()
    best_lib = score(ycal, o).max()
    best = best_lib; seen = set()
    t0 = time.time()
    for rnd in range(1, ROUNDS + 1):
        gp = TorchGP(X, Y, iters=0, init=meta['gp_p'])
        # seeds: top library points under current objective + random library points + previous designs
        mu_lib = gp.predict(Zlib)[0]
        top = np.argsort(-score(mu_lib, o))[:NSEED // 2]
        rnd_idx = rng.choice(len(Zlib), NSEED // 2, replace=False)
        z = torch.tensor(np.r_[Zlib[top], Zlib[rnd_idx]] + 0.05 * rng.standard_normal((NSEED, 2 * d)), dtype=torch.float64, requires_grad=True)
        z0 = z.detach().clone()
        opt = torch.optim.Adam([z], lr=0.05)
        if MODE == 'rand':
            with torch.no_grad():
                u = torch.randn_like(z); 
                for sl in (slice(0, d), slice(d, 2 * d)):
                    u[:, sl] = u[:, sl] / torch.linalg.norm(u[:, sl], dim=1, keepdim=True)
                z += DELTA * u * torch.rand(NSEED, 1, dtype=torch.float64)
        for s in range(STEPS if MODE == 'bo' else 0):
            m, sdv = gp.predict_t(z)
            a = acq(m, sdv, o, best)
            loss = -(torch.log(a + 1e-12) - penalty(z)).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():  # trust region: each block may move at most DELTA from its seed
                for sl in (slice(0, d), slice(d, 2 * d)):
                    dz = z[:, sl] - z0[:, sl]; nrm = torch.linalg.norm(dz, dim=1, keepdim=True)
                    z[:, sl] = z0[:, sl] + dz * torch.clamp(DELTA / (nrm + 1e-9), max=1.0)
        with torch.no_grad():
            m, sdv = gp.predict_t(z); a = (torch.log(acq(m, sdv, o, best) + 1e-12) - penalty(z)).numpy()
        sel = np.argsort(-a)[:Q] if MODE == 'bo' else rng.choice(NSEED, Q, replace=False)
        zsel = z.detach().numpy()[sel]
        acq_mu = m.numpy()[sel]
        # decode: greedy + stochastic samples for each monomer
        za = np.repeat(zsel[:, :d], 1 + NSAMP, 0); ze = np.repeat(zsel[:, d:], 1 + NSAMP, 0)
        sa = decode(model, D, zsel[:, :d], 'acid') ; se = decode(model, D, zsel[:, d:], 'epox')
        pairs = [(sa[i], se[i], i, 'greedy') for i in range(Q)]
        for k in range(NSAMP):
            sa2 = decode(model, D, zsel[:, :d], 'acid', temp=TEMP); se2 = decode(model, D, zsel[:, d:], 'epox', temp=TEMP)
            pairs += [(sa2[i], se2[i], i, 'sample') for i in range(Q)]
        n_dec = len(pairs)
        good = [(a_, e_, i, md) for a_, e_, i, md in pairs if fg_check(a_, 'acid') and fg_check(e_, 'epox')]
        good_u = []
        for p in good:
            if (p[0], p[1]) not in seen:
                seen.add((p[0], p[1])); good_u.append(p)
        if good_u:
            res = orc.score_pairs([p[0] for p in good_u], [p[1] for p in good_u])
            ok = res['ok']
            ga = [p[0] for p, k in zip(good_u, ok) if k]; ge = [p[1] for p, k in zip(good_u, ok) if k]
            if ga:
                zr = np.c_[encode(model, ga, 'acid')[0], encode(model, ge, 'epox')[0]]
                mu_re, sd_re = gp.predict(zr)
                ycal_o = res['cal'][ok]
                X = np.r_[X, zr]; Y = np.r_[Y, ycal_o]
                for j, (p, ) in enumerate(zip([p for p, k in zip(good_u, ok) if k])):
                    all_rows.append(dict(objective=oname, round=rnd, acid=p[0], epoxide=p[1], mode=p[3],
                                         acq_mu=acq_mu[p[2]], reenc_mu=mu_re[j], reenc_sd=sd_re[j],
                                         oracle_md=res['md'][ok][j], oracle_cal=ycal_o[j], oracle_sigma=res['sigma_cal'][ok][j],
                                         maxsim_lib=res['maxsim_lib'][ok][j], novel_acid=p[0] not in lib_acids,
                                         novel_epox=p[1] not in lib_epox, ru=np.array(res['ru'], dtype=object)[ok][j],
                                         n_decoded=n_dec, n_fg=len(good)))
                best = max(best, score(ycal_o, o).max())
        df = pd.DataFrame(all_rows)
        dr = df[(df.objective == oname) & (df['round'] == rnd)] if len(df) else df
        print(f'[{oname}] round {rnd}: decoded {n_dec}, fg-valid {len(good)}, new {len(good_u)}, '
              f'best-so-far {best:.1f} (library {best_lib:.1f}); round max {dr.oracle_cal.max() if len(dr) else float("nan"):.1f} '
              f'min {dr.oracle_cal.min() if len(dr) else float("nan"):.1f}  [{time.time() - t0:.0f}s]', flush=True)
        df.to_csv(OUT, index=False)
print('LIBMAX %.1f LIBMIN %.1f' % (LIBMAX, LIBMIN))
