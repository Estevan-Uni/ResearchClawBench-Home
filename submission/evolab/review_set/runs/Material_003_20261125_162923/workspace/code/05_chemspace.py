"""Step 5: curated combinatorial chemistry space.
Encode all unique acids/epoxides once with the dual encoders, fit a latent GP surrogate (calibrated Tg),
enumerate ~1e6 novel acid x epoxide combinations, screen them, and score a subsample + tails with the
virtual-MD + GP-calibration oracle."""
import sys, numpy as np, pandas as pd, pickle, time
sys.path.insert(0, 'code')
from latent_lib import *
from oracle_lib import Oracle

TAG = sys.argv[1] if len(sys.argv) > 1 else 'b01'
NCOMB = int(float(sys.argv[2])) if len(sys.argv) > 2 else 1_000_000
model, D = load(TAG)
calv = pd.read_csv('outputs/vitrimer_calibrated.csv')
idx = D['idx']
ycal = calv.tg_cal_best.values[idx]; ymd = D['tg']
acids = sorted(set(D['acid'])); epoxs = sorted(set(D['epox']))
ai = {s: i for i, s in enumerate(acids)}; ei = {s: i for i, s in enumerate(epoxs)}
t0 = time.time()
Za, LVa = encode(model, acids, 'acid'); Ze, LVe = encode(model, epoxs, 'epox')
print('encoded', len(acids), len(epoxs), 'in %.0fs' % (time.time() - t0))
pa = np.array([ai[s] for s in D['acid']]); pe = np.array([ei[s] for s in D['epox']])
Zlib = np.c_[Za[pa], Ze[pe]]
# --- latent GP surrogate: fit on 2500 random library pairs, test on 1500 held-out
rng = np.random.default_rng(1)
perm = rng.permutation(len(idx)); trn = perm[:2500]; tst = perm[2500:4000]
gp = TorchGP(Zlib[trn], ycal[trn])
mu_t, sd_t = gp.predict(Zlib[tst], noise=True)
r = ycal[tst] - mu_t
gpm = dict(MAE=float(np.abs(r).mean()), R2=float(1 - (r ** 2).sum() / ((ycal[tst] - ycal[tst].mean()) ** 2).sum()),
           cov95=float((np.abs(r) < 1.96 * sd_t).mean()), hyper=np.exp(gp.p.numpy()).tolist())
hd = head(model, Zlib[tst]); rh = ymd[tst] - hd
gpm['head_MAE_md'] = float(np.abs(rh).mean()); gpm['head_R2_md'] = float(1 - (rh ** 2).sum() / ((ymd[tst] - ymd[tst].mean()) ** 2).sum())
print('latent GP (calibrated Tg) held-out:', gpm)
# --- enumerate ~1e6 novel combinations
known = set(zip(pa.tolist(), pe.tolist()))
A = rng.integers(0, len(acids), int(NCOMB * 1.02)); E = rng.integers(0, len(epoxs), int(NCOMB * 1.02))
comb = np.unique(np.c_[A, E], axis=0)
mask = np.array([(a, e) not in known for a, e in comb])
comb = comb[mask][:NCOMB]
t0 = time.time()
mu_c, sd_c = gp.predict(np.c_[Za[comb[:, 0]], Ze[comb[:, 1]]], noise=True)
hd_c = np.concatenate([head(model, np.c_[Za[comb[b:b + 200000, 0]], Ze[comb[b:b + 200000, 1]]]) for b in range(0, len(comb), 200000)])
print('screened %d combos in %.0fs' % (len(comb), time.time() - t0))
mu_lib, sd_lib = gp.predict(Zlib)
print('library cal range %.1f-%.1f; space GP range %.1f-%.1f; frac>libmax %.4f frac<libmin %.4f' % (
    ycal.min(), ycal.max(), mu_c.min(), mu_c.max(), (mu_c > ycal.max()).mean(), (mu_c < ycal.min()).mean()))
# --- oracle on random subsample + tails
orc = Oracle()
order = np.argsort(mu_c)
sub = np.r_[rng.choice(len(comb), 2000, replace=False), order[:150], order[-150:]]
t0 = time.time()
res = orc.score_pairs([acids[i] for i in comb[sub, 0]], [epoxs[i] for i in comb[sub, 1]])
print('oracle scored %d in %.0fs; ok %d' % (len(sub), time.time() - t0, res['ok'].sum()))
grp = np.r_[np.zeros(2000), np.ones(150), 2 * np.ones(150)]
ores = pd.DataFrame(dict(acid=[acids[i] for i in comb[sub, 0]], epoxide=[epoxs[i] for i in comb[sub, 1]], group=grp,
                         gp_mu=mu_c[sub], gp_sd=sd_c[sub], head_md=hd_c[sub], oracle_md=res['md'], oracle_cal=res['cal'],
                         oracle_sigma=res['sigma_cal'], maxsim_lib=res['maxsim_lib']))
ores.to_csv('outputs/chemspace_oracle_subsample.csv', index=False)
for g, nm in [(0, 'random'), (1, 'GP-bottom150'), (2, 'GP-top150')]:
    q = ores[ores.group == g]
    print(nm, 'GP mu %.1f  oracle cal %.1f  corr %.2f' % (q.gp_mu.mean(), q.oracle_cal.mean(), np.corrcoef(q.gp_mu, q.oracle_cal)[0, 1]))
np.savez_compressed('outputs/chemspace.npz', comb=comb, mu=mu_c, sd=sd_c, head=hd_c, Za=Za, Ze=Ze, LVa=LVa, LVe=LVe,
                    Zlib=Zlib, ycal=ycal, ymd=ymd, mu_lib=mu_lib, trn=trn, tst=tst, mu_t=mu_t, sd_t=sd_t)
pickle.dump(dict(acids=acids, epoxs=epoxs, gp_metrics=gpm, gp_X=Zlib[trn], gp_y=ycal[trn], gp_p=gp.p.numpy()),
            open('outputs/chemspace_meta.pkl', 'wb'))
