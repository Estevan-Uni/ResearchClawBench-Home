"""Step 8: latent traversal analysis. Spherical interpolation (per monomer block) between a low-Tg and a
high-Tg library vitrimer; decode along the path; score with head, latent GP and the virtual-MD+GP oracle."""
import sys, numpy as np, pandas as pd, pickle
sys.path.insert(0, 'code')
from latent_lib import *
from oracle_lib import Oracle
from scipy.stats import spearmanr

TAG = sys.argv[1] if len(sys.argv) > 1 else 'b01'
model, D = load(TAG)
C = np.load('outputs/chemspace.npz'); meta = pickle.load(open('outputs/chemspace_meta.pkl', 'rb'))
Zlib, ycal = C['Zlib'], C['ycal']; d = Zlib.shape[1] // 2
gp = TorchGP(meta['gp_X'], meta['gp_y'], iters=0, init=meta['gp_p'])
orc = Oracle(); rng = np.random.default_rng(3); torch.manual_seed(3)


def slerp(a, b, t):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    om = np.arccos(np.clip(np.dot(a / na, b / nb), -1, 1))
    if om < 1e-6: return (1 - t) * a + t * b
    v = (np.sin((1 - t) * om) * a / na + np.sin(t * om) * b / nb) / np.sin(om)
    return v * ((1 - t) * na + t * nb)


order = np.argsort(ycal)
lows = order[[20, 60, 120]]; highs = order[[-20, -60, -120]]
rows = []
NT = 13
for k, (lo, hi) in enumerate(zip(lows, highs)):
    ts = np.linspace(0, 1, NT)
    Z = np.stack([np.r_[slerp(Zlib[lo, :d], Zlib[hi, :d], t), slerp(Zlib[lo, d:], Zlib[hi, d:], t)] for t in ts])
    mu, sd = gp.predict(Z); hd = head(model, Z)
    for j, t in enumerate(ts):
        # decode: greedy first, then up to 10 stochastic samples until a functional-group-valid pair is found
        cand = [(decode(model, D, Z[j:j + 1, :d], 'acid')[0], decode(model, D, Z[j:j + 1, d:], 'epox')[0])]
        for _ in range(10):
            cand.append((decode(model, D, Z[j:j + 1, :d], 'acid', temp=0.6)[0], decode(model, D, Z[j:j + 1, d:], 'epox', temp=0.6)[0]))
        ca = next((c[0] for c in cand if fg_check(c[0], 'acid')), None); ce = next((c[1] for c in cand if fg_check(c[1], 'epox')), None)
        if t == 0: ca, ce = D['acid'][lo], D['epox'][lo]
        if t == 1: ca, ce = D['acid'][hi], D['epox'][hi]
        oc = np.nan; om = np.nan
        if ca and ce:
            r = orc.score_pairs([ca], [ce]); oc = r['cal'][0]; om = r['md'][0]
        rows.append(dict(path=k, t=t, gp_mu=mu[j], gp_sd=sd[j], head_md=hd[j], acid=ca, epoxide=ce, oracle_md=om, oracle_cal=oc,
                         lib_lo=ycal[lo], lib_hi=ycal[hi]))
df = pd.DataFrame(rows); df.to_csv('outputs/latent_traversal.csv', index=False)
for k in range(3):
    q = df[df.path == k].dropna(subset=['oracle_cal'])
    print('path', k, 'Spearman(t, gp_mu)=%.2f  Spearman(t, head)=%.2f  Spearman(t, oracle)=%.2f  valid %d/%d' % (
        spearmanr(df[df.path == k].t, df[df.path == k].gp_mu)[0], spearmanr(df[df.path == k].t, df[df.path == k].head_md)[0],
        spearmanr(q.t, q.oracle_cal)[0], len(q), NT))
