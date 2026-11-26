"""Per-instance fidelity estimators (linear XEB, log-XEB, PT-MLE, MB) and circuit-level bootstrap pooling.
Writes outputs/per_instance_estimates.csv and outputs/pooled_estimates.csv"""
import os, pickle, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import beta
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMMA = 0.5772156649015329
RNG = np.random.default_rng(12345)

def xeb_terms(counts, probs, N):
    keys = list(counts)
    c = np.array([counts[k] for k in keys], float)
    x = np.array([probs[k] for k in keys]) * 2.0 ** N
    return c, x

def linear_xeb(c, x):
    n = c.sum(); m = (c * x).sum() / n
    sd = np.sqrt((c * (x - m) ** 2).sum() / (n - 1))
    return m - 1, sd / np.sqrt(n)

def log_xeb(c, x):
    n = c.sum(); lx = np.log(x); m = (c * lx).sum() / n
    sd = np.sqrt((c * (lx - m) ** 2).sum() / (n - 1))
    return GAMMA + m, sd / np.sqrt(n)

def pt_mle(c, x):
    f = lambda F: -(c * np.log(np.clip(1 - F + F * x, 1e-300, None))).sum()
    lo = max(-1.0, -1 / (x.max() - 1) + 1e-9) if x.max() > 1 else -1.0
    hi = min(3.0, 1 / (1 - x.min()) - 1e-9) if x.min() < 1 else 3.0
    res = minimize_scalar(f, bounds=(lo, hi), method='bounded')
    F = res.x; I = (c * (x - 1) ** 2 / (1 - F + F * x) ** 2).sum()
    return F, 1 / np.sqrt(I)

def mb(counts, target, N):
    n = sum(counts.values()); k = counts.get(target, 0)
    S = k / n; F = (S - 2.0 ** -N) / (1 - 2.0 ** -N)
    lo, hi = beta.ppf([0.16, 0.84], k + 0.5, n - k + 0.5)
    return F, np.sqrt(S * (1 - S) / n), lo, hi, k, n

def boot_mean(v, B=4000, rng=RNG):
    v = np.asarray(v); idx = rng.integers(0, len(v), (B, len(v)))
    bm = v[idx].mean(1)
    return v.mean(), bm.std(ddof=1), np.percentile(bm, [16, 84])

if __name__ == '__main__':
    recs = pickle.load(open(os.path.join(ROOT, 'outputs/data_cache.pkl'), 'rb'))
    rows = []
    for rec in recs:
        base = dict(family=rec['family'], N=rec['N'], d=rec['d'], r=rec['r'])
        if rec['type'] == 'XEB' and 'ideal_probs' in rec:
            c, x = xeb_terms(rec['counts'], rec['ideal_probs'], rec['N'])
            n = c.sum()
            F, s = linear_xeb(c, x); rows.append(dict(base, method='linXEB', F=F, sigma=s, shots=n))
            F, s = log_xeb(c, x); rows.append(dict(base, method='logXEB', F=F, sigma=s, shots=n))
            F, s = pt_mle(c, x); rows.append(dict(base, method='PTMLE', F=F, sigma=s, shots=n))
        elif rec['type'] == 'MB':
            F, s, lo, hi, k, n = mb(rec['counts'], rec['ideal_bitstring'], rec['N'])
            rows.append(dict(base, method='MB', F=F, sigma=s, shots=n, hits=k, ci68_lo=lo, ci68_hi=hi))
        elif rec['type'] == 'Transport_1QRB':
            t = np.array(rec['ideal_bitstring'])
            surv = []; n = 0
            for k, v in rec['counts'].items():
                surv += [np.mean(np.array(k) == t)] * v; n += v
            surv = np.array(surv)
            full = rec['counts'].get(rec['ideal_bitstring'], 0) / n
            rows.append(dict(base, method='TRB_bitsurv', F=surv.mean(), sigma=surv.std(ddof=1) / np.sqrt(n), shots=n, hits=full))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, 'outputs/per_instance_estimates.csv'), index=False)
    print(df.groupby(['method']).size())
    # pooled per (family,N,d,method) with circuit-level bootstrap
    pooled = []
    for (fam, N, d, meth), g in df[df.method != 'TRB_bitsurv'].groupby(['family', 'N', 'd', 'method']):
        m, sb, (lo, hi) = boot_mean(g.F.values)
        n = g.shots.mean()
        # shot-noise-only prediction of between-circuit variance
        if meth == 'linXEB': pred = (1 + 2 * m - m ** 2) / n
        elif meth == 'logXEB': pred = (np.pi ** 2 / 6 - m ** 2) / n
        elif meth == 'MB':
            S = m; pred = S * (1 - S) / n
        else: pred = np.mean(g.sigma ** 2)
        pooled.append(dict(family=fam, N=N, d=d, method=meth, n_circ=len(g), shots=n, F=m, sigma_boot=sb,
                           ci68_lo=lo, ci68_hi=hi, sigma_naive=g.F.std(ddof=1) / np.sqrt(len(g)),
                           var_between=g.F.var(ddof=1), var_shot_pred=pred, var_ratio=g.F.var(ddof=1) / pred))
    P = pd.DataFrame(pooled)
    P.to_csv(os.path.join(ROOT, 'outputs/pooled_estimates.csv'), index=False)
    pd.set_option('display.width', 250)
    print(P.round(4).to_string())

def pooled_pt_mle(c_list, x_list):
    """Single PT maximum-likelihood fit over all shots of all circuits (unbiased alternative to per-circuit MLE)."""
    c = np.concatenate(c_list); x = np.concatenate(x_list)
    return pt_mle(c, x)
