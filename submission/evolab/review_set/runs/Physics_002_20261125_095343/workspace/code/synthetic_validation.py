"""Validate linXEB / logXEB / PT-MLE estimators + circuit-level uncertainty on synthetic data with known fidelity.
50 circuits x 20 shots, N=14 all-to-all random circuits (assumed gate set), depolarized sampler p_F = F p_U + (1-F)/2^N."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from circuits import random_circuit, statevector
from estimators import linear_xeb, log_xeb, pt_mle, GAMMA
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, D, NC, NS, TRIALS = 14, 12, 50, 20, 300
rng = np.random.default_rng(7)
P = [np.abs(statevector(N, random_circuit(N, D, 'all2all', 1000 + i)).ravel()) ** 2 for i in range(NC)]
# sanity: <ln x> for uniform and ideal
x_all = np.concatenate([p * 2 ** N for p in P])
print('collision 2^N sum p^2 =', np.mean([2 ** N * (p ** 2).sum() for p in P]), '(PT: 2)')
print('uniform sampler <ln x> =', np.mean(np.log(x_all)), ' expected', -GAMMA)
print('ideal sampler <ln x> =', np.mean([(p * np.log(p * 2 ** N)).sum() for p in P]), ' expected', 1 - GAMMA)
rows = []
for F in [0.1, 0.3, 0.5, 0.8, 1.0]:
    for t in range(TRIALS):
        est = {'linXEB': [], 'logXEB': [], 'PTMLE': []}
        for p in P:
            q = F * p + (1 - F) / 2 ** N
            s = rng.choice(2 ** N, size=NS, p=q / q.sum())
            u, c = np.unique(s, return_counts=True)
            x = p[u] * 2 ** N; c = c.astype(float)
            est['linXEB'].append(linear_xeb(c, x)[0]); est['logXEB'].append(log_xeb(c, x)[0]); est['PTMLE'].append(pt_mle(c, x)[0])
        for k, v in est.items():
            v = np.array(v); m = v.mean(); s = v.std(ddof=1) / np.sqrt(NC)
            rows.append(dict(F_true=F, method=k, trial=t, F_hat=m, sigma=s, cover1=abs(m - F) < s, cover2=abs(m - F) < 2 * s))
df = pd.DataFrame(rows)
summ = df.groupby(['F_true', 'method']).agg(mean_F=('F_hat', 'mean'), sd_F=('F_hat', 'std'), mean_sigma=('sigma', 'mean'),
                                           cov1=('cover1', 'mean'), cov2=('cover2', 'mean')).reset_index()
summ['bias'] = summ.mean_F - summ.F_true
print(summ.round(4).to_string())
summ.to_csv(os.path.join(ROOT, 'outputs/synthetic_validation.csv'), index=False)
df.to_csv(os.path.join(ROOT, 'outputs/synthetic_validation_trials.csv'), index=False)
