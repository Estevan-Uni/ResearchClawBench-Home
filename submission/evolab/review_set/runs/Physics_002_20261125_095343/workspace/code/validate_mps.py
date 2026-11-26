import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from circuits import random_circuit, statevector
from mps import run_mps
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for geo in ['all2all', '2d', '1d']:
    for N in [12, 16]:
        for chi in [4, 8, 16, 32]:
            for seed in range(2):
                for d in [4, 8, 12]:
                    L = random_circuit(N, d, geo, 500 + seed)
                    logF, m = run_mps(N, L, chi)
                    ex = statevector(N, L).ravel(); ap = m.to_vector().ravel()
                    ov = abs(np.vdot(ex, ap)) ** 2
                    rows.append(dict(geometry=geo, N=N, chi=chi, seed=seed, d=d, F_est=np.exp(logF[-1]), F_exact=ov))
df = pd.DataFrame(rows); df.to_csv(os.path.join(ROOT, 'outputs/mps_validation.csv'), index=False)
print(df.groupby(['geometry', 'N', 'chi', 'd'])[['F_est', 'F_exact']].mean().round(4).to_string())
