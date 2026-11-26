"""MPS approximability runs at N=56 (depth scan) and d=12 (N scan). Usage: python classical_mps.py {depth|nscan}"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from circuits import random_circuit
from mps import run_mps
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
job = sys.argv[1]
rows = []
if job == 'depth':
    plan = [(g, chi, s) for g in ['1d', '2d', 'all2all'] for chi in [16, 32, 64] for s in ([0, 1] if chi == 64 else [0])]
    plan += [('1d', 128, 0), ('2d', 128, 0)]
    for g, chi, s in plan:
        t = time.time(); lf, _ = run_mps(56, random_circuit(56, 24, g, 9000 + s), chi)
        for d, v in enumerate(lf, 1): rows.append(dict(geometry=g, N=56, chi=chi, seed=s, d=d, logF=v))
        print(g, chi, s, f'{time.time()-t:.1f}s', lf[[7, 11, 19, 23]], flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'outputs/mps_N56_depth.csv'), index=False)
else:
    for g in ['all2all', '2d', '1d']:
        for N in [16, 24, 32, 40, 48, 56]:
            for chi in [32, 64]:
                t = time.time(); lf, _ = run_mps(N, random_circuit(N, 12, g, 7000 + N), chi)
                rows.append(dict(geometry=g, N=N, chi=chi, d=12, logF=lf[-1], logF_d8=lf[7]))
                print(g, N, chi, f'{time.time()-t:.1f}s', lf[-1], flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'outputs/mps_d12_Nscan.csv'), index=False)
print('done')
