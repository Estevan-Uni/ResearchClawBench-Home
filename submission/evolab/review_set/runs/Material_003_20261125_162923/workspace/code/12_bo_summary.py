"""Summarise BO vs random trust-region baseline vs 1e6 screen; extrapolation statistics."""
import numpy as np, pandas as pd
C = np.load('outputs/chemspace.npz'); ycal = C['ycal']; LMAX, LMIN = ycal.max(), ycal.min()
O = pd.read_csv('outputs/chemspace_oracle_subsample.csv')
rows = []
for f, meth in [('outputs/bo_designs.csv', 'latent BO'), ('outputs/bo_rand.csv', 'random trust-region')]:
    B = pd.read_csv(f)
    for o in ['max', 'min', 'target373', 'target300']:
        q = B[B.objective == o]; T = {'target373': 373, 'target300': 300}.get(o)
        cons = q[(q.reenc_mu - q.oracle_cal).abs() < 25]
        r = dict(method=meth, objective=o, n_valid_designs=len(q), fg_yield=q.drop_duplicates('round').n_fg.sum() / q.drop_duplicates('round').n_decoded.sum(),
                 novel_both=float((q.novel_acid & q.novel_epox).mean()), mean_maxsim_lib=q.maxsim_lib.mean(), consistency_yield=len(cons) / max(len(q), 1))
        if o == 'max':
            r.update(best=q.oracle_cal.max(), n_beyond_library=int((q.oracle_cal > LMAX).sum()), frac_beyond=float((q.oracle_cal > LMAX).mean()),
                     top10_mean=q.oracle_cal.nlargest(10).mean(), best_sigma=q.loc[q.oracle_cal.idxmax(), 'oracle_sigma'])
        elif o == 'min':
            r.update(best=q.oracle_cal.min(), n_beyond_library=int((q.oracle_cal < LMIN).sum()), frac_beyond=float((q.oracle_cal < LMIN).mean()),
                     top10_mean=q.oracle_cal.nsmallest(10).mean(), best_sigma=q.loc[q.oracle_cal.idxmin(), 'oracle_sigma'])
        else:
            e = (q.oracle_cal - T).abs()
            r.update(best=e.min(), hits_5K=int((e < 5).sum()), hit_rate_5K=float((e < 5).mean()), hits_10K=int((e < 10).sum()), top10_mean=e.nsmallest(10).mean())
        rows.append(r)
# screening baseline: GP-top/bottom 150 of 1e6 + random subsample for targets
for o in ['max', 'min', 'target373', 'target300']:
    if o == 'max': q = O[O.group == 2]; rows.append(dict(method='10^6 screen (GP top-150 -> oracle)', objective=o, n_valid_designs=150, best=q.oracle_cal.max(), n_beyond_library=int((q.oracle_cal > LMAX).sum()), frac_beyond=float((q.oracle_cal > LMAX).mean()), top10_mean=q.oracle_cal.nlargest(10).mean(), novel_both=0.0))
    elif o == 'min': q = O[O.group == 1]; rows.append(dict(method='10^6 screen (GP bottom-150 -> oracle)', objective=o, n_valid_designs=150, best=q.oracle_cal.min(), n_beyond_library=int((q.oracle_cal < LMIN).sum()), frac_beyond=float((q.oracle_cal < LMIN).mean()), top10_mean=q.oracle_cal.nsmallest(10).mean(), novel_both=0.0))
    else:
        T = 373 if o == 'target373' else 300; q = O[O.group == 0]; e = (q.oracle_cal - T).abs()
        rows.append(dict(method='random 2,000 of 10^6 space', objective=o, n_valid_designs=len(q), best=e.min(), hits_5K=int((e < 5).sum()), hit_rate_5K=float((e < 5).mean()), hits_10K=int((e < 10).sum()), top10_mean=e.nsmallest(10).mean(), novel_both=0.0))
    q = ycal
    if o.startswith('target'):
        T = 373 if o == 'target373' else 300; e = np.abs(ycal - T)
        rows.append(dict(method='library (8,424)', objective=o, n_valid_designs=len(ycal), best=e.min(), hits_5K=int((e < 5).sum()), hit_rate_5K=float((e < 5).mean()), hits_10K=int((e < 10).sum())))
    else:
        rows.append(dict(method='library (8,424)', objective=o, n_valid_designs=len(ycal), best=ycal.max() if o == 'max' else ycal.min(),
                         top10_mean=np.sort(ycal)[-10:].mean() if o == 'max' else np.sort(ycal)[:10].mean()))
S = pd.DataFrame(rows); S.to_csv('outputs/bo_summary.csv', index=False)
pd.set_option('display.width', 250)
print(S.round(3).to_string())
print('library min/max', LMIN, LMAX)
