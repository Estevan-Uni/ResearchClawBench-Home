"""Step 4: fit the MD-Tg surrogate ('virtual MD') on repeat-unit Tanimoto kernel; 5-fold CV."""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, 'code')
from oracle_lib import bitfp, tanimoto
from scipy.optimize import minimize_scalar

vit = pd.read_csv('outputs/vitrimer_repeat_units.csv')
X = bitfp(vit.repeat_unit); y = vit.tg.values; n = len(y)
K = tanimoto(X, X)
ym = y.mean()
# choose signal/noise ratio by 5-fold CV on a grid (kernel ridge == GP mean)
rng = np.random.default_rng(0); perm = rng.permutation(n); folds = np.array_split(perm, 5)
res = {}
for lam in [0.02, 0.05, 0.1, 0.2, 0.4]:
    pred = np.zeros(n)
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        a = np.linalg.solve(K[np.ix_(tr, tr)] + lam * np.eye(len(tr)), y[tr] - ym)
        pred[te] = ym + K[np.ix_(te, tr)] @ a
    res[lam] = (np.abs(pred - y).mean(), np.sqrt(((pred - y) ** 2).mean()), 1 - ((pred - y) ** 2).sum() / ((y - ym) ** 2).sum(), pred.copy())
    print(lam, res[lam][:3])
lam = min(res, key=lambda k: res[k][0])
pd.DataFrame(dict(tg_md=y, oracle_cv=res[lam][3])).to_csv('outputs/oracle_cv_pred.csv', index=False)
s = 1000.0  # signal variance (K^2) -- scale irrelevant for mean
alpha = np.linalg.solve(K + lam * np.eye(n), y - ym)
pickle.dump(dict(Xtr=X, alpha=alpha, m=ym, s=1.0, lam=lam, cv=dict(MAE=res[lam][0], RMSE=res[lam][1], R2=res[lam][2])),
            open('outputs/oracle.pkl', 'wb'))
print('chosen lam', lam, res[lam][:3])
