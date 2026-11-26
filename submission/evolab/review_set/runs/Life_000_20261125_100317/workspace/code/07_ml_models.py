"""Surrogate benchmarking, prospective (round-by-round) validation, exact Shapley attribution, embedding."""
import sys; sys.path.insert(0, 'code')
from style import *
import pandas as pd, numpy as np, itertools, warnings
from math import factorial
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedKFold
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance
warnings.filterwarnings('ignore')
A = pd.read_csv('outputs/all_measured_formulations.csv')
X = A[MON].values; y = A.Fa_glass.values; ly = np.log(y)
def models():
    return {
     'GP (Matérn-ARD)': GaussianProcessRegressor(C(1.0) * Matern(length_scale=np.ones(6) * 0.3, nu=2.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(0.1, (1e-4, 1)), normalize_y=True, n_restarts_optimizer=2, random_state=0),
     'Random forest': RandomForestRegressor(500, min_samples_leaf=2, random_state=0, n_jobs=-1),
     'Extra trees': ExtraTreesRegressor(500, min_samples_leaf=2, random_state=0, n_jobs=-1),
     'Gradient boosting': GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0),
     'kNN (k=5)': KNeighborsRegressor(5, weights='distance'),
     'Linear mixture (ridge)': RidgeCV(alphas=np.logspace(-4, 2, 20)),
     'Quadratic mixture (ridge)': make_pipeline(PolynomialFeatures(2, include_bias=False), RidgeCV(alphas=np.logspace(-4, 2, 20))),
    }
def score(yt, yp):
    return dict(spearman=stats.spearmanr(yt, yp)[0], R2_log=1 - np.sum((np.log(yt) - np.log(yp)) ** 2) / np.sum((np.log(yt) - np.log(yt).mean()) ** 2),
                MAE_kPa=np.mean(np.abs(yt - yp)))
rows = []
for dsname, mask in [('initial 180 DM', (A.source == 'DM').values), ('all 351 measured', np.ones(len(A), bool))]:
    Xd, lyd = X[mask], ly[mask]
    for name, mdl in models().items():
        preds = []
        for k, (tr, te) in enumerate(RepeatedKFold(n_splits=5, n_repeats=3, random_state=1).split(Xd)):
            m = models()[name]; m.fit(Xd[tr], lyd[tr]); p = np.exp(m.predict(Xd[te]))
            s = score(np.exp(lyd[te]), p); s.update(dataset=dsname, model=name, fold=k); rows.append(s)
        print(dsname, name, 'done')
cv = pd.DataFrame(rows); cvs = cv.groupby(['dataset', 'model'])[['spearman', 'R2_log', 'MAE_kPa']].agg(['mean', 'std']).round(3)
cvs.to_csv('outputs/ml_cv_benchmark.csv'); print(cvs)

# ---------- prospective validation ----------
pro = []; prov = []
for r in [1, 2, 3]:
    tr = A['round'] < r; te = A['round'] == r
    for name in ['GP (Matérn-ARD)', 'Random forest', 'Gradient boosting', 'Quadratic mixture (ridge)']:
        m = models()[name]; m.fit(X[tr], ly[tr]); p = np.exp(m.predict(X[te]))
        s = score(y[te], p); s.update(round=r, model=name, n_train=int(tr.sum()), n_test=int(te.sum()),
                                     max_train=y[tr].max(), max_obs=y[te].max(), max_pred=p.max(),
                                     frac_obs_above_train_max=float((y[te] > y[tr].max()).mean()))
        pro.append(s)
        prov.append(pd.DataFrame(dict(round=r, model=name, obs=y[te], pred=p)))
pro = pd.DataFrame(pro); pro.to_csv('outputs/ml_prospective_validation.csv', index=False); print(pro.round(3).to_string())
prov = pd.concat(prov)
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for j, r in enumerate([1, 2, 3]):
    ax = axs[j]
    for k, name in enumerate(['GP (Matérn-ARD)', 'Random forest']):
        s = prov[(prov['round'] == r) & (prov.model == name)]
        rr = pro[(pro['round'] == r) & (pro.model == name)].iloc[0]
        ax.scatter(s.pred, s.obs, s=16, color=CAT[k], alpha=0.75, edgecolor='white', lw=0.3, label=f"{name}: ρ={rr.spearman:.2f}")
    mx = A[A['round'] < r].Fa_glass.max()
    ax.axhline(mx, color=INK2, ls='--', lw=0.8); ax.text(375, mx + 5, f'training max {mx:.0f}', ha='right', fontsize=7.5, color=INK2)
    ax.plot([0, 380], [0, 380], ':', color=INK2, lw=1); ax.set_xlim(0, 380); ax.set_ylim(0, 380)
    ax.set_xlabel('predicted $F_a$ (kPa), trained on rounds < r'); ax.set_ylabel('measured $F_a$ (kPa)')
    ax.set_title(f'{"abc"[j]}  Round {r}: train n={int((A["round"] < r).sum())}, test n={int((A["round"] == r).sum())}', loc='left', fontsize=9.5)
    ax.legend(fontsize=7, loc='lower right')
fig.tight_layout(); save(fig, 'fig09_prospective_validation')

# ---------- exact interventional Shapley values (6 features -> 64 coalitions) ----------
rf = RandomForestRegressor(500, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(X, ly)
rng = np.random.default_rng(0); bg = X[rng.choice(len(X), 80, replace=False)]
coal = list(itertools.product([0, 1], repeat=6))
V = np.zeros((len(X), len(coal)))
for ci, c in enumerate(coal):
    c = np.array(c, bool)
    Z = np.repeat(bg[None], len(X), 0).copy()  # (n, nbg, 6)
    Z[:, :, c] = X[:, None, c]
    V[:, ci] = rf.predict(Z.reshape(-1, 6)).reshape(len(X), len(bg)).mean(1)
idx = {c: i for i, c in enumerate(coal)}
phi = np.zeros((len(X), 6))
for j in range(6):
    for c in coal:
        if c[j]: continue
        s = sum(c); w = factorial(s) * factorial(6 - s - 1) / factorial(6)
        c1 = list(c); c1[j] = 1
        phi[:, j] += w * (V[:, idx[tuple(c1)]] - V[:, idx[c]])
check = np.abs(phi.sum(1) + V[:, idx[(0,) * 6]] - V[:, idx[(1,) * 6]]).max()
print('Shapley efficiency check (should be ~0):', check)
sh = pd.DataFrame(phi, columns=MON); sh.to_csv('outputs/shapley_values_rf_logFa.csv', index=False)
imp = sh.abs().mean().sort_values(ascending=False)
pi = permutation_importance(rf, X, ly, n_repeats=20, random_state=0)
impdf = pd.DataFrame(dict(mean_abs_shapley=imp, perm_importance=pd.Series(pi.importances_mean, index=MON)[imp.index],
                          spearman_feature_vs_shap=[stats.spearmanr(A[m], sh[m])[0] for m in imp.index]))
impdf.to_csv('outputs/feature_importance.csv'); print(impdf)
fig, axs = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1.6, 1]})
ax = axs[0]
order = list(imp.index)
for k, m in enumerate(order[::-1]):
    v = A[m].values; yy = k + rng.normal(0, 0.12, len(v))
    sc = ax.scatter(sh[m], yy, c=v / max(v.max(), 1e-9), cmap='coolwarm', s=9, vmin=0, vmax=1, lw=0)
ax.set_yticks(range(6)); ax.set_yticklabels(order[::-1]); ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel('Shapley value for log $F_a$ (RF trained on all 351 gels)')
cb = plt.colorbar(sc, ax=ax); cb.set_label('feature value (scaled to max)')
ax.set_title('a  Exact interventional Shapley values (beeswarm)', loc='left', fontsize=9.5)
ax = axs[1]
yy = np.arange(6)[::-1]
ax.barh(yy + 0.18, impdf.mean_abs_shapley / impdf.mean_abs_shapley.max(), 0.36, color=CAT[0], label='mean |Shapley|')
ax.barh(yy - 0.18, impdf.perm_importance / impdf.perm_importance.max(), 0.36, color=CAT[1], label='permutation importance')
ax.set_yticks(yy); ax.set_yticklabels(order); ax.set_xlabel('relative importance'); ax.legend(fontsize=7.5)
ax.set_title('b  Global importance', loc='left', fontsize=9.5)
fig.tight_layout(); save(fig, 'fig10_shapley_importance')

# ---------- embedding: t-SNE (UMAP unavailable) ----------
L = pd.read_csv('outputs/ml_designs_all_lists.csv'); L = L[L.HEA.notna()]
emb_in = np.vstack([X, L[MON].values])
E = TSNE(2, perplexity=30, random_state=0, init='pca').fit_transform(emb_in + 0)
EA, EL = E[:len(X)], E[len(X):]
pd.DataFrame(dict(x=E[:, 0], y=E[:, 1], kind=['measured'] * len(X) + ['list'] * len(L))).to_csv('outputs/tsne_embedding.csv', index=False)
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
ax = axs[0]
rc = {0: '#c3c2b7', 1: CAT[0], 2: CAT[1], 3: CAT[2]}
for r in [0, 1, 2, 3]:
    s = A['round'] == r
    ax.scatter(EA[s, 0], EA[s, 1], s=12 if r == 0 else 22, color=rc[r], edgecolor=INK2 if r else 'none', lw=0.3,
               label='initial 184' if r == 0 else f'round {r} (n={s.sum()})', zorder=2 + r)
km = pd.read_csv('outputs/key_formulations.csv')
for nm, ix in [('G-042', A.index[A.ID == 'G-042'][0]), ('GPRFR-2', A.index[A.ID == 'GPRFR-2'][0])]:
    ax.scatter(*EA[ix], marker='*', s=150, color=CAT[7], edgecolor=INK, lw=0.5, zorder=9); ax.text(EA[ix, 0] + 1, EA[ix, 1] + 1, nm, fontsize=8)
ib = int(np.argmax(y)); ax.scatter(*EA[ib], marker='*', s=200, color=CAT[3], edgecolor=INK, lw=0.5, zorder=9); ax.text(EA[ib, 0] + 1, EA[ib, 1] - 3, 'R1-max', fontsize=8)
ax.legend(fontsize=7.5, loc='best'); ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2'); ax.set_title('a  Exploration by round (t-SNE of compositions)', loc='left', fontsize=9.5)
ax = axs[1]
sc = ax.scatter(EA[:, 0], EA[:, 1], c=y, cmap='Blues', s=14, edgecolor=INK2, lw=0.2)
r1 = L[(L['round'] == 1)]
for meth, mk, col in [('RFR-GP', 'o', CAT[1]), ('GP-GP', 's', CAT[6]), ('old-SM-GP', '^', CAT[3])]:
    s = (L['round'] == 1).values & (L.method == meth).values
    ax.scatter(EL[s, 0], EL[s, 1], marker=mk, s=40, facecolor='none', edgecolor=col, lw=1.3, label=f'round-1 {meth} lists')
plt.colorbar(sc, ax=ax, label='measured $F_a$ (kPa)')
ax.legend(fontsize=7.5, loc='best'); ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2'); ax.set_title('b  Landscape and GP-based round-1 proposals', loc='left', fontsize=9.5)
fig.tight_layout(); save(fig, 'fig11_embedding')

# ---------- grouped Shapley (simplex-aware summary): hydrophilic {HEA,CBEA,AAm}, hydrophobic+aromatic {BA,PEA}, cationic {ATAC}
GR = {'hydrophilic (HEA+CBEA+AAm)': [0, 2, 5], 'hydrophobic+aromatic (BA+PEA)': [1, 4], 'cationic (ATAC)': [3]}
gn = list(GR); gphi = np.zeros((len(X), 3))
def fcoal(gc):
    t = [0] * 6
    for g, on in zip(gn, gc):
        if on:
            for j in GR[g]: t[j] = 1
    return tuple(t)
for gi in range(3):
    for gc in itertools.product([0, 1], repeat=3):
        if gc[gi]: continue
        s = sum(gc); w = factorial(s) * factorial(3 - s - 1) / factorial(3)
        g1 = list(gc); g1[gi] = 1
        gphi[:, gi] += w * (V[:, idx[fcoal(g1)]] - V[:, idx[fcoal(gc)]])
gs = pd.DataFrame(gphi, columns=gn)
grp = pd.DataFrame(dict(mean_abs=gs.abs().mean(), spearman_groupcontent_vs_shap=[stats.spearmanr(A[[MON[j] for j in GR[g]]].sum(1), gs[g])[0] for g in gn]))
best = int(np.argmax(y)); grp['R1max_phi'] = gs.iloc[best].values; grp['G042_phi'] = gs.iloc[int(A.index[A.ID == 'G-042'][0])].values
grp.to_csv('outputs/grouped_shapley.csv'); print(grp)
