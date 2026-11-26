import sys; sys.path.insert(0, '/workspace/code')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from core import *
X, obs, var = load_rpe(); Xz = zscore(X)
med, lab = dedup_medoids(Xz, 0.9)
sizes = pd.Series(lab).value_counts()
Z, Vt, ev = pca(Xz, 20)
rho_age = [spearmanr(Z[:, i], obs['age'])[0] for i in range(20)]
cyc = obs['phase'] != 'G0'
rho_age_c = [spearmanr(Z[cyc, i], obs['age'][cyc])[0] for i in range(20)]
mean_int = Xz.mean(1)
pc1_int = np.corrcoef(Z[:, 0], mean_int)[0, 1]
batch_acc = knn_cv_accuracy(Z, obs['batch'], balanced=True)
fb = [abs(spearmanr(X[:, j], obs['batch'] == '1')[0]) for j in range(X.shape[1])]
summ = dict(n_cells=X.shape[0], n_features=X.shape[1], n_groups_r09=len(med),
            singletons=int((sizes == 1).sum()), pairs=int((sizes == 2).sum()), triplets_plus=int((sizes >= 3).sum()),
            pc1_var_frac=float(ev[0] / Xz.var(0).sum()), pc1_corr_mean_intensity=float(pc1_int),
            pc1_rho_age=float(rho_age[0]), batch_knn_bal_acc_allfeat=float(batch_acc), max_feature_batch_rho=float(max(fb)))
pd.Series(summ).to_csv('/workspace/outputs/rpe_overview_summary.csv')
pd.DataFrame({'pc': np.arange(1, 21), 'var_frac': ev / Xz.var(0).sum(), 'rho_age_all': rho_age, 'rho_age_cycling': rho_age_c}).to_csv('/workspace/outputs/rpe_pc_age.csv', index=False)
print(summ)
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
order = ['G0', 'G1', 'S', 'G2']; pal = dict(zip(order, sns.color_palette('Set2', 4)))
for p in order:
    ax[0, 0].hist(obs['age'][obs['phase'] == p], bins=40, alpha=.6, label=f'{p} (n={(obs["phase"]==p).sum()})', color=pal[p])
ax[0, 0].set(xlabel='annotated age (h)', ylabel='cells', title='a  Age distribution by phase'); ax[0, 0].legend()
ax[0, 1].bar(np.arange(1, 21), ev / Xz.var(0).sum(), color='0.6', label='variance fraction')
a2 = ax[0, 1].twinx(); a2.plot(np.arange(1, 21), np.abs(rho_age), 'o-', c='C3', label='|ρ(PC, age)| all'); a2.plot(np.arange(1, 21), np.abs(rho_age_c), 's--', c='C0', label='|ρ| cycling')
ax[0, 1].set(xlabel='PC', ylabel='variance fraction', title='b  PCs: variance vs age association'); a2.set_ylabel('|Spearman ρ|'); a2.legend(loc='upper right')
C = np.abs(np.corrcoef(Xz.T))
from scipy.cluster.hierarchy import linkage, leaves_list
o = leaves_list(linkage(squareform(np.clip(1 - C, 0, None), checks=False), 'average'))
sns.heatmap(C[np.ix_(o, o)], ax=ax[1, 0], cmap='viridis', xticklabels=False, yticklabels=False, cbar_kws={'label': '|r|'})
ax[1, 0].set_title(f'c  |corr| between 241 features ({len(med)} groups at |r|>0.9)')
for p in order:
    m = obs['phase'] == p; ax[1, 1].scatter(Z[m, 0], Z[m, 3], s=3, color=pal[p], label=p)
ax[1, 1].set(xlabel=f'PC1 (global intensity, r={pc1_int:.2f} with mean z)', ylabel='PC4', title='d  PC1 vs PC4 coloured by phase'); ax[1, 1].legend(markerscale=4)
plt.tight_layout(); plt.savefig('/workspace/report/images/fig1_data_overview.png', dpi=130)
