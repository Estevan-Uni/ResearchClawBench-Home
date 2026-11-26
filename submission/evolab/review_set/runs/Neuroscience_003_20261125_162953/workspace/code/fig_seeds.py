import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
s = pd.read_csv('/workspace/outputs/rpe_delve_faithful_scores.csv'); dg = pd.read_csv('/workspace/outputs/rpe_seed_diagnostics.csv').set_index('variant')
fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
comp = s.feature.str.split('_').str[-1]
for c in ['nuc', 'cell', 'cyto', 'ring']:
    ax[0].hist(s.dyn_score[comp == c], bins=30, alpha=.5, label=c)
ax[0].axvline(1 / 11, color='k', ls=':', label='exchangeable noise 1/(k+1)'); ax[0].set(xlabel='dynamic score Var(M_f)/Var(X_f)', ylabel='features', title='a  DELVE-faithful dynamic scores by compartment'); ax[0].legend()
dg[['frac_seeds_in_redundant_group', 'abs_r_seedPC1_global_intensity', 'frac_nuc_seeds', 'frac_nuc_top30']].plot.bar(ax=ax[1], rot=0)
ax[1].set_title('b  Seed-set diagnostics (label-free)'); ax[1].legend(fontsize=8)
ax[2].bar(dg.index, dg.n_seeds, color='0.6', label='# seeds'); ax[2].bar(dg.index, dg.n_distinct_groups_in_seeds, color='C3', alpha=.7, label='# distinct |r|>0.9 groups in seeds')
ax[2].set_title('c  Seed redundancy'); ax[2].legend()
plt.tight_layout(); plt.savefig('/workspace/report/images/fig2_seed_diagnostics.png', dpi=110)
