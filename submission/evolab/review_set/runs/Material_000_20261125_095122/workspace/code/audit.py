"""Data audit + cheap signal diagnostics (no sklearn needed)."""
import sys; sys.path.insert(0, '/workspace/code')
import numpy as np, pandas as pd, json, collections
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

D = load_cache(); res = {}
sets = ['pretrain', 'finetune', 'candidate']
rows = []
hashes = {}
for s in sets:
    gs = D[s]; y = np.array([g['y'] for g in gs])
    oh_ok = all(((g['xoh'].sum(1) == 1) & ((g['xoh'] == 0) | (g['xoh'] == 1)).all(1)).all().item() for g in gs)
    sym = 0; selfloop = 0; oddn = 0; oddm = 0; paired = 0
    for g in gs:
        ei = g['ei'].numpy(); ea = g['ea'].numpy()
        fw = {(a, b): tuple(np.round(e, 6)) for (a, b), e in zip(ei.T, ea)}
        sym += all((b, a) in fw and fw[(b, a)] == e for (a, b), e in fw.items())
        selfloop += (ei[0] == ei[1]).any()
        # are edges stored as consecutive (a,b),(b,a) pairs?
        paired += all(ei[0, 2*k] == ei[1, 2*k+1] and ei[1, 2*k] == ei[0, 2*k+1] for k in range(ei.shape[1] // 2))
        oddn += len(g['x']) % 2; oddm += (g['x'] < N_MAG).sum() % 2
    hs = [graph_hash(g) for g in gs]; hashes[s] = hs
    ex = [h[0] for h in hs]; inv = [h[1] for h in hs]
    rows.append(dict(set=s, graphs=len(gs), positives=int(y.sum()), pos_frac=y.mean(), meta_pos_ratio=D[s + '_meta']['pos_ratio'],
                     onehot_valid=oh_ok, edge_symmetric=sym, consecutive_pairs=paired, self_loops=int(selfloop), odd_atoms=int(oddn), odd_mag=int(oddm),
                     dup_exact=len(ex) - len(set(ex)), dup_invariant=len(inv) - len(set(inv)),
                     atoms_min=min(len(g['x']) for g in gs), atoms_max=max(len(g['x']) for g in gs),
                     mean_undirected_bonds=np.mean([g['ei'].shape[1] / 2 for g in gs])))
audit = pd.DataFrame(rows); print(audit.T)
audit.to_csv('outputs/audit_counts.csv', index=False)
cross = {}
for a in sets:
    for b in sets:
        if a < b:
            cross[f'{a}-{b}'] = dict(exact=len(set(h[0] for h in hashes[a]) & set(h[0] for h in hashes[b])),
                                     invariant=len(set(h[1] for h in hashes[a]) & set(h[1] for h in hashes[b])))
print('cross-set duplicates', cross); res['cross_dups'] = cross

# descriptors
desc = {s: pd.DataFrame([descriptors(g) for g in D[s]]) for s in sets}
for s in sets: desc[s]['y'] = [g['y'] for g in D[s]]; desc[s].to_csv(f'outputs/descriptors_{s}.csv', index=False)

# univariate AUCs
ua = []
for c in desc['pretrain'].columns[:-1]:
    r = {'descriptor': c}
    for s in sets:
        r[s] = auc(desc[s]['y'].values, desc[s][c].values)
    ua.append(r)
ua = pd.DataFrame(ua); ua.to_csv('outputs/univariate_auc.csv', index=False); print(ua.round(3).to_string())

# chi2 element presence vs label
chi = []
for s in sets:
    y = np.array([g['y'] for g in D[s]])
    pres = np.array([np.bincount(g['x'].numpy(), minlength=28) > 0 for g in D[s]])
    for e in range(28):
        t = np.array([[((pres[:, e]) & (y == 1)).sum(), ((pres[:, e]) & (y == 0)).sum()],
                      [((~pres[:, e]) & (y == 1)).sum(), ((~pres[:, e]) & (y == 0)).sum()]])
        if (t.sum(0) == 0).any() or (t.sum(1) == 0).any(): continue
        p = fisher_exact(t)[1] if t.min() < 5 else chi2_contingency(t)[1]
        chi.append(dict(set=s, element=ELEMENTS[e], p=p, frac_pos=t[0, 0] / max(1, y.sum()), frac_neg=t[0, 1] / max(1, (y == 0).sum())))
chi = pd.DataFrame(chi); chi.to_csv('outputs/chi2_elements.csv', index=False)
print('element chi2: n tests', len(chi), 'min p', chi.p.min(), 'n p<0.05', (chi.p < 0.05).sum())

# atom count category / magnetic count category
cat = []
for s in sets:
    for col in ['n_atoms', 'n_mag', 'n_bonds', 'n_species']:
        tab = pd.crosstab(desc[s][col], desc[s]['y'])
        tab = tab[tab.sum(1) >= 5]  # drop sparse rows
        if tab.shape[0] < 2 or tab.shape[1] < 2: continue
        c2, p, dof, _ = chi2_contingency(tab.values)
        cat.append(dict(set=s, variable=col, chi2=c2, dof=dof, p=p))
cat = pd.DataFrame(cat); cat.to_csv('outputs/chi2_categories.csv', index=False); print(cat.to_string())

# label vs file index / position artefacts and node-order artefacts
art = []
for s in sets:
    y = np.array([g['y'] for g in D[s]]); idx = np.arange(len(y))
    # runs test for serial structure
    runs = 1 + (np.diff(y) != 0).sum(); n1 = y.sum(); n0 = len(y) - n1
    mu = 2 * n1 * n0 / len(y) + 1; var = (mu - 1) * (mu - 2) / (len(y) - 1)
    from scipy.stats import norm
    zr = (runs - mu) / np.sqrt(var)
    first_mag = np.array([float(g['x'][0] < N_MAG) for g in D[s]])
    first_edge_src = np.array([float(g['ei'][0, 0]) / len(g['x']) if g['ei'].shape[1] else -1 for g in D[s]])
    mag_pos = np.array([np.mean(np.where(g['x'].numpy() < N_MAG)[0]) / len(g['x']) if (g['x'] < N_MAG).any() else 0.5 for g in D[s]])
    edge_sorted = np.array([float(np.all(np.diff(g['ei'][0].numpy()[::2]) >= 0)) for g in D[s]])
    art.append(dict(set=s, auc_index=auc(y, idx), runs_z=zr, runs_p=2 * norm.sf(abs(zr)),
                    auc_first_atom_magnetic=auc(y, first_mag), auc_first_edge_src_pos=auc(y, first_edge_src),
                    auc_mean_mag_position=auc(y, mag_pos), auc_edges_sorted=auc(y, edge_sorted),
                    frac_edges_sorted=edge_sorted.mean()))
art = pd.DataFrame(art); art.to_csv('outputs/artefact_checks.csv', index=False); print(art.T)

# ---------------- figures --------------------
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
col = {'pretrain': '#4C72B0', 'finetune': '#DD8452', 'candidate': '#55A868'}
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for s in sets:
    ax[0, 0].hist(desc[s]['n_atoms'], bins=np.arange(3.5, 25.5, 1), density=True, histtype='step', lw=2, label=s, color=col[s])
ax[0, 0].set(xlabel='atoms per graph', ylabel='density', title='(a) Graph size'); ax[0, 0].legend()
w = 0.27
for k, s in enumerate(sets):
    cnt = np.bincount(np.concatenate([g['x'].numpy() for g in D[s]]), minlength=28)
    ax[0, 1].bar(np.arange(28) + (k - 1) * w, cnt / cnt.sum(), w, color=col[s], label=s)
ax[0, 1].set_xticks(range(28)); ax[0, 1].set_xticklabels(ELEMENTS, rotation=90, fontsize=8)
ax[0, 1].axvline(13.5, color='k', ls=':'); ax[0, 1].set(ylabel='fraction of atoms', title='(b) Element frequency (left of dotted line: magnetic)')
allea = np.concatenate([g['ea'].numpy()[::2] for s in sets for g in D[s]])
for t, c in zip(range(3), ['#8172B2', '#C44E52', '#937860']):
    ax[1, 0].hist(allea[allea[:, 1] == t, 0], bins=40, histtype='step', lw=2, color=c, label=f'bond type {t}')
ax[1, 0].set(xlabel='bond value (edge_attr[:,0])', ylabel='count', title='(c) Bond values per bond type'); ax[1, 0].legend()
lab = audit.set_index('set')
xb = np.arange(3)
ax[1, 1].bar(xb, lab.positives, color=[col[s] for s in sets])
for i, s in enumerate(sets): ax[1, 1].text(i, lab.positives[s] + 30, f"{lab.positives[s]}/{lab.graphs[s]}\n({100*lab.pos_frac[s]:.1f}%)", ha='center')
ax[1, 1].set_xticks(xb); ax[1, 1].set_xticklabels(sets); ax[1, 1].set(ylabel='# positive labels (y=1)', title='(d) Label counts', ylim=(0, 2900))
plt.tight_layout(); plt.savefig('report/images/fig1_data_overview.png', dpi=150); plt.close()

# QQ of chi2 p-values
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for s in sets:
    p = np.sort(chi[chi.set == s].p.values); e = (np.arange(1, len(p) + 1) - 0.5) / len(p)
    ax[0].scatter(-np.log10(e), -np.log10(p), s=14, color=col[s], label=f'{s} (n={len(p)})')
m = 3.5; ax[0].plot([0, m], [0, m], 'k--', lw=1); ax[0].set(xlabel='expected -log10 p (uniform null)', ylabel='observed -log10 p', title='(a) QQ: element presence vs label'); ax[0].legend()
ua_m = ua.melt(id_vars='descriptor', var_name='set', value_name='AUC')
for k, s in enumerate(sets):
    v = ua[s].values; ax[1].scatter(np.full(len(v), k) + np.random.default_rng(0).uniform(-.15, .15, len(v)), v, s=14, color=col[s])
ax[1].axhline(0.5, color='k', ls='--', lw=1); ax[1].set_xticks(range(3)); ax[1].set_xticklabels(sets)
ax[1].set(ylabel='univariate ROC-AUC', title=f'(b) {len(ua)} descriptors vs label', ylim=(0.3, 0.7))
plt.tight_layout(); plt.savefig('report/images/fig2_signal_diagnostics.png', dpi=150); plt.close()

# per-atom-count label rate (pretrain)
tab = pd.crosstab(desc['pretrain']['n_atoms'], desc['pretrain']['y'])
json.dump({'pretrain_atomcount_table': tab.to_dict()}, open('outputs/atomcount_table.json', 'w'), default=str)
print(tab.T)
