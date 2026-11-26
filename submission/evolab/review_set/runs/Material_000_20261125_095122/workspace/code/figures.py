import sys, json, os; sys.path.insert(0, '/workspace/code')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e1',
                     'grid.linewidth': 0.6, 'axes.edgecolor': '#52514e', 'text.color': '#0b0b0b', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'legend.frameon': False, 'lines.linewidth': 2})
C = {'random': '#9a9893', 'gbdt': '#2a78d6', 'wl_lr': '#eda100', 'scratch': '#eb6834', 'pre_frozen': '#1baf7a', 'pre_ft': '#6250d6'}
NAME = {'random': 'Random', 'gbdt': 'Descriptor GBDT', 'wl_lr': 'WL-kernel LR', 'scratch': 'GNN scratch', 'pre_frozen': 'Pretrained GNN (frozen)', 'pre_ft': 'Pretrained GNN (fine-tuned)'}
M = list(C); IMG = '/workspace/report/images/'; O = '/workspace/outputs/'
what = sys.argv[1:] or ['pre', 'curves', 'main', 'hist', 'pr', 'perm', 'le', 'probe', 'res']

if 'pre' in what:
    P = json.load(open(O + 'pretrain_log_main.json')); L = pd.DataFrame(P['log']); b = P['baselines']
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(L.epoch, L.train_loss, color='#2a78d6', label='train'); ax[0].plot(L.epoch, L.val_loss, color='#eb6834', label='validation (500 held-out graphs)')
    ax[0].axhline(b['uniform_ce'], color='#52514e', ls='--', lw=1); ax[0].text(60, b['uniform_ce'] + .01, 'uniform ln 28', ha='right', va='bottom', color='#52514e')
    ax[0].axhline(b['marginal_ce'], color='#52514e', ls=':', lw=1); ax[0].text(60, b['marginal_ce'] - .015, 'element-frequency prior', ha='right', va='top', color='#52514e')
    ax[0].set(xlabel='epoch', ylabel='masked-element cross-entropy', title='(a) Masked-node reconstruction loss'); ax[0].legend(loc='center right')
    ax[1].plot(L.epoch, L.train_acc, color='#2a78d6', label='train'); ax[1].plot(L.epoch, L.val_acc, color='#eb6834', label='validation')
    ax[1].axhline(b['uniform_acc'], color='#52514e', ls='--', lw=1); ax[1].text(60, b['uniform_acc'] + .002, 'uniform 1/28', ha='right', va='bottom', color='#52514e')
    ax[1].axhline(b['majority_acc'], color='#52514e', ls=':', lw=1); ax[1].text(60, b['majority_acc'] + .002, 'majority element', ha='right', va='bottom', color='#52514e')
    ax[1].set(xlabel='epoch', ylabel='masked-element accuracy', title='(b) Reconstruction accuracy vs. baselines', ylim=(0, 0.13)); ax[1].legend(loc='lower right')
    plt.tight_layout(); plt.savefig(IMG + 'fig3_pretraining.png', dpi=150); plt.close()

if 'curves' in what:
    fig, ax = plt.subplots(2, 4, figsize=(16, 7.5))
    for r, tag in enumerate(['real', 'control']):
        J = json.load(open(O + f'curves_{tag}.json')); z = np.load(O + f'scores_{tag}.npz'); prev = z['ft_y'].mean()
        for c, (key, lab) in enumerate([('loss', 'weighted BCE loss'), ('acc', 'accuracy (thr 0.5)'), ('bacc', 'balanced accuracy'), ('ap', 'PR-AUC (average precision)')]):
            a = ax[r, c]
            for split, col, ls in [('train', '#2a78d6', '-'), ('val', '#eb6834', '-'), ('test', '#1baf7a', '--')]:
                arr = [pd.DataFrame(J['curves'][f'pre_ft_fold{f}'])[f'{split}_{key}'].values for f in range(5)]
                n = min(map(len, arr)); A = np.array([x[:n] for x in arr]); ep = np.arange(1, n + 1)
                a.plot(ep, A.mean(0), color=col, ls=ls, label={'train': 'train (inner)', 'val': 'validation (inner, early-stopping)', 'test': 'held-out outer fold'}[split])
                a.fill_between(ep, A.min(0), A.max(0), color=col, alpha=0.12, lw=0)
            if key == 'acc': a.axhline(1 - prev, color='#52514e', ls=':', lw=1); a.text(1, 1 - prev - .01, f'all-negative = {1-prev:.3f}', va='top', color='#52514e', fontsize=8)
            if key == 'bacc': a.axhline(0.5, color='#52514e', ls=':', lw=1); a.text(1, 0.505, 'chance 0.5', va='bottom', color='#52514e', fontsize=8)
            if key == 'ap': a.axhline(prev, color='#52514e', ls=':', lw=1); a.text(1, prev + .005, f'prevalence {prev:.3f}', va='bottom', color='#52514e', fontsize=8)
            a.set(xlabel='epoch', ylabel=lab); a.set_title(f"{'Real labels' if tag == 'real' else 'Positive control'}: {lab}", fontsize=10)
            if r == 0 and c == 0: a.legend(fontsize=8, loc='upper right')
    plt.suptitle('Fine-tuning curves of the pretrained gated-residual CGCNN (mean and min-max over 5 outer folds)', y=1.0)
    plt.tight_layout(); plt.savefig(IMG + 'fig4_finetune_curves.png', dpi=150, bbox_inches='tight'); plt.close()

if 'main' in what:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for k, tag in enumerate(['real', 'control']):
        R = pd.read_csv(O + f'metrics_{tag}.csv').set_index('method'); off = -0.18 if k == 0 else 0.18
        mk = 'o' if tag == 'real' else 's'
        for i, m in enumerate(M):
            ax[0].errorbar(i + off, R.cand_auc[m], yerr=[[R.cand_auc[m] - R.cand_auc_lo[m]], [R.cand_auc_hi[m] - R.cand_auc[m]]], fmt=mk, ms=8,
                           color=C[m], mfc=C[m] if tag == 'control' else 'white', mew=2, capsize=3, lw=1.5)
            ax[1].errorbar(i + off, R.cand_ap[m] / R.prevalence[m], yerr=[[(R.cand_ap[m] - R.cand_ap_lo[m]) / R.prevalence[m]], [(R.cand_ap_hi[m] - R.cand_ap[m]) / R.prevalence[m]]],
                           fmt=mk, ms=8, color=C[m], mfc=C[m] if tag == 'control' else 'white', mew=2, capsize=3, lw=1.5)
            ax[2].bar(i + off, R.hits50[m], 0.34, color=C[m] if tag == 'control' else 'white', edgecolor=C[m], lw=2)
            ax[2].text(i + off, R.hits50[m] + 0.4, str(R.hits50[m]), ha='center', fontsize=8, color='#52514e')
        ax[2].hlines(R.expected_hits50.iloc[0], -0.5 + (0 if k == 0 else 0), len(M) - 0.5, color='#52514e', ls=':' if k == 0 else '--', lw=1)
    ax[0].axhline(0.5, color='#52514e', ls=':', lw=1); ax[1].axhline(1, color='#52514e', ls=':', lw=1)
    for a in ax: a.set_xticks(range(len(M))); a.set_xticklabels(['Random', 'GBDT', 'WL-LR', 'GNN\nscratch', 'Pretrained\nfrozen', 'Pretrained\nfine-tuned'], fontsize=8.5)
    ax[0].set(ylabel='candidate ROC-AUC (95% bootstrap CI)', title='(a) Candidate ROC-AUC'); ax[1].set(ylabel='PR-AUC / prevalence (lift)', title='(b) Candidate PR-AUC lift over random')
    ax[2].set(ylabel='true positives in top 50', title='(c) Hits@50 (dotted/dashed: chance, real/control)')
    from matplotlib.lines import Line2D
    ax[0].legend(handles=[Line2D([], [], marker='o', ls='', mfc='white', mec='#52514e', mew=2, ms=8, label='real labels (hollow)'),
                          Line2D([], [], marker='s', ls='', color='#52514e', ms=8, label='positive control (filled)')], loc='upper left', fontsize=8)
    plt.tight_layout(); plt.savefig(IMG + 'fig5_main_results.png', dpi=150); plt.close()

if 'hist' in what:
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for r, tag in enumerate(['real', 'control']):
        z = np.load(O + f'scores_{tag}.npz'); y = z['cd_y']
        for c, m in enumerate(['pre_ft', 'scratch', 'gbdt']):
            s = z[f'cand_{m}']; bins = np.linspace(0, 1, 26); a = ax[r, c]
            a.hist(s[y == 0], bins=bins, color='#9a9893', alpha=0.8, label=f'true negatives (n={int((y==0).sum())})', edgecolor='white', lw=0.5)
            a2 = a.twinx() if False else a
            a.hist(s[y == 1], bins=bins, color=C[m], alpha=0.95, label=f'true positives (n={int(y.sum())})', edgecolor='white', lw=0.5)
            for t in (0.5, 0.8, 0.9): a.axvline(t, color='#52514e', ls=':', lw=1)
            a.set_yscale('log'); a.set(xlabel='raw predicted probability', ylabel='# candidates (log)', title=f"{'Real' if tag=='real' else 'Control'} - {NAME[m]}")
            a.legend(fontsize=8, loc='upper right')
    plt.tight_layout(); plt.savefig(IMG + 'fig6_probability_histograms.png', dpi=150); plt.close()

if 'pr' in what:
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for r, tag in enumerate(['real', 'control']):
        z = np.load(O + f'scores_{tag}.npz'); y = z['cd_y']
        for m in M:
            s = z[f'cand_{m}']; o = np.argsort(-s, kind='stable'); yy = y[o]; tp = np.cumsum(yy); k = np.arange(1, len(y) + 1)
            ax[r, 0].plot(tp / y.sum(), tp / k, color=C[m], lw=1.6, label=NAME[m])
            ax[r, 1].plot(k, tp, color=C[m], lw=1.6, label=NAME[m])
        ax[r, 0].axhline(y.mean(), color='#52514e', ls=':', lw=1); ax[r, 1].plot([0, len(y)], [0, y.sum()], color='#52514e', ls=':', lw=1, label='random expectation')
        ax[r, 1].plot([0, y.sum(), len(y)], [0, y.sum(), y.sum()], color='#52514e', ls='--', lw=1, label='ideal')
        T = 'Real labels' if tag == 'real' else 'Positive control'
        ax[r, 0].set(xlabel='recall', ylabel='precision', title=f'{T}: precision-recall on candidates', ylim=(0, 1.02))
        ax[r, 1].set(xlabel='# candidates screened (ranked)', ylabel='# true positives found', title=f'{T}: discovery curve', xlim=(0, 300)); ax[r, 1].axvline(50, color='#c3c2b7', lw=1)
        ax[r, 1].legend(fontsize=8, loc='lower right' if tag == 'control' else 'upper left')
    plt.tight_layout(); plt.savefig(IMG + 'fig7_pr_discovery.png', dpi=150); plt.close()

if 'perm' in what:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, tag in enumerate(['real', 'control']):
        z = np.load(O + f'scores_{tag}.npz'); y = z['cd_y']; s = z['cand_pre_ft']
        p, null = perm_p(y, s, ap, 2000); obs = ap(y, s)
        ax[k].hist(null, bins=40, color='#c3c2b7', edgecolor='white', label='label-permutation null (2000)')
        ax[k].axvline(obs, color=C['pre_ft'], lw=2.5, label=f'observed AP = {obs:.3f} (p = {p:.3f})')
        ax[k].set(xlabel='average precision on candidates', ylabel='count', title=f"{'Real labels' if tag=='real' else 'Positive control'}: fine-tuned pretrained GNN"); ax[k].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(IMG + 'fig8_permutation.png', dpi=150); plt.close()

if 'le' in what:
    L = pd.concat([pd.read_csv(f) for f in [O + 'label_eff_a.csv', O + 'label_eff_b.csv'] if os.path.exists(f)])
    L.to_csv(O + 'label_efficiency_all.csv', index=False)
    G = L.groupby(['method', 'n_pos']).agg(['mean', 'std', 'count']); G.to_csv(O + 'label_efficiency_summary.csv')
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for j, met in enumerate(['auc', 'ap', 'hits50']):
        for i, m in enumerate(['gbdt', 'scratch', 'pre_frozen', 'pre_ft']):
            if m not in L.method.values: continue
            g = L[L.method == m].groupby('n_pos')[met]; mu, sd = g.mean(), g.std()
            x = np.array(mu.index) * (1 + 0.03 * (i - 1.5))
            ax[j].errorbar(x, mu.values, yerr=sd.values, color=C[m], marker='o', ms=7, capsize=3, label=NAME[m])
        ax[j].set_xscale('log'); ax[j].set_xticks([10, 25, 50, 100]); ax[j].set_xticklabels([10, 25, 50, 100])
        ax[j].set(xlabel='# labelled positives in fine-tuning set', ylabel={'auc': 'candidate ROC-AUC', 'ap': 'candidate PR-AUC', 'hits50': 'hits@50 (of 50 positives)'}[met])
    ax[0].axhline(0.5, color='#52514e', ls=':', lw=1); ax[1].axhline(50 / 896, color='#52514e', ls=':', lw=1); ax[2].axhline(50 * 50 / 896, color='#52514e', ls=':', lw=1)
    ax[0].legend(fontsize=8); plt.suptitle('Label efficiency on the positive-control task (mean ± s.d. over seeds)', y=1.0)
    plt.tight_layout(); plt.savefig(IMG + 'fig9_label_efficiency.png', dpi=150, bbox_inches='tight'); plt.close()

if 'probe' in what:
    P = pd.read_csv(O + 'probes.csv'); fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k, met in enumerate(['R2', 'AUC']):
        Q = P[P.metric == met]; ts = Q.target.unique(); w = 0.38
        for i, (e, col) in enumerate([('main', '#6250d6'), ('random_init', '#9a9893')]):
            v = [Q[(Q.encoder == e) & (Q.target == t)].value.iloc[0] for t in ts]
            ax[k].bar(np.arange(len(ts)) + (i - .5) * w, v, w * 0.95, color=col, label={'main': 'masked-node pretrained', 'random_init': 'random-init encoder'}[e])
            for xx, vv in zip(np.arange(len(ts)) + (i - .5) * w, v): ax[k].text(xx, max(vv, 0) + 0.01, f'{vv:.2f}', ha='center', fontsize=8, color='#52514e')
        ax[k].set_xticks(range(len(ts))); ax[k].set_xticklabels(ts, fontsize=9)
        ax[k].set(ylabel='5-fold CV ' + ('R²' if met == 'R2' else 'ROC-AUC'), title='(a) Ridge probes (regression)' if met == 'R2' else '(b) Logistic probes (classification)')
        if met == 'AUC': ax[k].axhline(0.5, color='#52514e', ls=':', lw=1); ax[k].set_ylim(0.4, 1)
        ax[k].legend(fontsize=8, loc='upper right')
    plt.tight_layout(); plt.savefig(IMG + 'fig10_probes.png', dpi=150); plt.close()

if 'res' in what:
    J = json.load(open(O + 'residual_ablation.json')); fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for key, v in J.items():
        L_, r, s = key.split('_'); D_ = pd.DataFrame(v)
        col = {'L3_res1': '#2a78d6', 'L3_res0': '#9fc3ee', 'L6_res1': '#6250d6', 'L6_res0': '#eb6834'}[f'{L_}_{r}']
        lab = f"{L_[1:]} layers, residual {'on' if r == 'res1' else 'off'}" if s == 's0' else None
        ax[0].plot(D_.epoch, D_.train_loss, color=col, ls='-' if s == 's0' else '--', lw=1.6, label=lab)
        ax[1].plot(D_.epoch, D_.cand_auc, color=col, ls='-' if s == 's0' else '--', lw=1.6, label=lab)
    ax[0].set(xlabel='epoch', ylabel='training loss (weighted BCE)', title='(a) Training loss (solid seed 0, dashed seed 1)'); ax[0].legend(fontsize=8)
    ax[1].set(xlabel='epoch', ylabel='candidate ROC-AUC (monitor only)', title='(b) Held-out candidate AUC'); ax[1].axhline(0.5, color='#52514e', ls=':', lw=1)
    plt.tight_layout(); plt.savefig(IMG + 'fig11_residual_ablation.png', dpi=150); plt.close()
print('figs done', what)
