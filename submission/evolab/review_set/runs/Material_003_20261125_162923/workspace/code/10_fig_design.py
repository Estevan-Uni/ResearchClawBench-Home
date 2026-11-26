"""Figures for VAE training, latent space, 1e6 chemistry space, BO and traversal; candidate grid."""
import os, sys, pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 130})
C1, C2, C3, C4 = '#2a6f97', '#e07a5f', '#81b29a', '#6d597a'
IMG = 'report/images/'
# ---- VAE training curves
L = pd.read_csv('outputs/vae_b01_log.csv')
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
ax[0].plot(L.epoch, L.train_rec_acid, c=C1, ls='--', label='train acid'); ax[0].plot(L.epoch, L.val_rec_acid, c=C1, label='val acid')
ax[0].plot(L.epoch, L.train_rec_epox, c=C2, ls='--', label='train epoxide'); ax[0].plot(L.epoch, L.val_rec_epox, c=C2, label='val epoxide')
ax[0].set_ylabel('reconstruction NLL (nats/molecule)'); ax[0].set_xlabel('epoch'); ax[0].legend(fontsize=8)
a2 = ax[0].twinx(); a2.plot(L.epoch, L.val_kl, c='#777777', lw=1); a2.set_ylabel('val KL (nats)', color='#777777')
ax[1].plot(L.epoch, L.head_r2, c=C4, label='Tg head R2 (val)'); ax[1].set_ylim(0, 1); ax[1].set_xlabel('epoch'); ax[1].set_ylabel('R2')
a3 = ax[1].twinx(); a3.plot(L.epoch, L.head_mae_K, c=C3); a3.set_ylabel('head MAE (K)', color=C3); ax[1].legend(fontsize=8, loc='lower right')
for col, c, lab in [('fg_rec_acid', C1, 'recon acid'), ('fg_rec_epox', C2, 'recon epoxide'), ('fg_prior_acid', C1, 'prior acid'),
                    ('fg_prior_epox', C2, 'prior epoxide'), ('exact_acid', '#333333', 'exact recon acid'), ('exact_epox', '#999999', 'exact recon epoxide')]:
    ax[2].plot(L.epoch, L[col], c=c, ls='-' if 'rec' in col and 'exact' not in col else (':' if 'exact' in col else '--'), label=lab)
ax[2].set_ylim(0, 1.02); ax[2].set_xlabel('epoch'); ax[2].set_ylabel('fraction'); ax[2].legend(fontsize=7, ncol=2)
ax[2].set_title('Functional-group-correct decodes', fontsize=10)
plt.tight_layout(); plt.savefig(IMG + 'fig5_vae_training.png'); plt.close()

# ---- latent PCA + 1e6 space
C = np.load('outputs/chemspace.npz'); meta = pickle.load(open('outputs/chemspace_meta.pkl', 'rb'))
Zlib, ycal = C['Zlib'], C['ycal']
Zc = Zlib - Zlib.mean(0); U, S, Vt = np.linalg.svd(Zc, full_matrices=False); P = Zc @ Vt[:2].T
d = Zlib.shape[1] // 2
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sc = ax[0].scatter(P[:, 0], P[:, 1], c=ycal, s=3, cmap='viridis'); plt.colorbar(sc, ax=ax[0], label='calibrated Tg (K)')
ax[0].set_xlabel('PC1'); ax[0].set_ylabel('PC2'); ax[0].set_title('Joint latent z=[z_a,z_e] (8,424 vitrimers)', fontsize=10)
ax[1].scatter(C['mu_t'], ycal[C['tst']], s=4, alpha=.4, c=C1)
ax[1].plot([250, 490], [250, 490], 'k--', lw=1); ax[1].set_xlabel('latent-GP prediction (K)'); ax[1].set_ylabel('GP-calibrated Tg (K)')
g = meta['gp_metrics']; ax[1].set_title('Latent GP, held-out 1,500 pairs: MAE %.1f K, R2 %.2f' % (g['MAE'], g['R2']), fontsize=9)
b = np.linspace(230, 500, 60)
ax[2].hist(C['mu'], bins=b, density=True, alpha=.6, color=C4, label='10^6 novel combinations (latent GP)')
ax[2].hist(ycal, bins=b, density=True, alpha=.6, color=C3, label='8,424 library (calibrated)')
ax[2].axvline(ycal.max(), c='k', ls=':', lw=1); ax[2].axvline(ycal.min(), c='k', ls=':', lw=1)
ax[2].set_yscale('log'); ax[2].set_xlabel('Tg (K)'); ax[2].set_ylabel('density (log)'); ax[2].legend(fontsize=7)
ax[2].set_title('Curated combinatorial chemistry space', fontsize=10)
plt.tight_layout(); plt.savefig(IMG + 'fig6_latent_chemspace.png'); plt.close()
O = pd.read_csv('outputs/chemspace_oracle_subsample.csv')
fig, ax = plt.subplots(1, 1, figsize=(5, 4.2))
for gidx, c, lab in [(0, '#999999', 'random 2,000'), (1, C1, 'GP-screen bottom 150'), (2, C2, 'GP-screen top 150')]:
    q = O[O.group == gidx]; ax.scatter(q.gp_mu, q.oracle_cal, s=6, alpha=.6, c=c, label=lab)
ax.plot([240, 480], [240, 480], 'k--', lw=1); ax.set_xlabel('latent-GP screening Tg (K)'); ax.set_ylabel('virtual MD + GP calibration Tg (K)')
ax.legend(fontsize=8); ax.set_title('Screening vs oracle on 10^6 space', fontsize=10)
plt.tight_layout(); plt.savefig(IMG + 'fig7_chemspace_oracle.png'); plt.close()

# ---- BO convergence
if os.path.exists('outputs/bo_designs.csv'):
    B = pd.read_csv('outputs/bo_designs.csv'); R = pd.read_csv('outputs/bo_rand.csv') if os.path.exists('outputs/bo_rand.csv') else None
    objs = [('max', 'maximise Tg'), ('min', 'minimise Tg'), ('target373', 'target 373 K'), ('target300', 'target 300 K (near RT)')]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for k, (o, title) in enumerate(objs):
        for df, c, lab in [(B, C1, 'latent BO'), (R, '#999999', 'random trust-region')]:
            if df is None: continue
            q = df[df.objective == o]
            if not len(q): continue
            if o == 'max': f = q.oracle_cal; agg = 'max'
            elif o == 'min': f = q.oracle_cal; agg = 'min'
            else: f = (q.oracle_cal - (373 if o == 'target373' else 300)).abs(); agg = 'min'
            if o.startswith('target'):
                hr = (f < 5).groupby(q['round']).mean()
                ax[k].plot(hr.index, hr.values, 'o-', c=c, label=lab + ' hit rate (+/-5 K)')
                continue
            per = f.groupby(q['round']).agg(agg); bsf = per.cummax() if agg == 'max' else per.cummin()
            ax[k].plot(per.index, bsf.values, 'o-', c=c, label=lab + ' best-so-far')
            ax[k].scatter(q['round'] + (0.1 if df is R else -0.1), f, s=5, alpha=.3, c=c)
        if o == 'max': ax[k].axhline(ycal.max(), c='k', ls=':', label='library max'); ax[k].axhline(C['mu'].max(), c=C2, ls='--', label='10^6 screen max (GP)')
        if o == 'min': ax[k].axhline(ycal.min(), c='k', ls=':', label='library min'); ax[k].axhline(C['mu'].min(), c=C2, ls='--', label='10^6 screen min (GP)')
        if o.startswith('target'):
            T_ = 373 if o == 'target373' else 300
            ax[k].axhline((np.abs(ycal - T_) < 5).mean(), c='k', ls=':', label='library hit rate')
            ax[k].set_ylim(0, 0.6)
        ax[k].set_xlabel('BO round'); ax[k].set_ylabel('oracle Tg (K)' if o in ('max', 'min') else 'fraction of designs within +/-5 K')
        ax[k].set_title(title, fontsize=10); ax[k].legend(fontsize=6.5)
    plt.tight_layout(); plt.savefig(IMG + 'fig8_bo_convergence.png'); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    b = np.linspace(200, 520, 65)
    ax[0].hist(ycal, bins=b, density=True, color='#cccccc', label='library (calibrated)')
    for o, c in [('max', C2), ('min', C1), ('target373', C3), ('target300', C4)]:
        q = B[B.objective == o]; ax[0].hist(q.oracle_cal, bins=b, density=True, histtype='step', lw=1.8, color=c, label=f'BO {o} (n={len(q)})')
    ax[0].axvline(373, c=C3, ls=':'); ax[0].axvline(300, c=C4, ls=':')
    ax[0].set_xlabel('oracle-calibrated Tg (K)'); ax[0].legend(fontsize=7); ax[0].set_title('BO design distributions vs library', fontsize=10)
    ax[1].scatter(B.reenc_mu, B.oracle_cal, s=5, alpha=.4, c=C1)
    ax[1].plot([200, 520], [200, 520], 'k--', lw=1); ax[1].set_xlabel('latent GP at re-encoded design (K)'); ax[1].set_ylabel('oracle (K)')
    r = np.corrcoef(B.reenc_mu, B.oracle_cal)[0, 1]; ax[1].set_title('Re-encode consistency (r = %.2f)' % r, fontsize=10)
    plt.tight_layout(); plt.savefig(IMG + 'fig9_bo_designs.png'); plt.close()

# ---- traversal
if os.path.exists('outputs/latent_traversal.csv'):
    T = pd.read_csv('outputs/latent_traversal.csv')
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
    for k in range(3):
        q = T[T.path == k]
        ax[k].plot(q.t, q.gp_mu, 'o-', c=C1, label='latent GP (calibrated)')
        ax[k].fill_between(q.t, q.gp_mu - 2 * q.gp_sd, q.gp_mu + 2 * q.gp_sd, color=C1, alpha=.15)
        ax[k].plot(q.t, q.head_md, 's--', c='#999999', ms=3, label='Tg head (MD scale)')
        ax[k].plot(q.t, q.oracle_cal, '^-', c=C2, label='decoded -> virtual MD + calibration')
        ax[k].set_xlabel('slerp parameter t'); ax[k].set_title(f'path {k + 1}', fontsize=10)
    ax[0].set_ylabel('Tg (K)'); ax[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(IMG + 'fig10_latent_traversal.png'); plt.close()
print('ok')
