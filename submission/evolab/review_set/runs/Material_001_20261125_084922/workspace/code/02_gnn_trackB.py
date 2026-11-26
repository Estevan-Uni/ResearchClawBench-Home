"""Track B: CGCNN-style GNN trained on a structure-dependent surrogate formation-energy set
built from the K5 template of the dataset (5-atom clusters, fully connected graph).

Surrogate formation energy per atom (documented, eV/atom):
  E_f = (1/N) [ sum_{i<j} D_ij((1-exp(-α(r_ij-r0_ij)))^2 - 1)            (Morse bonding)
               - k_ion sum_{i<j} (χ_i-χ_j)^2 / r_ij                       (ionic/electroneg. stabilisation)
               + sum_i Δμ_Zi ]  + ε,  ε ~ N(0, 0.005^2)
with D_ij = sqrt(D_i D_j), r0_ij = R_i + R_j (covalent radii), Δμ_Z element offsets.
"""
import os, time
import numpy as np
from common import save_json, style, IMG, OUT, C, grad_check
import gnn_np as G

SEED = 0
rng = np.random.default_rng(SEED); print('seed', SEED)
# element table: Z, covalent radius (Å), Morse depth (eV), Pauling electronegativity, ref-energy offset (eV)
ELEM = {3: (1.28, 0.55, 0.98, 0.30), 5: (0.84, 1.20, 2.04, 0.10), 6: (0.76, 1.50, 2.55, 0.00),
        7: (0.71, 1.10, 3.04, 0.15), 8: (0.66, 0.90, 3.44, 0.20)}
ZS = sorted(ELEM); NE = len(ZS); N = 5; ALPHA = 1.6; KION = 0.35; NOISE = 0.005
EDGES = np.array([0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 4]).reshape(-1, 2)  # dataset edge list
MASK = np.zeros((N, N)); MASK[EDGES[:, 0], EDGES[:, 1]] = 1; MASK = MASK + MASK.T
# trigonal-bipyramid template (unit bond length) for the K5 cluster
TBP = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [-0.5, -np.sqrt(3) / 2, 0]], float)


def make_clusters(n, rng):
    Z = rng.choice(ZS, size=(n, N))
    R = np.vectorize(lambda z: ELEM[z][0])(Z)
    scale = (2 * R.mean(1) * rng.uniform(0.85, 1.25, n))[:, None, None]
    pos = TBP[None] * scale + rng.normal(0, 0.18, size=(n, N, 3))
    D = np.linalg.norm(pos[:, :, None] - pos[:, None], axis=-1)
    Dm = np.vectorize(lambda z: ELEM[z][1])(Z); chi = np.vectorize(lambda z: ELEM[z][2])(Z)
    mu = np.vectorize(lambda z: ELEM[z][3])(Z)
    iu = np.triu_indices(N, 1)
    r = D[:, iu[0], iu[1]]
    Dij = np.sqrt(Dm[:, iu[0]] * Dm[:, iu[1]]); r0 = R[:, iu[0]] + R[:, iu[1]]
    morse = Dij * ((1 - np.exp(-ALPHA * (r - r0))) ** 2 - 1)
    ion = -KION * (chi[:, iu[0]] - chi[:, iu[1]]) ** 2 / r
    Ef = (morse.sum(1) + ion.sum(1) + mu.sum(1)) / N + rng.normal(0, NOISE, n)
    return Z, D, Ef


def featurise(Z, D, centers=np.linspace(0.5, 4.5, 21), gamma=1 / 0.2 ** 2):
    X = (Z[..., None] == np.array(ZS)).astype(float)            # one-hot element
    E = np.exp(-gamma * (D[..., None] - centers) ** 2)          # Gaussian RBF expansion (CGCNN)
    return X, E


def ridge(Ftr, ytr, Fte, lam=1e-3):
    A = np.c_[Ftr, np.ones(len(Ftr))]; w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
    return np.c_[Fte, np.ones(len(Fte))] @ w


def metrics(y, yh):
    return dict(MAE=float(np.mean(np.abs(y - yh))), RMSE=float(np.sqrt(np.mean((y - yh) ** 2))),
                R2=float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)))


NTOT = 5000
Z, D, Ef = make_clusters(NTOT, rng)
np.savez(os.path.join(OUT, 'trackB_clusters.npz'), Z=Z, D=D, Ef=Ef)
X, E = featurise(Z, D)
perm = rng.permutation(NTOT); itr, iva, ite = perm[:4000], perm[4000:4500], perm[4500:]
ym, ys = Ef[itr].mean(), Ef[itr].std()
yz = (Ef - ym) / ys
print(f'E_f range {Ef.min():.3f}..{Ef.max():.3f} eV/atom, std {Ef.std():.3f}')

# gradient check on tiny batch with the real featurisation
p0 = G.init_params(NE, E.shape[-1], H=8, L=2, Hfc=6, seed=3)
gc = grad_check(lambda p: G.mse_loss_grad(p, X[:8], E[:8], MASK, yz[:8])[0],
                lambda p: G.mse_loss_grad(p, X[:8], E[:8], MASK, yz[:8])[1], p0, n=15, h=1e-5)
print('grad check max rel err', gc); assert gc < 1e-5

EPOCHS = int(os.environ.get('EPOCHS', 150)); EP2 = int(os.environ.get('EP2', 100))


def run(name, Etr, Eva, Ete, seed=SEED, ntr=None, epochs=EPOCHS):
    tr = itr if ntr is None else itr[:ntr]
    p = G.init_params(NE, Etr.shape[-1], H=24, L=3, Hfc=32, seed=seed)
    data = dict(Xtr=X[tr], Etr=Etr[:len(tr)],
                ytr=yz[tr], Xva=X[iva], Eva=Eva, yva=yz[iva], mask=MASK)
    t0 = time.time(); print(f'== {name} (n_train={len(tr)})', flush=True)
    best, hist, bep = G.train(p, data, epochs=epochs, bs=128, lr=3e-3, seed=seed, verbose=50)
    pred = G.forward(best, X[ite], Ete, MASK) * ys + ym
    m = metrics(Ef[ite], pred); m.update(best_epoch=bep, time_s=time.time() - t0)
    print('  test', m, flush=True)
    return best, hist, pred, m


res = {}
# full model
Etr, Eva, Ete = E[itr], E[iva], E[ite]
best, hist, pred, res['GNN (full)'] = run('GNN full', Etr, Eva, Ete)
emb = G.embed(best, X[ite], Ete, MASK)
res['GNN (full)']['embedding_std_across_graphs'] = float(emb.std(0).mean())
# ablation 1: no distance features (composition + topology only)
Z0 = np.zeros((NTOT, N, N, 0))
_, hist_nod, pred_nod, res['GNN (no distances)'] = run('GNN no-dist', Z0[itr], Z0[iva], Z0[ite], epochs=EP2)
# ablation 2: distance features shuffled across clusters (breaks structure-property link)
sh = rng.permutation(len(itr))
_, hist_sh, pred_sh, res['GNN (shuffled distances)'] = run('GNN shuffled', Etr[sh], Eva[rng.permutation(len(iva))], Ete[rng.permutation(len(ite))], epochs=EP2)

# baselines
res['Mean'] = metrics(Ef[ite], np.full(len(ite), Ef[itr].mean()))
comp = X.sum(1)                                            # element counts
res['Ridge (composition)'] = metrics(Ef[ite], ridge(comp[itr], Ef[itr], comp[ite]))
iu = np.triu_indices(N, 1)
rs = np.sort(D[:, iu[0], iu[1]], 1)
pairc = np.stack([(Z[:, iu[0]] == a) & (Z[:, iu[1]] == b) | (Z[:, iu[0]] == b) & (Z[:, iu[1]] == a)
                  for k, a in enumerate(ZS) for b in ZS[k:]], -1).sum(1)
Flin = np.c_[comp, rs, 1 / rs, pairc]
res['Ridge (comp + sorted distances + pair counts)'] = metrics(Ef[ite], ridge(Flin[itr], Ef[itr], Flin[ite]))
pred_lin = ridge(Flin[itr], Ef[itr], Flin[ite])

# learning curve
lc = {}
for n in [250, 500, 1000, 2000]:
    _, _, _, m = run(f'GNN n={n}', E[itr[:n]], Eva, Ete, ntr=n, epochs=EP2)
    lc[n] = m['MAE']
lc[4000] = res['GNN (full)']['MAE']  # note: full model trained 150 epochs
res['learning_curve_MAE'] = lc
res['setup'] = dict(seed=SEED, n_total=NTOT, split='4000/500/500', epochs_full=EPOCHS, epochs_ablation_lc=EP2, H=24, layers=3,
                    rbf='21 centres 0.5-4.5 Å, width 0.2 Å', grad_check=gc, noise_floor_MAE=NOISE * np.sqrt(2 / np.pi),
                    Ef_std=float(Ef.std()))
save_json(res, 'gnn_trackB_results.json')
np.savez(os.path.join(OUT, 'gnn_trackB_curves.npz'), tr=hist['train'], va=hist['val'], tr_nod=hist_nod['train'],
         va_nod=hist_nod['val'], tr_sh=hist_sh['train'], va_sh=hist_sh['val'], y=Ef[ite], pred=pred,
         pred_nod=pred_nod, pred_sh=pred_sh, pred_lin=pred_lin, emb=emb, comp=comp[ite])

# ---- figures
plt = style()
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
v2 = ys ** 2; ep = np.arange(1, EPOCHS + 1); ep2 = np.arange(1, EP2 + 1)
ax[0].semilogy(ep, np.array(hist['train']) * v2, c=C[0], label='full: train'); ax[0].semilogy(ep, np.array(hist['val']) * v2, c=C[0], ls='--', label='full: val')
ax[0].semilogy(ep2, np.array(hist_nod['train']) * v2, c=C[1], label='no distances: train'); ax[0].semilogy(ep2, np.array(hist_nod['val']) * v2, c=C[1], ls='--', label='no distances: val')
ax[0].semilogy(ep2, np.array(hist_sh['train']) * v2, c=C[5], label='shuffled dist.: train'); ax[0].semilogy(ep2, np.array(hist_sh['val']) * v2, c=C[5], ls='--', label='shuffled dist.: val')
ax[0].axhline(NOISE ** 2, c='k', ls=':', lw=1, label='label-noise floor')
ax[0].set(xlabel='epoch', ylabel='MSE (eV²/atom²)', title='(a) Training / validation loss'); ax[0].legend(fontsize=7)
lim = [Ef[ite].min() - 0.05, Ef[ite].max() + 0.05]
ax[1].scatter(Ef[ite], pred, s=8, c=C[0], alpha=0.6, label=f"GNN  MAE={res['GNN (full)']['MAE']*1000:.1f} meV/atom")
ax[1].scatter(Ef[ite], pred_lin, s=8, c=C[2], alpha=0.35, label=f"Ridge  MAE={res['Ridge (comp + sorted distances + pair counts)']['MAE']*1000:.1f} meV/atom")
ax[1].plot(lim, lim, 'k--', lw=1); ax[1].set(xlim=lim, ylim=lim, xlabel='surrogate $E_f$ (eV/atom)', ylabel='predicted $E_f$ (eV/atom)', title='(b) Test-set parity (n=500)'); ax[1].legend(fontsize=8)
ns = sorted(lc); ax[2].loglog(ns, [lc[n] * 1000 for n in ns], 'o-', c=C[0], label='GNN')
for k, cc in [('Mean', C[5]), ('Ridge (composition)', C[1]), ('Ridge (comp + sorted distances + pair counts)', C[2])]:
    ax[2].axhline(res[k]['MAE'] * 1000, c=cc, ls='--', label=k.replace('comp + sorted distances + pair counts', 'comp+dist+pairs'))
ax[2].axhline(NOISE * np.sqrt(2 / np.pi) * 1000, c='k', ls=':', label='noise floor')
ax[2].set(xlabel='training set size', ylabel='test MAE (meV/atom)', title='(c) Learning curve vs baselines'); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig2_gnn_trackB.png'))

# embedding PCA
U, S, Vt = np.linalg.svd(emb - emb.mean(0), full_matrices=False); pc = U[:, :2] * S[:2]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sc = ax[0].scatter(*pc.T, c=Ef[ite], s=8, cmap='viridis'); fig.colorbar(sc, ax=ax[0], label='$E_f$ (eV/atom)')
ax[0].set(title='(a) PCA of learned cluster embeddings', xlabel='PC1', ylabel='PC2')
names = ['GNN (full)', 'GNN (shuffled distances)', 'GNN (no distances)', 'Ridge (comp + sorted distances + pair counts)', 'Ridge (composition)', 'Mean']
vals = [res[k]['MAE'] * 1000 for k in names]
ax[1].barh(range(len(names)), vals, color=[C[0], C[5], C[1], C[2], C[3], C[4]])
for i, v in enumerate(vals): ax[1].text(v, i, f' {v:.1f}', va='center', fontsize=8)
ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels([n.replace('comp + sorted distances + pair counts', 'comp+dist+pairs') for n in names], fontsize=8)
ax[1].invert_yaxis(); ax[1].set(xlabel='test MAE (meV/atom)', title='(b) Ablations and baselines')
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig3_gnn_ablation_embedding.png'))
print('done')
