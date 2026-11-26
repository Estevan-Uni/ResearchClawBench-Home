"""Structure generation: numpy VAE on lattice-parameter pairs (a, b) [Å, assumed].
Encoder 2-16-16-(mu, logvar in R^2), decoder 2-16-16-2, tanh, Gaussian recon (MSE) + beta*KL,
KL warm-up. Manual backprop, Adam, gradient-checked. Outputs are inverse-standardised to Å."""
import os
import numpy as np
from scipy.stats import wasserstein_distance
from common import load_dataset, Adam, grad_check, save_json, style, IMG, OUT, C

SEED = 0; print('seed', SEED)
d = load_dataset(); Xraw = np.c_[d['lattice_a'], d['lattice_b']]
U = np.unique(np.round(Xraw, 4), axis=0)


def mlp_init(sizes, rng, pre):
    return {f'{pre}W{i}': rng.normal(0, np.sqrt(1 / a), (a, b)) for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))} | \
           {f'{pre}b{i}': np.zeros(b) for i, b in enumerate(sizes[1:])}


def mlp_fwd(p, pre, x, n):
    acts = [x]
    for i in range(n):
        x = x @ p[f'{pre}W{i}'] + p[f'{pre}b{i}']
        if i < n - 1: x = np.tanh(x)
        acts.append(x)
    return x, acts


def mlp_bwd(p, pre, acts, dout, n, g):
    for i in reversed(range(n)):
        if i < n - 1: dout = dout * (1 - acts[i + 1] ** 2)
        g[f'{pre}W{i}'] = acts[i].T @ dout; g[f'{pre}b{i}'] = dout.sum(0)
        dout = dout @ p[f'{pre}W{i}'].T
    return dout


def init(seed, H=16, Z=2):
    rng = np.random.default_rng(seed)
    return mlp_init([2, H, H, 2 * Z], rng, 'e') | mlp_init([Z, H, H, 2], rng, 'd')


def loss_grad(p, x, eps, beta, want_grad=True):
    B, Z = len(x), eps.shape[1]
    o, ea = mlp_fwd(p, 'e', x, 3); mu, lv = o[:, :Z], o[:, Z:]
    sd = np.exp(0.5 * lv); z = mu + sd * eps
    xh, da = mlp_fwd(p, 'd', z, 3)
    rec = np.sum((xh - x) ** 2) / B; kl = 0.5 * np.sum(np.exp(lv) + mu ** 2 - 1 - lv) / B
    L = rec + beta * kl
    if not want_grad: return L, rec, kl
    g = {}
    dz = mlp_bwd(p, 'd', da, 2 * (xh - x) / B, 3, g)
    dmu = dz + beta * mu / B; dlv = dz * eps * 0.5 * sd + beta * 0.5 * (np.exp(lv) - 1) / B
    mlp_bwd(p, 'e', ea, np.c_[dmu, dlv], 3, g)
    return L, rec, kl, g


def encode(p, x):
    o, _ = mlp_fwd(p, 'e', x, 3); return o[:, :2], o[:, 2:]


def decode(p, z):
    return mlp_fwd(p, 'd', z, 3)[0]


def train_vae(Xs, beta, epochs=3000, seed=0, lr=3e-3, warm=0.3):
    rng = np.random.default_rng(seed); p = init(seed); opt = Adam(p, lr=lr)
    H = {'rec': [], 'kl': [], 'beta_eff': []}
    for ep in range(epochs):
        b = beta * min(1.0, ep / (warm * epochs))     # linear KL warm-up
        eps = rng.normal(size=(len(Xs), 2))
        L, rec, kl, g = loss_grad(p, Xs, eps, b)
        opt.step(g); H['rec'].append(rec); H['kl'].append(kl); H['beta_eff'].append(b)
    return p, H


# ---- metrics (in Å)
def mmd(Xa, Xb, bw=0.1):
    k = lambda A, B: np.exp(-((A[:, None] - B[None]) ** 2).sum(-1) / (2 * bw ** 2))
    return float(k(Xa, Xa).mean() + k(Xb, Xb).mean() - 2 * k(Xa, Xb).mean())


def hist_kl(P, Q, bins=np.linspace(4.9, 6.1, 25)):
    hp = np.histogram2d(*P.T, bins=[bins, bins])[0] + 1e-3; hq = np.histogram2d(*Q.T, bins=[bins, bins])[0] + 1e-3
    hp /= hp.sum(); hq /= hq.sum(); return float(np.sum(hp * np.log(hp / hq)))


def prec_cov(real_u, gen):
    """kNN (k=1) precision/coverage on unique real points (Naeem et al. 2020 style)."""
    Dr = np.linalg.norm(real_u[:, None] - real_u[None], axis=-1); np.fill_diagonal(Dr, np.inf)
    rad = Dr.min(1)                                        # NN radius of each real point
    Dg = np.linalg.norm(gen[:, None] - real_u[None], axis=-1)
    precision = float(np.mean((Dg <= rad[None]).any(1)))
    coverage = float(np.mean((Dg <= rad[None]).any(0)))
    density = float(np.mean((Dg <= rad[None]).sum(1)) / 1)
    return precision, coverage


def evaluate(gen, real, real_u, name):
    nn = np.linalg.norm(gen[:, None] - real_u[None], axis=-1).min(1)
    P, Cv = prec_cov(real_u, gen)
    return dict(name=name, mean_a=float(gen[:, 0].mean()), mean_b=float(gen[:, 1].mean()), std_a=float(gen[:, 0].std()),
                std_b=float(gen[:, 1].std()), W1_a=wasserstein_distance(gen[:, 0], real[:, 0]),
                W1_b=wasserstein_distance(gen[:, 1], real[:, 1]), MMD=mmd(gen, real), KL_hist=hist_kl(real, gen),
                precision=P, coverage=Cv, novelty_frac=float(np.mean(nn > 0.02)), median_nn_dist=float(np.median(nn)),
                valid_frac=float(np.mean((gen > 5.0).all(1) & (gen < 6.0).all(1))))


def kde_sample(Xr, n, rng, bw=0.08):
    return Xr[rng.integers(len(Xr), size=n)] + rng.normal(0, bw, (n, 2))


if __name__ == '__main__':
    mu0, sd0 = Xraw.mean(0), Xraw.std(0); Xs = (Xraw - mu0) / sd0
    rng = np.random.default_rng(SEED)
    p = init(1); eps = rng.normal(size=(8, 2))
    gc = grad_check(lambda q: loss_grad(q, Xs[:8], eps, 0.7, False)[0], lambda q: loss_grad(q, Xs[:8], eps, 0.7)[3], p, n=20, h=1e-6)
    print('VAE grad check', gc); assert gc < 1e-5
    NG = 3000; models = {}; res = {'grad_check': gc, 'data': dict(mean=mu0, std=sd0, n_rows=len(Xraw), n_unique=len(U))}
    evals = []
    for beta in [0.1, 1.0]:
        pv, Hh = train_vae(Xs, beta, seed=SEED)
        zz = rng.normal(size=(NG, 2)); gen = decode(pv, zz) * sd0 + mu0      # inverse standardisation -> Å
        mu_z, lv_z = encode(pv, Xs)
        models[beta] = (pv, Hh, gen, mu_z, lv_z)
        e = evaluate(gen, Xraw, U, f'VAE beta={beta}'); e['final_rec'] = Hh['rec'][-1]; e['final_KL'] = Hh['kl'][-1]
        e['recon_MAE_A'] = float(np.mean(np.abs(decode(pv, mu_z) * sd0 + mu0 - Xraw)))
        evals.append(e); print(e)
    memo = Xraw[rng.integers(len(Xraw), size=NG)]
    gauss = rng.multivariate_normal(mu0, np.cov(Xraw.T), NG)
    evals += [evaluate(memo, Xraw, U, 'Memoriser'), evaluate(gauss, Xraw, U, 'Gaussian'), evaluate(kde_sample(Xraw, NG, rng), Xraw, U, 'KDE (bw 0.08 Å)')]
    for e in evals[2:]: print(e)
    res['generators'] = evals

    # leave-2-of-7-out rediscovery: train on rows whose pair is in 5 kept, sample, check held-out recovery
    from itertools import combinations
    red = {'VAE beta=1.0': [], 'VAE beta=0.1': [], 'KDE (bw 0.08 Å)': [], 'Memoriser': []}
    for k, hold in enumerate(combinations(range(7), 2)):
        keep = np.array([not any(np.allclose(x, U[h], atol=1e-3) for h in hold) for x in np.round(Xraw, 4)])
        Xk = Xraw[keep]; m1, s1 = Xk.mean(0), Xk.std(0)
        for beta in [1.0, 0.1]:
            pk, _ = train_vae((Xk - m1) / s1, beta, epochs=2000, seed=k)
            g = decode(pk, rng.normal(size=(NG, 2))) * s1 + m1
            red[f'VAE beta={beta}'].append([float(np.min(np.linalg.norm(g - U[h], axis=1))) for h in hold])
        g = kde_sample(Xk, NG, rng); red['KDE (bw 0.08 Å)'].append([float(np.min(np.linalg.norm(g - U[h], axis=1))) for h in hold])
        g = Xk[rng.integers(len(Xk), size=NG)]; red['Memoriser'].append([float(np.min(np.linalg.norm(g - U[h], axis=1))) for h in hold])
    res['leave2out'] = {k: dict(min_dist_median=float(np.median(v)), frac_within_0p05=float(np.mean(np.array(v) < 0.05)),
                                frac_within_0p10=float(np.mean(np.array(v) < 0.10))) for k, v in red.items()}
    print(res['leave2out'])
    save_json(res, 'vae_results.json')
    np.savez(os.path.join(OUT, 'vae_generated.npz'), gen_b1=models[1.0][2], gen_b01=models[0.1][2])

    plt = style()
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    for beta, cc in [(0.1, C[1]), (1.0, C[0])]:
        Hh = models[beta][1]
        ax[0, 0].semilogy(Hh['rec'], c=cc, label=f'recon, β={beta}'); ax[0, 0].semilogy(Hh['kl'], c=cc, ls='--', label=f'KL, β={beta}')
    ax2 = ax[0, 0].twinx(); ax2.plot(np.array(models[1.0][1]['beta_eff']), c='grey', lw=0.8); ax2.set_ylabel('effective β (β=1 run)', color='grey'); ax2.grid(False)
    ax[0, 0].set(xlabel='epoch', ylabel='loss (standardised units)', title='(a) VAE training: reconstruction & KL'); ax[0, 0].legend(fontsize=7, loc='center right')
    pv, Hh, gen, mu_z, lv_z = models[1.0]
    sc = ax[0, 1].scatter(*mu_z.T, c=Xraw[:, 0], cmap='viridis', s=40, edgecolor='k', lw=0.3)
    for i in range(len(mu_z)):
        if i < 7: ax[0, 1].add_patch(plt.matplotlib.patches.Ellipse(mu_z[i], 2 * np.exp(0.5 * lv_z[i, 0]), 2 * np.exp(0.5 * lv_z[i, 1]), fill=False, color='grey', lw=0.6))
    fig.colorbar(sc, ax=ax[0, 1], label='a (Å)'); ax[0, 1].set(title='(b) Latent means μ(x) (β=1), ellipses = 1σ', xlabel='$z_1$', ylabel='$z_2$')
    # latent traversal decoded grid
    g1 = np.linspace(-2.5, 2.5, 60); G1, G2 = np.meshgrid(g1, g1)
    dec = decode(pv, np.c_[G1.ravel(), G2.ravel()]) * Xraw.std(0) + Xraw.mean(0)
    cs = ax[0, 2].contourf(G1, G2, dec[:, 0].reshape(G1.shape), 20, cmap='viridis'); fig.colorbar(cs, ax=ax[0, 2], label='decoded a (Å)')
    ax[0, 2].set(title='(c) Decoder map: latent → a (β=1)', xlabel='$z_1$', ylabel='$z_2$')
    bins = np.linspace(4.9, 6.1, 61)
    for j, lab in enumerate(['a', 'b']):
        ax[1, j].hist(Xraw[:, j], bins=bins, density=True, color='k', alpha=0.35, label='training (101 rows)')
        ax[1, j].hist(models[1.0][2][:, j], bins=bins, density=True, histtype='step', color=C[0], lw=1.5, label='VAE β=1')
        ax[1, j].hist(models[0.1][2][:, j], bins=bins, density=True, histtype='step', color=C[1], lw=1.5, label='VAE β=0.1')
        ax[1, j].set(xlabel=f'{lab} (Å)', ylabel='density', title=f'({"de"[j]}) Marginal of {lab}: generated vs training'); ax[1, j].legend(fontsize=7)
    ax[1, 2].scatter(*models[1.0][2][:800].T, s=4, c=C[0], alpha=0.35, label='VAE β=1')
    ax[1, 2].scatter(*models[0.1][2][:800].T, s=4, c=C[1], alpha=0.35, label='VAE β=0.1')
    ax[1, 2].scatter(*U.T, s=120, marker='*', c='k', label='7 unique training pairs')
    ax[1, 2].set(xlabel='a (Å)', ylabel='b (Å)', title='(f) Generated (a,b) in Å'); ax[1, 2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig5_vae.png'))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    names = [e['name'] for e in evals]; x = np.arange(len(names))
    w = 0.2
    for k, (m, cc) in enumerate([('precision', C[0]), ('coverage', C[2]), ('novelty_frac', C[1]), ('valid_frac', C[3])]):
        ax[0].bar(x + (k - 1.5) * w, [e[m] for e in evals], w, color=cc, label=m)
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=8, rotation=15); ax[0].set(title='(a) Precision / coverage / novelty / validity', ylim=(0, 1.1)); ax[0].legend(fontsize=7, ncol=2)
    ks = list(res['leave2out']); ax[1].bar(range(len(ks)), [res['leave2out'][k]['frac_within_0p05'] for k in ks], color=C[0], label='< 0.05 Å')
    ax[1].bar(range(len(ks)), [res['leave2out'][k]['frac_within_0p10'] for k in ks], color=C[0], alpha=0.35, label='< 0.10 Å')
    ax[1].set_xticks(range(len(ks))); ax[1].set_xticklabels(ks, fontsize=8); ax[1].set(ylabel='fraction of held-out pairs rediscovered', title='(b) Leave-2-of-7-out rediscovery (21 folds)'); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig6_vae_validation.png'))
