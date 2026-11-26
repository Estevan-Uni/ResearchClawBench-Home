"""Iterative pore-model training (simulation): start from the additive (main-effect) approximation of the
r9.4.1 model, align simulated reads with bcDTW, re-estimate k-mer means, repeat."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simlib import *

rng = np.random.default_rng(SEED + 2)
M = Model("DNA r9.4.1 (6-mer)")
K = M.K
# additive initial model
y = M.mu; X = M.X
init = np.full_like(y, y.mean())
for p in range(K):
    m = np.bincount(X[:, p], weights=y, minlength=4) / np.bincount(X[:, p], minlength=4)
    init += m[X[:, p]] - y.mean()
NREADS, LK, ITERS = 100, 300, 4
reads = []
for r in range(NREADS):
    seq = random_seq(rng, LK + K - 1); codes = kmer_codes(seq, K)
    rd = simulate_read(rng, M, codes, noise_infl=2.0)
    mvb = move_table(rng, rd, jitter_strides=1.0)
    reads.append((codes, rd, mvb))
model = init.copy()
hist = []


def evaluate(model, it, accs, mads):
    d = model - M.mu
    hist.append(dict(iteration=it, pearson_true=float(np.corrcoef(model, M.mu)[0, 1]),
                     MAE_true=float(np.mean(np.abs(d))), RMSE_true=float(np.sqrt(np.mean(d ** 2))),
                     mean_align_acc=float(np.mean(accs)) if accs else None,
                     median_read_MAD=float(np.median(mads)) if mads else None))


evaluate(model, 0, [], [])
for it in range(1, ITERS + 1):
    S = np.zeros(4 ** K); N = np.zeros(4 ** K); accs = []; mads = []
    for codes, rd, mvb in reads:
        mu = model[codes]; sd = M.sd[codes] * 2.0
        lab, (s, b), _ = bcdtw(rd["raw"], mu, sd, mvb, W=5)
        z = (rd["raw"] - b) / s
        em, n = event_means(z, lab, len(codes))
        ok = n > 0
        mads.append(np.median(np.abs(em[ok] - mu[ok])))
        accs.append(np.mean(lab == rd["lab"]))
        np.add.at(S, codes[ok], em[ok] * n[ok]); np.add.at(N, codes[ok], n[ok])
    new = model.copy(); seen = N > 0
    new[seen] = S[seen] / N[seen]
    new = (new - new.mean()) / new.std()   # re-standardise like the distributed models
    model = new
    evaluate(model, it, accs, mads)
    print(hist[-1], "coverage", float(seen.mean()), flush=True)
hist = pd.DataFrame(hist); hist.to_csv(f"{OUT}/train_sim_history.csv", index=False)
pd.DataFrame(dict(kmer_code=np.arange(4 ** K), trained_mean=model, true_mean=M.mu, init_mean=init)).to_csv(
    f"{OUT}/train_sim_trained_model_r941.csv", index=False)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
axes[0].plot(hist.iteration, hist.RMSE_true, "o-", color="#1f5aa6"); axes[0].set_ylabel("RMSE vs true model (SD units)")
axes[0].set_xlabel("training iteration"); axes[0].set_title("Model error")
axes[1].plot(hist.iteration[1:], hist.mean_align_acc[1:], "o-", color="#e07b39"); axes[1].set_ylabel("sample assignment accuracy")
axes[1].set_xlabel("training iteration"); axes[1].set_title("Alignment accuracy with current model")
axes[2].scatter(M.mu, init, s=2, alpha=0.3, color="#999999", label=f"additive init (r={hist.pearson_true[0]:.3f})")
axes[2].scatter(M.mu, model, s=2, alpha=0.3, color="#1f5aa6", label=f"after {ITERS} iters (r={hist.pearson_true.iloc[-1]:.3f})")
axes[2].plot([-3, 3], [-3, 3], "k:"); axes[2].set_xlabel("true k-mer mean"); axes[2].set_ylabel("model k-mer mean")
axes[2].legend(frameon=False, markerscale=5, fontsize=8); axes[2].set_title("r9.4.1 6-mer model")
plt.tight_layout(); plt.savefig(f"{IMG}/fig12_sim_training.png", dpi=150); plt.close()
