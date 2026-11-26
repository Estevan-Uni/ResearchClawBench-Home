"""Posterior characterisation of the two BH mass-spin sample sets."""
import numpy as np, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker
from scipy.stats import gaussian_kde

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {"M33 X-7": "M33_X-7_samples.dat", "IRAS 09149-6206": "IRAS_09149-6206_samples.dat"}


def load(name):
    d = np.loadtxt(os.path.join(ROOT, "data", FILES[name]))
    return d[:, 0], d[:, 1]


def summ(x):
    p = np.percentile(x, [2.5, 16, 50, 84, 97.5])
    return dict(mean=float(x.mean()), std=float(x.std(ddof=1)), p2_5=p[0], p16=p[1], median=p[2], p84=p[3],
                p97_5=p[4], min=float(x.min()), max=float(x.max()), n_unique=int(len(np.unique(x))))


def hpd_levels(Z, fracs):
    z = np.sort(Z.ravel())[::-1]; c = np.cumsum(z); c /= c[-1]
    return [z[np.searchsorted(c, f)] for f in fracs]


if __name__ == "__main__":
    out = {}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, name in enumerate(FILES):
        M, a = load(name)
        st = dict(N=len(M), M=summ(M), a=summ(a), logM=summ(np.log10(M)),
                  corr_M_a=float(np.corrcoef(M, a)[0, 1]), corr_lnM_a=float(np.corrcoef(np.log(M), a)[0, 1]),
                  n_unique_pairs=int(len(np.unique(np.c_[M, a], axis=0))))
        from scipy.stats import spearmanr, skew
        st["spearman"] = float(spearmanr(M, a)[0]); st["skew_logM"] = float(skew(np.log10(M))); st["skew_a"] = float(skew(a))
        out[name] = st
        print(name, json.dumps(st, indent=1, default=float))
        lx = np.log10(M)
        ax = axes[row, 0]
        if row == 1:
            bins = np.logspace(lx.min(), lx.max(), 60); ax.hist(M, bins=bins, color="C0", alpha=.7, weights=np.full(len(M), 1 / (len(M) * np.diff(np.log10(bins))[0]))); ax.set_xscale("log"); ax.set_ylabel("density per dex")
        else:
            ax.hist(M, bins=50, color="C0", alpha=.7, density=True)
        ax.axvline(st["M"]["mean"], color="k", ls="-", label="mean"); ax.axvline(st["M"]["median"], color="k", ls="--", label="median")
        ax.set_xlabel(r"$M\ [M_\odot]$"); ax.set_ylabel("density per dex" if row == 1 else "density"); ax.set_title(f"{name}: mass marginal"); ax.legend(fontsize=8)
        ax = axes[row, 1]
        ax.hist(a, bins=50, color="C1", alpha=.7, density=True)
        ax.axvline(st["a"]["mean"], color="k"); ax.axvline(st["a"]["median"], color="k", ls="--")
        ax.set_xlabel(r"$a_*$"); ax.set_title(f"{name}: spin marginal")
        ax = axes[row, 2]
        ax.scatter(M, a, s=2, alpha=.3, color="0.4", rasterized=True)
        k = gaussian_kde(np.vstack([lx, a]))
        gx = np.linspace(lx.min(), lx.max(), 120); gy = np.linspace(a.min(), a.max(), 120)
        X, Y = np.meshgrid(gx, gy); Z = k(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        lv = hpd_levels(Z, [0.95, 0.68])
        ax.contour(10 ** X, Y, Z, levels=sorted(lv), colors=["C3", "C3"], linestyles=["--", "-"])
        if row == 1:
            ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(r"$M\ [M_\odot]$"); ax.set_ylabel(r"$a_*$")
        ax.set_title(f"joint (68/95% HPD), $\\rho$={st['corr_M_a']:.2f}")
    plt.tight_layout(); plt.savefig(os.path.join(ROOT, "report/images/fig1_posteriors.png"), dpi=140)
    json.dump(out, open(os.path.join(ROOT, "outputs/posterior_summary.json"), "w"), indent=1, default=float)
