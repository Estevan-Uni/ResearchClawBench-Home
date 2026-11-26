"""Data overview, lambda scan, certified reference solution of the Lasso."""
import json, time, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *

A, b, xt, meta = load_data()
m, n = A.shape
s = np.linalg.svd(A, compute_uv=False)
facts = dict(meta=str(meta), m=m, n=n, smax=float(s.max()), smin_nonzero=float(s.min()),
             L=float(s.max() ** 2), cond_A=float(s.max() / s.min()), rank=int(np.sum(s > 1e-10)),
             null_dim=int(n - np.sum(s > 1e-10)), nnz_true=int(np.count_nonzero(xt)),
             resid_true=float(np.linalg.norm(A @ xt - b)), lam_max=float(np.abs(A.T @ b).max()),
             F0=float(0.5 * b @ b), xt_absmin=float(np.abs(xt[xt != 0]).min()),
             xt_absmax=float(np.abs(xt).max()), norm_b=float(np.linalg.norm(b)))
print(facts)
lmax = facts['lam_max']; smax = s.max()

scan = {}
sols = {}
St = xt != 0
for frac in [0.005, 0.01, 0.02, 0.05, 0.1]:
    P = Problem(A, b, lam=frac * lmax, smax=smax)
    t = time.time(); x = reference(P, K=3000); P.set_ref(x)
    S = x != 0
    on, off, slack = P.kkt(x)
    scan[frac] = dict(lam=frac * lmax, F=float(P.Fs), nnz=int(S.sum()), TP=int(np.sum(S & St)),
                      FP=int(np.sum(S & ~St)), FN=int(np.sum(~S & St)),
                      relerr=float(np.linalg.norm(x - xt) / np.linalg.norm(xt)),
                      kkt_on=float(on), kkt_off=float(off), dgap=float(P.duality_gap(x)), time=time.time() - t)
    sols[frac] = x
    print(frac, scan[frac])

frac = 0.01
best = min(scan, key=lambda f: scan[f]['relerr'])
print('best frac by recovery', best)
lam = frac * lmax
P = Problem(A, b, lam=lam, smax=smax)
xs = sols[frac]; P.set_ref(xs)
S = xs != 0
AS = A[:, S]; ev = np.linalg.eigvalsh(AS.T @ AS)
on, off, slack = P.kkt(xs)
ref = dict(frac=frac, lam=lam, Fs=float(P.Fs), nnz=int(S.sum()), kkt_on=float(on), kkt_off=float(off),
           min_offsupport_slack=float(slack), fpr=float(P.fpr(xs)), dgap=float(P.duality_gap(xs)),
           mu_S=float(ev.min()), L_S=float(ev.max()), sqrt_muS_LS=float(np.sqrt(ev.min() * ev.max())),
           TP=int(np.sum(S & St)), FP=int(np.sum(S & ~St)), FN=int(np.sum(~S & St)),
           FP_maxabs=float(np.abs(xs[S & ~St]).max()) if np.any(S & ~St) else 0.0,
           FN_true_vals=[float(v) for v in xt[~S & St]],
           relerr=float(np.linalg.norm(xs - xt) / np.linalg.norm(xt)),
           min_abs_on_support=float(np.abs(xs[S]).min()))
print(ref)
np.save(f'{OUT}/xstar.npy', xs)
json.dump(dict(facts=facts, scan={str(k): v for k, v in scan.items()}, ref=ref, best_frac=best),
          open(f'{OUT}/reference.json', 'w'), indent=2)

# ---------------- data overview figure ----------------
fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
ax[0].plot(np.arange(1, m + 1), s, color='#1f5fa8', lw=1.5)
ax[0].axhline(0, color='k', lw=0.5)
ax[0].set_xlabel('index i'); ax[0].set_ylabel(r'$\sigma_i(A)$')
ax[0].set_title(f'(a) singular values: [{s.min():.0f}, {s.max():.0f}], rank {facts["rank"]}\n'
                f'+ {facts["null_dim"]}-dim null space ($\\mu=0$)', fontsize=9.5)
idx = np.nonzero(xt)[0]
ax[1].vlines(idx, 0, xt[idx], color='#444', lw=0.8, label=r'$x_{true}$')
ax[1].plot(np.nonzero(xs)[0], xs[xs != 0], 'o', ms=2.5, mfc='none', color='#d1495b', label=r'$x^\star$ (Lasso)')
ax[1].set_xlabel('coordinate j'); ax[1].set_title(f'(b) ground truth ({facts["nnz_true"]} nonzeros) and $x^\\star$', fontsize=9.5)
ax[1].legend(fontsize=8, frameon=False)
g0 = np.abs(A.T @ b)
ax[2].hist(g0, bins=60, color='#8aa', edgecolor='none')
yl = ax[2].get_ylim()[1]
for f, yy in [(0.01, 0.92), (0.1, 0.78)]:
    ax[2].axvline(f * lmax, color='#d1495b' if f == 0.01 else '#999', ls='--', lw=1)
    ax[2].text(f * lmax + 0.6, yl * yy, f'{f}$\\lambda_{{max}}$', fontsize=8)
ax[2].set_xlabel(r'$|A^\top b|_j$'); ax[2].set_title(r'(c) $|A^\top b|$, $\lambda_{max}$=%.2f' % lmax, fontsize=9.5)
fr = sorted(scan)
ax[3].semilogx(fr, [100 * scan[f]['relerr'] for f in fr], 'o-', color='#1f5fa8', ms=7)
for f in fr:
    ax[3].annotate(f"nnz={scan[f]['nnz']}\nFP={scan[f]['FP']}, FN={scan[f]['FN']}", (f, 100 * scan[f]['relerr']),
                   xytext=(-14, 9), textcoords='offset points', fontsize=7, color='#52514e')
ax[3].plot([0.01], [100 * scan[0.01]['relerr']], 'o', ms=12, mfc='none', mec='#d1495b', mew=1.5)
ax[3].set_ylim(0, 34)
ax[3].set_xlabel(r'$\lambda/\lambda_{max}$'); ax[3].set_ylabel(r'$\|x^\star_\lambda-x_{true}\|/\|x_{true}\|$ (%)')
ax[3].set_title('(d) $\\lambda$ scan (circled: value used)', fontsize=9.5)
plt.tight_layout(); plt.savefig(f'{IMG}/fig1_data_overview.png', dpi=150)
print('done')
