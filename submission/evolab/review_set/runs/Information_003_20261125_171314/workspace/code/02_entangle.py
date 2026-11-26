"""Data overview + entanglement evidence + SD diagnostics.
Outputs: report/images/{data_overview,entanglement_overlap,sd_kde,sd_corr}.png, outputs/entangle.json
"""
import sys, numpy as np
sys.path.insert(0, '/workspace/code')
from common import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, gaussian_kde
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

style()
D = load(); X = D['msg']; y = D['attack']; t = D['t']
H = np.load(f'{OUT}/sd_h.npy')
tr, va, te = chrono_split(len(y))
res = {}

# ---------- data overview ----------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
cnt = np.bincount(y, minlength=10); order = np.argsort(-cnt)
ax[0].barh([CLASSES[i] for i in order][::-1], cnt[order][::-1], color=[CLASS_COL[i] for i in order][::-1])
ax[0].set_xscale('log'); ax[0].set_xlabel('flows (log scale)'); ax[0].set_title('Class distribution (148,774 flows)')
for j, i in enumerate(order[::-1]):
    ax[0].text(cnt[i] * 1.1, j, f'{cnt[i]:,}', va='center', fontsize=7, color=INK2)
ax[0].grid(axis='y', visible=False)
bins = np.linspace(0, 86400, 49)
for c in [3, 6, 7, 0, 9]:
    h_, _ = np.histogram(t[y == c], bins); ax[1].plot(bins[:-1] / 3600, h_ / max(h_.sum(), 1), color=CLASS_COL[c], lw=1.5, label=CLASSES[c])
ax[1].axvline(t[tr[-1]] / 3600, color=INK2, ls='--', lw=1); ax[1].axvline(t[va[-1]] / 3600, color=INK2, ls='--', lw=1)
ax[1].text(t[tr[-1]] / 3600, ax[1].get_ylim()[1] * 0.95, ' val', fontsize=7, color=INK2)
ax[1].text(t[va[-1]] / 3600, ax[1].get_ylim()[1] * 0.95, ' test', fontsize=7, color=INK2)
ax[1].set_xlabel('time of day (h)'); ax[1].set_ylabel('fraction of class flows'); ax[1].set_title('Temporal profile + chronological split')
ax[1].legend(fontsize=7, ncol=2)
deg = np.bincount(np.concatenate([D['src_i'], D['dst_i']]), minlength=D['n_nodes'])
vals, cts = np.unique(deg, return_counts=True)
ax[2].loglog(vals, cts, 'o', ms=3, color=PAL[0]); ax[2].set_xlabel('node degree'); ax[2].set_ylabel('# nodes')
ax[2].set_title(f'Degree distribution ({D["n_nodes"]:,} IP:port nodes)')
plt.tight_layout(); plt.savefig(f'{IMG}/data_overview.png'); plt.close()
src_deg = np.bincount(D['src_i'], minlength=D['n_nodes'])
res['frac_flows_once_seen_src'] = float((src_deg[D['src_i']] == 1).mean())
res['n_nodes'] = int(D['n_nodes']); res['layer1_frac_src'] = float(D['layer_src'].mean()); res['layer1_frac_dst'] = float(D['layer_dst'].mean())
res['class_counts'] = cnt.tolist()
res['split_sizes'] = [len(tr), len(va), len(te)]
res['split_class_counts'] = [np.bincount(y[s], minlength=10).tolist() for s in (tr, va, te)]
# duplication
trset = set(map(bytes, X[tr]))
res['frac_test_rows_in_train'] = float(np.mean([bytes(r) in trset for r in X[te]]))
res['n_unique_rows'] = int(len(np.unique(X, axis=0)))

# ---------- entanglement: per-feature pairwise overlap 1-2|AUC-.5| ----------
attacks = [0, 1, 3, 4, 5, 6, 7, 8, 9]
rng = np.random.default_rng(0)
def sub(c, n=3000):
    idx = np.where(y == c)[0]; return idx if len(idx) <= n else rng.choice(idx, n, replace=False)
S = {c: sub(c) for c in range(10)}
def overlap_matrix(Z):
    M = np.zeros((10, 10)); P = np.zeros((10, 10, Z.shape[1]))
    for a in range(10):
        for b in range(a + 1, 10):
            ov = []
            for f in range(Z.shape[1]):
                za, zb = Z[S[a], f], Z[S[b], f]
                if np.ptp(np.r_[za, zb]) == 0: ov.append(1.0); continue
                auc = roc_auc_score(np.r_[np.zeros(len(za)), np.ones(len(zb))], np.r_[za, zb])
                ov.append(1 - 2 * abs(auc - 0.5))
            P[a, b] = P[b, a] = ov
            M[a, b] = M[b, a] = np.mean(np.sort(ov)[:5])   # best-5 features: low = separable
    return M, P
M_F, P_F = overlap_matrix(X)
res['overlap_best5_F'] = M_F.tolist()
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={'width_ratios': [1, 1.25]})
mask = np.eye(10, dtype=bool)
im = ax[0].imshow(np.where(mask, np.nan, M_F), cmap='Blues', vmin=0, vmax=1)
ax[0].set_xticks(range(10)); ax[0].set_xticklabels(CLASSES, rotation=60, ha='right'); ax[0].set_yticks(range(10)); ax[0].set_yticklabels(CLASSES)
for a in range(10):
    for b in range(10):
        if a != b: ax[0].text(b, a, f'{M_F[a,b]:.2f}', ha='center', va='center', fontsize=6, color='white' if M_F[a, b] > 0.6 else INK)
ax[0].grid(False); ax[0].set_title('Class-pair overlap (mean of 5 most separating features)\n0 = perfectly separable, 1 = identical')
plt.colorbar(im, ax=ax[0], fraction=0.046)
# per-feature overlap for selected attack pairs
pairs = [(3, 4), (0, 1), (3, 1), (8, 9), (6, 2), (7, 2)]
Pm = np.array([P_F[a, b] for a, b in pairs])
im = ax[1].imshow(Pm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
ax[1].set_yticks(range(len(pairs))); ax[1].set_yticklabels([f'{CLASSES[a]}–{CLASSES[b]}' for a, b in pairs])
ax[1].set_xlabel('feature index'); ax[1].grid(False); ax[1].set_title('Per-feature overlap for selected class pairs')
plt.colorbar(im, ax=ax[1], fraction=0.046)
plt.tight_layout(); plt.savefig(f'{IMG}/entanglement_overlap.png'); plt.close()
res['pair_overlap_mean_allfeat'] = {f'{CLASSES[a]}-{CLASSES[b]}': float(P_F[a, b].mean()) for a, b in pairs}

# ---------- SD diagnostics ----------
idx = rng.choice(len(y), 20000, replace=False)
def mean_abs_spearman(Z):
    Z = Z[idx]; keep = Z.std(0) > 0
    r = spearmanr(Z[:, keep]).correlation; r = np.abs(r[~np.eye(len(r), dtype=bool)])
    return float(np.nanmean(r))
res['mean_abs_spearman_F'] = mean_abs_spearman(X); res['mean_abs_spearman_h'] = mean_abs_spearman(H)
M_H, P_H = overlap_matrix(H)
res['overlap_best5_H'] = M_H.tolist()
iu = np.triu_indices(10, 1)
res['mean_pair_overlap_allfeat_F'] = float(P_F[iu].mean()); res['mean_pair_overlap_allfeat_H'] = float(P_H[iu].mean())
res['mean_pair_overlap_best5_F'] = float(M_F[iu].mean()); res['mean_pair_overlap_best5_H'] = float(M_H[iu].mean())
# kNN attack typing on F, h, [F||h] (attack flows only; quantile-transformed, train->test)
Fq = quantile_fit(X, tr); Hq = quantile_fit(H, tr)
att_tr = tr[y[tr] != BENIGN]; att_te = te[y[te] != BENIGN]
for name, Z in [('F', Fq), ('h', Hq), ('F||h', np.c_[Fq, Hq])]:
    knn = KNeighborsClassifier(5, n_jobs=2).fit(Z[att_tr], y[att_tr])
    p = knn.predict(Z[att_te])
    res[f'knn_attack_macroF1_{name}'] = float(f1_score(y[att_te], p, average='macro'))
print({k: v for k, v in res.items() if not isinstance(v, list)})

# KDE small multiples: pick features with largest overlap change for DoS/Exploits/Backdoor/Analysis/Benign
cls_show = [2, 3, 4, 1, 0]
sel = np.argsort([np.mean([P_F[a, b, f] - P_H[a, b, f] for a in cls_show for b in cls_show if a < b]) for f in range(40)])[::-1]
sel = [f for f in sel if X[:, f].std() > 0][:5]
res['kde_features'] = [int(f) for f in sel]
fig, ax = plt.subplots(2, 5, figsize=(14, 5))
for j, f in enumerate(sel):
    for r, (Z, nm) in enumerate([(Fq, 'F (quantile)'), (Hq, 'h = w⊙F (quantile)')]):
        a = ax[r, j]
        for c in cls_show:
            z = Z[S[c], f]
            if np.ptp(z) == 0: a.axvline(z[0], color=CLASS_COL[c] if c != 2 else GRAY, lw=1.5); continue
            xs = np.linspace(0, 1, 200); k = gaussian_kde(z, bw_method=0.15)(xs)
            a.plot(xs, k / k.max(), color=CLASS_COL[c] if c != 2 else GRAY, lw=1.5, label=CLASSES[c])
        ov = np.mean([(P_F if r == 0 else P_H)[a_, b_, f] for a_ in cls_show for b_ in cls_show if a_ < b_])
        a.set_title(f'feat {f} – {nm}\nmean pair overlap {ov:.2f}', fontsize=8); a.set_yticks([])
from matplotlib.lines import Line2D
ax[0, 0].legend(handles=[Line2D([0], [0], color=CLASS_COL[c] if c != 2 else GRAY, lw=1.5, label=CLASSES[c]) for c in cls_show], fontsize=7)
fig.suptitle('Five features with the largest overlap reduction after SD (selected post hoc; averaged over all 40 features the overlap is unchanged)', fontsize=9, y=1.01)
plt.tight_layout(); plt.savefig(f'{IMG}/sd_kde.png'); plt.close()

# correlation heatmaps of F vs h
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for a, (Z, nm) in zip(ax, [(X, 'raw features F'), (H, 'SD features h')]):
    Zs = Z[idx]; keep = Zs.std(0) > 0
    r = np.abs(spearmanr(Zs[:, keep]).correlation)
    im = a.imshow(r, cmap='Blues', vmin=0, vmax=1); a.grid(False)
    a.set_title(f'|Spearman| {nm}\nmean off-diag = {np.nanmean(r[~np.eye(len(r), dtype=bool)]):.3f}')
plt.colorbar(im, ax=ax, fraction=0.03)
plt.savefig(f'{IMG}/sd_corr.png'); plt.close()
save_json(res, f'{OUT}/entangle.json')
np.save(f'{OUT}/Fq.npy', Fq); np.save(f'{OUT}/Hq.npy', Hq)
