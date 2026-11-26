"""Representation visualisations: t-SNE, |corr| heatmaps, confusion matrices, scale attention,
training curves."""
import sys, json, glob, os, numpy as np
sys.path.insert(0, '/workspace/code')
from common import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
style()
RUN = f'{OUT}/runs'
J = lambda n: json.load(open(f'{RUN}/{n}.json')) if os.path.exists(f'{RUN}/{n}.json') else None
res = {}


def tsne_panel(ax, Z, y, title, classes):
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    T = TSNE(2, perplexity=30, init='pca', random_state=0).fit_transform(Zs)
    for c in classes:
        m = y == c
        ax.scatter(T[m, 0], T[m, 1], s=5 if c == BENIGN else 7, color=CLASS_COL[c], alpha=0.6 if c == BENIGN else 0.85,
                   label=CLASSES[c], edgecolors='none')
        cx, cy = np.median(T[m], 0)
        ax.text(cx, cy, CLASSES[c], fontsize=6.5, color=INK, ha='center', weight='bold')
    sil = silhouette_score(Zs, y); km = KMeans(len(classes), n_init=5, random_state=0).fit_predict(Zs)
    nmi = normalized_mutual_info_score(y, km)
    ax.set_title(f'{title}\nsilhouette={sil:.3f}  k-means NMI={nmi:.3f}', fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    return sil, nmi


# ---- 1. t-SNE of novel-class embeddings (few-shot protocol)
cands = [('Raw features [F||h]', None), ('BSNet-style', 'fewshot_bsnet_s0'), ('E-GraphSAGE', 'fewshot_egs_s0'),
         ('TGN', 'fewshot_tgn_s0'), ('DIDS-MFL', 'fewshot_dids_mfl_s0')]
cands = [c for c in cands if c[1] is None or os.path.exists(f'{RUN}/{c[1]}_emb.npz')]
rng = np.random.default_rng(0)
ref = np.load(f'{RUN}/fewshot_dids_mfl_s0_emb.npz') if os.path.exists(f'{RUN}/fewshot_dids_mfl_s0_emb.npz') else np.load(f'{RUN}/{cands[1][1]}_emb.npz')
gsel = np.r_[ref['gid'][ref['y'] != BENIGN], rng.choice(ref['gid'][ref['y'] == BENIGN], 400, replace=False)]
Fq = np.load(f'{OUT}/Fq.npy'); Hq = np.load(f'{OUT}/Hq.npy'); yall = np.load(f'{OUT}/data.npz')['attack']
fig, axs = plt.subplots(1, len(cands), figsize=(4 * len(cands), 4))
res['tsne_novel'] = {}
for ax, (lab, run) in zip(axs, cands):
    if run is None:
        Z = np.c_[Fq, Hq][gsel]; yy = yall[gsel]
    else:
        E = np.load(f'{RUN}/{run}_emb.npz'); pos = {g: i for i, g in enumerate(E['gid'])}
        keep = [g for g in gsel if g in pos]; Z = E['z'][[pos[g] for g in keep]]; yy = yall[keep]
    m = yy != BENIGN   # metrics on novel classes only
    s, n = tsne_panel(ax, Z, yy, lab, [BENIGN] + NOVEL)
    res['tsne_novel'][lab] = {'silhouette_with_benign': s, 'kmeans_nmi_with_benign': n,
                              'silhouette_novel_only': float(silhouette_score((Z[m] - Z[m].mean(0)) / (Z[m].std(0) + 1e-6), yy[m]))}
axs[0].legend(markerscale=2.5, fontsize=7, loc='lower left')
plt.tight_layout(); plt.savefig(f'{IMG}/tsne_novel.png'); plt.close()

# ---- 2. t-SNE known protocol (test sample)
cands = [('E-GraphSAGE', 'known_egs_s0'), ('TGN', 'known_tgn_s0'), ('DIDS-MFL', 'known_dids_mfl_s0')]
cands = [c for c in cands if os.path.exists(f'{RUN}/{c[1]}_emb.npz')]
fig, axs = plt.subplots(1, len(cands), figsize=(4.6 * len(cands), 4.4))
res['tsne_known'] = {}
for ax, (lab, run) in zip(np.atleast_1d(axs), cands):
    E = np.load(f'{RUN}/{run}_emb.npz'); idx = []
    for c in range(10):
        w = np.where(E['y'] == c)[0]; idx += list(rng.choice(w, min(200, len(w)), replace=False))
    idx = np.array(idx)
    s, n = tsne_panel(ax, E['z'][idx], E['y'][idx], lab, list(range(10)))
    res['tsne_known'][lab] = {'silhouette': s, 'kmeans_nmi': n}
np.atleast_1d(axs)[0].legend(markerscale=2.5, fontsize=6.5, loc='lower left', ncol=2)
plt.tight_layout(); plt.savefig(f'{IMG}/tsne_known.png'); plt.close()

# ---- 3. |corr| heatmaps of learned embeddings (known protocol, test sample)
pairs = [('E-GraphSAGE  z', 'known_egs_s0', 'z'), ('TGN  z', 'known_tgn_s0', 'z'), ('DIDS-MFL  z', 'known_dids_mfl_s0', 'z'),
         ('TGN  memory-scale z_m', 'known_tgn_s0', 'zm'), ('DIDS-MFL w/o RD  z_m', 'known_mfl_noRD_s0', 'zm'),
         ('DIDS-MFL  z_m', 'known_dids_mfl_s0', 'zm')]
pairs = [p for p in pairs if os.path.exists(f'{RUN}/{p[1]}_emb.npz')]
fig, axs = plt.subplots(2, 3, figsize=(12, 7.8)); axs = axs.ravel()
res['corr'] = {}
for ax, (lab, run, key) in zip(axs, pairs):
    E = np.load(f'{RUN}/{run}_emb.npz'); Z = E[key]; Z = Z[:, Z.std(0) > 1e-8]
    C = np.abs(np.corrcoef(Z.T)); off = C[~np.eye(len(C), dtype=bool)].mean()
    o = np.argsort(-C.mean(0))
    im = ax.imshow(C[np.ix_(o, o)], cmap='Blues', vmin=0, vmax=1); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{lab}\nmean off-diag |r| = {off:.3f}', fontsize=9); res['corr'][lab] = float(off)
for ax in axs[len(pairs):]: ax.axis('off')
fig.colorbar(im, ax=axs.tolist(), fraction=0.02)
plt.savefig(f'{IMG}/repr_corr.png'); plt.close()

# per-class representation correlation (paper Fig 2e/f analogue): DIDS-MFL vs E-GraphSAGE
res['corr_by_class'] = {}
for lab, run in [('E-GraphSAGE', 'known_egs_s0'), ('TGN', 'known_tgn_s0'), ('DIDS-MFL', 'known_dids_mfl_s0')]:
    if not os.path.exists(f'{RUN}/{run}_emb.npz'): continue
    E = np.load(f'{RUN}/{run}_emb.npz'); d = {}
    for c in range(10):
        Z = E['z'][E['y'] == c]; Z = Z[:, Z.std(0) > 1e-8]
        if len(Z) > 20:
            C = np.abs(np.corrcoef(Z.T)); d[CLASSES[c]] = float(C[~np.eye(len(C), dtype=bool)].mean())
    res['corr_by_class'][lab] = d

# ---- 4. confusion matrices
cms = [('RF', 'known_rf_s0'), ('E-GraphSAGE', 'known_egs_s0'), ('DIDS-MFL', 'known_dids_mfl_s0')]
cms = [c for c in cms if J(c[1])]
fig, axs = plt.subplots(1, len(cms), figsize=(5 * len(cms), 4.5))
for ax, (lab, run) in zip(np.atleast_1d(axs), cms):
    cm = np.array(J(run)['test']['cm'], float); cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1); ax.grid(False)
    for i in range(10):
        for j in range(10):
            if cmn[i, j] >= 0.01: ax.text(j, i, f'{cmn[i,j]:.2f}', ha='center', va='center', fontsize=5.5, color='white' if cmn[i, j] > 0.6 else INK)
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=60, ha='right', fontsize=7); ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES, fontsize=7)
    ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title(f'{lab} (row-normalised, test)')
plt.tight_layout(); plt.savefig(f'{IMG}/confusion.png'); plt.close()

# ---- 5. scale attention per class
r = J('known_dids_mfl_s0'); rf = J('fewshot_dids_mfl_s0')
if r and 'scale_attention_by_class' in r:
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, rr, ttl in [(axs[0], r, 'known protocol'), (axs[1], rf, 'few-shot protocol (novel classes unseen)')]:
        if rr is None: continue
        A = rr['scale_attention_by_class']; cl = [c for c in CLASSES if c in A]; M = np.array([A[c] for c in cl])
        bottom = np.zeros(len(cl))
        for j, nm in enumerate(['edge/statistical z_e', 'memory z_m', 'diffusion z_d'][:M.shape[1]]):
            ax.bar(range(len(cl)), M[:, j], bottom=bottom, color=PAL[j], label=nm, edgecolor='#fcfcfb', linewidth=1)
            bottom += M[:, j]
        ax.set_xticks(range(len(cl))); ax.set_xticklabels(cl, rotation=30, ha='right'); ax.set_title(f'MFL scale attention ({ttl})')
        ax.grid(axis='x', visible=False)
    axs[0].set_ylabel('mean attention weight'); axs[1].legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.12), ncol=3)
    plt.tight_layout(); plt.savefig(f'{IMG}/scale_attention.png'); plt.close()
    res['scale_attention_known'] = r['scale_attention_by_class']

# ---- 6. training curves
fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
for j, (lab, m) in enumerate([('TGN', 'tgn'), ('3D-IDS', '3dids'), ('DIDS-MFL', 'dids_mfl')]):
    for ls, prot in [('-', 'known'), ('--', 'fewshot')]:
        for s in range(3):
            rr = J(f'{prot}_{m}_s{s}')
            if rr is None: continue
            v = [h['val_macro_f1'] for h in rr['history']]
            axs[0].plot(range(1, len(v) + 1), v, ls, color=PAL[j], lw=1.2, alpha=0.8, label=f'{lab} ({prot})' if s == 0 else None)
axs[0].set_xlabel('epoch'); axs[0].set_ylabel('val macro-F1 (label space)'); axs[0].legend(fontsize=7); axs[0].set_title('Validation curves (3 seeds)')
for j, (lab, n) in enumerate([('DIDS-MFL 8 ep', 'known_dids_mfl_s0'), ('DIDS-MFL 20 ep', 'known_dids_mfl_s0_long'),
                              ('w/o diffusion 20 ep', 'known_mfl_noDiff_s0_long')]):
    rr = J(n)
    if rr: axs[1].plot([h['val_macro_f1'] for h in rr['history']], color=PAL[j], marker='o', ms=3, label=lab)
axs[1].set_title('Longer budget (known)'); axs[1].set_xlabel('epoch'); axs[1].legend(fontsize=7)
rr = J('known_dids_mfl_s0')
if rr:
    for j, k in enumerate(['int', 'rd', 'dis', 'ep']):
        axs[2].plot([h[k] for h in rr['history']], color=PAL[j], marker='o', ms=3, label={'int': 'L_int', 'rd': 'L_RD', 'dis': 'L_Dis', 'ep': 'L_episode'}[k])
    axs[2].set_title('DIDS-MFL loss components (known, s0)'); axs[2].set_xlabel('epoch'); axs[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{IMG}/training_curves.png'); plt.close()
save_json(res, f'{OUT}/viz_stats.json')
print(json.dumps(res, indent=1)[:3000])
