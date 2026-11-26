"""Shared episodic evaluation harness (same episodes for every method)."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score, accuracy_score
from common import OUT

EP = None


def episodes():
    global EP
    if EP is None:
        EP = dict(np.load(f'{OUT}/episodes.npz'))
    return EP


def head_euclid(S, Qe):
    P = S.mean(1)
    d = ((Qe[:, None, :] - P[None]) ** 2).sum(-1)
    return -d


def head_cosine(S, Qe):
    P = S.mean(1); P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    Qn = Qe / (np.linalg.norm(Qe, axis=1, keepdims=True) + 1e-9)
    return Qn @ P.T


def evaluate(emb_fn, head_fn, pools=('all', 'dedup'), settings=('4way', '5way'), Ks=(1, 5, 10),
             n_ep=500, kmeans_ep=50):
    """emb_fn(global_idx[np.int64]) -> embeddings [n,d] ; head_fn(S[N,K,d], Q[nq,d]) -> scores [nq,N]"""
    E = episodes(); res = {}
    for pool in pools:
        for st in settings:
            for K in Ks:
                S_all = E[f'{pool}_{st}_K{K}_S'][:n_ep]; Q_all = E[f'{pool}_{st}_K{K}_Q'][:n_ep]
                N, Qn = S_all.shape[1], Q_all.shape[2]
                uniq = np.unique(np.r_[S_all.ravel(), Q_all.ravel()])
                Z = emb_fn(uniq); pos = {g: i for i, g in enumerate(uniq)}
                f1s, nmis, accs, km = [], [], [], []
                yq = np.repeat(np.arange(N), Qn)
                for e in range(len(S_all)):
                    S = Z[[pos[g] for g in S_all[e].ravel()]].reshape(N, K, -1)
                    Qe = Z[[pos[g] for g in Q_all[e].ravel()]]
                    pred = np.asarray(head_fn(S, Qe)).argmax(1)
                    f1s.append(f1_score(yq, pred, average='macro', zero_division=0))
                    nmis.append(normalized_mutual_info_score(yq, pred)); accs.append(accuracy_score(yq, pred))
                    if e < kmeans_ep:
                        lab = KMeans(N, n_init=1, random_state=e).fit_predict(Qe)
                        km.append(normalized_mutual_info_score(yq, lab))
                ci = lambda v: float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v)))
                res[f'{pool}_{st}_K{K}'] = {'f1': float(np.mean(f1s)), 'f1_ci': ci(f1s), 'nmi': float(np.mean(nmis)),
                                            'nmi_ci': ci(nmis), 'acc': float(np.mean(accs)),
                                            'kmeans_nmi': float(np.mean(km)) if km else float('nan'), 'n_ep': len(S_all)}
    return res
