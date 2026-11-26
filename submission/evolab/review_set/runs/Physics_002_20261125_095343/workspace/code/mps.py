"""Truncated MPS simulator with SWAP routing (qubit permutation tracked, no swap-back) and
discarded-weight fidelity estimate F ~ prod(1 - delta)."""
import numpy as np, scipy.linalg as sla
from circuits import ZZ

SWAP = np.zeros((2, 2, 2, 2)); 
for a in range(2):
    for b in range(2): SWAP[b, a, a, b] = 1
ZZ_SWAP = np.einsum('abcd,cdef->abef', SWAP, ZZ)  # first ZZ then swap

def svd(m):
    try: return np.linalg.svd(m, full_matrices=False)
    except np.linalg.LinAlgError: return sla.svd(m, full_matrices=False, lapack_driver='gesvd')

class MPS:
    def __init__(self, N, chi):
        self.N, self.chi = N, chi
        self.A = [np.zeros((1, 2, 1), complex) for _ in range(N)]
        for t in self.A: t[0, 0, 0] = 1
        self.center = 0; self.logF = 0.0
        self.pos = list(range(N))  # pos[q] = site of qubit q
        self.q_at = list(range(N))  # q_at[site] = qubit
    def move_center(self, i):
        while self.center < i:
            c = self.center; t = self.A[c]; Dl, _, Dr = t.shape
            q, r = np.linalg.qr(t.reshape(Dl * 2, Dr)); self.A[c] = q.reshape(Dl, 2, -1)
            self.A[c + 1] = np.tensordot(r, self.A[c + 1], axes=(1, 0)); self.center += 1
        while self.center > i:
            c = self.center; t = self.A[c]; Dl, _, Dr = t.shape
            q, r = np.linalg.qr(t.reshape(Dl, 2 * Dr).T); self.A[c] = q.T.reshape(-1, 2, Dr)
            self.A[c - 1] = np.tensordot(self.A[c - 1], r.T, axes=(2, 0)); self.center -= 1
    def apply1(self, U, q):
        s = self.pos[q]; self.A[s] = np.einsum('ab,lbr->lar', U, self.A[s])
    def apply2_sites(self, G, i):
        """G[a',b',a,b] on sites (i,i+1)"""
        self.move_center(i)
        th = np.einsum('lar,rbs->labs', self.A[i], self.A[i + 1])
        th = np.einsum('xyab,labs->lxys', G, th)
        Dl, _, _, Dr = th.shape
        u, s, vh = svd(th.reshape(Dl * 2, 2 * Dr))
        s2 = s ** 2; tot = s2.sum(); k = min(self.chi, int((s > 1e-14 * s[0]).sum()))
        kept = s2[:k].sum(); self.logF += np.log(kept / tot)
        s = s[:k] / np.sqrt(kept)
        self.A[i] = u[:, :k].reshape(Dl, 2, k); self.A[i + 1] = (s[:, None] * vh[:k]).reshape(k, 2, Dr)
        self.center = i + 1
    def zz(self, qa, qb):
        pa, pb = self.pos[qa], self.pos[qb]
        if pa > pb: pa, pb = pb, pa; qa, qb = qb, qa
        # bring qb leftward next to qa; final op applies ZZ then swap? simpler: swap until adjacent, then ZZ
        if pb - pa == 1:
            self.apply2_sites(ZZ, pa); return
        # route: move qb from pb to pa+1 via swaps (choose the side closest to the center)
        for s in range(pb - 1, pa, -1):
            self.apply2_sites(SWAP, s)
            q1, q2 = self.q_at[s], self.q_at[s + 1]
            self.q_at[s], self.q_at[s + 1] = q2, q1; self.pos[q2], self.pos[q1] = s, s + 1
        self.apply2_sites(ZZ, pa)
    def to_vector(self):
        v = self.A[0]
        for t in self.A[1:]: v = np.tensordot(v, t, axes=(v.ndim - 1, 0))
        v = v.reshape((2,) * self.N)
        # sites -> qubits
        return np.transpose(v, np.argsort(self.q_at) if False else [self.pos[q] for q in range(self.N)])

def run_mps(N, layers, chi, record=True):
    m = MPS(N, chi); out = []
    for us, prs in layers:
        for q, U in enumerate(us): m.apply1(U, q)
        # order pairs to reduce routing: sort by the left position
        for a, b in sorted(prs, key=lambda p: min(m.pos[p[0]], m.pos[p[1]])):
            m.zz(a, b)
        out.append(m.logF)
    return np.array(out), m
