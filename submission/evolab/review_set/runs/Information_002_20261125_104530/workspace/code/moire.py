"""Continuum model + plane-wave Hartree-Fock for AB-stacked MoTe2/WSe2 (arXiv:2111.01152).

Conventions (follow the paper / gold answers of the 16-step template chain):
  * electron picture, valley tau=+1/-1, layer l in {b (MoTe2), t (WSe2)}
  * H_tau = [[-hbar^2 k^2/2m_b + Delta_b(r),  Delta_T,tau(r)],
             [Delta_T,tau^dag(r), -hbar^2 (k - tau kappa)^2/2m_t + V_zt]]   (Delta_t = 0, paper convention)
  * Delta_b(r) = 2 V_b sum_{j=1,3,5} cos(g_j.r + psi_b)
  * Delta_T,tau(r) = tau w (1 + w^tau e^{i tau g2.r} + w^{2tau} e^{i tau g3.r}),  omega = e^{2 pi i/3}
  * plane-wave matrix element  h_{(l,G),(l',G')}(k) = (1/A) int e^{-i(k+G).r} H(r) e^{i(k+G').r}
  * hole basis: h~ = -h^T ; interaction V(q) = 2 pi e^2 tanh(qd)/(eps q)
Units: meV, nm.
"""
import numpy as np

HB2M = 38.0998          # hbar^2 / (2 m_e)  [meV nm^2]
E2 = 1439.964           # e^2 (Gaussian) [meV nm]
A_B, A_T = 0.3575, 0.332
A_M = A_B * A_T / abs(A_B - A_T)        # moire period [nm]
M_B, M_T = 0.65, 0.35
OMEGA = np.exp(2j * np.pi / 3)
GABS = 4 * np.pi / (np.sqrt(3) * A_M)
KAPPA = np.array([4 * np.pi / (3 * A_M), 0.0])


def gvec(j):
    """g_j = 4pi/(sqrt3 a_M) (-sin(pi(j-1)/3), cos(pi(j-1)/3)), j = 1..6"""
    th = np.pi * (j - 1) / 3
    return GABS * np.array([-np.sin(th), np.cos(th)])


B1, B2 = gvec(1), gvec(2)            # primitive reciprocal vectors (g3 = g2 - g1)
A_CELL = np.sqrt(3) / 2 * A_M ** 2   # moire unit-cell area


def gset(nshell):
    """integer pairs (m,n), G = m b1 + n b2, with hexagonal distance <= nshell"""
    out = []
    for m in range(-nshell, nshell + 1):
        for n in range(-nshell, nshell + 1):
            if max(abs(m), abs(n), abs(m + n)) <= nshell:
                out.append((m, n))
    out.sort(key=lambda p: (np.linalg.norm(p[0] * B1 + p[1] * B2), p))
    return np.array(out)


def int_coords(vec):
    """integer coordinates of a reciprocal lattice vector in (b1,b2) basis"""
    M = np.column_stack([B1, B2])
    c = np.linalg.solve(M, vec)
    r = np.rint(c).astype(int)
    assert np.allclose(c, r, atol=1e-8)
    return tuple(r)


class Model:
    def __init__(self, w=12.0, Vb=7.0, psi_deg=-14.0, Vzt=-20.0, nshell=2,
                 variant="gold"):
        """variant: 'gold' (paper), 'both_shift' (GPT-4/auto slip: bottom layer also shifted by tau kappa),
        'auto_plus' (auto.md step-2/6 slip: +hbar^2(k - tau kappa)^2/2m in both layers),
        'no_tau_T' (drop tau prefactor of Delta_T; pure gauge)"""
        self.w, self.Vb, self.psi, self.Vzt = w, Vb, np.deg2rad(psi_deg), Vzt
        self.variant = variant
        self.G = gset(nshell)
        self.nG = len(self.G)
        self.Gc = self.G[:, 0:1] * B1 + self.G[:, 1:2] * B2      # cartesian
        self.idx = {tuple(g): i for i, g in enumerate(self.G)}
        self._build_potential()

    # ---- Fourier components of the potentials (step 7) ----
    def fourier_components(self, tau):
        """dicts {(m,n): coefficient}:  Delta(r) = sum_q c_q e^{i q.r}"""
        Db = {}
        for j in (1, 3, 5):
            q = int_coords(gvec(j))
            Db[q] = Db.get(q, 0) + self.Vb * np.exp(1j * self.psi)
            mq = (-q[0], -q[1])
            Db[mq] = Db.get(mq, 0) + self.Vb * np.exp(-1j * self.psi)
        pref = tau * self.w if self.variant != "no_tau_T" else self.w
        DT = {(0, 0): pref,
              int_coords(tau * gvec(2)): pref * OMEGA ** tau,
              int_coords(tau * gvec(3)): pref * OMEGA ** (2 * tau)}
        return Db, DT

    def _build_potential(self):
        """potential part of h^(tau)  (k-independent), shape (2nG, 2nG), order [b G..., t G...]"""
        self.Upot = {}
        nG = self.nG
        for tau in (+1, -1):
            Db, DT = self.fourier_components(tau)
            U = np.zeros((2 * nG, 2 * nG), complex)
            for i, Gi in enumerate(self.G):
                for jj, Gj in enumerate(self.G):
                    d = (Gi[0] - Gj[0], Gi[1] - Gj[1])   # G - G'
                    if d in Db:
                        U[i, jj] += Db[d]
                    if d in DT:                          # <b,G|Delta_T|t,G'>
                        U[i, nG + jj] += DT[d]
            U[nG:, :nG] = U[:nG, nG:].conj().T           # Delta_T^dagger block
            U[nG:, nG:] += self.Vzt * np.eye(nG)
            self.Upot[tau] = U

    def h(self, k, tau):
        """electron-picture plane-wave Hamiltonian h^(tau)(k)"""
        k = np.asarray(k, float)
        kG = k[None, :] + self.Gc
        s = -1.0
        if self.variant == "auto_plus":
            s = +1.0
        if self.variant in ("both_shift", "auto_plus"):
            Eb = s * HB2M / M_B * np.sum((kG - tau * KAPPA) ** 2, 1)
        else:
            Eb = s * HB2M / M_B * np.sum(kG ** 2, 1)
        Et = s * HB2M / M_T * np.sum((kG - tau * KAPPA) ** 2, 1)
        return self.Upot[tau] + np.diag(np.concatenate([Eb, Et]))

    def htilde(self, k, tau):
        """hole-basis single-particle matrix  h~ = -h^T"""
        return -self.h(k, tau).T


# ---------------- k meshes and paths ----------------
def fold_ws(kvec):
    """fold k into the Wigner-Seitz moire BZ (closest to gamma); returns folded k and integer offset F (k = kf + F)"""
    best = None
    for m_ in range(-2, 3):
        for n_ in range(-2, 3):
            kf = kvec - m_ * B1 - n_ * B2
            key = (round(np.linalg.norm(kf), 9), m_, n_)
            if best is None or key < best[0]:
                best = (key, kf, (m_, n_))
    return best[1], best[2]


def kmesh(N, fold=True):
    """Gamma-centred N x N mesh, k = (i b1 + j b2)/N, folded into the WS BZ so that the plane-wave
    cutoff |k+G| is centred; returns (N*N,2) k vectors and integer labels (i,j)"""
    ij = np.array([(i, j) for i in range(N) for j in range(N)])
    k = ij[:, 0:1] / N * B1 + ij[:, 1:2] / N * B2
    if fold:
        k = np.array([fold_ws(kk)[0] for kk in k])
    return k, ij


def kpath(npts=40):
    """gamma -> kappa' (K) -> M -> gamma -> K' path in the moire BZ"""
    Kp = (2 * B1 + B2) / 3   # a BZ corner
    Kq = (B1 + 2 * B2) / 3   # = rotated corner
    M = B1 / 2
    g = np.zeros(2)
    nodes = [g, Kp, M, g, Kq]
    labels = [r"$\gamma$", r"$\kappa$", r"$m$", r"$\gamma$", r"$\kappa'$"]
    pts, x, xt = [], [], [0.0]
    s = 0.0
    for a, b in zip(nodes[:-1], nodes[1:]):
        L = np.linalg.norm(b - a)
        for t in np.linspace(0, 1, npts, endpoint=False):
            pts.append(a + t * (b - a)); x.append(s + t * L)
        s += L; xt.append(s)
    pts.append(nodes[-1]); x.append(s)
    return np.array(pts), np.array(x), np.array(xt), labels


# ---------------- Chern number (Fukui-Hatsugai-Suzuki) ----------------
def shift_index(model, shift):
    """for basis (layer,G): index of (layer, G+shift) or -1"""
    nG = model.nG
    out = -np.ones(2 * nG, int)
    for i, g in enumerate(model.G):
        j = model.idx.get((g[0] + shift[0], g[1] + shift[1]))
        if j is not None:
            out[i] = j; out[nG + i] = nG + j
    return out


def chern_fhs(U, N, model, kpts=None):
    """Fukui-Hatsugai-Suzuki Chern number.  U: (N*N, dim, nocc) occupied eigenvectors at mesh points kpts
    (mesh labels i*N+j).  Links to a neighbour whose stored representative differs by a reciprocal vector B use
    u_{k+B}(G) = u_k(G+B)."""
    if kpts is None:
        kpts, _ = kmesh(N)
    nG = model.nG
    Mb = np.column_stack([B1, B2])

    def shifted(u, Bint):
        if Bint == (0, 0):
            return u
        v = np.zeros_like(u)
        for a, g in enumerate(model.G):
            t = model.idx.get((g[0] + Bint[0], g[1] + Bint[1]))
            if t is not None:
                v[a] = u[t]; v[nG + a] = u[nG + t]
        return v

    def state(i, j, kref_target):
        ii, jj = i % N, j % N
        idx = ii * N + jj
        Bc = np.linalg.solve(Mb, kref_target - kpts[idx])
        Bint = tuple(np.rint(Bc).astype(int))
        assert np.allclose(Bc, Bint, atol=1e-6)
        return shifted(U[idx], Bint)

    def link(a, b):
        d = np.linalg.det(a.conj().T @ b)
        return d / abs(d)
    F = 0.0
    for i in range(N):
        for j in range(N):
            k0 = kpts[i * N + j]
            u00 = U[i * N + j]
            u10 = state(i + 1, j, k0 + B1 / N)
            u11 = state(i + 1, j + 1, k0 + (B1 + B2) / N)
            u01 = state(i, j + 1, k0 + B2 / N)
            F += np.angle(link(u00, u10) * link(u10, u11) * link(u11, u01) * link(u01, u00))
    return F / (2 * np.pi)


# ---------------- Hartree-Fock ----------------
def Vq(q, eps, d=5.0):
    q = np.asarray(q, float)
    out = np.empty_like(q)
    small = q < 1e-10
    out[~small] = 2 * np.pi * E2 * np.tanh(q[~small] * d) / (eps * q[~small])
    out[small] = 2 * np.pi * E2 * d / eps
    return out


class HF:
    """Valley-diagonal plane-wave HF in the hole basis.  rho[tau][k, i, j] = <b^dag_{k,i,tau} b_{k,j,tau}>,
    i=(l,G). Hartree:  Sigma^H_{(l,G),(l,G')} = (1/A) V(|G-G'|) n(G-G'),  n(q)=sum_{k,tau,l,G} rho_{(l,G),(l,G+q)}
    Fock:     Sigma^F_{(l2,G2),(l1,G4)}(k) = -(1/A) sum_{k',Q} V(|k'-k+Q|) rho_{(l1,G4+Q),(l2,G2+Q)}(k')
    (Q=0 Hartree component dropped: uniform neutralizing background)."""

    def __init__(self, model, N, eps, d=5.0):
        self.m, self.N, self.eps, self.d = model, N, eps, d
        self.k, self.ij = kmesh(N)
        self.Nk = N * N
        self.A = self.Nk * A_CELL
        nG = model.nG
        self.dim = 2 * nG
        self.h0 = {tau: np.array([model.htilde(k, tau) for k in self.k]) for tau in (1, -1)}
        # Q set = all differences G - G'
        Qs = sorted({(a[0] - b[0], a[1] - b[1]) for a in model.G for b in model.G})
        self.Qs = Qs
        # for each Q: pairs (i -> i') with G_i' = G_i + Q (both in set), for layer-doubled basis
        self.shiftmaps = []
        for Q in Qs:
            src, dst = [], []
            for i, g in enumerate(model.G):
                j = model.idx.get((g[0] + Q[0], g[1] + Q[1]))
                if j is not None:
                    src.append(i); dst.append(j)
            src = np.array(src); dst = np.array(dst)
            src2 = np.concatenate([src, src + nG]); dst2 = np.concatenate([dst, dst + nG])
            self.shiftmaps.append((src2, dst2))
        Qc = np.array([q[0] * B1 + q[1] * B2 for q in Qs])
        # V_Q[k2, k1] = V(|k1 - k2 + Q|)
        dk = self.k[None, :, :] - self.k[:, None, :]
        self.VQ = np.array([Vq(np.linalg.norm(dk + Qc[a], axis=-1), eps, d) for a in range(len(Qs))])
        self.VH = {Q: Vq(np.array([np.linalg.norm(Qc[a])]), eps, d)[0] for a, Q in enumerate(Qs)}

    def sigma(self, rho):
        nG = self.m.nG
        # ---- Hartree: n(q) summed over valleys, layers ----
        Hmat = np.zeros((self.dim, self.dim), complex)
        tot = rho[1].sum(0) + rho[-1].sum(0)        # sum over k
        for a, Q in enumerate(self.Qs):
            if Q == (0, 0):
                continue
            src, dst = self.shiftmaps[a]
            nq = 0.0
            # n(Q) = sum_{l,G} rho_{(l,G),(l,G+Q)}
            nq = np.sum(tot[src, dst])
            # Sigma^H_{(l,G'),(l,G'')} with G' - G'' = Q  ->  entries (dst? ) : G2 - G3 = G4 - G1 = Q
            # G2 = G3 + Q : rows dst, cols src
            Hmat[dst, src] += self.VH[Q] * nq / self.A
        S = {}
        for tau in (1, -1):
            r = rho[tau]
            F = np.zeros((self.Nk, self.dim, self.dim), complex)
            for a, Q in enumerate(self.Qs):
                src, dst = self.shiftmaps[a]
                # rhoQ[k1, i, j] = rho[k1, i+Q, j+Q]  for i,j in src
                sub = r[:, dst][:, :, dst]                       # (Nk, n, n) = rho_{(G4+Q),(G2+Q)}
                conv = np.tensordot(self.VQ[a], sub, axes=(1, 0))   # (Nk, n, n) indexed [k2, G4, G2]
                F[:, src[:, None], src[None, :]] -= np.transpose(conv, (0, 2, 1)) / self.A
            S[tau] = F + Hmat[None]
        return S

    def diag(self, S):
        E, U = {}, {}
        for tau in (1, -1):
            Hk = self.h0[tau] + S[tau]
            Hk = 0.5 * (Hk + np.conj(np.transpose(Hk, (0, 2, 1))))
            E[tau], U[tau] = np.linalg.eigh(Hk)
        return E, U

    def density(self, E, U, nu, valley_fix=None):
        """aufbau: occupy the nu*Nk lowest hole states.
        valley_fix=None: free aufbau over both valleys;
        valley_fix=(n_plus, n_minus): fixed valley fillings per moire cell (TRS ansatz (nu/2,nu/2), VP ansatz (1,0))"""
        occ = {1: np.zeros((self.Nk, self.dim)), -1: np.zeros((self.Nk, self.dim))}
        if valley_fix is None:
            nocc = int(round(nu * self.Nk))
            allE = np.concatenate([E[1].ravel(), E[-1].ravel()])
            order = np.argsort(allE, kind="stable")[:nocc]
            flat = np.zeros(allE.size); flat[order] = 1.0
            occ[1] = flat[:E[1].size].reshape(E[1].shape); occ[-1] = flat[E[1].size:].reshape(E[-1].shape)
        else:
            for tau, nv in zip((1, -1), valley_fix):
                nocc = int(round(nv * self.Nk))
                order = np.argsort(E[tau].ravel(), kind="stable")[:nocc]
                flat = np.zeros(E[tau].size); flat[order] = 1.0
                occ[tau] = flat.reshape(E[tau].shape)
        rho = {}
        for tau in (1, -1):
            Uo = U[tau] * occ[tau][:, None, :]
            rho[tau] = np.einsum('kin,kjn->kij', Uo.conj(), U[tau])   # <b^dag_i b_j> = sum_n u_i^* u_j
        return rho, occ

    def energy(self, rho, S):
        e = 0.0
        for tau in (1, -1):
            e += np.einsum('kij,kij->', self.h0[tau], rho[tau]).real
            e += 0.5 * np.einsum('kij,kij->', S[tau], rho[tau]).real
        return e / self.Nk

    def symmetrize_T(self, rho):
        if not hasattr(self, "Tm"):
            self.Tm = build_T_map(self.m, self)
        v = self.Tm[2]
        mask = np.einsum('ki,kj->kij', v, v)
        rp = np.where(mask, 0.5 * (rho[1] + apply_T(rho[-1], self.Tm)), rho[1])
        rm = np.where(mask, apply_T(rp, self.Tm), rho[-1])
        return {1: rp, -1: rm}

    def run(self, nu, rho0, maxit=150, mix=0.6, tol=1e-7, valley_fix=None, verbose=False, trs=False):
        rho = self.symmetrize_T(rho0) if trs else rho0
        hist = []
        for it in range(maxit):
            S = self.sigma(rho)
            E, U = self.diag(S)
            rnew, occ = self.density(E, U, nu, valley_fix)
            if trs:
                rnew = self.symmetrize_T(rnew)
            Etot = self.energy(rnew, self.sigma(rnew)) if False else self.energy(rho, S)
            diff = max(np.abs(rnew[t] - rho[t]).max() for t in (1, -1))
            hist.append((Etot, diff))
            self.hist = hist
            if verbose:
                print(it, Etot, diff, flush=True)
            rho = {t: mix * rnew[t] + (1 - mix) * rho[t] for t in (1, -1)}
            if diff < tol:
                break
        S = self.sigma(rho); E, U = self.diag(S); rho_f, occ = self.density(E, U, nu, valley_fix)
        if trs:
            rho_f = self.symmetrize_T(rho_f)
        self.rho, self.S, self.E, self.U, self.occ = rho_f, self.sigma(rho_f), E, U, occ
        return dict(E=E, U=U, occ=occ, hist=hist, Etot=self.energy(rho_f, self.sigma(rho_f)), niter=it + 1)

    def gap(self, E, occ):
        eo = max(E[t][occ[t] > 0].max() for t in (1, -1) if occ[t].sum() > 0)
        eu = min(E[t][occ[t] < 1].min() for t in (1, -1))
        return eu - eo

    def sigma_at(self, kvec, tau):
        """HF self-energy at arbitrary k (for band plots) from converged rho"""
        nG = self.m.nG
        Hmat = self.S[tau][0] - self._fock_at(self.k[0], tau)     # Hartree part (k-independent)
        return Hmat + self._fock_at(kvec, tau)

    def _fock_at(self, kvec, tau):
        r = self.rho[tau]
        F = np.zeros((self.dim, self.dim), complex)
        for a, Q in enumerate(self.Qs):
            Qc = Q[0] * B1 + Q[1] * B2
            v = Vq(np.linalg.norm(self.k - kvec[None, :] + Qc, axis=1), self.eps, self.d)
            src, dst = self.shiftmaps[a]
            sub = r[:, dst][:, :, dst]
            conv = np.tensordot(v, sub, axes=(0, 0))
            F[src[:, None], src[None, :]] -= conv.T / self.A
        return F


# ---------------- time-reversal map on the Gamma-centred mesh ----------------
def build_T_map(model, hf):
    """(T rho)_-(k)_{(l,G),(l',G')} = s_l s_l' conj(rho_+(k')_{(l,-G-B),(l',-G'-B)}),  k' = -k + B on mesh,
    s_b=+1, s_t=-1 (from Delta_T,-K = -Delta_T,+K^*).  Returns (kp index, perm, valid, sign)."""
    N = hf.N; nG = model.nG
    kp = np.zeros(hf.Nk, int); perms = []; valids = []
    for ik, (i, j) in enumerate(hf.ij):
        ii, jj = (-i) % N, (-j) % N
        kp[ik] = ii * N + jj
        Bc = np.linalg.solve(np.column_stack([B1, B2]), hf.k[kp[ik]] + hf.k[ik])   # k' + k = B
        B = tuple(np.rint(Bc).astype(int))
        perm = -np.ones(2 * nG, int)
        for a, g in enumerate(model.G):
            t = model.idx.get((-g[0] - B[0], -g[1] - B[1]))
            if t is not None:
                perm[a] = t; perm[nG + a] = nG + t
        perms.append(perm); valids.append(perm >= 0)
    sign = np.concatenate([np.ones(nG), -np.ones(nG)])
    return kp, np.array(perms), np.array(valids), sign


def apply_T(rho_src, Tm):
    kp, perms, valids, sign = Tm
    out = np.zeros_like(rho_src)
    for ik in range(len(kp)):
        p = perms[ik]; v = valids[ik]
        sub = np.conj(rho_src[kp[ik]][np.ix_(np.where(v, p, 0), np.where(v, p, 0))])
        sub = sub * np.outer(sign, sign) * np.outer(v, v)
        out[ik] = sub
    return out
