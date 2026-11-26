"""Exact operator-algebra verification of steps 8-15 of the template chain (and discriminating wrong variants).
Fock-space operators are built with the Jordan-Wigner construction (scipy.sparse)."""
import json, itertools
import numpy as np
import scipy.sparse as sp

rng = np.random.default_rng(42)


def jw_ops(n):
    I = sp.identity(2, format="csr"); Z = sp.diags([1, -1]).tocsr()
    a = sp.csr_matrix(np.array([[0, 1], [0, 0]]))   # annihilation on |1> -> |0>
    ops = []
    for j in range(n):
        mats = [Z] * j + [a] + [I] * (n - j - 1)
        M = mats[0]
        for X in mats[1:]:
            M = sp.kron(M, X, format="csr")
        ops.append(M)
    return ops


def opnorm(A):
    A = A.tocoo()
    return np.abs(A.data).max() if A.nnz else 0.0


res = {}
# ---------------- Steps 8-9: particle-hole transformation + normal ordering ----------------
n = 5
c = jw_ops(n); cd = [x.getH() for x in c]
X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)); h = X + X.conj().T
H_part = sum(cd[i] @ c[j] * h[i, j] for i in range(n) for j in range(n))
b = cd; bd = c                      # b_i = c_i^dagger
Id = sp.identity(2 ** n, format="csr")
H_step8 = sum(b[i] @ bd[j] * h[i, j] for i in range(n) for j in range(n))           # sum b h b^dag (step 8 gold)
H_gold9 = np.trace(h) * Id - sum(bd[i] @ b[j] * h.T[i, j] for i in range(n) for j in range(n))
H_conj9 = np.trace(h) * Id - sum(bd[i] @ b[j] * np.conj(h)[i, j] for i in range(n) for j in range(n))  # gold yaml form (h^*)
H_notr = - sum(bd[i] @ b[j] * h.T[i, j] for i in range(n) for j in range(n))
H_notransp = np.trace(h) * Id - sum(bd[i] @ b[j] * h[i, j] for i in range(n) for j in range(n))
H_sign = np.trace(h) * Id + sum(bd[i] @ b[j] * h.T[i, j] for i in range(n) for j in range(n))
res["step8_sum_b_h_bdag_vs_original"] = opnorm(H_step8 - H_part)
res["step9_gold_Trh_minus_bdag_hT_b"] = opnorm(H_gold9 - H_part)
res["step9_gold_yaml_form_h_conj"] = opnorm(H_conj9 - H_part)
res["step9_WRONG_drop_trace"] = opnorm(H_notr - H_part)
res["step9_WRONG_no_transpose"] = opnorm(H_notransp - H_part)
res["step9_WRONG_plus_sign"] = opnorm(H_sign - H_part)

# ---------------- Steps 10-13: Wick / quadratic / swap on a 12-mode ring ----------------
L, nf = 6, 2                  # 6 momenta on a ring x 2 flavours (layer/valley composite index)
modes = [(k, f) for k in range(L) for f in range(nf)]
nm = len(modes); idx = {m: i for i, m in enumerate(modes)}
bb = jw_ops(nm); bbd = [x.getH() for x in bb]
Vk = {q: 1.0 / (1.0 + 0.7 * min(q, L - q)) + 0.3 for q in range(L)}    # any even function V(|q|)
A = 1.0 * L

def V(k1, k4):
    return Vk[(k1 - k4) % L]

terms = []   # (coef, i1,i2,i3,i4) for b1^dag b2^dag b3 b4 with flavour structure l1,l2,l2,l1
for k1, k2, k3, k4 in itertools.product(range(L), repeat=4):
    if (k1 + k2 - k3 - k4) % L: continue
    for f1, f2 in itertools.product(range(nf), repeat=2):
        terms.append((V(k1, k4) / (2 * A), idx[(k1, f1)], idx[(k2, f2)], idx[(k3, f2)], idx[(k4, f1)]))
Hint = sp.csr_matrix((2 ** nm, 2 ** nm), dtype=complex)
for cf, i1, i2, i3, i4 in terms:
    Hint = Hint + cf * (bbd[i1] @ bbd[i2] @ bb[i3] @ bb[i4])

def slater(Np):
    """random Slater determinant: occupied orbitals = random unitary columns (no symmetry assumed)"""
    Xr = rng.normal(size=(nm, nm)) + 1j * rng.normal(size=(nm, nm))
    Q, _ = np.linalg.qr(Xr)
    orb = Q[:, :Np]
    vac = np.zeros(2 ** nm, complex); vac[0] = 1.0       # |00..0> is index 0 in this JW ordering? check
    # in our convention a|1>=|0>; basis state |0> is index 0 => all-empty is index 0
    psi = vac
    for n_ in range(Np):
        dn = sum(orb[i, n_] * bbd[i] for i in range(nm))
        psi = dn @ psi
    psi /= np.linalg.norm(psi)
    rho = orb.conj() @ orb.T          # rho_ij = <b_i^dag b_j> = sum_n u_i^* u_j
    return psi, rho

def wick_energy(rho, prefH=1.0, prefF=1.0):
    E = 0.0
    for cf, i1, i2, i3, i4 in terms:
        E += cf * (prefH * rho[i1, i4] * rho[i2, i3] - prefF * rho[i1, i3] * rho[i2, i4])
    return E

def mf_operator(rho, form):
    """form: 'wick6' (6 terms, 1/2A, incl. c-numbers), 'quad4' (4 quadratic terms, 1/2A),
    'hf2' (2 terms, 1/A), 'hf2_halfA' (WRONG: 2 terms with 1/2A)"""
    M = np.zeros((nm, nm), complex); const = 0.0
    for cf, i1, i2, i3, i4 in terms:
        if form in ("wick6", "quad4"):
            M[i2, i3] += cf * rho[i1, i4]; M[i1, i4] += cf * rho[i2, i3]
            M[i2, i4] -= cf * rho[i1, i3]; M[i1, i3] -= cf * rho[i2, i4]
            if form == "wick6":
                const += cf * (-rho[i1, i4] * rho[i2, i3] + rho[i1, i3] * rho[i2, i4])
        else:
            s = 2.0 if form == "hf2" else 1.0
            M[i2, i3] += s * cf * rho[i1, i4]; M[i2, i4] -= s * cf * rho[i1, i3]
    return M, const

def quad_op(M, const=0.0):
    return sum(bbd[i] @ bb[j] * M[i, j] for i in range(nm) for j in range(nm) if abs(M[i, j]) > 0) + const * sp.identity(2 ** nm)

wick = {"E_exact_minus_E_wick": [], "Ewick6_minus_exact": [], "WRONG_wick_no_cnumbers": [],
        "grad_check_hf2": [], "quad4_vs_hf2_operator": [], "WRONG_hf2_halfA_grad": [], "WRONG_hartree_only": []}
for Np in (3, 5, 6):
    psi, rho = slater(Np)
    Eex = np.vdot(psi, Hint @ psi)
    Ew = wick_energy(rho)
    wick["E_exact_minus_E_wick"].append(abs(Eex - Ew))
    M6, c6 = mf_operator(rho, "wick6")
    E6 = np.vdot(psi, quad_op(M6, c6) @ psi)
    wick["Ewick6_minus_exact"].append(abs(E6 - Eex))
    E4 = np.vdot(psi, quad_op(M6) @ psi)
    wick["WRONG_wick_no_cnumbers"].append(abs(E4 - Eex))
    # HF operator = dE/drho^T : numerical gradient of E_wick wrt rho_ij
    M2, _ = mf_operator(rho, "hf2"); Mh, _ = mf_operator(rho, "hf2_halfA")
    G = np.zeros((nm, nm), complex); epsd = 1e-6
    for i in range(nm):
        for j in range(nm):
            dr = np.zeros((nm, nm), complex); dr[i, j] = epsd
            G[i, j] = (wick_energy(rho + dr) - wick_energy(rho - dr)) / (2 * epsd)
    # E = sum_ij M_ij rho_ij /2 ... dE/drho_ij = M_ij
    wick["grad_check_hf2"].append(np.abs(G - M2).max())
    wick["WRONG_hf2_halfA_grad"].append(np.abs(G - Mh).max())
    wick["quad4_vs_hf2_operator"].append(np.abs(M6 - M2).max())
    MH = np.zeros((nm, nm), complex)
    for cf, i1, i2, i3, i4 in terms:
        MH[i2, i3] += 2 * cf * rho[i1, i4]
    wick["WRONG_hartree_only"].append(np.abs(G - MH).max())
res["steps11-13"] = {k: float(max(v)) for k, v in wick.items()}

# ---------------- Steps 14-15: momentum reduction (k in BZ + reciprocal lattice) ----------------
Nk, NG = 4, 5               # 1D: k = 2pi n/(Nk a), n=0..Nk-1 ; q = 2pi m/a, m=-2..2
kk = np.arange(Nk) / Nk; qq = np.arange(-2, 3) * 1.0   # in units of 2pi/a
Vf = lambda p: 1.0 / (0.4 + abs(p))
Aar = Nk * 1.0
# translation invariant rho: rho[(k,q),(k',q')] = delta_{kk'} r_k[q,q']
rk = []
for _ in range(Nk):
    Xr = rng.normal(size=(NG, NG)) + 1j * rng.normal(size=(NG, NG)); rk.append(Xr @ Xr.conj().T / 10)
def rho_full(k1, q1, k4, q4):
    return rk[k1][q1, q4] if k1 == k4 else 0.0
P = lambda k, q: kk[k] + qq[q]
def same_mom(a, b):
    return abs(a - b) < 1e-9
H_un = np.zeros((Nk, NG, Nk, NG), complex); F_un = np.zeros_like(H_un)
for k1, k2, k3, k4 in itertools.product(range(Nk), repeat=4):
    for q1, q2, q3, q4 in itertools.product(range(NG), repeat=4):
        if not same_mom(P(k1, q1) + P(k2, q2), P(k3, q3) + P(k4, q4)): continue
        H_un[k2, q2, k3, q3] += Vf(P(k1, q1) - P(k4, q4)) * rho_full(k1, q1, k4, q4) / Aar
        F_un[k2, q2, k4, q4] -= Vf(P(k1, q1) - P(k4, q4)) * rho_full(k1, q1, k3, q3) / Aar
H_red = np.zeros_like(H_un); F_red = np.zeros_like(H_un); F_wrong = np.zeros_like(H_un); H_wrongdelta = np.zeros_like(H_un)
for k1, k2 in itertools.product(range(Nk), repeat=2):
    for q1, q2, q3, q4 in itertools.product(range(NG), repeat=4):
        if not same_mom(qq[q1] + qq[q2], qq[q3] + qq[q4]): continue
        H_red[k2, q2, k2, q3] += Vf(qq[q1] - qq[q4]) * rk[k1][q1, q4] / Aar
        F_red[k2, q2, k2, q4] -= Vf(P(k1, q1) - P(k2, q4)) * rk[k1][q1, q3] / Aar
        F_wrong[k2, q2, k2, q4] -= Vf(qq[q1] - qq[q4]) * rk[k1][q1, q3] / Aar
res["step14_hartree_reduced_vs_unreduced"] = float(np.abs(H_red - H_un).max())
res["step15_fock_reduced_vs_unreduced"] = float(np.abs(F_red - F_un).max())
res["step15_WRONG_fock_with_V(q1-q4)"] = float(np.abs(F_wrong - F_un).max())
res["scale_of_coefficients"] = float(np.abs(F_un).max())
res = {k: (float(v) if not isinstance(v, dict) else v) for k, v in res.items()}
json.dump(res, open("/workspace/outputs/algebra_checks.json", "w"), indent=1)
print(json.dumps(res, indent=1))
