"""Random-circuit generator (ASSUMED gate set: Haar SU(2) on every qubit, then ZZ(pi/2)=exp(-i pi/4 Z⊗Z)
on a pairing of the qubits) for three geometries, plus a dense statevector simulator."""
import numpy as np

def haar_su2(rng):
    z = (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    q = q * (np.diag(r) / np.abs(np.diag(r)))
    return q / np.sqrt(np.linalg.det(q))

ZZ = np.diag(np.exp(-1j * np.pi / 4 * np.array([1, -1, -1, 1]))).reshape(2, 2, 2, 2)

def pairing(N, geometry, layer, rng, Lx=None):
    if geometry == 'all2all':
        p = rng.permutation(N)
        return [tuple(sorted((int(p[2 * i]), int(p[2 * i + 1])))) for i in range(N // 2)]
    if geometry == '1d':
        s = layer % 2
        return [(i, i + 1) for i in range(s, N - 1, 2)]
    if geometry == '2d':  # ABCD pattern on Lx x Ly grid
        Lx = Lx or int(np.round(np.sqrt(N)))
        while N % Lx: Lx -= 1
        Ly = N // Lx
        pat = layer % 4
        pairs = []
        for x in range(Lx):
            for y in range(Ly):
                q = x * Ly + y
                if pat in (0, 1) and (y % 2 == pat) and y + 1 < Ly: pairs.append((q, q + 1))
                if pat in (2, 3) and (x % 2 == pat - 2) and x + 1 < Lx: pairs.append((q, q + Ly))
        return pairs
    raise ValueError(geometry)

def random_circuit(N, depth, geometry, seed):
    rng = np.random.default_rng(seed)
    layers = []
    for l in range(depth):
        layers.append(([haar_su2(rng) for _ in range(N)], pairing(N, geometry, l, rng)))
    return layers

def apply_1q(psi, U, q):
    psi = np.tensordot(U, psi, axes=([1], [q]))
    return np.moveaxis(psi, 0, q)

def apply_zz(psi, a, b):
    # diagonal: multiply by phase depending on parity of qubits a,b
    sl = [slice(None)] * psi.ndim
    ph = np.exp(-1j * np.pi / 4); phm = np.exp(1j * np.pi / 4)
    for va in (0, 1):
        for vb in (0, 1):
            sl[a] = va; sl[b] = vb
            psi[tuple(sl)] *= ph if va == vb else phm
    return psi

def statevector(N, layers, upto=None):
    psi = np.zeros((2,) * N, complex); psi[(0,) * N] = 1
    for l, (us, prs) in enumerate(layers[:upto]):
        for q, U in enumerate(us): psi = apply_1q(psi, U, q)
        for a, b in prs: psi = apply_zz(psi, a, b)
    return psi

def statevector_series(N, layers):
    """yield psi after each layer"""
    psi = np.zeros((2,) * N, complex); psi[(0,) * N] = 1
    for us, prs in layers:
        for q, U in enumerate(us): psi = apply_1q(psi, U, q)
        for a, b in prs: psi = apply_zz(psi, a, b)
        yield psi
