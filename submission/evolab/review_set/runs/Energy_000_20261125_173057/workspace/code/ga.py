"""Real-coded genetic algorithm on the unit hypercube: tournament selection, simulated binary crossover (SBX),
polynomial mutation, elitism.  `fobj(P)` must evaluate a whole population (n, d) -> (n,) (vectorised)."""
import numpy as np


def sbx(p1, p2, rng, eta=15.0, pc=0.9):
    c1, c2 = p1.copy(), p2.copy()
    n, d = p1.shape
    do = (rng.random((n, d)) < 0.5) & (rng.random((n, 1)) < pc)
    u = rng.random((n, d))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    m, s = 0.5 * (p1 + p2), 0.5 * np.abs(p1 - p2)
    sign = np.where(rng.random((n, d)) < 0.5, 1, -1)
    c1 = np.where(do, m - sign * beta * s, p1); c2 = np.where(do, m + sign * beta * s, p2)
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def poly_mut(x, rng, eta=20.0, pm=None):
    n, d = x.shape; pm = 1.0 / d if pm is None else pm
    do = rng.random((n, d)) < pm
    u = rng.random((n, d))
    dl = np.where(u < 0.5, (2 * u + (1 - 2 * u) * (1 - x) ** (eta + 1)) ** (1 / (eta + 1)) - 1,
                  1 - (2 * (1 - u) + 2 * (u - 0.5) * x ** (eta + 1)) ** (1 / (eta + 1)))
    return np.clip(np.where(do, x + dl, x), 0, 1)


def ga(fobj, d, pop=200, gens=100, seed=0, elite=0.1, k_tour=3, init=None, callback=None):
    rng = np.random.default_rng(seed)
    P = rng.random((pop, d)) if init is None else init.copy()
    F = fobj(P); n_eval = pop
    hist = dict(best=[float(F.min())], median=[float(np.median(F))], n_eval=[n_eval])
    ne = max(1, int(elite * pop))
    for g in range(gens):
        def tour():
            idx = rng.integers(0, pop, (pop // 2, k_tour))
            return P[idx[np.arange(pop // 2), np.argmin(F[idx], 1)]]
        c1, c2 = sbx(tour(), tour(), rng)
        C = poly_mut(np.vstack([c1, c2]), rng)
        FC = fobj(C); n_eval += len(C)
        el = np.argsort(F)[:ne]
        allP = np.vstack([P[el], C]); allF = np.concatenate([F[el], FC])
        keep = np.argsort(allF)[:pop]
        P, F = allP[keep], allF[keep]
        hist['best'].append(float(F.min())); hist['median'].append(float(np.median(F))); hist['n_eval'].append(n_eval)
        if callback: callback(g, P, F)
    return P, F, hist
