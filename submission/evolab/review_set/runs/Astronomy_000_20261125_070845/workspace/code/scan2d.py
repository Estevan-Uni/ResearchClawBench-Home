"""2-D scan of the survival probability in the (mu, g=1/f) plane for two self-interaction models."""
import numpy as np, os, time, sys, json
import physics as ph, stats as st
from data_overview import load, ROOT

G_GRID = np.logspace(-20, -10, 51)          # g = 1/f [GeV^-1]
F_GRID = 1.0 / G_GRID
MODELS = [("bosenova", 0.5), ("bosenova", 5.0), ("bosenova", 50.0),
          ("mixing", 1 / 25.), ("mixing", 1.0), ("mixing", 25.0)]


def psurv_f(M, a, mu, tage, model, c, rate_model="calibrated", lmax=ph.LMAX):
    Mb, ab, fb = M[:, None], a[:, None], F_GRID[None, :]
    u = np.zeros((len(M), len(F_GRID)), bool)
    for l in range(1, lmax + 1):
        u |= ph.t_spindown(Mb, ab, mu, l, f=fb, model=model, c=c, rate_model=rate_model) < tage
    return 1 - u.mean(0)


def run(name, mus, ages, tag):
    M, a = load(name)
    out = dict(mus=mus, g=G_GRID, ages=ages)
    t0 = time.time()
    for model, c in MODELS:
        for ia, t in enumerate(ages):
            P = np.ones((len(mus), len(G_GRID)))
            for i, mu in enumerate(mus):
                p0 = st.p_surv(M, a, mu, t, rate_model="calibrated")
                if p0 >= 0.999:      # no unstable samples even without self-interactions
                    continue
                P[i] = psurv_f(M, a, mu, t, model, c)
            out[f"{model}_{c:g}_t{ia}"] = P
            print(name, model, c, t, f"{time.time()-t0:.0f}s", flush=True)
    np.save(os.path.join(ROOT, f"outputs/scan2d_{tag}.npy"), out, allow_pickle=True)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "M33":
        run("M33 X-7", np.logspace(np.log10(1.5e-13), np.log10(1.5e-11), 121), [st.T_SALPETER, 3e6] + list(st.AGES["M33 X-7"]["marg"]), "M33")
    else:
        run("IRAS 09149-6206", np.logspace(-20.5, -17.5, 91), [st.T_SALPETER, 1e10], "IRAS")
