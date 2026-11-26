"""Single-particle checks: step-7 plane-wave matrix elements vs real-space FFT, Hermiticity, T symmetry,
first-band Chern numbers for the gold kinetic term and for the LLM slips (steps 1-2), band plot."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/workspace/code")
from moire import *

out = {}
m = Model(Vzt=-20.0)

# ---------- Step 7: FFT of real-space potentials over one moire cell ----------
# real-space lattice vectors a1,a2 with a_i . b_j = 2 pi delta_ij
Bm = np.column_stack([B1, B2]); Am = 2 * np.pi * np.linalg.inv(Bm).T
a1, a2 = Am[:, 0], Am[:, 1]
Ng = 48
s = np.arange(Ng) / Ng
S1, S2 = np.meshgrid(s, s, indexing="ij")
R = S1[..., None] * a1 + S2[..., None] * a2
def Delta_b(R):
    return 2 * m.Vb * sum(np.cos(R @ gvec(j) + m.psi) for j in (1, 3, 5))
def Delta_T(R, tau):
    return tau * m.w * (1 + OMEGA ** tau * np.exp(1j * tau * R @ gvec(2)) + OMEGA ** (2 * tau) * np.exp(1j * tau * R @ gvec(3)))
fft_res = {}
maxerr = 0.0
for tau in (1, -1):
    Db_c, DT_c = m.fourier_components(tau)
    for name, field, ref in (("Delta_b", Delta_b(R), Db_c), ("Delta_T", Delta_T(R, tau), DT_c)):
        # c_(m,n) = (1/Ng^2) sum_r f(r) e^{-i (m b1 + n b2).r}  ; e^{-i 2pi(m s1 + n s2)}
        C = np.fft.fft2(field) / Ng ** 2        # C[m,n] coefficient of e^{+i 2 pi (m s1 + n s2)}
        nz = {}
        for mm in range(-4, 5):
            for nn in range(-4, 5):
                c = C[mm % Ng, nn % Ng]
                if abs(c) > 1e-10:
                    nz[(mm, nn)] = c
        assert set(nz) == set(ref), (name, tau, set(nz), set(ref))
        err = max(abs(nz[q] - ref[q]) for q in ref)
        maxerr = max(maxerr, err)
        fft_res[f"{name}_tau{tau:+d}"] = {str(q): [round(ref[q].real, 6), round(ref[q].imag, 6)] for q in ref}
    # matrix entries: check that h^(tau) potential block equals Fourier coefficients
    U = m.Upot[tau]; nG = m.nG
    for i, Gi in enumerate(m.G):
        for j, Gj in enumerate(m.G):
            d = (Gi[0] - Gj[0], Gi[1] - Gj[1])
            assert abs(U[i, j] - Db_c.get(d, 0)) < 1e-12
            assert abs(U[i, nG + j] - DT_c.get(d, 0)) < 1e-12
out["step7_fft_max_abs_error_meV"] = maxerr
out["step7_fourier_components"] = fft_res
# matrix element of plane-wave h at a random k vs direct numerical integration <k+G|H|k+G'>
rng = np.random.default_rng(0)
k0 = rng.normal(size=2) * 0.3
hk = m.h(k0, 1)
err_int = 0.0
for (i, j) in [(0, 1), (1, 0), (0, m.nG + 0), (2, m.nG + 5), (m.nG + 3, 4)]:
    li, lj = i // m.nG, j // m.nG
    Gi, Gj = m.Gc[i % m.nG], m.Gc[j % m.nG]
    if li == 0 and lj == 0:
        f = Delta_b(R)
    elif li == 0 and lj == 1:
        f = Delta_T(R, 1)
    elif li == 1 and lj == 0:
        f = np.conj(Delta_T(R, 1))
    else:
        f = m.Vzt * np.ones_like(S1)
    val = np.mean(np.exp(-1j * R @ (Gi - Gj)) * f)
    if i == j: continue
    err_int = max(err_int, abs(val - hk[i, j]))
out["step7_direct_integral_err"] = err_int

# ---------- Hermiticity and T symmetry ----------
herm = max(np.abs(m.h(k, t) - m.h(k, t).conj().T).max() for k in rng.normal(size=(5, 2)) for t in (1, -1))
tsym = 0.0
for k in rng.normal(size=(5, 2)) * 0.5:
    tsym = max(tsym, np.abs(np.linalg.eigvalsh(m.h(k, -1)) - np.linalg.eigvalsh(m.h(-k, 1))).max())
out["hermiticity_err"] = herm
out["T_symmetry_err_meV(truncation)"] = tsym
out["a_M_nm"] = A_M
out["hbar2kappa2_2mb_meV"] = HB2M / M_B * KAPPA @ KAPPA
out["e2_over_eps10_aM_meV"] = E2 / (10 * A_M)

# ---------- Chern numbers of the topmost valence band for variants ----------
def first_band_chern(model, N=18, band="valence"):
    res = {}
    k, _ = kmesh(N)
    for tau in (1, -1):
        Ue, Uh = [], []
        for kk in k:
            e, u = np.linalg.eigh(model.h(kk, tau))
            sel = -1 if band == "valence" else 0
            Ue.append(u[:, [sel]])
            eh, uh = np.linalg.eigh(model.htilde(kk, tau))
            Uh.append(uh[:, [0 if band == "valence" else -1]])
        res[tau] = (round(chern_fhs(np.array(Ue), N, model), 3), round(chern_fhs(np.array(Uh), N, model), 3))
    return res

cherns = {}
for var, lab in (("gold", "gold (top layer shifted only)"), ("both_shift", "slip: both layers shifted (GPT-4 & auto step 1-2)"),
                 ("auto_plus", "slip: +hbar^2 (k - tau kappa)^2/2m, both layers (auto steps 1,2,6)"),
                 ("no_tau_T", "variant: tau prefactor of Delta_T dropped")):
    mm = Model(Vzt=-20.0, variant=var)
    band = "conduction" if var == "auto_plus" else "valence"
    c = first_band_chern(mm, band=band)
    cherns[lab] = {"C_+K (electron)": c[1][0], "C_-K (electron)": c[-1][0], "C_+K (hole basis)": c[1][1], "C_-K (hole basis)": c[-1][1],
                   "band": band}
    print(lab, cherns[lab], flush=True)
out["chern_variants_Vzt-20"] = cherns

# Vzt scan of gold first-band Chern number (electron picture)
scan = {}
for V in (-60, -50, -40, -30, -25, -20, -10, 0):
    c = first_band_chern(Model(Vzt=V), N=12)
    scan[V] = c[1][0]
out["gold_chern_vs_Vzt(w=12,Vb=7)"] = scan
print(scan)

# ---------- band structure figure (gold vs both-shift slip) ----------
pts, x, xt, lab = kpath(40)
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
for ax, var, title in zip(axs, ("gold", "both_shift", "auto_plus"),
                          ("Gold steps 1-4 (paper)", "Slip: bottom layer also shifted", "auto.md: + sign, both shifted")):
    mm = Model(Vzt=-20.0, variant=var)
    for tau, col in ((1, "tab:orange"), (-1, "tab:blue")):
        E = np.array([np.linalg.eigvalsh(mm.h(p, tau)) for p in pts])
        if var == "auto_plus":
            E = E[:, :4]
        else:
            E = E[:, -4:]
        ax.plot(x, E, color=col, lw=1.3, label=("+K" if tau == 1 else "-K"))
    for t in xt: ax.axvline(t, color="k", lw=0.5)
    ax.set_xticks(xt); ax.set_xticklabels(lab); ax.set_xlim(x[0], x[-1])
    c = cherns[[l for l in cherns if (var == 'gold' and l.startswith('gold')) or (var == 'both_shift' and 'both layers shifted' in l) or (var == 'auto_plus' and l.startswith('slip: +'))][0]]
    ax.set_title(f"{title}\nC(+K)={c['C_+K (electron)']:+.0f}, C(-K)={c['C_-K (electron)']:+.0f} ({c['band']} band 1)", fontsize=10)
    ax.set_ylabel("E (meV, electron picture)")
h, l = axs[0].get_legend_handles_labels(); axs[0].legend(h[::4], l[::4], loc="lower left")
if True:
    axs[0].set_ylim(-80, 12)
plt.tight_layout()
plt.savefig("/workspace/report/images/single_particle_bands.png", dpi=150)
json.dump(out, open("/workspace/outputs/single_particle_checks.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != "step7_fourier_components"}, indent=1, default=str))
