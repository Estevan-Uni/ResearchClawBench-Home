"""Production plane-wave HF runs.  usage: python hf_production.py <job>
jobs: nu2_mesh  nu1_mesh  nu2_scan  nu1_scan  shell3"""
import sys, json, time
import numpy as np
sys.path.insert(0, "/workspace/code")
from moire import *

job = sys.argv[1]
W, VB = 12.0, 7.0


def tasym(hf, E, nb=4):
    Tm = build_T_map(hf.m, hf)
    return float(max(np.abs(E[1][k, :nb] - E[-1][Tm[0][k], :nb]).max() for k in range(hf.Nk)))


def solve(nu, Vzt, eps, N, ansatz, nshell=2, maxit=200, mix=0.6):
    m = Model(w=W, Vb=VB, Vzt=Vzt, nshell=nshell)
    hf = HF(m, N, eps)
    z = {1: np.zeros_like(hf.h0[1]), -1: np.zeros_like(hf.h0[1])}
    E0, U0 = hf.diag(z)
    if ansatz == "TRS":
        vf, trs = (nu / 2, nu / 2), True
    elif ansatz == "VP":
        vf, trs = (nu, 0.0), False
    elif ansatz == "VP_free":       # start from VP density but free aufbau across valleys
        vf, trs = None, False
    rho0, occ0 = hf.density(E0, U0, nu, vf if ansatz != "VP_free" else (nu, 0.0))
    gap0 = hf.gap(E0, occ0)
    t = time.time()
    r = hf.run(nu, rho0, maxit=maxit, mix=mix, tol=1e-6, valley_fix=vf, trs=trs)
    E, U, occ = r["E"], r["U"], r["occ"]
    out = dict(nu=nu, Vzt=Vzt, eps=eps, N=N, ansatz=ansatz, nshell=nshell, niter=r["niter"],
               converged=bool(r["hist"][-1][1] < 1e-5), Etot_meV_per_cell=r["Etot"],
               gap_meV=float(hf.gap(E, occ)), bare_gap_meV=float(gap0), time_s=time.time() - t,
               valley_filling={"+K": float(occ[1].sum() / hf.Nk), "-K": float(occ[-1].sum() / hf.Nk)},
               hist=[(float(a), float(b)) for a, b in r["hist"]])
    herm = max(np.abs(hf.S[tt] - np.conj(np.transpose(hf.S[tt], (0, 2, 1)))).max() for tt in (1, -1))
    out["sigma_hermiticity_err"] = float(herm)
    E_mono = np.array([h[0] for h in r["hist"]])
    out["energy_monotone_after_5"] = bool(np.all(np.diff(E_mono[5:]) < 1e-6)) if len(E_mono) > 6 else True
    # Chern numbers of occupied bands per valley (hole basis)
    ch = {}
    for tt in (1, -1):
        nb = int(round(occ[tt].sum() / hf.Nk))
        fullband = np.allclose(occ[tt][:, :nb], 1.0) and np.allclose(occ[tt][:, nb:], 0.0)
        if nb > 0 and fullband:
            ch["+K" if tt == 1 else "-K"] = round(float(chern_fhs(U[tt][:, :, :nb], N, m, hf.k)), 3)
        else:
            ch["+K" if tt == 1 else "-K"] = None if nb > 0 else 0
    out["chern_hole_basis"] = ch
    out["T_asymmetry_lowest4_meV"] = tasym(hf, E)
    out["flag"] = ("SLOW(>60 it)" if r["niter"] > 60 else "") + ("" if out["converged"] else " NOT_CONVERGED")
    return out, hf


def bands_on_path(hf, nb=4):
    pts, x, xt, lab = kpath(30)
    Eb = {}
    for tt in (1, -1):
        Es = []
        for p in pts:
            Hk = hf.m.htilde(p, tt) + hf.sigma_at(p, tt)
            Hk = 0.5 * (Hk + Hk.conj().T)
            Es.append(np.linalg.eigvalsh(Hk)[:nb])
        Eb[tt] = np.array(Es)
    E0 = {tt: np.array([np.linalg.eigvalsh(hf.m.htilde(p, tt))[:nb] for p in pts]) for tt in (1, -1)}
    return dict(x=x.tolist(), xt=xt.tolist(), lab=lab, E_hf={str(t): Eb[t].tolist() for t in Eb},
                E_bare={str(t): E0[t].tolist() for t in E0})


results = []
if job == "nu2_mesh":
    for N in (6, 9, 12):
        o, hf = solve(2.0, -20.0, 15.0, N, "TRS")
        o["mu_occ_max"] = float(max(hf.E[t][hf.occ[t] > 0].max() for t in (1, -1)))
        o["mu_unocc_min"] = float(min(hf.E[t][hf.occ[t] < 1].min() for t in (1, -1)))
        if N == 9:
            o["bands"] = bands_on_path(hf)
        print({k: v for k, v in o.items() if k not in ("hist", "bands")}, flush=True)
        results.append(o)
elif job == "nu1_mesh":
    for N in (6, 9, 12):
        o, hf = solve(1.0, -10.0, 15.0, N, "VP")
        o["mu_occ_max"] = float(max(hf.E[t][hf.occ[t] > 0].max() for t in (1, -1) if hf.occ[t].sum() > 0))
        o["mu_unocc_min"] = float(min(hf.E[t][hf.occ[t] < 1].min() for t in (1, -1)))
        if N == 9:
            o["bands"] = bands_on_path(hf)
        print({k: v for k, v in o.items() if k not in ("hist", "bands")}, flush=True)
        results.append(o)
    o, hf = solve(1.0, -10.0, 15.0, 6, "VP_free")
    print({k: v for k, v in o.items() if k not in ("hist", "bands")}, flush=True)
    results.append(o)
elif job == "nu2_scan":
    for eps in (15.0, 10.0):
        for V in (-50, -40, -30, -25, -20, -10, 0):
            o, hf = solve(2.0, float(V), eps, 6, "TRS")
            print({k: v for k, v in o.items() if k not in ("hist", "bands")}, flush=True)
            results.append(o)
elif job == "nu2_fine":
    for eps in (15.0, 10.0):
        for V in (-38.0, -36.0, -34.0, -32.0):
            o, hf = solve(2.0, float(V), eps, 6, "TRS")
            print({k: v for k, v in o.items() if k not in ("hist", "bands")}, flush=True)
            results.append(o)
elif job == "nu1_scan":
    for V in (-40, -30, -25, -20, -15, -10, -5, 0):
        o, hf = solve(1.0, float(V), 15.0, 6, "VP")
        print({k: v for k, v in o.items() if k not in ("hist", "bands")}, flush=True)
        results.append(o)
elif job == "shell":
    for ns in (1, 3):
        for (nu, V, an) in ((2.0, -20.0, "TRS"), (1.0, -10.0, "VP")):
            o, hf = solve(nu, V, 15.0, 6, an, nshell=ns)
            print({k: v for k, v in o.items() if k not in ("hist", "bands")}, flush=True)
            results.append(o)
json.dump(results, open(f"/workspace/outputs/hf_{job}.json", "w"), indent=1)
print("DONE", job)
