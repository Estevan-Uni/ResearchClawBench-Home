"""Analysis of the water MD: partial RDFs, CN, H-bonds, molecular integrity, block convergence."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, IMG
from ase.io import read
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
tag = sys.argv[1] if len(sys.argv) > 1 else "L12"
DISCARD_FS = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0
frames = read(f"{OUT}/water_{tag}.traj", ":")
log = np.genfromtxt(f"{OUT}/water_{tag}_log.csv", delimiter=",", names=True)
L = frames[0].cell[0, 0]; dt_frame = 2.5  # fs (every 5 steps of 0.5 fs)
times = np.arange(len(frames)) * dt_frame
O = np.where(frames[0].numbers == 8)[0]; H = np.where(frames[0].numbers == 1)[0]
rmax, dr = L / 2, 0.05; edges = np.arange(0, rmax + 1e-9, dr); rc = 0.5 * (edges[1:] + edges[:-1]); shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
def mic(d): return d - L * np.round(d / L)
def rdf(fr, A, B):
    h = np.zeros(len(rc))
    for a in fr:
        p = a.positions
        d = np.linalg.norm(mic(p[A][:, None] - p[B][None]), axis=-1)
        if A is B: d = d[np.triu_indices(len(A), 1)]; npair = len(A) * (len(A) - 1) / 2
        else: d = d.ravel(); npair = len(A) * len(B)
        h += np.histogram(d, edges)[0]
    rho_pairs = npair / L ** 3
    return h / (len(fr) * rho_pairs * shell)
def feats(g, rho_B):
    m = (rc > 2.3) & (rc < 3.4); ip = np.argmax(np.where(m, g, -1)); rp, gp = rc[ip], g[ip]
    m2 = (rc > rp + 0.2) & (rc < 4.2); im = np.argmin(np.where(m2, g, 99)); rmn = rc[im]
    cn = float(np.sum(4 * np.pi * rc[: im + 1] ** 2 * g[: im + 1] * dr) * rho_B)
    return {"r_peak": float(rp), "g_peak": float(gp), "r_min": float(rmn), "g_min": float(g[im]), "CN": cn}
def analysis(fr):
    gOO, gOH, gHH = rdf(fr, O, O), rdf(fr, O, H), rdf(fr, H, H)
    return gOO, gOH, gHH, feats(gOO, len(O) / L ** 3)
prod = [f for f, t in zip(frames, times) if t >= DISCARD_FS]
gOO, gOH, gHH, fOO = analysis(prod)
# intramolecular / dissociation / H-bonds
roh, ang, ndis, nhb = [], [], [], []
for a in frames:
    p = a.positions
    dOH = np.linalg.norm(mic(p[H][:, None] - p[O][None]), axis=-1)  # (nH, nO)
    owner = dOH.argmin(1); counts = np.bincount(owner, minlength=len(O))
    ndis.append(int(np.sum(counts != 2)))
    roh.append(dOH.min(1).mean())
    angs = []
    for io in range(len(O)):
        hh = np.where(owner == io)[0]
        if len(hh) == 2:
            v1 = mic(p[H[hh[0]]] - p[O[io]]); v2 = mic(p[H[hh[1]]] - p[O[io]])
            angs.append(np.degrees(np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2))))
    ang.append(np.mean(angs))
    # geometric H-bond: O...O < 3.5 A and angle H-Od...Oa < 30 deg
    dOO = np.linalg.norm(mic(p[O][:, None] - p[O][None]), axis=-1); hb = 0
    for ih in range(len(H)):
        od = owner[ih]; vh = mic(p[H[ih]] - p[O[od]])
        for oa in np.where((dOO[od] < 3.5) & (np.arange(len(O)) != od))[0]:
            vo = mic(p[O[oa]] - p[O[od]])
            if np.degrees(np.arccos(np.dot(vh, vo) / np.linalg.norm(vh) / np.linalg.norm(vo))) < 30: hb += 1
    nhb.append(2 * hb / len(O))  # H-bonds per molecule (each bond shared by 2 molecules -> counts per molecule = 2*N_hb/N_mol)
roh, ang, ndis, nhb = map(np.array, (roh, ang, ndis, nhb))
pm = times >= DISCARD_FS
# block analysis
blocks = []
tmax = times[-1]
for t0, t1 in ([(250, 500), (500, 750), (750, 1000.1)] if tag == "L12" else [(150, 300), (300, 500.1)]):
    fr = [f for f, t in zip(frames, times) if t0 <= t < t1]
    if len(fr) < 10: continue
    _, _, _, fb = analysis(fr); fb.update({"t0": t0, "t1": min(t1, 1000), "nframes": len(fr), "HB_per_mol": float(nhb[(times >= t0) & (times < t1)].mean())}); blocks.append(fb)
lt = log["time_fs"]; T = log["T_K"]; Ep = log["Epot_eV"]
def trend(mask):
    c = np.polyfit(lt[mask], Ep[mask], 1); return float(c[0] * 1000)  # eV/ps
res = {"tag": tag, "L": float(L), "density_g_cm3": 32 * 18.015 / 6.02214e23 / (L * 1e-8) ** 3, "steps_done": int(log["step"][-1]), "n_frames": len(frames),
       "discard_fs": DISCARD_FS, "n_prod_frames": len(prod), "gOO": fOO,
       "gOH_2nd_peak": float(rc[(rc > 1.5) & (rc < 2.4)][np.argmax(gOH[(rc > 1.5) & (rc < 2.4)])]),
       "gOH_2nd_peak_height": float(gOH[(rc > 1.5) & (rc < 2.4)].max()),
       "gHH_peak": float(rc[(rc > 1.3) & (rc < 2.0)][np.argmax(gHH[(rc > 1.3) & (rc < 2.0)])]),
       "mean_T_prod": float(T[lt >= DISCARD_FS].mean()), "std_T_prod": float(T[lt >= DISCARD_FS].std()),
       "mean_T_last500fs": float(T[lt >= 500].mean()),
       "Epot_trend_last250fs_eV_per_ps": trend(lt >= 750) if lt[-1] > 800 else None,
       "Epot_trend_500_1000_eV_per_ps": trend(lt >= 500) if lt[-1] > 600 else None,
       "Epot_mean_blocks": [float(Ep[(lt >= a) & (lt < b)].mean()) for a, b in [(250, 500), (500, 750), (750, 1000.1)] if (lt >= a).any()],
       "max_dissociated": int(ndis.max()), "mean_rOH": float(roh[pm].mean()), "mean_HOH_deg": float(ang[pm].mean()),
       "HB_per_mol_prod": float(nhb[pm].mean()), "blocks": blocks,
       "wall_s_per_step": float(log["wall_s"][-1] / max(1, log["step"][-1]))}
json.dump(res, open(f"{OUT}/water_{tag}_analysis.json", "w"), indent=1)
np.savez(f"{OUT}/water_{tag}_rdf.npz", r=rc, gOO=gOO, gOH=gOH, gHH=gHH)
print(json.dumps({k: v for k, v in res.items() if k != "blocks"}, indent=0)); print(blocks)
# ---- figures ----
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a_, g, lab, c in zip(ax, [gOO, gOH, gHH], ["O–O", "O–H", "H–H"], ["#1f5f8b", "#c0392b", "#2e8b57"]):
    a_.plot(rc, g, color=c, lw=1.8); a_.set_xlabel("r (Å)"); a_.set_ylabel(f"g$_{{{lab.replace('–','')}}}$(r)"); a_.set_title(f"{lab} RDF ({len(prod)} frames, t ≥ {DISCARD_FS:.0f} fs)", fontsize=10); a_.set_xlim(0, L / 2)
ax[0].axvline(2.8, color="gray", ls="--", lw=1); ax[0].text(2.85, ax[0].get_ylim()[1] * 0.9, "expt. peak ≈ 2.8 Å", fontsize=8, color="gray")
if tag == "L12": ax[0].set_ylim(bottom=0)
ax[1].set_ylim(0, 3); ax[2].set_ylim(0, 3)
plt.tight_layout(); plt.savefig(f"{IMG}/water_rdf_{tag}.png", dpi=150); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(lt, T, lw=0.6, color="#c0392b"); ax[0].axhline(330, color="k", ls="--", lw=1); ax[0].set_xlabel("t (fs)"); ax[0].set_ylabel("T (K)")
ax[1].plot(lt, Ep, lw=0.8, color="#1f5f8b"); ax[1].set_xlabel("t (fs)"); ax[1].set_ylabel("E$_{pot}$ (eV)")
ax[2].plot(times, nhb, lw=0.8, color="#2e8b57", label="H-bonds / molecule"); ax[2].plot(times, ndis, lw=0.8, color="k", label="# non-intact H$_2$O"); ax[2].set_xlabel("t (fs)"); ax[2].legend(fontsize=8)
for a_ in ax: a_.axvline(DISCARD_FS, color="gray", ls=":", lw=1)
plt.tight_layout(); plt.savefig(f"{IMG}/water_thermo_{tag}.png", dpi=150); plt.close()
if blocks:
    fig, ax = plt.subplots(figsize=(6, 4))
    for b, c in zip(blocks, ["#9ecae1", "#4292c6", "#08306b"]):
        fr = [f for f, t in zip(frames, times) if b["t0"] <= t < b["t1"] + 0.1]
        ax.plot(rc, rdf(fr, O, O), color=c, label=f"{b['t0']:.0f}–{b['t1']:.0f} fs: peak {b['r_peak']:.2f} Å, g={b['g_peak']:.2f}")
    ax.set_xlabel("r (Å)"); ax.set_ylabel("g$_{OO}$(r)"); ax.legend(fontsize=8); ax.set_xlim(2, 6); ax.set_title("Block convergence of g$_{OO}$", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{IMG}/water_blocks_{tag}.png", dpi=150); plt.close()
