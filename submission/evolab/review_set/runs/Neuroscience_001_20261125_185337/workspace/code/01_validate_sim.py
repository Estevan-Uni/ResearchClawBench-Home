"""Validate the fast sparse simulator against flyvis native simulation
(flash responses + a subset of moving-edge responses) for model 000, and time it."""
import sys, time, json
sys.path.insert(0, "/workspace/code")
import numpy as np
import fastsim as fs
from flyvis import NetworkView
from flyvis.analysis.stimulus_responses import flash_responses, moving_edge_responses
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = {}
nv = NetworkView(fs.MODEL_NAMES[0])
t0 = time.time()
fr = flash_responses(nv, dt=1 / 200)
out["native_flash_time_s"] = time.time() - t0
stim = fr["stimulus"].values[:, :, 0]  # (S, T, hex)
resp_native = fr["responses"].values[0]  # (S,T,n_types)

fs.load_state(0)
P = fs.semantic_params()
sim = fs.FastSim(P)
t0 = time.time()
resp_fast = sim.simulate(stim.astype(np.float32), dt=1 / 200)
out["fast_flash_time_s"] = time.time() - t0
err = np.abs(resp_fast - resp_native)
out["flash_max_abs_err"] = float(err.max())
out["flash_max_abs_resp"] = float(np.abs(resp_native).max())
print(out)

# moving edges: one speed, all 12 angles x 2 intensities
t0 = time.time()
me = moving_edge_responses(nv, speeds=(19,), dt=1 / 200)
out["native_edge_time_s"] = time.time() - t0
stim_e = me["stimulus"].values[:, :, 0]
t0 = time.time()
resp_fast_e = sim.simulate(stim_e.astype(np.float32), dt=1 / 200)
out["fast_edge_time_s"] = time.time() - t0
err_e = np.abs(resp_fast_e - me["responses"].values[0])
out["edge_max_abs_err"] = float(err_e.max())
out["edge_n_samples"] = int(stim_e.shape[0]); out["edge_n_frames"] = int(stim_e.shape[1])
print(out)
json.dump(out, open("/workspace/outputs/01_validation.json", "w"), indent=1)

types = fs.central_type_order()
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
time_ax = fr["time"].values
for ct, c in zip(["L1", "Mi1", "Tm3", "T4c", "T5c"], plt.cm.tab10.colors):
    j = types.index(ct)
    s = int(np.where((fr["radius"].values == 6) & (fr["intensity"].values == 1))[0][0])
    axs[0].plot(time_ax, resp_native[s, :, j], color=c, lw=3, alpha=.4, label=f"{ct} flyvis")
    axs[0].plot(time_ax, resp_fast[s, :, j], "--", color=c, lw=1, label=f"{ct} fast")
axs[0].set(xlabel="time (s)", ylabel="voltage (a.u.)", title="ON flash (r=6), model 000")
axs[0].legend(fontsize=6, ncol=2)
axs[1].scatter(resp_native.ravel()[::50], resp_fast.ravel()[::50], s=1)
axs[1].set(xlabel="flyvis native", ylabel="fast sparse sim", title=f"flashes: max|err|={err.max():.1e}")
te = me["time"].values
j = types.index("T4c")
for a, c in zip([90, 270], ["C0", "C3"]):
    s = int(np.where((me["angle"].values == a) & (me["intensity"].values == 1))[0][0])
    axs[2].plot(te, me["responses"].values[0, s, :, j], color=c, lw=3, alpha=.4, label=f"T4c ON {a}° flyvis")
    axs[2].plot(te, resp_fast_e[s, :, j], "--", color=c, lw=1, label=f"T4c ON {a}° fast")
axs[2].set(xlabel="time (s)", title=f"moving edges 19°/s: max|err|={err_e.max():.1e}")
axs[2].legend(fontsize=7)
plt.tight_layout()
plt.savefig("/workspace/report/images/fig_sim_validation.png", dpi=150)
