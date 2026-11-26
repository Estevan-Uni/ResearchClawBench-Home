"""Architecture / pipeline schematic: stimulus -> hexagonal photoreceptor lattice ->
connectome-constrained DMN (45,669 neurons, 65 types) -> decoder -> optic flow; with the
learned vs. fixed parameter sets and the training objective."""
import sys
sys.path.insert(0, "/workspace/code")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, RegularPolygon
import fastsim as fs
import textures as tx

T = fs.topo()
fig = plt.figure(figsize=(17, 6.2))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 17); ax.set_ylim(0, 6.2); ax.axis("off")


def box(x, y, w, h, text, fc="#eef3fb", fs_=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="k", lw=1))
    ax.text(x + w / 2, y + h - 0.12, text, ha="center", va="top", fontsize=fs_)


def arrow(x0, y0, x1, y1, text=""):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18, lw=1.5))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.15, text, ha="center", fontsize=8)


# 1. stimulus
img = tx.pink_noise(512, seed=3)
a1 = fig.add_axes([0.01, 0.38, 0.13, 0.45]); a1.imshow(img, cmap="gray"); a1.set_xticks([]); a1.set_yticks([])
a1.set_title("1. visual stimulus\n(Sintel movies in training;\nflashes, edges, textures here)", fontsize=9)
arrow(2.45, 3.7, 2.95, 3.7)
# 2. hex lattice
mv, _ = tx.translation_movies(img, angles_deg=[0], n_frames=1)
a2 = fig.add_axes([0.175, 0.38, 0.13, 0.45])
x, y = np.sqrt(3) * (T.u_in + T.v_in / 2), 1.5 * T.v_in
a2.scatter(x, y, c=mv[0, 0], cmap="gray", marker="h", s=9); a2.set_aspect("equal"); a2.axis("off")
a2.set_title("2. BoxEye: 721 hexagonal columns\n(extent 15, kernel 13 px) → R1–R8", fontsize=9)
arrow(5.3, 3.7, 5.8, 3.7)
# 3. DMN
box(5.85, 0.7, 5.2, 5.0, "3. Deep mechanistic network (connectome-constrained)", fc="#fdf3e7")
ax.text(8.45, 5.2, "45,669 point neurons · 65 cell types · 1,513,231 synapses (604 type pairs)", ha="center", fontsize=8)
ax.text(6.0, 4.75, r"$\tau_{t_i}\,\dot V_i = -V_i + V^{rest}_{t_i} + \sum_j w_{ij}\,\mathrm{ReLU}(V_j) + e_i(t)$", fontsize=10)
ax.text(6.0, 4.3, r"$w_{ij} = \sigma_{t_it_j}\cdot N_{t_it_j}(\Delta u,\Delta v)\cdot\alpha_{t_it_j}$", fontsize=10)
ax.text(6.0, 3.65, "FIXED by connectome (2,959):\n  sign σ (604) · synapse counts N per offset (2,355)", fontsize=8, color="#7a3e00")
ax.text(6.0, 2.95, "LEARNED by task (734):\n  resting potentials (65) · time constants (65)\n  unit synaptic strengths α (604)", fontsize=8, color="#00457a")
layers = [("R1-8", 0.55), ("L1-5, Lawf, Am, C2/3", 0.45), ("Mi, Tm, CT1, T2/3", 0.35), ("T4a-d · T5a-d · TmY", 0.25)]
for k, (lab, yy) in enumerate(layers):
    for c in range(6):
        ax.add_patch(RegularPolygon((6.25 + c * 0.32, 1.0 + k * 0.38), 6, radius=0.15, fc=plt.cm.viridis(k / 4), ec="k", lw=.3))
    ax.text(8.3, 1.0 + k * 0.38, lab, va="center", fontsize=7.5)
arrow(11.1, 3.7, 11.6, 3.7, "ReLU(V) of\n34 output types")
# 4. decoder
box(11.65, 2.4, 2.2, 2.6, "4. decoder\n(hex conv 34→8→2+1 ch.,\nkernel 5, softplus,\nbatchnorm, dropout 0.5)", fc="#eef7ee")
arrow(13.9, 3.7, 14.35, 3.7)
# 5. flow
a5 = fig.add_axes([0.85, 0.36, 0.14, 0.45])
ii = np.arange(0, 721, 23)
a5.quiver(x[ii], y[ii], np.ones_like(ii), 0.3 * np.ones_like(ii), color="C3", scale=9, width=0.012)
a5.set_aspect("equal"); a5.axis("off"); a5.set_title("5. optic flow (u,v)\nper column", fontsize=9)
ax.text(8.45, 0.25, "Training (not re-run here): BPTT through Euler-integrated dynamics (dt = 20 ms, 19-frame Sintel clips, augmentation),"
        " loss = L2 norm of flow error, Adam lr 5e-5→5e-6, 250k iterations, batch 4;\n"
        "activity penalty keeps resting activity above baseline; ensemble of 50 independently trained models (shipped configs identical) ranked by validation loss (5.14–5.68).",
        ha="center", fontsize=8, style="italic")
plt.savefig("/workspace/report/images/fig_schematic.png", dpi=150)
