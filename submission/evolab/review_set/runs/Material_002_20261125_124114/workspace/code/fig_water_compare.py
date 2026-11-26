import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, IMG
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
a = np.load(f"{OUT}/water_L12_rdf.npz"); b = np.load(f"{OUT}/water_L9865_rdf.npz")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(a["r"], a["gOO"], color="#1f5f8b", label="L=12 Å, ρ=0.55 g/cm³ (prescribed; 250–1000 fs)")
ax.plot(b["r"], b["gOO"], color="#e67e22", label="L=9.865 Å, ρ=1.00 g/cm³ (sensitivity; 150–310 fs, not equilibrated)")
ax.axvline(2.8, color="gray", ls="--", lw=1, label="expt. first peak ≈ 2.8 Å")
ax.set_xlim(2, 6); ax.set_xlabel("r (Å)"); ax.set_ylabel("g$_{OO}$(r)"); ax.legend(fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{IMG}/water_density_compare.png", dpi=150)
