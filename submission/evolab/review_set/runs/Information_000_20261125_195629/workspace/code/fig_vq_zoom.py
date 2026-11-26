"""Zoomed crops: input vs VQ reconstruction on text regions (glyph fidelity of the generation encoder)."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
from PIL import Image
O = "/workspace/outputs/probes"
crops = [("equation", (20, 150, 370, 235)), ("doge", (5, 105, 250, 140)), ("doge", (220, 118, 384, 156))]
fig, ax = plt.subplots(len(crops), 2, figsize=(11, 6.5))
for i, (n, box) in enumerate(crops):
    for j, s in enumerate(["input384", "vqrecon"]):
        im = Image.open(f"{O}/{n}_{s}.png").crop(box); ax[i, j].imshow(im, interpolation="nearest"); ax[i, j].axis("off")
        ax[i, j].set_title(f"{n} — {'model input (384²)' if j == 0 else 'VQ encode→decode'}", fontsize=9)
plt.suptitle("Generation encoder (VQ, 16× downsampling) keeps layout and colour but loses glyph identity", fontsize=11)
plt.tight_layout(); plt.savefig("/workspace/report/images/fig7b_vq_text_zoom.png", dpi=150); plt.close()
