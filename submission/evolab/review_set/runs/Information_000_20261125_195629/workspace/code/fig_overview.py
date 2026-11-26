"""Figure 1: data overview (the two provided probe images) + Figure 2: Janus decoupled-encoding schematic."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
W = "/workspace"
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for a, f, t in zip(ax, ["equation.png", "doge.png"], ["(a) equation.png — formula OCR / LaTeX probe", "(b) doge.png — meme-understanding probe"]):
    im = Image.open(f"{W}/data/{f}"); a.imshow(im); a.set_title(f"{t}\n{im.size[0]}×{im.size[1]} px, mode {im.mode}", fontsize=10); a.axis("off")
plt.tight_layout(); plt.savefig(f"{W}/report/images/fig1_data_overview.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(12, 5.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
def box(x, y, w, h, txt, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)
def arr(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color="#333"))
U, G, L = "#cfe3f7", "#fde2c4", "#e4e4e4"
box(0.2, 4.2, 1.8, 0.9, "Image\n(384×384)", U); box(2.5, 4.2, 2.0, 0.9, "Und. encoder\nSigLIP-L/16\n→ 576×1024", U)
box(5.0, 4.2, 1.6, 0.9, "Und. adaptor\n2-layer MLP", U)
box(0.2, 0.9, 1.8, 0.9, "Image\n(384×384)", G); box(2.5, 0.9, 2.0, 0.9, "Gen. encoder\nVQ tokenizer (×16)\n→ 576 ids / 16384", G)
box(5.0, 0.9, 1.6, 0.9, "Gen. adaptor\nemb(8)→MLP", G)
box(7.1, 2.3, 1.9, 1.6, "Unified\nautoregressive\nTransformer\n(DeepSeek-LLM 1.3B)", L)
box(0.2, 2.75, 1.8, 0.7, "Text tokenizer", L); arr(2.0, 3.1, 7.1, 3.1)
box(9.5, 4.2, 2.3, 0.9, "Text head\n(LM vocab) → answer", U); box(9.5, 0.9, 2.3, 0.9, "Image head (16384)\n+ CFG → VQ decoder", G)
for (a, b) in [((2.0, 4.65), (2.5, 4.65)), ((4.5, 4.65), (5.0, 4.65)), ((2.0, 1.35), (2.5, 1.35)), ((4.5, 1.35), (5.0, 1.35))]: arr(*a, *b)
arr(6.6, 4.65, 7.1, 3.6); arr(6.6, 1.35, 7.1, 2.6); arr(9.0, 3.6, 9.5, 4.65); arr(9.0, 2.6, 9.5, 1.35)
ax.text(6, 5.6, "Janus: decoupled visual encoding, single shared Transformer", ha="center", fontsize=12, weight="bold")
ax.text(3.4, 3.75, "understanding pathway (high-level semantics)", color="#1f5f9f", fontsize=9, ha="center")
ax.text(3.4, 0.45, "generation pathway (fine-grained, discrete codes)", color="#b5651d", fontsize=9, ha="center")
plt.savefig(f"{W}/report/images/fig2_architecture.png", dpi=150, bbox_inches="tight"); plt.close()
