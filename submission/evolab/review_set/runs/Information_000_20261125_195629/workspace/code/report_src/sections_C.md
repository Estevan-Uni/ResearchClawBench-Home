## 5. Results III — Mechanistic evidence for decoupling inside the released model

The paper's central argument is that understanding and generation need *different* visual representations. Showing that the decoupled model works does not test this. We therefore probed whether the two encoders of the released model actually carry different information, and whether the shared Transformer can use one in place of the other.

### 5.1 Pathway swap: can the understanding head read the generation encoder?
Each probe image was encoded with the generation encoder (VQ → 576 codes). The codes were embedded with `gen_embed → gen_aligner`, which is exactly how the model consumes its own generated image tokens. We then inserted these 576 vectors into the `<image_placeholder>` slots of the understanding prompt, **in place of** the SigLIP → aligner vectors, and decoded greedily (160 tokens). We used two settings:
- as is;
- with the three image special tokens (`<begin_of_image>`, `<end_of_image>`, `<image_placeholder>`) banned through `bad_words_ids`.

The SigLIP baseline was produced through the same manual code path. It reproduces the official pipeline's answers token for token, which validates the harness.

| Image | Visual input to the LLM | Decoded answer (first characters, verbatim) |
|---|---|---|
| equation | SigLIP → und. adaptor | `Sure, here is the LaTeX code for the formula: \[ A_n = a_0 \left[ 1 + \frac{3}{4} …` (exact, as in §3.1) |
| equation | VQ → gen. adaptor | `<end_of_image>` × 160 (all 160 tokens) |
| equation | VQ → gen. adaptor, image tokens banned | `������[��������������ﻥ����������������市场上���…` (invalid byte tokens; no words) |
| doge | SigLIP → und. adaptor | `The meme compares two different approaches to visual encoding: "Decoupling Visual Encoding" and "Single Visual Encoder." …` (as in §3.2) |
| doge | VQ → gen. adaptor | `<end_of_image>` × 160 |
| doge | VQ → gen. adaptor, image tokens banned | `��������[��[��[��[��[������[����[���…` |

When the language model sees generation-pathway embeddings, it behaves as if it were **in image-generation mode**: it tries to close the image (`<end_of_image>`). When that is forbidden, it collapses to byte-level noise. The shared Transformer has learned to treat the two input subspaces as *functionally different*. The VQ code embeddings carry no understanding signal that the text head can use, even though they come from the same image. This is direct behavioural evidence that the released model relies on the decoupled understanding encoder for comprehension.

### 5.2 What the generation encoder preserves: reconstruction and glyph fidelity
The VQ encode→decode round-trip on the 384² model inputs gives:

| Image | Reconstruction PSNR | Unique codes used (of 576 positions) |
|---|---|---|
| equation | 24.5 dB | 213 |
| doge | 22.5 dB | 311 |

Figures 7 and 7b show that **global layout, colour, the dogs' shapes and the bracket geometry are preserved, but glyph identity is not**:
- "Decoupling Visual Encoding" becomes an illegible string of letter-like blobs.
- In the formula, `a₀`→"o₀", `3`→"8"-like, `4`→"Λ", `9`→"θ", and `k=1` is smeared.

At 16× downsampling, each 8-d code covers a 16×16 patch, and its job is to support pixel synthesis, not symbol identity. A single VQ encoder shared by both tasks would therefore hand the LLM an input in which the very characters it must transcribe are already corrupted. This is exactly the conflict the decoupled design avoids. The SigLIP pathway reads the same 384² input perfectly (§3.1).

![Pathway probe](images/fig7_pathway_probe.png)

*Figure 7. Columns 1–2: the model input and its VQ reconstruction. Column 3: L2 norms of the LLM input vectors for SigLIP→aligner tokens (blue, broad, median ≈48–65), VQ→gen_aligner tokens (orange, tightly peaked, ≈12.3) and text token embeddings (grey, ≈6.9); log x-axis. Column 4: cosine similarities in the shared 2048-d input space.*

![VQ zoom](images/fig7b_vq_text_zoom.png)

*Figure 7b. Zoomed crops: the VQ generation encoder keeps the layout but loses glyph identity.*

### 5.3 Geometry of the two adaptors in the shared input space
The quantities below are measured over the 576 image positions per image (`outputs/probes/*_geometry.npz`):
- **Norms (Fig. 7, col. 3).** The three input populations occupy disjoint norm shells:
  - understanding tokens have median ‖·‖ ≈ 48 (equation) / 65 (doge);
  - generation tokens ≈ 12.3–12.5, with a very narrow spread;
  - text embeddings ≈ 6.9.
  The understanding adaptor maps to vectors 4–5× larger than the generation adaptor's.
- **Alignment.** For the same image patch, the cosine between the understanding vector and the generation vector has mean **0.021** for both images. The two pathways embed the same content in essentially **orthogonal** directions.
- **Relation to text.** The nearest text-token embedding has cosine ≈0.08 for understanding tokens and ≈0.10 for generation tokens. Neither pathway is simply "translated into words". Both are continuous, non-lexical inputs.
- **Within-image diversity.** The mean pairwise cosine among the 576 tokens is 0.32/0.23 for understanding (equation/doge) and 0.47/0.32 for generation. Generation tokens are more mutually similar, consistent with an 8-d codebook projected into 2048-d.

Together with §5.1: the single Transformer hosts **two separable input subspaces** and routes them to different behaviours (text answer vs. image continuation). Unification happens in the shared weights, not in a shared visual representation.

### 5.4 Encoder dissociation on natural images (Imagenette)
To test the "semantic vs. pixel" trade-off beyond the two probe images, we encoded 120 natural images with both pathways: the first 12 validation images of each of the 10 Imagenette classes (ImageNet subset; tench … parachute, WNID `n03888257`). We then fitted two kinds of **linear** read-outs:
- **Semantics:** logistic-regression class probe (10-way; chance 0.10).
- **Pixel detail:** ridge regression to the image's 8×8 RGB thumbnail (192 targets; variance-weighted R²).

Both use 5-fold cross-validation, repeated with 3 shuffling seeds (mean ± s.d. over seeds). Features are taken *at the LLM's input*, i.e. after each adaptor, and are either mean-pooled over the 576 positions or average-pooled to a 4×4 spatial grid. The two pathways have identical dimensionality at each pooling level (`code/probe_encoders.py`, `code/analyze_encoder_probe.py`).

| Representation | Dim | Class-probe accuracy | Thumbnail R² |
|---|---|---|---|
| SigLIP output, mean-pooled | 1024 | 0.983 ± 0.000 | 0.109 ± 0.012 |
| SigLIP → und. adaptor, mean-pooled | 2048 | 0.983 ± 0.007 | 0.104 ± 0.010 |
| SigLIP → und. adaptor, 4×4 grid | 32768 | 0.972 ± 0.004 | 0.112 ± 0.009 |
| VQ → gen. adaptor, mean-pooled | 2048 | 0.394 ± 0.004 | 0.307 ± 0.002 |
| VQ → gen. adaptor, 4×4 grid | 32768 | 0.567 ± 0.027 | 0.516 ± 0.009 |

The mean VQ round-trip PSNR over these images is 24.2 ± 2.6 dB.

![Encoder probe](images/fig8_encoder_probe.png)

*Figure 8. Double dissociation between the two visual pathways of the same trained model.*

The result is a clean **double dissociation**:
- **Semantics:** the understanding pathway makes object category almost linearly separable (≈0.97–0.98 with only about 96 training images per fold). The generation pathway gives 0.39–0.57, far above chance but much worse.
- **Pixel detail:** the generation pathway's features predict the thumbnail far better (R² 0.31 mean-pooled, 0.52 with a 4×4 grid). SigLIP's features stay at R² ≈ 0.11 *even when spatial pooling is kept*: semantic training has discarded most colour and layout information.

The understanding adaptor fully preserves SigLIP's class separability (0.983 for both raw and adapted features). This is the quantitative form of the paper's motivating claim. Within the released model, *the information required for understanding and the information required for generation sit in different encoders*, and neither encoder is a good stand-in for the other. Caveat: n = 120 images and linear probes only. A nonlinear read-out could recover more from either representation.

