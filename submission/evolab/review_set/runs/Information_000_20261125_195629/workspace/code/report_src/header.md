# Decoupling Visual Encoding for Unified Multimodal Understanding and Generation: a CPU Reproduction and Mechanistic Analysis of Janus-1.3B

## Abstract
We study a unified autoregressive framework that **decouples visual encoding**. Understanding inputs go through a semantic encoder (SigLIP) and an understanding adaptor. Generation uses a discrete VQ image tokenizer with its own input adaptor and output head. One shared Transformer (a 1.3B-parameter LLM) processes both streams. This is the Janus design of Wu et al. (2024).

We had a CPU-only machine and so could not retrain. Instead we ran the **released Janus-1.3B checkpoint with the official code** end to end and evaluated the paper's three showcase behaviours on the provided data.

1. **Formula → LaTeX.** The model transcribes `equation.png` *exactly* (normalised edit distance 0; the output compiles). It stays exact under strong noise, moderate blur and 4× down-scaling, and degrades gracefully under extreme down-scaling and rotation (9-condition sweep).
2. **Meme understanding.** On `doge.png` it reads both captions verbatim, binds "Decoupling Visual Encoding" to the muscular dog and "Single Visual Encoder" to the sitting dog, and, when asked, states that decoupling is the "better", "more efficient and powerful" approach. It misses the crying cue, only partly explains the joke, and hallucinates technical glosses.
3. **Text-to-image of the two-faced god Janus.** Across two prompts and four samples, the model produces two-faced compositions in 3 of 4 samples and the requested cool vs. warm palette contrast in 4 of 4. Faces "looking in opposite directions" and a clean "young, beardless" second face were achieved only partially.

We then provide mechanistic evidence for *why* decoupling is needed, all from inside the released model:
- **(i) Pathway swap.** Generation-pathway (VQ) embeddings placed in an understanding prompt make the LLM emit only `<end_of_image>`, or byte noise when those tokens are banned.
- **(ii) Glyph loss.** The VQ tokenizer loses glyph identity in reconstruction.
- **(iii) Separate subspaces.** The two adaptors embed the same patch in near-orthogonal subspaces (cosine 0.02) with disjoint norm shells.
- **(iv) Double dissociation.** On 120 natural images, linear probes give class accuracy **0.98 vs. 0.39** (understanding vs. generation features) and pixel-thumbnail R² **0.10 vs. 0.31** (0.11 vs. 0.52 with spatial pooling).

The 1.3B LLM itself is the bottleneck for reasoning: it cannot compute the limit of the formula it transcribed perfectly.

**Key results at a glance**

| Probe | Setting (n) | Outcome |
|---|---|---|
| Formula → LaTeX, clean | greedy (1) | **exact**, NED 0.000, compiles |
| Formula → LaTeX, 9 degradations | greedy (1 each) | 4/9 exact; NED ≤ 0.13 for 8/9; rotation 30° fails (0.51) |
| Formula reasoning (n→∞ limit) | greedy (1) | ✗ answers "1"; correct is 8/5·a₀ |
| Meme OCR | greedy (1) | ✓ both captions exact |
| Meme symbolism / evaluation | greedy (5 prompts) | ✓ muscular = decoupling = "better"; ~ humour; ✗ "same dog", crying missed, invented glosses |
| Janus T2I | CFG 5, 2 prompts × 2 samples | 3/4 two faces, 4/4 palette contrast, 1/4 clear young face, 0/4 opposite gaze |
| CFG sweep (Janus prompt) | CFG 1 / 5 / 10, n = 1 / 2 / 1 | guidance is needed for attribute adherence; CFG 5 is the best trade-off |
| README showcase T2I | CFG 5, n = 2 | both landmarks and the starry-swirl sky present in 2/2; entity fusion in 1/2 |
| Pathway swap (VQ → und. prompt) | 2 images × 2 settings | 0/4 meaningful outputs vs. 2/2 for SigLIP |
| Encoder dissociation (Imagenette) | 120 images, 5-fold × 3 | class acc. 0.98 vs. 0.39; thumbnail R² 0.10 vs. 0.31 |

## 0. Data overview
The task provides two probe images, shown in Figure 1. Both also appear in the official repository's `images/` folder and in the paper's qualitative figures.
- **`data/equation.png`**: 1050×344 RGB, black-on-white typeset formula A_n = a₀[1 + ¾ Σ_{k=1}^{n} (4/9)^k]. It contains subscripts, a fraction, a summation with limits, a parenthesised fraction raised to the power k, and tall delimiters. It tests OCR and formula→LaTeX conversion.
- **`data/doge.png`**: 1200×799 RGB. The "Swole Doge vs. Cheems" meme template: a muscular standing Shiba under "Decoupling Visual Encoding" (left), and a small, sitting, tearful Shiba under "Single Visual Encoder" (right). It tests OCR in context, text–object binding and high-level, humorous interpretation.

Both images are letter-boxed to a square with grey (127,127,127) and resized to 384×384 by the model's processor, as the 384² inputs in Fig. 7 show. The formula therefore occupies only about 126 px of height in the model's view.

We used two auxiliary resources:
- 120 Imagenette validation images (12 per class × 10 classes), for the encoder probe of §5.4;
- OpenAI CLIP ViT-B/32, for CLIPScore.

![Data overview](images/fig1_data_overview.png)

*Figure 1. The two provided probe images.*

![Architecture](images/fig2_architecture.png)

*Figure 2. The decoupled-encoding architecture as instantiated in Janus-1.3B. Dimensions are verified from the checkpoint config.*
