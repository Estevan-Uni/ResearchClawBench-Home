## 6. Discussion

### 6.1 What the evidence says about decoupled visual encoding
The three showcase probes and the mechanistic probes support one consistent picture.

1. **Understanding relies on a semantic encoder, and that encoder is sufficient for fine-grained reading.**
   - The SigLIP pathway recovers the formula exactly, including scripts, limits and delimiter sizing.
   - It stays exact under heavy noise, moderate blur and 4× down-scaling.
   - It reads both meme captions verbatim and binds each to the correct dog.
2. **The generation encoder is a poor substitute for understanding, for two independent reasons.**
   - *Information:* a VQ round trip of the same 384² input visibly corrupts glyph identity (`a₀`→"o₀", 4→"Λ", "Visual"→illegible). The information needed for OCR is partly destroyed at 16× downsampling.
   - *Interface:* even the information that survives is unusable through the generation adaptor. Fed to the understanding prompt, VQ embeddings make the LLM emit `<end_of_image>` or byte noise.
   The released model has learned two functionally separate, nearly orthogonal input subspaces (same-patch cosine ≈ 0.02; norm shells ≈ 50–65 vs ≈ 12).
   On 120 natural images the dissociation is quantitative:
   - class-probe accuracy is 0.98 for SigLIP→adaptor vs. 0.39 for VQ→adaptor (mean-pooled);
   - thumbnail R² is 0.10 vs. 0.31, and 0.11 vs. 0.52 with 4×4 pooling.
3. **Generation relies on the discrete pixel-level codebook, and works.** With one 1.3B Transformer and no diffusion module, the model renders the requested Janus composition with the requested colour semantics (§4).

These observations are exactly the motivation of the Janus design. A single encoder must either be semantic, which hurts pixel reconstruction and therefore generation, or pixel-faithful and discrete, which hurts OCR and semantics and therefore understanding. Decoupling lets each task use the representation it needs, while the *shared Transformer* still supplies cross-modal knowledge for both: language priors when reading, and world knowledge about "Janus" or "Roman god" when drawing.

### 6.2 Limits of the released 1.3B model
- **Reasoning ≠ perception.** The model transcribes the formula perfectly but evaluates its limit as "1" with invalid algebra; the correct value is 8/5·a₀. The understanding encoder is not the bottleneck here. The 1.3B LLM is.
- **Shallow interpretation.** The meme explanation gets the symbolism right but misses the crying cue and does not spell out the joke. It also invents a technical meaning for "decoupling" (splitting data "into different layers").
- **Hallucinated structure.** Large square brackets trigger a "matrix" reading in the open-ended description, and again under 30° rotation.
- **Compositional generation.** Two faces sharing one head, with per-face attributes, is a hard binding problem. Success is seed-dependent (§4).

### 6.3 Threats to validity
- **No training.** Every result uses the released checkpoint. We did not re-run the three-stage training, the ablations of the paper (e.g., single VQ encoder vs. decoupled), or the benchmark suite (MMBench, POPE, GenEval, …). Our "decoupling helps" evidence is therefore *mechanistic and qualitative within one trained model*. It is not a controlled comparison against a single-encoder model trained under the same budget.
- **Numerics.** We ran in fp32 on CPU, while the paper uses bf16 on GPU. Greedy decoding can diverge between the two, so our verbatim answers need not match the paper's figures word for word.
- **Sample size.** Each understanding condition is one greedy decode. T2I uses n = 2 images per prompt and 1–2 seeds, and CLIPScore is only supporting evidence.
- **Ground truth.** The formula ground truth is our own transcription. We checked it by rendering it next to the input (Fig. 3).

### 6.4 Conclusion
We ran a unified autoregressive model with decoupled visual encoding end to end on CPU (Janus-1.3B, official code, released weights). It reproduced the paper's showcase behaviours:
- exact formula→LaTeX conversion;
- correct OCR and symbolic reading of the decoupling meme;
- a two-faced Janus image with contrasting palettes.

Our probes show *why* the decoupling matters:
- The generation tokenizer cannot carry the fine glyph information that understanding needs, and its embeddings are not interpretable by the language head.
- The semantic encoder that does carry this information cannot reconstruct pixels.
- Both are served by one shared Transformer that keeps them in separate subspaces.
