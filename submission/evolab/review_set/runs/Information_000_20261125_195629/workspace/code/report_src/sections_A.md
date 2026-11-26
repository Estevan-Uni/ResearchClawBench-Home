## 1. Introduction

Unified multimodal models try to understand images (visual question answering, OCR, reasoning) and generate them (text-to-image) with one autoregressive Transformer. Earlier unified designs feed one visual encoder to both tasks: Chameleon's VQ tokenizer, LWM, Show-o, and others. The two tasks pull that encoder in opposite directions:

- **Understanding** needs *high-level semantics*: what objects are present, what the text says, what is meant. Semantic encoders such as SigLIP (Zhai et al., 2023), trained contrastively on image–text pairs, supply this, and LLaVA-style models (Liu et al., 2023) project such features into the LLM.
- **Generation** needs *low-level, spatially precise detail* in a *discrete* code that a next-token predictor can sample, and that a decoder can turn back into pixels. VQ tokenizers in the LlamaGen line (Sun et al., 2024) supply this.

Janus (Wu et al., 2024, "Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation") resolves this tension by **decoupling the visual encoding**. Understanding images go through a SigLIP encoder and an understanding adaptor. Generated images are represented by VQ codes, which enter through a generation adaptor and leave through a dedicated image head. Both streams are processed by *the same* autoregressive Transformer (DeepSeek-LLM 1.3B). Figure 2 summarises the architecture.

**Our goals:**
1. Run the framework end-to-end on the provided probe data, plus the paper's T2I showcase subject.
2. Verify the qualitative claims of the paper's showcase figures:
   - formula→LaTeX OCR;
   - explaining the "Decoupling Visual Encoding vs. Single Visual Encoder" meme;
   - a text-to-image rendering of the two-faced god Janus.
3. Look for *mechanistic* evidence, inside the released model, that the two pathways encode different information. We test this with:
   - a pathway-swap experiment;
   - VQ reconstruction;
   - embedding-geometry analysis;
   - a semantic-vs-pixel probe of the two encoders.

**Scope and honesty statement.** No GPU was available. The machine had 2 CPU cores on a heavily shared host (load average about 190), so the 3-stage Janus training (adaptor warm-up, unified pre-training, supervised fine-tuning) on about 10⁸ samples could not be repeated. **Every result below comes from inference with the released `deepseek-ai/Janus-1.3B` checkpoint.** We ran the official open-source code, patched only to run on CPU. Our own contributions are:
- the evaluation protocol;
- the robustness sweep;
- the probes;
- the analysis.

At about 4 s per decoded token, one understanding query took 1–29 min and one 2-image generation took 39–65 min. Sample sizes are therefore small, and they are stated for every experiment.

## 2. Method

### 2.1 The decoupled framework (as implemented in the released model)
Table 1 lists the components. Each entry was verified from `config.json` and the official code.

| Component | Understanding path | Generation path |
|---|---|---|
| Visual encoder | SigLIP-Large/16 @384 (`siglip_large_patch16_384`), last layer → 24×24 = 576 tokens × 1024-d | VQ tokenizer (LlamaGen-style VQ-16), downsample ×16 → 24×24 = 576 discrete codes, codebook 16 384 × 8-d |
| Adaptor into the LLM | `aligner`: 2-layer GELU MLP 1024→2048 | `gen_embed` (16 384×8) followed by `gen_aligner`, a 2-layer GELU MLP 8→2048 |
| Shared backbone | DeepSeek-LLM 1.3B (24 layers, d = 2048, 16 heads, vocab 102 400), causal attention | same weights |
| Output head | LM head over the text vocabulary | `gen_head`: 2-layer MLP 2048→16 384 image-code logits |
| Decoding | greedy / sampling of text tokens | 576 sampled codes → VQ decoder → 384×384 RGB |

In an understanding sequence, the 576 aligned SigLIP vectors replace the `<image_placeholder>` tokens between `<begin_of_image>` and `<end_of_image>`. Generation follows `… prompt … <begin_of_image>`: the model autoregressively predicts 576 codes, and each sampled code is fed back through `gen_embed→gen_aligner`. Classifier-free guidance (CFG) is applied to the logits:

ℓ = ℓ_uncond + s·(ℓ_cond − ℓ_uncond)

The unconditional branch is the same sequence with the prompt tokens replaced by `<pad>`. This replacement mirrors the 10% text dropout used in training.

### 2.2 Execution environment and patches
We used the official repository `deepseek-ai/Janus`, downloaded as a GitHub zip, together with the `deepseek-ai/Janus-1.3B` weights (4.18 GB safetensors). The weights were fetched with our own resumable, parallel HTTP-range downloader, `code/download_hf.py`. Software: `torch 2.15 (CPU)`, `transformers 4.44.2`, `timm 1.0`.

Two patches were needed, and neither changes model numerics beyond kernel choice:
1. The obsolete `attrdict` package does not work on Python ≥3.10. We replaced it with a 10-line `AttrDict(dict)` shim in `code/shims/attrdict.py`.
2. The checkpoint config requests `flash_attention_2`, which does not run on CPU. We overrode `language_config._attn_implementation = "sdpa"`.

The model runs in **fp32**, whereas the official GPU scripts use bf16. Small numerical differences can therefore change greedy-decoded text relative to the paper's figures. The model loads in about 7 s. The shared loader is `code/janus_common.py`.

### 2.3 Protocols
- **Understanding** (`code/understand.py`):
  - Uses the official `deepseek_old` SFT template with the official system prompt, and `<image_placeholder>\n{question}` as the user turn.
  - Images are padded to a square with grey (127,127,127) and resized to 384².
  - Greedy decoding, `max_new_tokens = 512`, as in the official `inference.py`.
  - Multi-turn questions re-feed the full dialogue history.
  - Every raw answer is stored in `outputs/understanding_*.jsonl`.
- **Formula scoring** (`code/score_latex.py`):
  - We hand-transcribed the ground truth from `data/equation.png`: `A_n = a_0 \left[ 1 + \frac{3}{4} \sum_{k=1}^{n} \left( \frac{4}{9} \right)^k \right]`.
  - The formula is extracted from the answer's `\[..\]`, `$$..$$` or fenced code.
  - Normalisation removes `\left`/`\right`, whitespace and `$`, and collapses single-character braces (`{x}`→`x`).
  - We report the Levenshtein distance and its normalised version (NED), check bracket balance, and **re-render** the LaTeX with matplotlib mathtext as a compile check.
- **Robustness sweep:** 9 degraded copies of the formula image. A 14-condition grid was planned and cut to 9 because of CPU time:
  - down-scaling to 0.25/0.12;
  - Gaussian blur σ = 3/5;
  - additive Gaussian noise σ = 50/80;
  - rotation 15°/30°;
  - JPEG quality 10.
  Each copy is scored by NED (`code/robust_tf.py`).
- **Text-to-image** (`code/generate_t2i.py`):
  - The official CFG loop: temperature 1, 576 tokens, `gen_head` logits, and re-embedding of each sampled code with `prepare_gen_img_embeds`, followed by `decode_code` to 384².
  - Default CFG s = 5, n = 2 images per prompt, fixed seeds.
  - We inspected each image visually. CLIPScore (ViT-B/32, 100·max(cos, 0)) is supporting evidence only.
- **Mechanistic probes** (`code/probe_pathways.py`, `code/probe_encoders.py`): see §5.
