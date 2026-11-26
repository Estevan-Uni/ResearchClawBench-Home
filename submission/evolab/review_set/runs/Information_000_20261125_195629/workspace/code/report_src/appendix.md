## Appendix A — Reproducibility

| Item | Location |
|---|---|
| Parallel resumable HF downloader | `code/download_hf.py` |
| CPU loader (attrdict shim, SDPA override, fp32) | `code/janus_common.py`, `code/shims/attrdict.py` |
| Understanding runs (official template, greedy, 512 tokens) | `code/understand.py`; job specs in `outputs/jobs_*.json`; raw answers in `outputs/understanding_core.jsonl`, `outputs/understanding_robust.jsonl`, `outputs/understanding_selfcheck.jsonl` |
| LaTeX scoring (normalisation, Levenshtein, bracket balance) | `code/score_latex.py`; scores in `outputs/latex_scores.json` |
| Degradations | `code/robust_tf.py` |
| Text-to-image (official CFG loop) | `code/generate_t2i.py`; images, codes and timings in `outputs/t2i/` (`log.jsonl`, `*_codes.npy`) |
| CLIPScore | `code/clipscore.py` → `outputs/clipscore.json` |
| Pathway swap, VQ reconstruction, geometry | `code/probe_pathways.py` → `outputs/probes/pathway_probe.json`, `*_geometry.npz`, `*_vqrecon.png` |
| Encoder dissociation | `code/probe_encoders.py` (feature extraction) and `code/analyze_encoder_probe.py` (CV probes) → `outputs/probes/encoder_feats.npz`, `encoder_probe.json` |
| Figures | `code/fig_overview.py`, `code/fig_results.py`, `code/fig_t2i.py`, `code/fig_vq_zoom.py` |
| Report assembly | `code/build_report.py` |

**Determinism.**
- Understanding uses greedy decoding and is deterministic given fp32 CPU kernels.
- T2I seeds are set with `torch.manual_seed(seed)` before sampling. Batches with different n consume random numbers differently, so `janus_s0_0` and `janus_cfg*_s0_0` are *not* the same noise trajectory.
- Cross-validation uses fixed seeds 0–2.

**Compute.** 2 CPU cores on a heavily shared host. Measured costs:
- about 4 s per decoded text token;
- 39–65 min per 2-image generation;
- about 3.5–30 min per understanding query;
- about 45 min of feature extraction for 120 images.

No GPU, and no training or fine-tuning of any kind was performed.

## References
- Wu, C., Chen, X., Wu, Z., *et al.* (2024). *Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation.* arXiv:2410.13848. Code and weights: github.com/deepseek-ai/Janus, huggingface.co/deepseek-ai/Janus-1.3B.
- Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L. (2023). *Sigmoid Loss for Language Image Pre-Training* (SigLIP). ICCV.
- Sun, P., Jiang, Y., Chen, S., *et al.* (2024). *Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation* (LlamaGen; VQ tokenizer).
- Liu, H., Li, C., Wu, Q., Lee, Y. J. (2023). *Visual Instruction Tuning* (LLaVA). NeurIPS.
- Chameleon Team (2024). *Chameleon: Mixed-Modal Early-Fusion Foundation Models* (single VQ tokenizer for both tasks).
- Hessel, J., *et al.* (2021). *CLIPScore: A Reference-free Evaluation Metric for Image Captioning.* EMNLP.
