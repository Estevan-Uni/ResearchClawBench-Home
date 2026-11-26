### 3.3 Robustness of formula recognition to image degradations
We asked whether the exact transcription is a fragile best case. The same prompt was run on 9 degraded copies of `equation.png`, one greedy decode per condition, about 4 min each. The full answers are in `outputs/understanding_robust.jsonl`.

| Condition | NED | Extracted LaTeX (verbatim) | Error type |
|---|---|---|---|
| clean | **0.000** | `A_n = a_0 \left[ 1 + \frac{3}{4} \sum_{k=1}^{n} \left( \frac{4}{9} \right)^k \right]` | none |
| scale 0.25 (262 px wide) | **0.000** | identical | none |
| scale 0.12 (126 px wide) | 0.111 | `… \left( \frac{4}{9} \right)^{n-1} \right]` | exponent `k` → `n-1` (prior-driven guess) |
| Gaussian blur σ = 3 | **0.000** | identical | none |
| Gaussian blur σ = 5 | 0.073 | `A_n = a_0` ⏎ `1 + \frac{3}{4} \sum_{k=1}^{n} \left( \frac{4}{9} \right)^n` | outer `[ ]` dropped; exponent `k`→`n` |
| Gaussian noise σ = 50 | **0.000** | identical | none |
| Gaussian noise σ = 80 | **0.000** | identical | none |
| rotation 15° | 0.128 | `A_n = a_0 \left[ \left\| \left\{ 1 + … \right)^k \right\} \right\| \right]` | spurious `\|…\|` and `\{…\}` delimiters |
| rotation 30° | 0.507 | `A_n = a_0 \| \left\{ \begin{matrix} 1 + \frac{3}{n} \sum_{k=1}^{n} \frac{4}{9} \end{matrix} \right\} \|` | structure lost: `\frac{3}{n}`, missing power and parentheses, matrix hallucination |
| JPEG quality 10 | 0.049 | `A_n = a_0` ⏎ ` 1 + \frac{3}{4} \sum_{k=1}^{n} \left( \frac{4}{9} \right)^k` | outer `[ ]` dropped |

(⏎ marks a line break inside the model's ```latex block. `\|` in the table stands for a literal `|`.)

![OCR robustness](images/fig4_ocr_robustness.png)

*Figure 4. Degraded inputs (top two rows) and normalised edit distance of the extracted LaTeX (bottom). n = 1 greedy decode per condition, so differences of a few characters should not be over-interpreted.*

**Findings.**
1. Transcription is **exact under strong additive noise (σ up to 80/255), moderate blur and 4× down-scaling**. The SigLIP semantic encoder at 384² tolerates photometric corruption well.
2. When the image falls below roughly 130 px or is heavily blurred, the model keeps the overall *structure* but substitutes a **language-prior guess** for the smallest glyph (the exponent `k` → `n-1` or `n`). It also tends to drop the large brackets.
3. **Geometric** corruption hurts most. Rotation by 30° breaks the 2-D layout parsing (NED 0.51) and triggers the same "matrix" hallucination seen in §3.1. SigLIP and the OCR fine-tuning data presumably contain few rotated formulas.
4. All outputs remained bracket-balanced, syntactically valid LaTeX.
