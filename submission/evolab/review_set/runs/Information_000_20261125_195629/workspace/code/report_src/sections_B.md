## 3. Results I — Multimodal understanding through the SigLIP pathway

All answers below are **complete and verbatim**: greedy decoding, `max_new_tokens = 512`, fp32 CPU, released Janus-1.3B. The raw records are in `outputs/understanding_core.jsonl`. Wall-clock times are for a single query on 2 shared CPU cores.

### 3.1 Showcase 1 — Formula → LaTeX (`data/equation.png`)

**Prompt:** `Convert the formula into latex code.` (the official `inference.py` example, 316 s)

```text
Sure, here is the LaTeX code for the formula:

\[ A_n = a_0 \left[ 1 + \frac{3}{4} \sum_{k=1}^{n} \left( \frac{4}{9} \right)^k \right] \]

This is the LaTeX code for the given formula.
```

![Formula OCR](images/fig3_latex_ocr.png)

*Figure 3. (a) Input image. (b) The model's LaTeX, extracted from the `\[ … \]` block and re-rendered with matplotlib mathtext. (c) Our hand-transcribed ground truth, rendered the same way.*

**Analysis (match):**

| Check | Result |
|---|---|
| Exact match after normalisation | **yes**: edit distance 0, NED 0.000 |
| Sub-structures recovered | subscripts `A_n`, `a_0`; the fraction `\frac{3}{4}`; summation limits `_{k=1}^{n}`; the inner fraction `\frac{4}{9}`; the outer exponent `^k` placed *outside* the parentheses; auto-sized delimiters `\left[ … \right]` and `\left( … \right)` matching the tall brackets in the image |
| Brackets balanced | yes |
| Compiles / renders | yes (mathtext, Fig. 3b; visually identical to the input) |
| Raw (un-normalised) string vs. ground truth | identical character for character |

This reproduces the formula-recognition showcase of the Janus paper qualitatively. Two follow-up questions on the same image test how far the understanding goes beyond transcription.

**Follow-up A, `Describe this image in detail.`** (1 729 s; verbatim):

```text
The image displays a mathematical expression involving a matrix and summation. The expression is:

\[ A_n = a_0 \left[ 1 + \frac{3}{4} \sum_{k=1}^{n} \left( \frac{4}{9} \right)^k \right] \]

Here is a detailed breakdown of the components:

1. **Matrix Representation**:
   - The expression is written as a matrix, where each element is represented by a variable.
   - The matrix is denoted as \( A_n \).

2. **Matrix Elements**:
   - The first element of the matrix is \( a_0 \).
   - The second element is \( 1 + \frac{3}{4} \sum_{k=1}^{n} \left( \frac{4}{9} \right)^k \).
   - The third element is \( \left( \frac{4}{9} \right)^k \).

3. **Summation**:
   - The summation is denoted by \( \sum_{k=1}^{n} \left( \frac{4}{9} \right)^k \).
   - This summation is repeated \( n \) times, indicating that the sum is calculated for each index \( k \) from 1 to \( n \).

4. **Summation Term**:
   - The term \( \left( \frac{4}{9} \right)^k \) is a mathematical expression involving the fraction \( \frac{4}{9} \).
   - This fraction is repeated \( k \) times, indicating that the sum is calculated for each \( k \) from 1 to \( n \).

5. **Mathematical Operations**:
   - The expression involves arithmetic operations such as addition, multiplication, and exponentiation.
   - The exponentiation is performed using the exponentiation operator \( \left( \frac{4}{9} \right)^k \).

6. **Mathematical Context**:
   - This expression is likely used in a mathematical or computational context, possibly for solving a problem involving matrix operations or for performing calculations involving fractions and exponents.

In summary, the image presents a matrix expression involving summation and arithmetic operations, where the matrix elements are calculated using a specific formula involving fractions and exponents.
```

The transcription is again exact. The interpretation, however, is **hallucinated**: the expression is called a "matrix" with "elements", most likely because the large square brackets resemble matrix delimiters. The summation and exponent descriptions are correct but superficial.

**Follow-up B (second dialogue turn), `What is the value of A_n as n approaches infinity? Show your reasoning.`** (811 s; verbatim):

```text
As n approaches infinity, the value of A_n will approach 1. This is because the sum of the terms in the series converges to 1 as n increases without bound. Specifically, the sum of the terms in the series is given by:

\[ \sum_{k=1}^{n} \left( \frac{4}{9} \right)^k = \frac{4}{9} \sum_{k=1}^{n} k^9 = \frac{4}{9} \left( 1 + \frac{3}{4} \sum_{k=1}^{n} k^4 \right) \]

As n approaches infinity, the term \(\frac{4}{9} \sum_{k=1}^{n} k^9\) approaches 1, and the sum of the terms \(\sum_{k=1}^{n} k^4\) approaches 1. Therefore, as n approaches infinity, the value of A_n will approach 1.
```

The correct value is A_∞ = a₀[1 + ¾·(4/9)/(1−4/9)] = a₀(1 + ¾·⅘) = **8/5·a₀**. The model's answer of 1 is wrong. Its algebra is invalid: (4/9)^k becomes k⁹, and a divergent Σk⁹ is said to "approach 1". It also drops a₀.

So the 1.3B model *perceives* the formula perfectly but cannot *reason* about it. This is consistent with the paper's positioning of Janus-1.3B, a small LLM with limited mathematical ability. It also shows that the OCR success is a perceptual result of the SigLIP pathway, not general mathematical competence.

### 3.2 Showcase 2 — Meme understanding (`data/doge.png`)

The image (Fig. 1b) shows a muscular, standing "Swole Doge" under the caption **"Decoupling Visual Encoding"**. Opposite it, a small, sitting, tearful Shiba ("Cheems") sits under **"Single Visual Encoder"**. The intended message: decoupled encoding is the strong approach and a single shared encoder is the weak one.

**Prompt 1 (standard open-ended meme prompt): `Explain this meme.`** (463 s; verbatim, complete):

```text
The meme compares two different approaches to visual encoding: "Decoupling Visual Encoding" and "Single Visual Encoder." The left side of the meme shows a muscular, cartoonish representation of a dog, which is a visual encoding technique that separates the visual data into different layers or components. The right side of the meme shows a more relaxed, cartoonish representation of the same dog, which is a single visual encoder that processes the visual data as a single unit. The meme uses the image of the dog to visually represent the concept of decoupling and the single encoder, with the muscular dog symbolizing the decoupling technique and the more relaxed dog symbolizing the single encoder.
```

**Prompt 2: `What text is written in this image? Transcribe it exactly.`** (48 s; verbatim):

```text
Decoupling Visual Encoding Single Visual Encoder
```

**Prompt 3: `Which dog looks stronger, the left one or the right one? What is written above each dog, and what does this comparison suggest?`** (344 s; verbatim):

```text
The left dog, which is the muscular one, appears to be stronger than the right dog. This comparison suggests that the left dog represents a method or system that is more robust or capable of handling more complex tasks or data, as it is depicted with a muscular build and a confident posture. In contrast, the right dog, which is the sitting dog, represents a simpler or less robust method, as it is depicted in a more relaxed posture and does not appear to have the same level of strength or capability.
```

**Prompt 4: `Why is this meme funny? Describe the appearance and posture of each dog, the text above each one, and what each dog represents.`** (1956 s; verbatim, complete):

```text
The meme is funny because it uses a visual pun to contrast two different approaches to image encoding. The image of the muscular dog on the left represents the "Decoupling Visual Encoding," which is a technique where the visual features of an image are encoded separately from the semantic meaning. This is visually represented by the dog's exaggerated muscles, which symbolize the emphasis on visual features over semantic meaning.

The image of the sitting dog on the right represents the "Single Visual Encoder," which is a more traditional approach to image encoding where the visual features of an image are encoded as a single unit. This is visually represented by the dog's more relaxed posture and the lack of emphasis on muscles, which symbolize the focus on the semantic meaning of the image.

The text above each dog reads "Decoupling Visual Encoding" and "Single Visual Encoder," respectively, which further emphasizes the contrast between the two approaches. The humorous aspect lies in the visual pun, where the muscular dog represents the decoupling of visual features from semantic meaning, while the sitting dog represents the single visual encoder approach.
```

**Prompt 5: `This meme was made by the authors of a research paper. Which approach do they think is better, and how does the image convey that?`** (1176 s; verbatim, complete):

```text
The meme uses the image of the Shiba Inu dog to convey that the approach of "Decoupling Visual Encoding" is better than the approach of "Single Visual Encoder." This is because the image of the muscular, less-than-fit Shiba Inu represents the idea of a more efficient and powerful encoding system, which is the approach of "Decoupling Visual Encoding." In contrast, the image of the sitting Shiba Inu represents the "Single Visual Encoder," which is the approach of "Decoupling Visual Encoding." The visual content of the image is used to illustrate the difference between these two approaches, with the muscular dog symbolizing the more efficient and powerful encoding system.
```


**Element-by-element annotation** (✓ correct, ~ partially correct, ✗ wrong or hallucinated):

| Element | What the model said | Verdict |
|---|---|---|
| Embedded text (OCR) | "Decoupling Visual Encoding" and "Single Visual Encoder", spelled and capitalised exactly (P1, P2) | ✓ |
| Text ↔ image binding | Left/muscular ↔ "Decoupling Visual Encoding"; right ↔ "Single Visual Encoder" (P1) | ✓ |
| Topic of the meme | "compares two different approaches to visual encoding" (P1) | ✓ |
| Left dog: appearance and pose | "muscular, cartoonish" (P1); "muscular build and a confident posture" (P1, P3) | ✓ (the "cartoonish" reading fits the photoshopped body) |
| Right dog: appearance and pose | "more relaxed" (P1); "the sitting dog … more relaxed posture" (P3) | ~ the sitting pose is right; the **tears and sad expression are missed**, and "relaxed" softens the intended "weak / pathetic" reading |
| Identity of the dogs | "the same dog" (P1); "the Shiba Inu dog" (P5) | ~ the breed is right (P5), but these are two different meme characters (Swole Doge and Cheems), not "the same dog" |
| Symbolism | "muscular dog symbolizing the decoupling technique and the more relaxed dog symbolizing the single encoder" (P1) | ✓ |
| Evaluative message (who wins) | P1: only implicit. P3: the left is "stronger … more robust or capable of handling more complex tasks"; the right is "a simpler or less robust method … does not appear to have the same level of strength or capability". P5: "convey that the approach of 'Decoupling Visual Encoding' is better than the approach of 'Single Visual Encoder'", with the muscular dog as "a more efficient and powerful encoding system" | ✓ when asked directly (P3, P5); ~ in the open-ended explanation |
| Internal consistency | P5: "muscular, less-than-fit Shiba Inu"; the sitting dog "represents the 'Single Visual Encoder,' which is the approach of 'Decoupling Visual Encoding'" | ✗ self-contradictory slips |
| Technical gloss of "decoupling" | "separates the visual data into different layers or components" (P1); "visual features of an image are encoded separately from the semantic meaning … muscles … symbolize the emphasis on visual features over semantic meaning" (P4) | ✗ hallucinated. In Janus, decoupling means *separate encoders for understanding (semantic) and generation (pixel/VQ)*; the muscles stand for strength, not "visual features" |
| Technical gloss of "single encoder" | "processes the visual data as a single unit" (P1) | ~ vague but not wrong |
| Humour | P1: not explained. P4: "the humorous aspect lies in the visual pun …", which names a pun but never says the joke is the exaggerated strong-vs-pathetic (crying) contrast as a boast about the authors' own method | ~ only partly explained |
| Text placement | P4: "The text above each dog reads 'Decoupling Visual Encoding' and 'Single Visual Encoder,' respectively" | ✓ |
| Instruction following | P3 asked "what is written above each dog?" and the answer omits it | ✗ partial omission |

**Match against the paper's showcase.** The paper's Figure 5 presents Janus explaining this meme as a contrast between the strong decoupled design and the weak single-encoder design. Our run of the released 1.3B checkpoint reproduces:
- the OCR;
- the left↔right binding of captions to dogs;
- the "muscular = decoupling, weaker = single encoder" symbolism;
- with a direct question, the "stronger / more capable vs. less robust" evaluation (P3) and the conclusion that the authors consider decoupling "better" (P5).

It does **not** reproduce:
- an explicit statement of the joke in the open-ended prompt;
- the crying, sad cue.

It also adds hallucinated technical glosses (P1, P4) and a self-contradiction (P5). The divergence may partly come from fp32 vs. bf16 numerics under greedy decoding. It may also reflect that the paper's figure answers are the best qualitative examples rather than typical ones.

Overall, the SigLIP pathway delivers the semantic and OCR content needed to understand the meme. The 1.3B language model's commentary on that content is shallow.
