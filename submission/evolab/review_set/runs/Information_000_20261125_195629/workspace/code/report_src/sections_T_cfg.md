### 4.5 Effect of classifier-free guidance
We re-ran the attempt-1 Janus prompt at CFG s = 1 (no guidance; the conditional logits alone) and s = 10, with seed 0 and n = 1 each. These runs took 31–32 min each, and we compare them with the two s = 5 samples. Because the batch size differs, the s = 1/10 samples do not share the noise trajectory of the s = 5 samples, so this is a qualitative trend at n = 1 per setting.

![CFG sweep](images/fig10_t2i_cfg_sweep.png)

*Figure 10. The same prompt at CFG 1, 5 (both samples) and 10.*

| CFG | Faces | Palette contrast | Anatomy and quality | CLIPScore (own prompt) |
|---|---|---|---|---|
| 1 | 1 (old) | weak: mostly grey-blue, faint orange accents | distorted, bulging eyes; warped mouth; noisy drapery | 27.6 |
| 5 | 1 and 2 | strong blue vs. orange | clean sculptural detail; the second face in 1 of 2 samples | 31.7 / 31.0 |
| 10 | 1 (old, frontal) | moderate: blue face with an orange right flank | clean, symmetric, slightly over-saturated "painted" look | 31.1 |

Guidance is what makes the prompt's attributes appear. Without it (s = 1), the image keeps the gist ("old bearded sculpture") but loses both coherence and the colour instruction. Raising s from 5 to 10 improves anatomical cleanliness but does not add the second face, and it reduces diversity. CFG 5, the official default, gave the best trade-off in our small sample. CLIPScore follows the same ordering (s = 1 lowest), but the differences between s = 5 and s = 10 are within noise.

The README images score 34.0 and 28.9 on their own prompt, against 12.7–12.8 for the control.
