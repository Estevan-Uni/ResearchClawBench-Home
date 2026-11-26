## 4. Results II — Text-to-image generation through the VQ pathway

### 4.1 Showcase 3 — the two-faced Roman god Janus
The paper's T2I showcase depicts its namesake: Janus, the Roman god with two faces that look in opposite directions, toward the past and the future. Typical renderings contrast an old face with a young one. We did not have the paper's exact prompt text, so we wrote two detailed prompts that state the required attributes explicitly:
- two faces on one head;
- old vs. young;
- cool vs. warm palette.

We generated n = 2 images per prompt with the official sampler (CFG 5, temperature 1). Each 2-image batch took 39–65 min on the shared CPU.

**Attempt 1** (seed 0). *Prompt:* "A highly detailed portrait of Janus, the two-faced Roman god, depicted as a single head with two faces looking in opposite directions. The left face is an old, wise man with a long white beard and deep wrinkles, painted in cool silver and blue tones; the right face is a young, beardless man with smooth skin, painted in warm golden and orange tones. Classical marble sculpture style, dramatic lighting, dark background, masterpiece, 8k."

![Janus attempt 1](images/fig5_t2i_janus_attempt1.png)

*Figure 5. Attempt 1, both samples (384×384, unedited).*

**Attempt 2** (seed 1, rephrased toward classical "double-herm" iconography). *Prompt:* "Janus, the ancient Roman god of beginnings and endings, a marble bust with two faces joined at the back of one head, facing left and right in profile. One face is a wrinkled elderly man with a flowing white beard, lit in cold blue moonlight; the other face is a youthful smooth-skinned man, lit in warm golden sunlight. Symmetrical composition, highly detailed, dramatic contrast, dark background."

![Janus attempt 2](images/fig6_t2i_janus_attempt2.png)

*Figure 6. Attempt 2, both samples.*

**Visual inspection** (by us, image by image):

| Sample | Two faces present? | Old vs. young contrast | Cool vs. warm palette contrast | Faces looking in opposite directions / one head? | Style and artefacts |
|---|---|---|---|---|---|
| attempt 1, #0 | ✗ one face only | ✗ only the old, bearded, deeply wrinkled face | ✓ split palette: the left half of the face is silver-blue, the right half orange-gold, as if the two colour schemes were applied to the two *halves* of one face | ✗ | Convincing marble or painted-sculpture texture, sharp hair and beard; orange "flame" artefacts at the right edge |
| attempt 1, #1 | ✓ two faces | ✓ large old, white-bearded, wrinkled face; small young, beardless, smooth-skinned face at lower right | ✓ old face cool blue-grey; young face warm flesh/orange tones | ✗ two separate heads side by side, not one two-faced head; the young face is cropped by the frame | Clean, little texture noise; mismatched scales |
| attempt 2, #0 | ✓ two profiles fused into one bust (shared hair mass) | ~ front face old and white-bearded; rear face less wrinkled but also bearded | ✓ front face cold blue/white, rear face warm gold | ~ one head ✓, but both profiles face the **same** way (right) instead of opposite | Fused hairline and a doubled nose/brow contour where the profiles overlap |
| attempt 2, #1 | ✓ two faces on one marble bust, including the bust's carved base | ~ front face old with a long white beard; rear face younger-looking but with a beard | ✓ front cold blue-white stone, rear golden-lit | ~ one bust ✓, but both face right (offset double profile) | The most sculptural and coherent sample; minor smearing of the rear face's ear |

**Match / mismatch with the showcase.**
- **Matched:**
  - The model recognises the concept "two-faced Janus" well enough to produce a *multi-face* image in 3 of 4 samples.
  - It reliably honours the **colour semantics** in 4 of 4 samples: the "cool silver/blue" vs. "warm golden/orange" split appears every time and is assigned to the right face (the old face is cool).
  - It renders a convincing classical-sculpture style.
  - Attempt 2 produces the characteristic *one bust, two faces* structure of Roman Janus sculptures.
- **Not matched:**
  - In no sample do the faces point in **opposite** directions; they form a parallel or offset double profile.
  - The "young, beardless" attribute is applied only once (attempt 1 #1). Elsewhere the second face inherits the beard of the first.

  This is the familiar *attribute-binding* failure of small T2I models. Attributes such as "old" and "beard" leak from one entity to the other, and spatial relations such as "opposite directions" are weakly controlled. The follow-up Janus-Pro report (bundled with the official code) likewise describes the original Janus as less stable and less detailed than its successor, at a resolution of only 384×384.

### 4.2 Closing the loop: the understanding pathway inspects its own generations
Because both tasks live in one model, the *understanding* pathway (SigLIP) can evaluate images that the *generation* pathway (VQ) produced. We asked: `How many faces does this sculpture have? Describe each face (age, beard, colour of lighting) and the direction it is looking.` (greedy, verbatim):

**`janus_v2_s1_1.png`** (670 s):

```text
The sculpture has two faces. The face on the left is older, with a long, wavy beard and a more mature, weathered appearance. The face on the right is younger, with a shorter beard and a more defined, clean-shaven look. The older face is looking to the left, while the younger face is looking forward.
```

**`janus_s0_1.png`** (848 s):

```text
The sculpture has two faces. The first face is an older man with a long, wavy beard and a stern expression. The second face is a younger man with a more neutral expression. The older man's face is on the left side of the image, and the younger man's face is on the right side. The older man's face is looking slightly to the left, while the younger man's face is looking straight ahead.
```

For both images, the understanding pathway correctly reports **two faces** and an **older vs. younger** pairing. Its other claims are less reliable:
- **Wrong:** the gaze directions. In `janus_v2_s1_1`, both profiles face right.
- **Self-contradictory:** "a shorter beard and a more defined, clean-shaven look".
- **Ignored:** the lighting colours that the question asked about.

The two images were rendered by the VQ pathway and read through an independent encoder. The pathways therefore agree on the coarse semantics the prompt asked for. They are not mere code-book artefacts of the generator.

CLIP_PLACEHOLDER

EXTRA_T2I_PLACEHOLDER
