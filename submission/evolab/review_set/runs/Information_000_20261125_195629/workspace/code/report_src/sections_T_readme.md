### 4.4 A second showcase prompt (official README example)
To check that the Janus result is not specific to one subject, we also ran the T2I prompt that ships with the official `generation_inference.py`: "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue." Settings were seed 0, CFG 5, n = 2, and the batch took 34 min.

![README showcase](images/fig9_t2i_readme_showcase.png)

*Figure 9. Official README prompt, both samples.*

**Inspection.**
- **Both landmarks:** present in both samples. The Eiffel Tower lattice and the Opera House's white shell roofs are recognisable.
- **Juxtaposition:** "sitting next to" is respected.
- **Sky:** in both images the sky is a van-Gogh-like field of blue swirls and radial streaks with yellow star bursts, matching "roiling energy, exploding yellow stars, radiating swirls of blue".
- **Artefacts:** in #0 the lower Eiffel structure fuses with the Opera House arches into a hybrid bridge-like form. In #1 the Opera House is reduced to a small fragment at the lower left, and the tower's lattice is irregular.

This is the same pattern as for Janus: global semantics and style are honoured, and the fine object structure and the boundaries between entities are where errors appear.
