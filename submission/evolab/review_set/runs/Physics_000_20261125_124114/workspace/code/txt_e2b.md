| δ | Runs | Outcome | Annealed F1 / recall | Chirality index χ₂₃ |
|---|---|---|---|---|
| 0.06 (window centre) | 2 | **(2,3), Ch1→Ch2** in both | F1 0.88 / 0.88 (s0) | +0.77 (s0) |
| 0.02 (window edge) | 1 | **(2,3)** rather than conservative (1,4) | F1 0.81 / 0.85 | +0.52 |
| 0.08 (upper edge) | 1 | **(0,4)** Mackay, reverse step Ch1→MC | F1 0.88 / 0.96 | — |
| 0.00 | 1 | E2B_DELTA0 | | |

Run s1 at δ = 0.06 gave E2B_S1_RESULT.

**Hand inheritance.** In every run that produced (2,3), the chirality index is positive. The grown Ch2 shell has the **same hand as the seed**, and its mirror image (3,2) never formed. This is the lattice prediction: (1,3)→(2,3) is a NN step, while (1,3)→(3,2) is not.

**Equal-N test at δ = 0.02.** 212 atoms were deposited, the size of (1,4). The grown (2,3)+20-adatom cluster lies 47 ε above the ideal seed+(1,4) cluster, so (2,3) is again the kinetically selected, not the lowest-energy, continuation. At δ = 0 the gap is 57 ε. The ideal seed+(1,4) cluster is thermally stable: after 200 τ at T* = 0.22 and cooling it returns to the ideal energy with (1,4) F1 = 1.0 (`outputs/thermal_stability.json`).

**Two-stage layer formation.** The δ = 0.06 trajectory (Fig. 6) shows growth in two stages. A Mackay-like (0,4) partial layer (recall up to 0.7) forms first, then converts collectively into (2,3) at about 130–150 deposited atoms. Early (0,3)/(0,4)-like partial layers occur on both chiral seeds. They suggest that islands initially occupy positions that continue the five-fold Mackay registry at the vertices, and reorganise only once the layer is dense.
