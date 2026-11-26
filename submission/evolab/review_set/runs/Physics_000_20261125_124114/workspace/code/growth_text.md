### 7.1 Overview

![Fig. 6](images/fig6_growth_coverage_traj.png)

**Figure 6.** Coverage (recall) of every candidate template against the number of deposited atoms, for each run (line style = run). Panels: E2 (seed (1,2)), E2b (seed (1,3)), E3a (seed (0,1), δ=0.22) and E3b (seed (1,1), δ=0.09). Recall overlaps between templates are substantial for T ≥ 9 (see §2.6), so only the end points are used for assignment, via F1.

**Table 9.** Growth outcomes per experiment. "Target completed" means (0,2) recall ≥ 0.9 for E1, (1,1) ≥ 0.8 for E3a, and (1,2) or (2,1) ≥ 0.75 for E3b. Classes are F1-based.

{{TABLE_GROWTH}}

### 7.2 E1 — conservative MC→MC step at zero/small mismatch, and its kinetics

![Fig. 5](images/fig5_growth_E1_conservative.png)

**Figure 5.** E1: growth of 42 atoms on an A₁₃ icosahedron. (a) Fraction of runs completing the Mackay (0,2) shell (recall ≥ 0.9), with Wilson 95% CI, for each kinetic condition. (b) Per-run recall of (0,2) and (1,1). (c) Annealed energy relative to the ideal Mackay 55-atom cluster (same truncated LJ).

At δ = 0 the outcome is controlled by the deposition interval:

| τ_dep | Mackay (0,2) completed |
|---|---|
| 15 τ | 0/8 as grown (CI 0–0.32), 2/8 after anneal |
| 60 τ | 8/10 (CI 0.49–0.94) |
| 240 τ | 6/6 (CI 0.61–1.0) |

The mean energy above the ideal Mackay cluster falls correspondingly: 10.4 → 6.2 → 4.3 ε after anneal.

The runs that do not complete the Mackay shell have anti-Mackay-like overlayers. Their best template is (1,1), with F1 < 0.6. The trajectories (Fig. 7, left) show that **every** run (38/38) first builds an anti-Mackay (BG, class 1) island at 15–30 deposited atoms, and that most runs later convert it into the Mackay shell. This is the known anti-Mackay → Mackay overlayer transition of LJ growth.

So **the conservative step is the thermodynamically selected one at δ ≈ 0**, and its failures are kinetic trapping. They are suppressed by slow deposition and enhanced by fast deposition. Lowering T* from 0.22 to 0.18 at τ_dep = 60 reduces completion to 4/6, consistent with reduced mobility. At δ = 0.03 (just below the MC→MC optimum of 0.051) completion is 7/8 (CI 0.53–0.98).

The A₁₃ seed is a single-species core here. At T* = 0.22 the central atom is occasionally exchanged, so identification uses the atom nearest the centre of mass and a radial layer definition.

### 7.3 E3 — mismatch-driven steps from MC and BG seeds; spontaneous hand selection

- **E3a, δ = 0.22.** On A₁₃ (formula δ* of (0,1)→(1,1) = 0.265; LJ window 0.08–0.36), all 4 runs grow the anti-Mackay **(1,1)** shell (BG). Recall is 0.72–0.88, and the best F1 is (1,1) in 4/4. This is the class-raising step MC→BG.
- **E3b, δ = 0.09.** On the achiral BG seed A₁₃@B₃₂ (LJ window of (1,1)→(1,2): 0–0.16), all 3 runs grow a **single-handed (1,2)-type shell** (Ch1), with recall 0.75–0.92. The hand was +1, +1 and −1 (chirality indices +0.84, +0.50 and −0.74). Because (1,2) and (2,1) are degenerate continuations of an achiral seed, the hand is chosen by spontaneous symmetry breaking during island growth. Once chosen, it propagates over the whole shell; no racemic shells were found.

### 7.4 E2 — chiral (1,2) seed: class-raising Ch1→BG(2,2) transition

On the 117-atom seed A₁₃@B₃₂@C₇₂ (hand +1), the LJ stability windows are:
- (1,3): 0–0.02 (conservative);
- (2,2): 0.04–0.12 (class-raising, class index 1→2);
- (0,3): ≥ 0.14 (reverse).

Results:
- **δ = 0.08**: 2/2 runs grow the **(2,2)** shell, with annealed recall 0.96 and 0.88 and F1 0.92 and 0.84. This is a chirality-class transition, Ch1 → BG(2,2): the shell becomes achiral and its class index rises from 1 to 2.
- **δ = 0.03**: 2/2 runs also grow (2,2), not the conservative (1,3).

δ = 0.03 lies within 0.01 of δ*((1,2)→(1,3)) = 0.032, but just outside the Δ-based conservative window (0–0.02). This is not, however, why (2,2) grew; the δ = 0 runs and the equal-N test below settle that.

E2_DELTA0_TEXT

The coverage trajectories (Fig. 6) show that the partial layer at 20–90 deposited atoms is best matched by the (0,3) template. Because of template overlap, this early assignment is not conclusive. The (2,2) recall grows steadily and dominates beyond about 100 deposited atoms.

### 7.5 E2b — chiral (1,3) seed: the Ch1→Ch2 transition and hand inheritance

On the 249-atom seed (outer shell (1,3), Ch1, hand +1), the LJ stability windows are:
- (1,4): 0–0.02 (conservative);
- (2,3): 0.04–0.08 (Ch1→Ch2);
- (0,4): ≥ 0.10 (reverse).

E2B_TEXT

### 7.6 Sequential species (dataset alternating sequence)

E3C_TEXT

### 7.7 Benchmark-style summaries

![Fig. 7](images/fig7_class_vs_deposition.png)

**Figure 7.** (i) Dominant shell class against number of deposited atoms, in each MD run. Left: small-δ runs (conservative step predicted), where chiral seeds nevertheless jump to class 2. Middle: class-changing δ. Right: sequential species (E3c) and the BG seed (E3b). Runs are offset vertically for visibility. Colour = class label; y = class index; the seed class is plotted at n = 0. (i′) Lattice-model class frequency per shell step for the alkali sequence, the "mixed" sequence and constant δ = 0.08.

![Fig. 8](images/fig8_running_mismatch.png)

**Figure 8.** (ii) Running mean mismatch of the deposited species relative to the outer seed species, for all MD runs. (ii′) Shell-wise running mean mismatch for alkali design sequences, compared with the dataset's `growth_results` (dotted, squares coloured by the listed class).

![Fig. 9](images/fig9_step_statistics.png)

**Figure 9.** (iii) Step-type statistics from the lattice model: 500 paths × 5 shells with p_r = 0.1, for the thermodynamic and softened λ, alongside the dataset's `path_selection_stats`. (iii′) MD step outcomes per experiment (Wilson 95% CI), compared with three single-step predictions from the same seed and δ: elastic model with thermodynamic λ (bar), elastic model with λ/30 (×), and the T = 0 LJ stability-map winner (magenta diamond).

**Table 10.** Lattice-model statistics.

{{TABLE_E4}}

**Table 11.** MD outcomes against the single-step models.

{{TABLE_MDMODEL}}

**Table 12.** Per-run final identification (annealed; F1-based).

{{TABLE_RUNS}}

![Fig. 10](images/fig10_snapshots.png)

**Figure 10.** (iv) Annealed grown clusters viewed down a five-fold axis (front hemisphere only), coloured by species diameter. Titles give the best-matching shell, its recall, and the chirality index.
