# Trajectory-preserving selection of dynamic features from single-cell readouts: a DELVE-style two-step method on RPE 4i cell-cycle data and a simulated benchmark

## Abstract
We re-implemented a DELVE-style unsupervised feature selector and evaluated it on two kinds of data:

* **Real data.** 241 iterative-indirect-immunofluorescence (4i) imaging features from 2,759 retinal pigment epithelial (RPE) cells, sampled continuously through the cell cycle.
* **Simulated data.** A new count simulator with linear, bifurcating and tree trajectories, ground-truth differentially expressed (DE) genes, and controlled sweeps of SNR, library size and dropout.

The selector works in two steps:

1. Find modules of dynamically expressed features on a kernel-herding sketch of cell neighbourhoods, and keep only those that pass a permutation test.
2. Rank all features by their Laplacian score on a graph built from the seed modules.

We froze the method configuration and its label-free variants before computing any label-based metric.

**The literal, faithful version fails on RPE.** Its seed modules are dominated by redundant cell/cyto/ring copies of signalling proteins. The first PC of those seeds is a global-intensity axis (|r| = 0.91 with mean intensity). At k = 30 the panel reaches only 0.37 phase accuracy and a DPT–age Spearman ρ of −0.06.

**Two pre-registered, label-free variants recover the cell-cycle trajectory.** Both change only the step-1 graph:

* **`+dedup`** builds the graph on one representative feature per group of features with |r| > 0.9. At k = 30 it gives:
  * kNN phase accuracy 0.88;
  * k-means NMI 0.51;
  * DPT ρ 0.85 on all cells, including G0;
  * DPT ρ 0.87 on cycling cells.

  This is significantly better than the best label-free baseline, HVG dispersion (paired Δρ = +0.036 [0.020, 0.053]). It matches the supervised age oracle and mixes batches far better than the full feature set.
* **`+whiten`** gives DPT ρ 0.82.

The PHATE embedding of the `+dedup` top-30 panel shows a single continuous G1→S→G2 trajectory with G0 branching off. The embedding of all 241 features does not.

**In simulation, the DELVE-style method is not better than simpler graph-smoothness scores.** On clean data it is equivalent to the Laplacian score, MCFS and Moran's I. It degrades *faster* than they do when the DE signal is weak, because its seed step locks onto a strong trajectory-independent confounder programme. We report this negative result as it came out and discuss when the two-step design helps (RPE: redundancy plus global-intensity confounding) and when it hurts (weak signal plus a dominant coherent confounder).

---

## 1. Data overview

The input file `data/adata_RPE.h5ad` was read with h5py and its contents checked by assertion:

* **Size.** 2,759 cells × 241 preprocessed features.
* **Features.** Median, mean-edge, std and integrated intensities of 48 proteins plus DNA, measured in the cell, cytoplasm (cyto), nucleus (nuc) and perinuclear ring, plus nuclear area.
* **Phase labels.** G0 402, G1 1,128, S 891, G2 338. There is no M phase.
* **State labels.** Arrested 402, cycling 2,174, and "nan" 183.
* **Age.** `annotated_age` ranges from 0 to 25.07 h. Both G0 and G1 cells span the full age range (Fig. 1a).
* **Batches.** Two.

Labels were used **only** inside `evaluate()` and to orient the pseudotime root.

{{T_overview}}

Three properties of the data shaped the analysis (Fig. 1):

1. **Strong redundancy.** Average-linkage clustering on 1 − |r| with a cut at |r| > 0.9 collapses the 241 features into 154 groups: 104 singletons, 13 pairs and 37 groups of three or more. Most groups are cell/cyto/ring copies of one protein.
2. **A dominant global-intensity factor that has nothing to do with the trajectory.** PC1 explains 24.8 % of the variance. It correlates at r = 0.985 with mean z-scored intensity and at ρ = 0.001 with age. The age signal sits in later PCs (Fig. 1b; `outputs/rpe_pc_age.csv`).
3. **A weak but multivariate batch effect.** The largest single-feature association with batch is |ρ| = 0.15. Even so, a 5-NN classifier on all features separates the batches with balanced accuracy 0.83.

![Data overview](images/fig1_data_overview.png)

**Figure 1.** (a) Age distribution by phase. (b) Variance fraction per PC (bars) and |ρ(PC, age)| (lines). (c) Clustered |correlation| matrix of the 241 features. (d) PC1 (global intensity) vs PC4, coloured by phase.

## 2. Methods

### 2.1 DELVE-style two-step selection (re-implementation)
The DELVE paper was **not** among the provided `related_work/` files (Scanpy, Monocle 3, and two unrelated papers). The method below was therefore reconstructed from the general description of DELVE (dynamic selection of locally covarying features) and is not a verified port of it.

All steps are label-free (`code/core.py::delve`):

1. **Graph.** Z-score the features, take PCA (20 PCs) and build a kNN graph (k = 10).
2. **Sketch.** Use kernel herding to pick 500 anchor cells that represent the cell distribution. The kernel is Gaussian, approximated with 1,000 random Fourier features and a median-heuristic bandwidth.
3. **Neighbourhood means.** For each anchor, average the z-scored features over the anchor and its 10 neighbours. This gives a matrix M (anchors × features).
4. **Dynamic score.** For each feature f, compute d_f = Var(M_f)/Var(X_f). For exchangeable noise this is about 1/(k+1) ≈ 0.09.
5. **Modules.** Cluster the features into √(p/2) modules with k-means on their standardised M-profiles. Repeat with 5 random seeds.
6. **Permutation test.** Keep a module if its mean dynamic score **and** its mean pairwise profile correlation (coherence) both exceed those of 500 random feature sets of the same size (p < 0.05). Features retained in at least 60 % of the seeds are the **seed features**.
7. **Ranking.** Build a kNN graph (k = 10, heat kernel) on the seed features and compute the Laplacian score (He et al., 2005) for **every** feature, including redundant copies. The ranking is ascending Laplacian score, and no features are appended at the end.

Variants were named and frozen in advance (`outputs/config_frozen.json`, frozen at 2026-11-25T16:32:22, before any call to `evaluate()`):

| variant | change (all label-free) |
|---|---|
| **DELVE-faithful** (primary) | steps 1–7 as written |
| +noise-filter | before module detection, iteratively drop features at or below the 99th percentile of a cell-permutation null of d_f, then rebuild the step-1 graph |
| +dedup | build the step-1 graph and modules on medoids of the \|r\| > 0.9 groups; the final ranking still covers all 241 features |
| +whiten | use unit-variance PCs for the step-1 kNN graph |

The simulation used 300 anchors, 300 permutations and 3 seeds to save compute.

### 2.2 Baselines (each returns a full ranking over all features)
* random
* all features (reference)
* variance
* HVG-style dispersion (var/mean)
* PCA loadings (max |loading|·√eigenvalue over 10 PCs)
* Laplacian score on the full-feature kNN graph
* **MCFS** (Cai et al., 2010): top-5 non-trivial eigenvectors of the normalised graph; each is regressed on all features by LARS-lasso to 30 non-zero coefficients (100 in simulation); score = max |coef|. Uses scikit-learn `lars_path`.
* **Hotspot-like Moran's I** on the kNN graph
* neighbourhood variance (a Geary-type ratio)
* cluster F-statistic (k-means with 10 clusters, ANOVA F)
* max |corr| with diffusion components 1–5
* a **supervised oracle**: |ρ(feature, age)| on RPE and |ρ(feature, true t)| in simulation. This is a reference only.

### 2.3 Evaluation
Each panel is z-scored and reduced to 10 PCs. Panels were evaluated at k ∈ {5, 10, 20, 30, 50}; **k = 30 is the headline**. The metrics are:

* **Phase accuracy.** 5-fold cross-validated 5-NN accuracy on phase, over all cells.
* **NMI.** k-means (4 clusters) vs phase, averaged over 5 seeds.
* **Pseudotime.** Diffusion pseudotime (DPT, Haghverdi et al. 2016, re-implemented: 15 diffusion components, adaptive Gaussian kNN kernel with k = 15). Spearman ρ with `annotated_age` is reported **twice**:
  * on all cells, G0 included, with DPT computed on the all-cell graph;
  * on cycling (non-G0) cells, with DPT recomputed on the cycling-only graph.

  The root is the medoid of the youngest 2 % of non-G0 cells. Labels are used only to orient the root.
* **Batch.** Balanced accuracy of a 5-NN batch classifier (0.5 = perfectly mixed) and the mixing entropy of 30-NN neighbourhoods (maximum 1 bit).
* **Uncertainty.** 500 cell bootstraps of the k = 30 DPT ρ, and paired bootstrap differences against the best label-free baseline.

**PHATE.** We used the `phate` package (v2.0.0; knn = 5, other settings default), which became installable during the run. A numpy re-implementation (α-decay kernel → von Neumann-entropy knee for t → potential distances → classical MDS → SMACOF) is kept in `core.py` as a fallback but was not used for the figures.

### 2.4 Simulation benchmark (`code/simulate.py`, `code/run_sim.py`)
We wrote our own numpy simulator (no splatter or dyngen):

* **Size.** 1,500 cells × 500 genes, of which 100 are DE.
* **Topologies.**
  * *linear*: three consecutive segments;
  * *bifurcating*: root → A, B;
  * *tree*: root → A, B, then B → B1, B2.
* **Cells.** Each cell draws a terminal lineage and a pseudotime t ~ U(0, 1). Its milestone label is the segment it falls in.
* **Gene means.** log μ = base + effect · profile(t, lineage), with base ~ N(0, 1).
  * Each DE gene is tied to a random segment.
  * Its profile is either a sigmoid switch inside that segment that persists downstream, or a Gaussian bump within the segment.
  * In both cases the profile is active only on lineages that pass through the segment.
  * Effect = U(1, 2.5) × ±1 × effect_scale.
* **Confounder.** 40 non-DE genes share a trajectory-independent latent factor (loading 0.8), standing in for cell-size or stress programmes. This is the "confounding variation" a selector should avoid.
* **Counts.** NB(lib_i · μ, θ = 5), with lognormal library sizes (σ = 0.4). Optional dropout zeroes each entry with probability logistic(−(log μ − m)). Preprocessing is total-count normalisation followed by log1p.
* **Noise sweeps** (one factor at a time; level 0 = clean):
  * DE effect scale 1, 0.6, 0.4, 0.25;
  * library-size scale 1, 0.3, 0.1, 0.03;
  * dropout midpoint m = none, 0, 1, 2.
* **Replicates.** 3 topologies × 10 settings × 2 replicates = 60 datasets.
* **Metrics** (top-100 panel):
  * precision@100 for true DE genes;
  * 5-NN accuracy on milestone labels;
  * k-means NMI vs milestones;
  * DPT ρ with the true t, rooted at the medoid of the 1 % earliest cells;
  * the fraction of confounder genes selected.

Zero fractions per setting:

{{T_zero_frac}}

## 3. Results on RPE

### 3.1 What the selector picks: seed diagnostics
**Table 1.** Label-free diagnostics of the seed sets (`outputs/rpe_seed_diagnostics.csv`).

{{T_diag}}

The faithful version's 62 seeds cover only 23 distinct |r| > 0.9 groups. 96.8 % of them belong to redundant groups, and none is nuclear. They are cell/cyto/ring triplets of p38, pp38, S6, ERK, YAP, GSK3b, p16, pH2AX, pCHK1 and pp53, and their first PC is the global-intensity axis (|r| = 0.91).

Redundant copies are almost perfectly correlated across neighbourhoods. They therefore form highly coherent, high-dynamic-score modules that win the permutation test, so redundancy inflates apparent dynamics. The resulting seed graph encodes cell brightness, not cell-cycle position.

`+dedup` seeds are 14 distinct, mostly nuclear (93 %) readouts:

* cyclin A, cyclin B1, CDK2, E2F1, Skp2;
* RB and pRB, pp21, pH2AX, cJun;
* PCNA texture (std), integrated DNA, nuclear area.

`+whiten` also moves toward nuclear cell-cycle readouts (54 % nuclear seeds). Fig. 2 summarises these diagnostics.

![Seed diagnostics](images/fig2_seed_diagnostics.png)

**Figure 2.** (a) Dynamic scores by compartment. Nearly all features lie far above the exchangeable-noise level 1/(k+1); this is why the noise filter kept 238 of 241 features and had almost no effect on RPE. (b) Seed redundancy, alignment of the seed PC1 with global intensity, and nuclear fraction. (c) Number of seeds vs number of distinct redundancy groups among them.

The `+dedup` top-30 panel (`outputs/rpe_top50_features.csv`) is, in order:

* cycA_nuc, Intg_DNA_nuc, pRB_nuc, pp21_nuc, Skp2_nuc, E2F1_nuc;
* the cycB1 ring/cyto/cell copies, nuclear area, pH2AX_nuc, CDK2_nuc;
* Std_PCNA_nuc, RB_nuc, cycB1_nuc, cJun_nuc, p21_nuc, cycD1_nuc;
* Cdh1_nuc, Cdt1_nuc, p27_nuc, among others.

This is a textbook cell-cycle panel: replication licensing (Cdt1), the G1/S switch (cycD1, pRB, E2F1, p21, p27), S-phase progression (PCNA texture, DNA content, cycA, CDK2, Skp2) and G2 (cycB1).

### 3.2 PHATE: all features vs selected panels
![PHATE](images/fig3_phate_panels.png)

**Figure 3.** PHATE (knn = 5) of all 241 features and of the top-30 panels of DELVE-faithful, DELVE +dedup, variance and Laplacian (full graph). Columns are coloured by phase, annotated age and batch.

* **All 241 features.** The embedding is a diffuse cloud. Phases overlap, and the main axis follows intensity rather than age.
* **DELVE-faithful and Laplacian top-30.** These are equally unstructured with respect to the cycle.
* **DELVE +dedup top-30.** The embedding resolves one continuous arc, G1 → S → G2, along which age increases monotonically. G0/arrested cells form a short side branch off the G1 end, and the two batches are largely intermixed along the arc.
* **Variance top-30.** It recovers the trend but is noisier, with a broad, diffuse S phase.

### 3.3 Quantitative comparison at k = 30
**Table 2.** RPE metrics at k = 30. "all_features" uses all 241 features. Batch metrics: lower balanced accuracy and higher entropy mean better mixing.

{{T_rpe_k30}}

**Table 3.** Bootstrap means and 95 % CIs of DPT–age ρ at k = 30.

{{T_boot}}

**Table 4.** Paired bootstrap difference in DPT ρ, each DELVE variant minus the best label-free baseline (dispersion). `p_le0` is the bootstrap fraction ≤ 0.

{{T_paired}}

Main observations:

* **The primary, faithful configuration fails on RPE.** ρ = −0.06 and accuracy 0.37, which is *below* random panels (ρ 0.15, accuracy 0.62). The noise-filter variant fails in the same way.
* **All single-step graph-smoothness baselines also fail at k = 30.** Laplacian ρ = 0.05, Moran's I 0.06, neighbourhood variance 0.05, MCFS 0.18, diffusion-corr −0.05, cluster-F −0.07, PCA loadings −0.03. The full-feature kNN graph they share is dominated by the intensity and redundancy structure. The Laplacian score and neighbourhood variance give identical rankings here; the two quantities are almost equivalent on a kNN graph.
* **`+dedup` is best overall.** Accuracy 0.882, NMI 0.511, ρ 0.852 on all cells and 0.874 on cycling cells.
  * It beats dispersion, the best label-free baseline, by Δρ = +0.036 [0.020, 0.053].
  * It matches or exceeds the supervised oracle (ρ 0.817 all / 0.872 cycling; accuracy 0.816).
  * It mixes batches much better than all features: batch balanced accuracy 0.590 vs 0.718, and entropy 0.909. Only dispersion is comparable (0.573, 0.909).
* **`+whiten` is statistically tied with dispersion** (Δρ = +0.003 [−0.014, 0.021]).
* **G0 matters little.** Excluding G0 raises ρ only slightly for the good panels (0.85 → 0.87). Including the arrested cells does not break the ordering, because they attach near the G1 root.

### 3.4 Dependence on panel size
![k-curves](images/fig4_rpe_kcurves.png)

**Figure 4.** Accuracy, NMI, DPT ρ (all cells and cycling cells) and batch balanced accuracy as a function of k. DELVE variants are thick lines, baselines dashed, and the black dotted line is all 241 features.

![k=30 bars](images/fig5_rpe_k30_bars.png)

**Figure 5.** k = 30 summary. Red bars are DELVE variants, grey is the supervised oracle, and the dotted line is all features.

**Table 5.** DPT ρ (all cells) for every k. We report all k and did not cherry-pick.

{{T_rpe_kcurve_dpt_rho_all}}

Several curves are not monotone in k:

* `+dedup` is weak at k = 5 (ρ 0.26). Its top-5 holds only G1/S and S readouts (cycA, DNA, pRB, pp21, Skp2), which do not resolve G2 or G0; it is stable at 0.81–0.85 for k = 10–50.
* `+whiten` dips at k = 10 (0.39).
* Dispersion reaches 0.82 only at k = 30.
* Random panels look deceptively good at small k (ρ ≈ 0.5 at k = 5–10). This comes from a single random draw that happened to contain nuclear cell-cycle features, which is a reminder that small-k results have high variance.

### 3.5 Stability
**Table 6.** Jaccard overlap of the top-30 panel between the full data and three 80 % cell subsamples.

{{T_stab}}

* `+dedup` (0.80 ± 0.19) and `+whiten` (0.80 ± 0.08) are moderately stable.
* The faithful variant is not (0.42 ± 0.22).
* Univariate scores such as variance (0.98) and dispersion (0.92) are the most stable, as expected.
* The two-step methods inherit variability from module detection on a sketch.

### 3.6 Robustness to feature corruption on RPE (optional analysis)
Gaussian noise was added to every feature (sd = 0.5, 1 or 2 × the feature sd), or random entries were set to 0 (20 % or 50 %). Selection was then re-run and the k = 30 panel evaluated.

**Table 7.** DPT ρ (all cells) at k = 30 under feature corruption (`outputs/rpe_noise.csv`).

{{T_rpe_noise_dpt}}

![RPE noise](images/fig9_rpe_noise.png)

**Figure 9.** DPT ρ and phase accuracy at k = 30 under Gaussian noise (left) and random zeroing (right). Level 0 is uncorrupted data.

* **`+whiten` is the most robust variant.**
  * Under Gaussian noise: ρ 0.79, 0.61 and 0.39 at the three levels, against 0.79, 0.67 and 0.19 for dispersion.
  * Under 20 % zeroing: 0.74 vs 0.49 for dispersion.
* **`+dedup` collapses to the faithful result under any corruption.** Added noise destroys the |r| > 0.9 correlations, so no groups merge, the dedup step does nothing, and the global-intensity failure returns. A fixed correlation threshold is therefore a brittle way to remove redundancy; a noise-aware criterion (e.g. a rank-based or partial-correlation test) would be needed.

## 4. Simulation benchmark

### 4.1 Clean data: sanity gate and topology comparison
**Table 8.** Precision@100 for true DE genes on clean data (mean of 2 replicates).

{{T_sim_clean_precision}}

**Table 9.** DPT ρ with the true pseudotime on clean data.

{{T_sim_clean_dpt_rho}}

**Table 10.** Milestone kNN accuracy on clean data.

{{T_sim_clean_acc}}

**Table 11.** k-means NMI vs milestones on clean data.

{{T_sim_clean_nmi}}

**Table 12.** Fraction of the top-100 that are confounder genes (40 confounders exist; 0.40 means all of them were selected).

{{T_sim_clean_conf}}

![Sim clean](images/fig6_sim_clean.png)

**Figure 6.** Clean-simulation metrics for each method and topology.

**Sanity gate: passed.** On clean data, the Laplacian score (0.60–0.63), MCFS (0.48–0.66) and Moran's I (0.60–0.63) reach precision@100 three times the random level (about 0.2). The baseline implementations therefore behave as intended.

DELVE-faithful is equivalent to these baselines:

* precision 0.58–0.61;
* DPT ρ 0.91 (bifurcating), 0.98 (linear) and 0.78 (tree);
* milestone accuracy 0.92–0.95.

The tree is the hardest topology for every method.

Precision is capped near 0.6 for every unsupervised graph method, and the cause is identifiable. The 40 confounder genes vary smoothly on the cell graph because they share a latent factor, so every graph-smoothness method selects essentially all of them (confounder fraction about 0.40). The DE genes they displace are the weakest ones (low base expression or small transient bumps). The supervised oracle selects almost no confounders (0.00–0.03) and reaches 0.77–0.90.

Unsupervised smoothness alone cannot tell a trajectory programme from a coherent non-trajectory programme. This holds for the two-step DELVE design as well.

`+noise-filter` hurts on branching topologies: DPT ρ 0.51 (bifurcating) and 0.37 (tree), against 0.91 and 0.78 for the faithful variant. Iterating the filter and rebuilding the graph can lock onto a subset of programmes.

### 4.2 Noise sweeps
![Noise sweeps](images/fig7_sim_noise_sweeps.png)

**Figure 7.** Precision@100, milestone accuracy, NMI and DPT ρ against SNR (DE effect scale), library-size scale and dropout midpoint. Each point is the mean over 3 topologies × 2 replicates.

![Per-topology DPT](images/fig8_sim_topology_dpt.png)

**Figure 8.** DPT ρ against each noise factor, per topology.

**Table 13.** Precision@100 across sweeps; columns are (factor, level), and level 0 is clean.

{{T_sweep_precision}}

**Table 14.** DPT ρ across sweeps.

{{T_sweep_dpt_rho}}

**Table 15.** Loss in DPT ρ from clean to the hardest level of each factor. Smaller means more robust.

{{T_sweep_drop_dpt}}

Findings:

* **Low SNR.** DELVE-faithful degrades *faster* than single-step graph scores.
  * At effect scale 0.6, precision is 0.48 vs 0.58 for Laplacian and Moran's I.
  * At 0.4, it collapses to 0.12 (below random) while Laplacian keeps 0.45 and diffusion-corr 0.51.

  The mechanism is visible in the confounder fraction. As the DE signal weakens, the most coherent and most dynamic module left is the confounder module, whose strength does not depend on the effect scale. The permutation test then keeps essentially only that module as seeds, and the seed graph encodes the confounder. All 40 confounders are selected, and the rest of the panel is noise.

  This is the same failure as on RPE. **The two-step design amplifies whichever coherent programme dominates the sketch.** When that programme is the trajectory, the method is excellent; when it is a confounder, the method is worse than one-step smoothness scores.
* **Library size.** All graph methods degrade together. At scale 0.03, about 95 % of entries are zero and every method falls to DPT ρ ≤ 0.17. DELVE stays within a few points of the Laplacian score and Moran's I at each level (precision 0.55, 0.45 and 0.25, vs 0.60, 0.48 and 0.27 for Laplacian). Diffusion-corr and cluster-F hold up best on precision.
* **Dropout.** Graph-based methods are fairly robust: precision stays ≥ 0.46 at m = 2, with 87 % zeros. The exceptions are neighbourhood variance (0.29) and the univariate variance (0.28) and dispersion (0.38) scores. DELVE-faithful (0.49) sits in the pack.
  * `+noise-filter` was the most robust variant here: precision 0.53 and DPT ρ 0.74 at m = 2, vs 0.66 for DELVE-faithful and 0.63 for Laplacian. Dropout pushes the confounder module below the filter threshold (its confounder fraction falls to 0.05).
  * Among graph methods, neighbourhood variance degrades most (precision 0.29, DPT ρ 0.40).
* **Overall.** Pooled over the three noise sources, **no method dominates**, and the expected pattern ("DELVE degrades least") was **not** observed in our simulator.

## 5. Ablations and leakage controls
* **Variants.** The comparison of faithful, `+noise-filter`, `+dedup` and `+whiten` (Tables 1–7) is the ablation: the step-1 graph decides success or failure. The step-2 Laplacian ranking is identical across variants and ranks every feature.
* **Leakage audit.** `outputs/leakage_grep.txt` confirms that the selection code (`core.py::delve`, `baseline_scores`, `simulate.py`) never reads age, phase, batch, true t or DE status. The only exceptions are the explicitly labelled oracle and the DPT root orientation inside `evaluate()`.
* **Freeze.** The configuration was frozen at 16:32:22, before any label metric (`outputs/config_frozen.json`). No method was changed after labels were inspected; all four variants are reported.
* **Implementation changes after the freeze.** Two were made, neither of them a method change:
  * scikit-learn became available mid-run, so MCFS on RPE was recomputed with true LARS instead of the OMP fallback. All reported MCFS numbers, RPE and simulation, use LARS.
  * The simulation was restarted once for the same reason.
* **Reproducibility.** Every table in this report was generated by `code/make_tables.py` from the CSVs in `outputs/`.

## 6. Discussion

**When does trajectory-preserving selection help?** On the RPE 4i data, using all features hides the cell cycle. PHATE and DPT on 241 features give ρ ≈ 0.10, because a brightness axis (PC1) and redundant compartment copies dominate the neighbourhood graph. Selecting a few dozen nuclear cell-cycle readouts raises DPT ρ from 0.10 to 0.85. It also more than doubles NMI relative to variance selection, and it reduces batch separability. So the right 30 features not only preserve the continuous trajectory but also reduce confounding variation.

The DELVE-style idea of seeding the selection on dynamically co-varying modules achieves this, but **only if the step-1 graph is not itself confounded**. Our pre-registered `+dedup` and `+whiten` variants supply that condition. The literal version fails in the same way as every single-step graph method built on the confounded graph.

**Why the faithful method fails, and why this is general.** The dynamic score and module coherence reward features that co-vary across neighbourhoods. Redundant copies, or any shared latent programme (cell intensity in imaging, or library size, stress or cell-cycle genes in scRNA-seq), create exactly such co-variation. In the simulation, a confounder programme takes over the seed set as soon as the trajectory signal weakens.

Practical recommendations:

1. Collapse near-duplicate features before module detection, or use whitened or decorrelated graphs. Thresholds should be robust to measurement noise.
2. Report seed-set diagnostics, such as compartment, redundancy and alignment with global intensity or library size, as a routine label-free quality check.
3. Consider regressing out known nuisance axes before step 1.

**Relevance to neuroscience.** The task framing covers neural lineage progression, glial activation and neurodegeneration-related state transitions. These settings combine continuous trajectories with strong nuisance programmes: activity-dependent immediate-early genes, dissociation stress, ambient RNA and cell size. Our results suggest that DELVE-style seeding will be most useful when the trajectory programme dominates neighbourhood co-variation, and risky otherwise. Any panel should be checked against known nuisance modules. These are extrapolations; we did not analyse neural data.

## 7. Limitations
* **Method reconstruction.** The DELVE paper was not available, so module construction, the permutation-test statistic and the consistency rule are our reconstruction. Results for the official implementation may differ.
* **Baselines and tools are re-implementations.** MCFS uses scikit-learn's LARS; Hotspot is approximated by Moran's I with no count model; DPT is re-implemented. PHATE is the official package with default parameters except knn = 5, and its SGD-MDS gave convergence warnings.
* **One real dataset.** The cell-cycle "trajectory" is cyclic and partly linear, and the labels (age, phase) are themselves derived from imaging features, so they are not fully independent ground truth.
* **Simulator.** It is simple: independent genes given log μ, a single confounder module, and no batch effects. Only 2 replicates per setting were run, and there are no error bars on the sweep curves. The confounder strength was not swept, and it strongly shapes the precision ceiling and DELVE's low-SNR collapse.
* **Small-k random baseline.** It is a single draw and is noisy.
* **Neural extrapolation.** The neural relevance in §6 is extrapolated from a retinal-epithelium cell-cycle dataset and from simulation, not demonstrated on neural data.

## 8. Files
* **Code** (`code/`):
  * `core.py`: methods, baselines, DPT, PHATE fallback and metrics;
  * `simulate.py` and `run_sim.py`;
  * `rpe_select.py`, `rpe_eval.py`, `rpe_phate.py`, `rpe_stability.py`, `rpe_noise.py`, `rpe_diagnose.py`, `rpe_overview.py`;
  * `fig_*.py`, `make_tables.py`, `build_report.py`.
* **Outputs** (`outputs/`): the frozen configuration, rankings (`rpe_rankings.npz`), top-50 feature lists, all metric CSVs, bootstrap samples, simulation results, `tables.md` and the leakage grep.
