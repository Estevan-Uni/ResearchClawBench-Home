E3c follows the dataset's `deposition_sequences` entry "Ag13 + Cu": Cu₂₀Ag₁₀Cu₂₀ deposited on an Ag₁₃ seed. The Cu/Ag diameter ratio is 1.28/1.44 = 0.889 (δ = −0.111), with equal-ε LJ, and 3 runs were made.

- No well-defined outer shell forms on the Ag₁₃ frame: the best F1 is (1,1) at 0.49–0.64, which is unresolved.
- The composition analysis (`outputs/E3c_composition.json`) shows why. In all three runs the smaller Cu atoms migrate **into** the core:
  - the central atom becomes Cu;
  - 8–11 of the 13 innermost sites are Cu;
  - the original Ag₁₃ seed atoms are displaced outward (mean distance of Cu from the centre 1.44–1.50, of deposited Ag 1.69–1.74).
- Size alone therefore restores the positive outward mismatch that the lattice theory requires.
- The running mismatch (Fig. 8, E3c curve) is negative throughout, so this sequence is not a valid lattice path. The dataset's "Ag13@Cu45" is likewise not a valid single-shell cluster (45 ≠ 10T+2).
