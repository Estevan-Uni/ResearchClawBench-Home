**Conservative test at δ = 0.** At δ = 0, the (1,3) shell is the lowest-Δ continuation at T = 0. Even so, both runs grew **(2,2)**: annealed recall 0.93 and 0.82, F1 0.89 and 0.79.

**Equal-N energy test.** E2 deposits exactly 132 atoms, the size of the conservative (1,3) shell, so grown and ideal clusters can be compared at the same atom count (`outputs/equalN_energy.json`). The annealed grown clusters (a (2,2) layer plus 10 adatoms) lie *above* the ideal seed+(1,3) cluster by:

| δ | Energy above ideal seed+(1,3) |
|---|---|
| 0 | 36–39 ε |
| 0.03 | 20–28 ε |
| 0.08 | 7–11 ε *below* (grown is lower) |

So the conservative shell is the T = 0 ground state at δ ≤ 0.03, and the class-raising (2,2) shell is the ground state at δ = 0.08.

**Thermal stability.** The ideal seed+(1,3) cluster, run for 200 τ at T* = 0.22 and then cooled, returns to the ideal structure: identical energy, (1,3) recall 1.0 (`outputs/thermal_stability.json`). The conservative structure is therefore thermally stable. Its absence in growth is a **growth-pathway (kinetic) effect**, not an equilibrium one.

**Slow deposition.** SLOW_E2_TEXT
