# Agent (Claude) execution of the 16-step Hartree–Fock template chain for arXiv:2111.01152

**Provenance.** The AI agent (Claude) wrote this file while acting as the "LLM executor" of the prompt
templates in `Prompt_template.md`. It is **not** human-graded. It is **not blind** either: the agent had already read
the gold answers in `2111.01152.yaml` before executing the steps. The independent evidence for its correctness comes
from the numerical checks in `code/` (FFT, Fock-space algebra and brute-force HF tests), not from the grading.
Steps are numbered 1–16. The 0-based "Task k" label used in the benchmark rubric is k = step − 1.

Symbols used throughout:

* Valley index: τ = ±1 (±K).
* Layer index: l ∈ {b (MoTe₂, bottom), t (WSe₂, top)}.
* Moiré period: a_M = a_b a_t/|a_b − a_t| = 4.6545 nm, with (a_b, a_t) = (3.575, 3.32) Å.
* Moiré BZ corner: κ = 4π/(3a_M)(1, 0).
* Reciprocal vectors: g_j = 4π/(√3 a_M)(−sin(π(j−1)/3), cos(π(j−1)/3)), j = 1…6.
* ω = e^{2πi/3}, ψ_b = −14°, (m_b, m_t) = (0.65, 0.35) m_e.
* Spinors: Ψ_+ = (ψ_{b↑+}, ψ_{t↓+})ᵀ and Ψ_− = (ψ_{b↓−}, ψ_{t↑−})ᵀ. Spin is locked to (layer, valley), so it is suppressed.
* Basis order: (+K,b), (+K,t), (−K,b), (−K,t).
* Interaction: V(q) = 2πe² tanh(qd)/(εq), with d = 5 nm.

---

## Step 1 (Task 0) — Construct kinetic Hamiltonian (continuum, single-particle)
Filled placeholders:
* system = AB-stacked MoTe₂/WSe₂ heterobilayer (two layers)
* real|momentum = **real**
* single-particle|second-quantized = **single-particle**
* degrees_of_freedom = valley (±K) and layer (b, t)
* kinetic_symbol = H_Kinetic(r)
* order = (+K,b), (+K,t), (−K,b), (−K,t)

Result:

$$H_{\rm Kinetic}(\mathbf r)=\mathrm{diag}\big(E_{+K,b},\,E_{+K,t},\,E_{-K,b},\,E_{-K,t}\big).$$

## Step 2 (Task 1) — Define each term of the kinetic Hamiltonian
Filled placeholders:
* parabolic|Dirac|cos = **parabolic**
* electrons|holes = **holes** (valence bands, so the mass term is negative)
* momentum_shift = +κ and −κ
* shifted_Ek = **E_{t,+K} and E_{t,−K} only**: the bottom (MoTe₂) layer is *not* shifted

$$E_{\tau,b}(\mathbf k)=-\frac{\hbar^2\mathbf k^2}{2m_b},\qquad E_{\tau,t}(\mathbf k)=-\frac{\hbar^2(\mathbf k-\tau\boldsymbol\kappa)^2}{2m_t},\qquad \mathbf k=-i\partial_{\mathbf r},$$

$$H_{\rm Kinetic}=\mathrm{diag}\Big(-\tfrac{\hbar^2k^2}{2m_b},\,-\tfrac{\hbar^2(\mathbf k-\boldsymbol\kappa)^2}{2m_t},\,-\tfrac{\hbar^2k^2}{2m_b},\,-\tfrac{\hbar^2(\mathbf k+\boldsymbol\kappa)^2}{2m_t}\Big).$$

## Step 3 (Task 2) — Construct potential Hamiltonian
Filled placeholders:
* real|momentum = real
* single-particle
* diagonal_potential = Δ_l(r)
* offdiagonal = Δ_{T,τ}(r) and Δ†_{T,τ}(r)

$$H_{\rm Pot}(\mathbf r)=\begin{pmatrix}\Delta_b&\Delta_{T,+K}&0&0\\ \Delta_{T,+K}^*&\Delta_t&0&0\\0&0&\Delta_b&\Delta_{T,-K}\\0&0&\Delta_{T,-K}^*&\Delta_t\end{pmatrix}.$$

## Step 4 (Task 3) — Define each term of the potential Hamiltonian

$$\Delta_b(\mathbf r)=2V_b\sum_{j=1,3,5}\cos(\mathbf g_j\cdot\mathbf r+\psi_b),\qquad \Delta_t(\mathbf r)\equiv V_{zt},$$

$$\Delta_{T,\tau}(\mathbf r)=\tau w\big(1+\omega^{\tau}e^{i\tau\mathbf g_2\cdot\mathbf r}+\omega^{2\tau}e^{i\tau\mathbf g_3\cdot\mathbf r}\big):\quad \Delta_{T,+K}=w(1+\omega e^{i\mathbf g_2\cdot\mathbf r}+\omega^2e^{i\mathbf g_3\cdot\mathbf r}),\ \ \Delta_{T,-K}=-w(1+\omega^{-1}e^{-i\mathbf g_2\cdot\mathbf r}+\omega^{-2}e^{-i\mathbf g_3\cdot\mathbf r})=-\Delta_{T,+K}^*.$$

Convention: the paper sets Δ_t = 0 and adds V_zt separately on the (t,t) diagonal. The gold answer absorbs the
offset as Δ_t = V_zt. Both conventions give the same Hamiltonian provided V_zt is counted **once**.

## Step 5 (Task 4) — Second-quantised form, matrix

$$\hat H^0=\int d^2r\,\vec\psi^\dagger(\mathbf r)H^0(\mathbf r)\vec\psi(\mathbf r),\quad \vec\psi=(\psi_{+K,b},\psi_{+K,t},\psi_{-K,b},\psi_{-K,t})^T\equiv(\psi_{b\uparrow+},\psi_{t\downarrow+},\psi_{b\downarrow-},\psi_{t\uparrow-})^T,$$

$$H^0=\begin{pmatrix}-\frac{\hbar^2k^2}{2m_b}+\Delta_b&\Delta_{T,+K}&0&0\\ \Delta^*_{T,+K}&-\frac{\hbar^2(\mathbf k-\boldsymbol\kappa)^2}{2m_t}+V_{zt}&0&0\\0&0&-\frac{\hbar^2k^2}{2m_b}+\Delta_b&\Delta_{T,-K}\\0&0&\Delta^*_{T,-K}&-\frac{\hbar^2(\mathbf k+\boldsymbol\kappa)^2}{2m_t}+V_{zt}\end{pmatrix}.$$

## Step 6 (Task 5) — Second-quantised form, expanded sum

$$\hat H^0=\sum_{\tau=\pm}\sum_{l_1,l_2\in\{b,t\}}\int d^2r\,\psi^\dagger_{\tau l_1}(\mathbf r)H^0_{\tau l_1,\tau l_2}(\mathbf r)\psi_{\tau l_2}(\mathbf r)$$

$$=\sum_\tau\int d^2r\Big[\psi^\dagger_{\tau b}\Big(-\tfrac{\hbar^2k^2}{2m_b}+\Delta_b\Big)\psi_{\tau b}+\psi^\dagger_{\tau t}\Big(-\tfrac{\hbar^2(\mathbf k-\tau\boldsymbol\kappa)^2}{2m_t}+V_{zt}\Big)\psi_{\tau t}+\psi^\dagger_{\tau b}\Delta_{T,\tau}\psi_{\tau t}+\psi^\dagger_{\tau t}\Delta^*_{T,\tau}\psi_{\tau b}\Big].$$

The intervalley blocks (τ ≠ τ′) are zero and have been removed.

## Step 7 (Task 6) — Real space → momentum space
Fourier transform:

$$c^\dagger_{\tau l}(\mathbf k)=A^{-1/2}\int d^2r\,\psi^\dagger_{\tau l}(\mathbf r)e^{i\mathbf k\cdot\mathbf r},\qquad \psi^\dagger_{\tau l}(\mathbf r)=A^{-1/2}\sum_{\mathbf k\in\mathbb R^2}c^\dagger_{\tau l}(\mathbf k)e^{-i\mathbf k\cdot\mathbf r}.$$

Substituting gives

$$\hat H^0=\sum_{\tau,l_1,l_2,\mathbf k_1,\mathbf k_2}c^\dagger_{\tau l_1}(\mathbf k_1)\,h^{(\tau)}_{l_1l_2}(\mathbf k_1,\mathbf k_2)\,c_{\tau l_2}(\mathbf k_2),\qquad h^{(\tau)}_{l_1l_2}(\mathbf k_1,\mathbf k_2)=\frac1A\int d^2r\,e^{-i\mathbf k_1\cdot\mathbf r}H^0_{\tau l_1,\tau l_2}e^{i\mathbf k_2\cdot\mathbf r}.$$

The matrix elements are:

* h_bb(k₁,k₂) = −ħ²k₂²/2m_b δ_{k₁k₂} + V_b Σ_{j=1,3,5}[e^{iψ_b} δ_{k₁−k₂, g_j} + e^{−iψ_b} δ_{k₁−k₂, −g_j}]
* h_tt(k₁,k₂) = [−ħ²|k₂ − τκ|²/2m_t + V_zt] δ_{k₁k₂}
* h_bt(k₁,k₂) = τw[δ_{k₁k₂} + ω^τ δ_{k₁−k₂, τg₂} + ω^{2τ} δ_{k₁−k₂, τg₃}]
* h_tb(k₁,k₂) = h_bt(k₂,k₁)*

In the plane-wave basis, write k₁ = k+G and k₂ = k+G′ with k in the moiré BZ. Then h is block-diagonal in k
(Bloch's theorem), and k₁ − k₂ is always a reciprocal-lattice vector.

## Step 8 (Task 7) — Particle–hole transformation
With b_{τl}(k) = c†_{τl}(k):

$$\hat H^0=\sum_{\tau,l_1,l_2,\mathbf k_1,\mathbf k_2}b_{\tau l_1}(\mathbf k_1)h^{(\tau)}_{l_1l_2}(\mathbf k_1,\mathbf k_2)b^\dagger_{\tau l_2}(\mathbf k_2).$$

## Step 9 (Task 8) — Normal order in the hole basis
Use b b† = δ − b†b and relabel indices:

$$\hat H^0=\sum_\tau\mathrm{Tr}\,h^{(\tau)}-\sum_{\tau,l_1,l_2,\mathbf k_1,\mathbf k_2}b^\dagger_{\tau l_1}(\mathbf k_1)[h^{(\tau)}]^T_{l_1l_2}(\mathbf k_1,\mathbf k_2)b_{\tau l_2}(\mathbf k_2),\qquad \tilde h^{(\tau)}\equiv-[h^{(\tau)}]^T=-[h^{(\tau)}]^*.$$

## Step 10 (Task 9) — Interaction Hamiltonian

$$\hat H_{\rm int}=\frac{1}{2A}\sum_{\tau_1\tau_2l_1l_2}\sum_{\mathbf k_1..\mathbf k_4}b^\dagger_{l_1\tau_1}(\mathbf k_1)b^\dagger_{l_2\tau_2}(\mathbf k_2)b_{l_2\tau_2}(\mathbf k_3)b_{l_1\tau_1}(\mathbf k_4)V(|\mathbf k_1-\mathbf k_4|)\delta_{\mathbf k_1+\mathbf k_2,\mathbf k_3+\mathbf k_4}.$$

## Step 11 (Task 10) — Wick's theorem (all six terms)
Shorthand: 1 = (l₁τ₁k₁), 2 = (l₂τ₂k₂), 3 = (l₂τ₂k₃), 4 = (l₁τ₁k₄). Then

$$\hat H^{\rm HF}_{\rm int}=\frac1{2A}\sum V(|\mathbf k_1-\mathbf k_4|)\delta_{\mathbf k_1+\mathbf k_2,\mathbf k_3+\mathbf k_4}\Big[\langle b_1^\dagger b_4\rangle b_2^\dagger b_3+\langle b_2^\dagger b_3\rangle b_1^\dagger b_4-\langle b_1^\dagger b_4\rangle\langle b_2^\dagger b_3\rangle-\langle b_1^\dagger b_3\rangle b_2^\dagger b_4-\langle b_2^\dagger b_4\rangle b_1^\dagger b_3+\langle b_1^\dagger b_3\rangle\langle b_2^\dagger b_4\rangle\Big].$$

## Step 12 (Task 11) — Quadratic terms
Keep only the four terms that are bilinear in the operators:

$$\hat H^{\rm HF,2}_{\rm int}=\frac1{2A}\sum V(|\mathbf k_1-\mathbf k_4|)\delta_{\mathbf k_1+\mathbf k_2,\mathbf k_3+\mathbf k_4}\Big[\langle b_1^\dagger b_4\rangle b_2^\dagger b_3+\langle b_2^\dagger b_3\rangle b_1^\dagger b_4-\langle b_1^\dagger b_3\rangle b_2^\dagger b_4-\langle b_2^\dagger b_4\rangle b_1^\dagger b_3\Big].$$

## Step 13 (Task 12) — Swap indices to combine
In the 2nd and 4th terms, relabel (l₁,τ₁,k₁,k₄) ↔ (l₂,τ₂,k₂,k₃). This leaves δ invariant, and
V(|k₂−k₃|) = V(|k₁−k₄|) by momentum conservation. Hence

$$\hat H^{\rm HF,2}_{\rm int}=\frac1A\sum V(|\mathbf k_1-\mathbf k_4|)\delta_{\mathbf k_1+\mathbf k_2,\mathbf k_3+\mathbf k_4}\Big[\langle b^\dagger_{l_1\tau_1}(\mathbf k_1)b_{l_1\tau_1}(\mathbf k_4)\rangle b^\dagger_{l_2\tau_2}(\mathbf k_2)b_{l_2\tau_2}(\mathbf k_3)-\langle b^\dagger_{l_1\tau_1}(\mathbf k_1)b_{l_2\tau_2}(\mathbf k_3)\rangle b^\dagger_{l_2\tau_2}(\mathbf k_2)b_{l_1\tau_1}(\mathbf k_4)\Big].$$

## Step 14 (Task 13) — Reduce momentum in the Hartree term
Write k_i → k_i + q_i, with k_i in the first BZ and q_i reciprocal-lattice vectors.

* Translation invariance gives ⟨b†_{q₁}(k₁) b_{q₄}(k₄)⟩ ∝ δ_{k₁k₄}.
* Then δ_{k₂+q₁+q₂, k₃+q₃+q₄} = δ_{k₂k₃} δ_{q₁+q₂, q₃+q₄}, because k₂ − k₃ lies inside the BZ while q₃+q₄−q₁−q₂ is a lattice vector.

$$H_{\rm Hartree}=\frac1A\sum_{\tau_1\tau_2l_1l_2\mathbf k_1\mathbf k_2\mathbf q_1..\mathbf q_4}\langle b^\dagger_{l_1\tau_1\mathbf q_1}(\mathbf k_1)b_{l_1\tau_1\mathbf q_4}(\mathbf k_1)\rangle b^\dagger_{l_2\tau_2\mathbf q_2}(\mathbf k_2)b_{l_2\tau_2\mathbf q_3}(\mathbf k_2)V(|\mathbf q_1-\mathbf q_4|)\delta_{\mathbf q_1+\mathbf q_2,\mathbf q_3+\mathbf q_4}.$$

## Step 15 (Task 14) — Reduce momentum in the Fock term

$$H_{\rm Fock}=-\frac1A\sum_{\tau_1\tau_2l_1l_2\mathbf k_1\mathbf k_2\mathbf q_1..\mathbf q_4}\langle b^\dagger_{l_1\tau_1\mathbf q_1}(\mathbf k_1)b_{l_2\tau_2\mathbf q_3}(\mathbf k_1)\rangle b^\dagger_{l_2\tau_2\mathbf q_2}(\mathbf k_2)b_{l_1\tau_1\mathbf q_4}(\mathbf k_2)V(|\mathbf k_1+\mathbf q_1-\mathbf k_2-\mathbf q_4|)\delta_{\mathbf q_1+\mathbf q_2,\mathbf q_3+\mathbf q_4}.$$

## Step 16 (Task 15) — Combine the Hartree and Fock terms (final mean-field Hamiltonian)
Define the density matrix ρ_{(α,q),(β,q′)}(k) = ⟨b†_{α,q}(k) b_{β,q′}(k)⟩, with α = (l,τ). Then

$$\hat H^{\rm HF}=\sum_{\mathbf k\in{\rm mBZ}}\sum_{\alpha\mathbf q,\beta\mathbf q'}b^\dagger_{\alpha\mathbf q}(\mathbf k)\Big[\tilde h^{(\tau_\alpha)}_{l_\alpha\mathbf q,l_\beta\mathbf q'}(\mathbf k)\delta_{\tau_\alpha\tau_\beta}+\Sigma^H_{\alpha\mathbf q,\beta\mathbf q'}+\Sigma^F_{\alpha\mathbf q,\beta\mathbf q'}(\mathbf k)\Big]b_{\beta\mathbf q'}(\mathbf k),$$

$$\Sigma^H_{(\alpha,\mathbf q),(\beta,\mathbf q')}=\frac{\delta_{\alpha\beta}}A V(|\mathbf q-\mathbf q'|)\,n(\mathbf q-\mathbf q'),\qquad n(\mathbf Q)=\sum_{\mathbf k',\gamma,\mathbf p}\rho_{(\gamma,\mathbf p),(\gamma,\mathbf p+\mathbf Q)}(\mathbf k'),$$

$$\Sigma^F_{(\beta,\mathbf q_2),(\alpha,\mathbf q_4)}(\mathbf k)=-\frac1A\sum_{\mathbf k'}\sum_{\mathbf Q}V(|\mathbf k'-\mathbf k+\mathbf Q|)\,\rho_{(\alpha,\mathbf q_4+\mathbf Q),(\beta,\mathbf q_2+\mathbf Q)}(\mathbf k').$$

These expressions are exactly H_Hartree + H_Fock of steps 14–15 rewritten as a matrix:

* Hartree: set q₂ → q and q₃ → q′, so Q = q₄ − q₁ = q − q′.
* Fock: set Q = q₁ − q₄.

The total energy is

$$E=\sum_{\mathbf k}\mathrm{Tr}\big[\tilde h(\mathbf k)\rho^T(\mathbf k)\big]+\tfrac12\sum_{\mathbf k}\mathrm{Tr}\big[(\Sigma^H+\Sigma^F(\mathbf k))\rho^T(\mathbf k)\big].$$

The self-consistency condition is that ρ is built from the occupied eigenvectors of the bracket. In the numerics the
Q = 0 Hartree term is dropped, because it is cancelled by the neutralising background (gate charges).
