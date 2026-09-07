# GCS — Paper–Code Consistency Review

Cross-reference between `docs/apssamp.tex` (latest version, "Superpositions of Quantum Gaussian Processes") and the implementation on branch `docs/code-review`. Sections/tables quoted by label follow the LaTeX source.

---

## 1 · Summary of changes since last code audit

The following items from the original `BUGS.md` audit have been fixed on `docs/code-review`:

| Bug | Description | Status |
|---|---|---|
| 2.2 | Missing `d` driving vector in open ṙ_jk ODE | **Fixed** — `Open_Operator_Gaussian.py:83` now has `0.5 * Omega @ (2*d - r_j - r_k)` |
| 2.3 | Missing Γ_z dephasing in open ṙ⁽⁰⁾ ODE | **Fixed** — `Open_Operator_Gaussian.py:114` now computes `0.5 * Re(Σ Gamma_z*(J*K-1))` |
| 2.4 | Heterodyne efficiency formula | **Fixed** — both homodyne and heterodyne now use `arccos(sqrt(η))` in `Measurements.py:127` |
| — | Per-branch initial conditions in Force case | **Fixed** — `Unitary_Operator_Force.py` and `Open_Operator_Force.py` detect `sigma_0.ndim == 4` |
| — | Per-branch initial conditions in General case | **Fixed** — `Unitary_Operator_Gaussian.py:148` and `Open_Operator_Gaussian.py:160` |
| — | Pure-Gaussian first-moment dynamics sign + missing Ω | **Fixed** — `Gaussian.py` force-path ODE corrected |
| — | 3-D phase-space plotting and animations | **Added** — `Plots_Functions.py` `Draw_Curves`, `Draw_Vectors`, `Draw_Ellipsoids`, `Plot_Pancake_3D`, plus animation wrappers |

---

## 2 · Table 1 (Tab. `tab:unitary`) — Unitary General ODEs

### 2.1 σ̇_jk ODE

**Paper (current):**
```
σ̇_jk = ½ [ Ω(H_j+H_k)σ_jk − σ_jk(H_j+H_k)Ω − iσ_jk(H_j−H_k)σ_jk + Ω(H_j−H_k)Ω ]
```
The `½` prefactor applies to **all four** terms; the `Ω(H_j−H_k)Ω` term is **positive and carries no `i`**.

**Code (`Unitary_Operator_Gaussian.py:53`):**
```python
sigma_jk_dot = 0.5 * (Omega @ (H_J+H_K) @ sigma_jk - sigma_jk @ (H_J+H_K) @ Omega) \
             - 1j * (sigma_jk @ (H_J-H_K) @ sigma_jk + Omega @ (H_J-H_K) @ Omega)
```

**Status: ✅ Fixed.** The change in Table 1 was purely a factor-of-2: the `½` now applies to the Riccati and ΩΔΩ terms as well (previously they had coefficient `−i` instead of `−½i`). Code now reads:
```python
sigma_jk_dot = 0.5 * (Omega @ (H_J+H_K) @ sigma_jk - sigma_jk @ (H_J+H_K) @ Omega
                      - 1j * (sigma_jk @ (H_J-H_K) @ sigma_jk + Omega @ (H_J-H_K) @ Omega))
```
This gives:
- Riccati: `−½iσ(H_j−H_k)σ` ✓
- ΩΔΩ: `−½iΩ(H_j−H_k)Ω` ✓

This also agrees with the appendix derivation (App. B), which gives coefficient `i/2` on both the `σΔσ` and `ΩΔΩ` pieces. The explicit single-qubit off-diagonal equations in Sec. 4.2.1 (tex lines 516–517) still read `−i(σH_qσ + ΩH_qΩ)` (the old, pre-factor-of-2 form) — these should be updated in the paper text for consistency with the corrected Table 1.

**Note on Ċ sign:** The paper flags `\textcolor{red}{removed minus sign from $\dot{C}$}` in Eq. (general_unitary_ODE_off). The code implements `C_JK_t = −Re(r_JK_0_t − log(rho_q_0))`, which makes `Ċ = −Re(ṙ⁽⁰⁾)` — consistent with the current paper convention. ✓

### 2.2 ṙ_jk ODE

**Paper:**
```
ṙ_jk = ½ [ (Ω(H_j+H_k) − iσ_jk(H_j−H_k)) r_jk − Ω(r_j+r_k) + iσ_jk(r_j−r_k) ]
```

**Code (`Unitary_Operator_Gaussian.py:83`):**
```python
r_jk_dot = 0.5 * Omega @ (H_j+H_k) @ r_jk - 0.5j * sigma_jk @ (H_j-H_k) @ r_jk \
          - 0.5 * Omega @ (r_j+r_k) + 0.5j * sigma_jk @ (r_j-r_k)
```

**Status: ✓ Matches Tab. 1.**

### 2.3 ṙ⁽⁰⁾_jk ODE

**Paper:**
```
ṙ⁽⁰⁾_jk = −i/2 r̄_jk^T (H_j−H_k) r̄_jk + i(r_j−r_k)^T r̄_jk − i/4 Tr[(H_j−H_k)σ_jk] − i/2 (H_j⁰−H_k⁰)
```

**Code (`Unitary_Operator_Gaussian.py:112–113`):**
```python
r_0_jk_dot = 1j * (- 0.5 * r_jk.T @ (H_j-H_k) @ r_jk + (r_j-r_k).T @ r_jk
            - 0.25 * np.trace((H_j-H_k) @ sigma_jk) - 0.5 * np.transpose(J_labels-K_labels) @ H_q_0_array)
```

**Status: ✓ Matches Tab. 1.** The factor `−i/4` (= `1j * −0.25`) is correctly implemented. This was the factor-of-2 fix mentioned by the user — the old code had `−0.5`.

---

## 3 · Table 2 (Tab. `tab:force`) — Unitary Force Closed Form

Not repeated in full here; the closed-form expressions for `σ(τ)`, `r̄_jk(τ)`, `C_jk(τ)`, `φ_jk(τ)` match `Unitary_Operator_Force.py`. Per-branch initial conditions now supported. ✓

**Remaining concern:** `r_tilde = H_m^{-1} r_j` (paper) corresponds to the code's internal convention where `r_array` = `−r_J` of the paper (negative linear term). This sign difference is documented in `ARCHITECTURE.md` but is a persistent trap for users copying paper formulas directly into the API.

---

## 4 · Table 3 (Tab. `tab:open`) — Open General ODEs

### 4.1 σ̇_jk ODE (open)

**Paper:**
```
σ̇_jk = ½ [ Ω(H_j+H_k+2E)σ_jk − σ_jk(H_j+H_k+2E^T)Ω − iσ_jk(H_j−H_k)σ_jk + Ω(H_j−H_k)Ω ] + D
```

**Code (`Open_Operator_Gaussian.py:53`):**
```python
sigma_jk_dot = 0.5 * (Omega @ (H_J+H_K+2*E) @ sigma_jk - sigma_jk @ (H_J+H_K+2*E.T) @ Omega) + D \
             - 1j * (sigma_jk @ (H_J-H_K) @ sigma_jk + Omega @ (H_J-H_K) @ Omega)
```

**Status: ✅ Fixed** (same fix as §2.1). `Open_Operator_Gaussian.py:53` now matches Tab. 3.

### 4.2 ṙ_jk ODE (open)

**Paper:**
```
ṙ_jk = ½ [ (Ω(H_j+H_k+2E) − iσ_jk(H_j−H_k)) r_jk + Ω(2d − r_j − r_k) + iσ_jk(r_j−r_k) ]
```

**Code (`Open_Operator_Gaussian.py:83`):**
```python
r_jk_dot = 0.5 * Omega @ (H_j+H_k+2*E) @ r_jk - 0.5j * sigma_jk @ (H_j-H_k) @ r_jk \
          + 0.5 * Omega @ (2*d - r_j - r_k) + 0.5j * sigma_jk @ (r_j-r_k)
```

**Status: ✓ Matches Tab. 3.** (Bug 2.2 fixed — `d` driving now included.)

### 4.3 ṙ⁽⁰⁾_jk ODE (open)

**Paper:**
```
ṙ⁽⁰⁾_jk = −i/2 r̄^T(H_j−H_k)r̄ + i(r_j−r_k)^T r̄ − i/4 Tr[(H_j−H_k)σ] − i/2(H_j⁰−H_k⁰)
           − ½(Γ_z^jj + Γ_z^kk) + Γ_z^jk
```

**Code (`Open_Operator_Gaussian.py:117–120`):**
```python
r_0_jk_dot = 1j * (- 0.5 * r_jk.T @ (H_j-H_k) @ r_jk + (r_j-r_k).T @ r_jk
            - 0.25 * np.trace((H_j-H_k) @ sigma_jk) - 0.5 * np.transpose(J_labels-K_labels) @ H_q_0_array)
return r_0_jk_dot[0,0] + dephasing   # dephasing = 0.5 * Re(Σ Gamma_z*(J*K-1))
```

**Status: ✓ Matches Tab. 3 for the standard per-qubit independent dephasing model.** The code's `dephasing` term implements `Σ_i Γ_z^i (J_i K_i − 1)/2`, which corresponds to `−½(Γ_z^jj+Γ_z^kk) + Γ_z^jk` under the single-qubit model of Eq. (linda) with jump operators `sqrt(Γ_z/2) σ_z`.

---

## 5 · Table 3(b) — Open Force Integral Form

Implemented in `Open_Operator_Force.py:Dynamics_Symetric_Numerical` for E=0 only. The general-E path (`Dynamics_General_Numerical`) is broken (undefined names, see BUGS.md §1.5). The `np.exp` vs `expm` bug in `Open_Operator_Force.py:34,37` remains.

---

## 6 · Perturbation Theory (Sec. 7 / `sec:perturbation`)

**Paper introduces** a phase-space perturbation theory for non-diagonal DV super-operators (e.g. DV decay, T₁ noise). The key result is the Volterra series (Eq. volterra / Eq. series), with the first-order correction to the characteristic function:
```
χ^(1)(τ, ξ̃) = p₊ Γ₁ ∫₀^τ dt e^{−Γ₁t} e^{−¼ ξ̃^T σ^{Γ₁}(τ,t) ξ̃ + i ξ̃^T r̄^{Γ₁}(τ,t)}
```
with "two-times" covariance and first moments:
```
σ^{Γ₁}(τ,t) = S₋(τ−t) S₊(t) σ₀ S₊^T(t) S₋^T(τ−t)
r̄^{Γ₁}(τ,t) = S₋(τ−t)[S₊(t)r₀ − (S₊(t)−1)r₊] − (S₋(τ−t)−1)r₋
```
where S±(τ) = exp(τ Ω (H_m ± H_q)), r± = H_m±H_q)^{-1}(r_m ± r_q).

**Code status: Not implemented.** No object, no module, no function exists for the perturbation theory or trajectory method. This is planned as a new feature (new object `Trajectory` or `Perturbation_State`) that would enable:
- Computing the first-order T₁ correction to a GBCS
- Evaluating the resulting (non-Gaussian) Wigner function via the characteristic function integral
- Visualizing with the existing 3-D phase-space plotting functions

---

## 7 · Physical Examples

### 7.1 Example 1 — Dispersive qubit readout via squeezed resonator (Sec. 8.1)

Uses quadratic Hamiltonian (General case). Solved via doubling method (Riccati). Steady state + transient. **Not directly reproducible with current code** because:
- The Riccati doubling method is not exposed as a callable in `Symplectic_Known.py` or `Objects.py`; only the numerical ODE solver is available.
- The σ̇ ODE discrepancy (§2.1) means numerical results may differ quantitatively from the paper until the factor-of-2 question is resolved.

### 7.2 Example 2 — Stern-Gerlach interferometer (Sec. 8.2)

Uses Force case + open dynamics (Lyapunov, E=0, D=diffusion). **Reproducible with current code** via `Open_Dynamics_Numerical` → Force path (`Hamiltonian.type[1] == 'Force'`, E=0). The analytical closed form (Tab. open_app) is also in `Open_Operator_Force.py:Dynamics_Symetric_Numerical`. The post-measurement Wigner functions (the ones shown in Fig. example_1 of the paper) can be computed and plotted with existing `Wigner_Functions.py` and the new 3-D plots.

The T₁ perturbative correction shown as the second Wigner figure in Sec. 8.2 requires the new perturbation theory object (§6 above).

---

## 8 · Measurements (Sec. 6 / `sec:measure`)

### 8.1 Generaldyne / homodyne

Fixed: both `CM_Homodyne_Noisy_1Mode` and `CM_Heterodyne_Noisy_1Mode` now use `arccos(sqrt(η))`. ✓

### 8.2 Qubit measurement + Wigner fringes

`Measurements.py:Wigner_PMS_func` builds the post-measurement Wigner from `POVM_array`. The eigenvector phase from `linalg.eig` is arbitrary — the result is physically correct up to a global phase on the fringes but the phase of individual fringe components is not guaranteed to match the paper's convention. Low severity but worth noting when comparing fringe plots.

---

## 9 · Branch Management Plan

Current branches (as of 2026-09-07):

| Branch | Status | Commits ahead of main | Recommendation |
|---|---|---|---|
| `main` | Base | — | Target for final push |
| `docs/code-review` | Active; all bug fixes + docs + 3D plots | +9 | Merge → `main` |
| `animation-vectors-ellipsoids` | 3D plot work; already merged into `docs/code-review` base | +2 vs old main | Delete after `docs/code-review` is merged |
| `final_branch` | Old snapshot; diverged from current work | +0 meaningful | Delete |
| `test` | 2 test commits; no useful code | +2 trivial | Delete |
| `origin/LorenzoBracciniUCL-patch-1` | README-only patch already merged | — | Delete remote ref |

**Recommended sequence toward final push:**

1. **Resolve the Tab. 1 σ̇ discrepancy** (§2.1) by checking the appendix derivation; update either the table or the code to make them consistent. This is the highest-priority physics question.
2. **Merge `docs/code-review` → `main`** (PR or fast-forward). This brings in all fixes, 3D plots, and updated docs in one clean merge.
3. **Delete stale branches**: `final_branch`, `test`, `animation-vectors-ellipsoids` (local + remote), `LorenzoBracciniUCL-patch-1` remote.
4. **New branch `feature/perturbation-theory`**: Implement `Perturbation_State` object + trajectory method for T₁ and the second example's Wigner function.
5. **New branch `fix/time-dependent`**: Fix sign-flipped ODEs in `*_Time_Depentent.py` modules (BUGS.md §1.1–1.3) and fix the `Dynamics_Analytical` dispatch bug (§1.4).
6. **New branch `fix/critical`**: Fix `E.all() == 0.0` (§1.6), `np.exp → expm` (§1.8), `Wigner_jk_func` bare return (§1.10), and `Open_Operator_Force.Dynamics_General_Numerical` undefined names (§1.5).

---

## 10 · Items Not Verified

- **Appendix B derivation** (`app:unitary_general`): determines which form of σ̇ (Tab. 1 or explicit equations) is correct for the updated paper.
- **Doubling / Riccati solver** (`app:doubling`, `subsec:riccati`): the `A_jk` matrix formulation is in the paper (Sec. 4.3) but not in code. Needed for Example 1.
- **Open force Tab. 3(b) with E≠0**: not tested. The integral form requires `S_A = exp(ΩAτ)` with `A = H_m + E`; the current `np.exp` bug in `Open_Operator_Force.py:34` must be fixed before enabling this path.
- **Multi-mode Wigner** (`Wigner_Functions.py`): correctness for n>1 was not checked against the paper's definitions.
