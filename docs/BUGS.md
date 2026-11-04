# QCAT / GCS — Bug & Inconsistency Audit

Status updated 2026-09-07 on branch `docs/code-review` (tip `c077aff`). Cross-referenced against `docs/apssamp.tex` (paper "Superpositions of Quantum Gaussian Processes", updated version). See `docs/REVIEW.md` for the full paper–code consistency report.

Emoji legend: ✅ = Fixed on `docs/code-review` | ⚠️ = New / updated finding | still listed without emoji = open.

Items are grouped by severity. **Critical** = currently raises an exception on call or produces a clearly-wrong numerical result; **High** = silent wrong answer in a documented code path; **Medium** = limits the library (hardcoded N=2 / n=1, etc.); **Low** = style / cleanup.

---

## 1 · Critical — time-dependent modules are broken

### 1.1 `src/gbcs/Unitary_Operator_Time_Dependent.py:65` and `:98` — sign-flipped returns

```python
return - sigma_jk_dot.flatten()   # line 65
return - r_jk_dot.flatten()       # line 98
```

The constant-Hamiltonian counterpart (`src/gbcs/Unitary_Operator_Gaussian.py:55, :85`) returns the derivative without negation. The body of `sigma_jk_dot` (line 63) and `r_jk_dot` (line 96) compute the same RHS as the constant version, so the outer minus integrates the system backwards in time. Either remove the `-` or rewrite the RHS to be its negative — but as written, the two paths give opposite results for the same physics.

### 1.2 `src/gbcs/Open_Operator_Time_Dependent.py:62, 95, 130` — σ̇, ṙ, ṙ⁽⁰⁾ all wrong

The RHS construction puts the commutator terms in reversed order (e.g. line 62: `σ(H+H)Ω − Ω(H+H)σ` instead of `Ω(H+H)σ − σ(H+H)Ω`), then the final `return - x.flatten()` flips signs again. The two cancellations work out only for the linear part of σ̇; the Riccati term `−i(σ(H−H)σ + Ω(H−H)Ω)` survives with the **wrong sign** (`+i` instead of `−i`).

The ṙ ODE (line 95) ends up with `−½i σ(r_J − r_K)` where the paper Tab. 1 requires `+½i σ(r_J − r_K)`.

The ṙ⁽⁰⁾ ODE (line 130) ends up the **negative of the paper's Tab. 1 formula** in every term, *and* has coefficient `+½ Tr((H−H)σ)` instead of the paper's `−¼ Tr((H−H)σ)` (factor-of-two error on the trace).

### 1.3 `src/gbcs/Open_Operator_Time_Dependent.py` lacks all dissipative terms

The file is named "Open" but the ODEs do not contain `E`, `D`, or any Lindblad contribution. Compare with the corresponding constant module `src/gbcs/Open_Operator_Gaussian.py:53, 83` which carries `+ 2E`, `+ D`. The time-dependent open code is effectively a (sign-flipped) copy of the unitary time-dependent code.

### 1.4 `src/gbcs/Objects.py:224, 249, 290` — dispatch to non-existent functions

| line | calls | exists? |
|---|---|---|
| 224 | `Unitary_Operator_Time_Dependent.Dynamics_Numerical` | ✓ (defined at `:155`) |
| 249 | `Unitary_Operator_Time_Dependent.Dynamics_Analytical` | ✗ no such function in that module |
| 290 | `Open_Operator_Time_Dependent.Dynamics_Numerical` | ✗ that module defines `Unitary_Gaussian_General_Time_Dep_Numerical` instead |

So `Quantum_State.Unitary_Dynamics_Analytical(..., Symplectic.type == 'Time', ...)` and `Quantum_State.Open_Dynamics_Numerical(..., Hamiltonian.type[0] == 'Time', ...)` raise `AttributeError` at runtime.

### 1.5 `src/gbcs/Open_Operator_Force.py:Dynamics_General_Numerical` (`:115`) and `Dy_Analytical` (`:147`) — undefined names

Body references `Compute_R_JK_Unitary_Gaussian_Force_Numerical`, `Compute_C_JK_Unitary_Gaussian_Force_Numerical`, `Compute_Sigma_JK_Unitary_Gaussian_Force_Numerical`, and a bare `S_m_t` — none of which are defined or imported in this module. Any call lands on `NameError`.

`Quantum_State.Open_Dynamics_Numerical` reaches this dead branch when `Hamiltonian.type[1] == 'Force'` **and** `E ≠ 0` (line 275–278 of `Objects.py`); only `E == 0` works today.

### 1.6 `src/gbcs/Objects.py:271` — wrong numpy idiom for zero-check

```python
if E.all() == 0.0:
```

`E.all()` returns a Python `bool` (True if every element is truthy). Comparing that bool to `0.0` is `True == 0.0` → `False`, `False == 0.0` → `True`. So the condition fires whenever **any** element of `E` is zero, *not* when all are zero. Should be `np.all(E == 0)` (or `not np.any(E)`).

This silently misroutes between `Dynamics_Symetric_Numerical` (working) and `Dynamics_General_Numerical` (broken — see §1.5).

### 1.7 `src/gbcs/Objects.py:326` — class attribute used as instance attribute

```python
elif Quantum_State.type == 'Cat':
```

`Quantum_State.type` is not defined at class level; only `self.type` is set in `Initialize_Gaussian_State`. The check always raises `AttributeError`. Should be `self.type`.

### 1.8 `src/gbcs/Gaussian.py:Compute_Sigma_Open_Gaussian_Numerical_not_working` and `Compute_Sigma_Open_Gaussian_Analytical`

Both broken, the first one is honestly named:

- `:141, :144`: `S_tau = np.exp(t_array[i] * Omega @ (H_m+E))` — `np.exp` is **elementwise** scalar exponential, not matrix exponential. The propagator must use `scipy.linalg.expm`. The `print(additional_term[0]...)` on `:151` is also a debug leftover.
- `:182`: `Compute_Sigma_Open_Gaussian_Analytical` defines an inner `integrand` that is never used, and `solve_ivp(..., y0, ...)` references a name `y0` that is never defined in that scope. Raises `NameError`.

Same `np.exp` bug repeats in `src/gbcs/Open_Operator_Force.py:34, 37`.

### 1.9 `src/gbcs/Symplectic_Known.py:Symplectic_QHO_N` is structurally broken

```python
def Symplectic_QHO_t_1(omega, t):
    s_m_t = np.array([...])
    # no return
```

The inner helper doesn't return anything; the outer `Symplectic_QHO_t` then writes `vec_s_t[i] = Symplectic_QHO_t_1(...)` into a `(n_modes,)` real array but expects a `(2, 2)` matrix. The subsequent `np.block` comprehension references `mat` and `i, j` in a way that doesn't match what `vec_s_t` actually holds.

`Objects.Symplectic.Initialize_Constant_Symplectic('QHO_n', …)` calls this and would `TypeError`/`IndexError` immediately.

### 1.10 `src/gbcs/Wigner_Functions.py:205` — bare `return`

`Wigner_jk_func` ends with `return` (no value); the function silently returns `None`. `Wigner_jk_t_Func` (line 165) calls it expecting a `(2^N, 2^N, steps, steps)` array — will `TypeError` on the subsequent assignment.

---

## 2 · High — silent wrong answers

### 2.1 `src/gbcs/Gaussian.py:equation_dot_r` (`:200`) inconsistent with `Compute_r_t_S_m` (`:48`)

ODE form:
```python
r_dot = Omega @ (H_m + E) @ r - r_m
```

Closed-form solution used in the unitary path:
```python
r_t = S_m @ (r_0 + r_tilde_m) - r_tilde_m,   r_tilde_m = H_m⁻¹ r_m
```

Differentiating the closed form gives `ṙ = ΩH·r + Ω·r_m`. The ODE form has `-r_m` (no `Ω`). Both cannot be right; the closed form matches the paper's Tab. 2 verbatim, so the ODE is the one to fix:

```python
r_dot = Omega @ (H_m + E) @ r + Omega @ r_m   # consistent with the closed form / paper sign
```

Note: this depends on the chosen sign for `r_m` in the input Hamiltonian (`+r_mᵀ r̂` in the code vs `−r_Jᵀ r̂` in paper Eq. 29). The code's `r_array` is effectively the **negative** of the paper's `r_J`. Document this in `ARCHITECTURE.md`/`CLAUDE.md` so users don't paste paper numbers in directly.

### ✅ 2.2 `src/gbcs/Open_Operator_Gaussian.py:83` — missing driving term (FIXED)

Fixed: `Open_Operator_Gaussian.py:83` now correctly has `0.5 * Omega @ (2*d - r_j - r_k)`. The function accepts an optional `d` argument (default zeros).

### ✅ 2.3 `src/gbcs/Open_Operator_Gaussian.py:112` — no qubit dephasing (Γ_z) (FIXED)

Fixed: `Open_Operator_Gaussian.py:114` now computes `dephasing = 0.5 * np.real(np.sum(Gamma_z * (J_labels * K_labels - 1)))` and adds it to `r_0_jk_dot`. The function accepts an optional `Gamma_z` array (default zeros). Implements the single-qubit independent dephasing model of Tab. 3 (Eq. linda).

### ✅ 2.4 `src/gbcs/Measurements.py:102` vs `:127` — inconsistent efficiency-angle formulas (FIXED)

Fixed: both `CM_Homodyne_Noisy_1Mode` and `CM_Heterodyne_Noisy_1Mode` now use `arccos(sqrt(η))`, consistent with the generaldyne POVM derivation.

### 2.5 `src/gbcs/Measurements.py:153, 155` — `det` is not a determinant

```python
if quadrature == 'Position':
    det = (sigma_jk[i,j] + sigma_measure)[2*mode_number, 2*mode_number]
elif quadrature == 'Momentum':
    det = (sigma_jk[i,j] + sigma_measure)[2*mode_number+1, 2*mode_number+1]
```

Named `det` but it is one diagonal entry of the matrix (a single variance), not `np.linalg.det`. Used downstream as if it were a `det` in the Gaussian normalisation `(π^n sqrt(det))⁻¹`. For a generaldyne measurement projecting onto a single quadrature the right object is the marginal variance, so this *might* be intentional — but the name is actively misleading.

### 2.6 `src/gbcs/Wigner_Functions.py:Create_XX_PP` (`:13`) plot range uses variances, not std-devs

```python
max_x = np.max(np.real(GCS.r_JK_t[...,0,0] + sigma_para * GCS.sigma_JK_t[...,0,0]))
```

For a Gaussian, the n-σ contour lies at `r + n · sqrt(Σ_xx)`. The current expression adds the **variance** instead of `sqrt(variance)`. Likely cause of "plot too small/too large depending on units".

---

## 3 · Medium — hardcoded N = 2 / n = 1

### 3.1 `src/gbcs/Plots_Functions.py:_INDEX_MAP_3D` (`:755`) — 4 entries, 2-qubit only

Used by `Draw_Curves`, `Draw_Vectors`, `Draw_Ellipsoids` and every 3-D plotter/animation. For N > 2 you have 2^(2N) (j,k) pairs; the labels `real+ / real- / off / off*` no longer cover the index space. Replace with a generated map: `[((j,k), label_jk, role_jk) for j in range(2**N) for k in range(2**N)]`, distinguishing diagonal (j == k) from off-diagonal by role.

### 3.2 `src/gbcs/Measurements.py:Negativity_Qubits` (`:195`) — 2-qubit hardcoded

```python
J = (len(rho) - 1) / 4
dim = int(2*J + 1)
dimensions_list = [dim, dim]
mask = [0, 1]
```

The `J = (len - 1)/4` formula gives `dim = 2` only for `len(rho) == 4` (i.e. N = 2). For N > 2: `dimensions_list = [2] * N`, plus the user must specify which bipartition to test (the `mask`).

### 3.3 `src/gbcs/Measurements.py:75, 148` — `r_measure` 1-mode hardcoded

```python
r_measure = np.array([[r_measure_array[k]], [r_measure_array[k]]])
```

This is a `(2, 1)` vector — only works for `n_modes == 1`. For multi-mode, `r_measure` would need shape `(2n, 1)` with values placed at the measured modes (zeros elsewhere, or whatever the measurement convention dictates).

### 3.4 `src/gbcs/Symplectic_Known.py:Symplectic_2QHO_XX` — hardcoded `4 × 4`

Only valid for 2 modes. The variable name is honest but there is no n > 2 generalisation.

### 3.5 `src/gbcs/Gaussian.py:log_nega_2_modes` — hardcoded 2 modes

Name is again honest; would need an n-mode equivalent for multi-mode log-negativity.

---

## 4 · Medium — dead / unreachable code

### 4.1 `src/gbcs/Open_Operator_Force.py:28` — `E` parameter is silently ignored

```python
def Compute_Sigma_JK_Open(N_qubits, n_modes, sigma_0, H_m, E, D, t_array):
    ...
    S_m_tau = np.exp(omega @ H_m * tau)   # no E here
```

The function takes `E` but never uses it; the propagator is built from `H_m` alone. Then `Dynamics_Symetric_Numerical:91` calls it with `E = zeros((2n, 2n))` (correct), and `Dynamics_General_Numerical:123` calls it with `(N, n, σ_0, H_m, D, t_array)` — 6 positional args — so `D` binds to `E` and `t_array` is missing entirely. `TypeError` at call.

### 4.2 `src/gbcs/QIT_Functions.py:14–35` — `Extract_Qubit_Labels_Array_old`

Older buggy version (see the awkward `np.sum(ranges*J_labels) + ranges[i] + 0.1` heuristic). Superseded by the cleaner `format(index, '0Nb')` version below it. Delete.

### 4.3 `src/gbcs/Plots_Functions.py:222 (Plot_Wigner_4_Times), :292 (Plot_Wigner_Fringes), :407 (generate_pi_ticks)`, and others — unreferenced helpers

Several plotting helpers and the paper-figure utilities (`plot_paper_1st`, `plot_frame_presentation`, `plot_wigner_paper_1`, `set_up_figure`) are not called from anywhere inside the library nor from any tracked notebook. Either drop or move to an `examples/` subdir.

### 4.4 `src/gbcs/Wigner_Functions.py:165, :178` — `Wigner_jk_t_Func` / `Wigner_jk_func`

Unused. Combined with the bare-`return` bug (§1.10) they are pure cruft.

---

## 5 · Medium — naming / convention traps

### 5.1 File `Unitary_Operator_Gaussian.py` is the **General** case

Counter-intuitive: the "Gaussian" path (no qubit-mode coupling) lives in `Gaussian.py`; the "Gaussian-process-valued operator H_J" lives in `Unitary_Operator_Gaussian.py`. Rename to `Unitary_Operator_General.py` (and the matching `Open_Operator_General.py`).

### 5.2 Typos in filenames and identifiers

| Repo string | Should be |
|---|---|
| ~~`Time_Depentent`~~ | `Time_Dependent` — done (files and all references renamed) |
| `Simpletic` (e.g. `Gaussian.py:96`) | `Symplectic` |
| `Generadyne` (Measurements.py) | `Generaldyne` |
| `Measurament` (Objects.py, Measurements.py — pervasive) | `Measurement` |
| `Indipendent` (Objects.py print strings) | `Independent` |
| `eficency` / `efficency` (Measurements.py) | `efficiency` |
| `unkonwn` (docstrings) | `unknown` |
| `Caovariant` / mixed cases (docstrings) | `Covariance` |

Rename in a single PR to avoid scattering renames across unrelated work.

### 5.3 Sign of `r_m` vs paper `r_J`

The code's analytical closed form `r(τ) = S(r_0 + r̃) − r̃` corresponds to the Hamiltonian convention `H = ½ r̂ᵀ H_m r̂ + r_mᵀ r̂` (positive linear term). The paper Eq. 29 uses `H_J = ½ r̂ᵀ H_J r̂ − r_Jᵀ r̂` (negative linear term). So **`r_array` in code is the additive inverse of `r_J` in the paper**. Document prominently or rename `r_array → minus_r_array` for clarity.

---

## 6 · Low — style / cleanup

### 6.1 Debug `print()` calls left in hot paths

| File | Lines |
|---|---|
| `src/gbcs/Unitary_Operator_Gaussian.py` | 162 (`print(rho_q_0_JK)`), 176, 177 (`print(phi_JK_t[...])`) |
| `src/gbcs/Measurements.py` | 158 (`print(det)`) |
| `src/gbcs/Gaussian.py` | 151 (`print(additional_term[0].reshape(...))` inside the loop) |
| `src/gbcs/Objects.py` | 53, 56, 59, 77, 80, 83, 145, 162 (in tests path), 265, 305, 311, 312, 326–330, 334, 342–345, 355, 358, 389, 422 |

Remove or replace with `logging.debug`.

### 6.2 dtype inconsistency

`src/gbcs/Open_Operator_Gaussian.py`, `src/gbcs/Unitary_Operator_Force.py`, `src/gbcs/Open_Operator_Force.py`, the time-dependent modules, and the measurement code allocate arrays as `complex64`. `src/gbcs/Unitary_Operator_Gaussian.py:139–143` uses `complex128`. Scipy's `solve_ivp` returns `float64`/`complex128` regardless. Storing in `complex64` introduces single-precision round-off on every read-back. Pick one (`complex128` everywhere is the safe default).

### 6.3 `import X as X`

Every module does `import Gaussian as Gaussian`, `import QIT_Functions as QIT_Functions`, etc. The alias is identical to the module name and provides no benefit — drop the `as` clause.

### 6.4 Missing `__init__.py` content

`src/__init__.py` is empty. With no contents and no `setup.py` / `pyproject.toml`, the project is consumed via `sys.path` hacks in notebooks. A minimal `__init__` exposing the public API (`Quantum_State`, `Hamiltonian`, `Symplectic`, `Decoherence`) would let users `from qcat import Quantum_State` after a `pip install -e .`.

### 6.5 `Unitary_Operator_Force.py:Dynamics_Numerical` doesn't use Hermitian-conjugate symmetry

Lines 70–81 loop the full `(2^N)²` grid; the analytical version (`:106`) already exploits `j ≤ k`. Apply the same optimisation to halve work.

### 6.6 Untracked artifacts in the repo root

`SqueezedThermalStates.png`, `ThermalStates.png`, `output{,_2,…_6}.png`, `notebooks/*.gif`, `Mass Independence/`, and similar plot output files are sitting at the repo root unignored. Add to `.gitignore` (`*.png`, `*.gif`, `output*` if they're never meant to be committed) — keeps `git status` readable.

---

## 7 · ⚠️ New findings — paper update vs code (2026-09-07)

These items arise from comparing the code against the **updated** `docs/apssamp.tex` ("Superpositions of Quantum Gaussian Processes"). The paper has been revised since the original audit.

### ✅ ⚠️ 7.1 Tab. 1 σ̇ ODE — factor-of-2 + i-factor discrepancy (FIXED in code)

The updated Table 1 (label `tab:unitary`) now reads:
```
σ̇_jk = ½ [ Ω(H_j+H_k)σ_jk − σ_jk(H_j+H_k)Ω − iσ_jk(H_j−H_k)σ_jk + Ω(H_j−H_k)Ω ]
```
The `½` applies to all four terms, and the `Ω(H_j−H_k)Ω` term is **positive with no `i`**.

The code (`Unitary_Operator_Gaussian.py:53`) still implements:
```python
0.5 * (linear terms) - 1j * (Riccati + Omega @ (H_J-H_K) @ Omega)
```
Two errors relative to the updated table:
1. **Factor of 2**: Riccati `−iσΔσ` should be `−½iσΔσ`.
2. **Wrong sign and i-factor** on ΩΔΩ: code has `−iΩΔΩ`, table has `+½ΩΔΩ`.

The same discrepancy appears in `Open_Operator_Gaussian.py:53`.

**Internal paper inconsistency:** The explicit off-diagonal equations in Sec. 4.2.1 (tex lines 516–517) still read `−i(σH_qσ + ΩH_qΩ)`, matching the **old** table and the code, not the current Table 1. One of these needs updating. The appendix derivation (`app:unitary_general`) is the tie-breaker. Until resolved, the code's behavior matches the in-body equations but not the table.

### ⚠️ 7.2 Same σ̇ discrepancy in Tab. 3 (open, HIGH)

Table 3 (label `tab:open`) has the same updated form for the Riccati+ΩΔΩ block. Same correction applies to `Open_Operator_Gaussian.py:53` once §7.1 is resolved.

### ⚠️ 7.3 ṙ⁽⁰⁾ Tr factor — now consistent (resolved ambiguity)

The ṙ⁽⁰⁾ ODE in Tab. 1 shows `−i/4 Tr[(H_j−H_k)σ_jk]`. The code now uses `- 0.25 * np.trace(...)` (`Unitary_Operator_Gaussian.py:113`, `Open_Operator_Gaussian.py:118`). **This is consistent with Tab. 1.** The old bug (factor ½ instead of ¼) has been fixed.

### ⚠️ 7.4 Perturbation theory (Sec. 7) — not implemented (MISSING FEATURE)

The paper introduces a phase-space perturbation theory for non-diagonal DV super-operators (T₁ qubit decay). The first-order correction is a "two-times" Gaussian integral (Eq. decay_qubit):
```
χ^(1)(τ,ξ̃) = p₊Γ₁ ∫₀^τ dt e^{−Γ₁t} e^{−¼ξ̃^T σ^{Γ₁}(τ,t) ξ̃ + iξ̃^T r̄^{Γ₁}(τ,t)}
```
with two-times parameters involving `S₊(t)` and `S₋(τ−t)`.

**No implementation exists in any module.** This is a required feature for the second physical example's Wigner function (Sec. 8.2 / T₁ case). Will require:
- A new object (e.g. `Perturbation_State` or `Trajectory`) to hold the two-times parameters
- Numerical integration of the first-order characteristic function correction
- Extension of `Wigner_Functions.py` to evaluate the integral form

### ✅ 7.5 Doubling / Riccati solver — resolved

Sec. 4.3 (`subsec:riccati`) gives a systematic method to solve the Riccati equation analytically by doubling the variables: `σ_jk = Y_jk X_jk^{-1}` where `(X, Y)^T` evolves under a linear `A_jk` matrix. The paper uses this to solve Example 1 (dispersive readout).

Implemented in `src/gbcs/Riccati_Linear.py` (`Doubling_Matrix`, `Dynamics_Linear`) and exposed as `solver='linear'` on `Unitary_Dynamics_Numerical` and `Open_Dynamics_Numerical`, for the Constant + General case. It extends the doubling to `r` and `s`, so all three come from one matrix exponential per branch pair. See `tests/test_riccati_linear.py` and `notebooks/Riccati_Linear_Map.ipynb`.

### ⚠️ 7.6 ✅ Ċ sign — fixed (note for record)

The paper flags `\textcolor{red}{removed minus sign from $\dot{C}$}` in Eq. (general_unitary_ODE_off). The code correctly defines `C_JK_t = −Re(r_JK_0_t − log(rho_q_0))`, which gives `Ċ = −Re(ṙ⁽⁰⁾)`. This is consistent with the updated paper. ✓

---

## 9 · Things I did not verify

- The numerical accuracy of `log_nega_2_modes` (the algebra looks consistent with the standard 2-mode formula, but I did not run it against a known case).
- Whether the `Wigner_PMS_func` POVM contraction (line 74) sign / order is right — it builds `POVM[a,i,j] · ρ_q[i,j]` then sums; the natural ordering depends on whether `POVM_array` was built as `|outcome⟩⟨outcome|` in qubit basis or its conjugate, which is set in `Measurements.Qubit_Ideal_Measurament` line 28–42 (uses `linalg.eig` whose eigenvector phase is arbitrary).
- `Plots_Functions.Plot_Phase_Space_First_QRDM_Func` and the legacy paper-figure helpers (`plot_paper_1st`, `set_up_figure`, etc.) — I only verified they're either unused or syntactically valid; physics not checked.
- Performance: every `Wigner_*` function has a 4- or 5-level nested Python loop. For a 2-qubit/1-mode/`steps=200` grid that's already minutes per frame; for animations it would benefit from `np.einsum` vectorisation. Not a correctness bug but a real ergonomics issue.

---

## 10 · Suggested fix order (updated 2026-09-07)

**Before final push — blockers:**

1. **Resolve Tab. 1 σ̇ discrepancy** (§7.1–7.2): check appendix derivation, then update either Table 1 or the code + explicit in-body equations to be consistent. This is a physics question, not a style question.
2. **Block time-dependent wrong-answer paths** (§1.1–1.3): fix sign errors or raise `NotImplementedError`.
3. **Fix dispatch to non-existent functions** (§1.4, §1.5): `Dynamics_Analytical` for time-dependent, `Dynamics_General_Numerical` undefined names in Force open module.
4. **Fix `E.all() == 0.0`** (§1.6): one-line fix → `np.all(E == 0)`.
5. **Replace `np.exp` with `linalg.expm`** (§1.8 / §4.1).
6. **Fix `Wigner_jk_func` bare return** (§1.10).

**New features for final push:**

7. **Implement perturbation theory / trajectory method** (§7.4): new object for T₁ correction; needed for the second example's Wigner function.
8. ✅ **Expose Riccati doubling solver** (§7.5): done, `src/gbcs/Riccati_Linear.py`.

**Cleanup (can follow final push):**

9. Fix `_INDEX_MAP_3D` to support N > 2 (§3.1).
10. Fix `Symplectic_QHO_N` (§1.9).
11. Strip debug prints, unify dtype to `complex128`, add `__init__.py` exports, gitignore output artifacts (§6).
12. Rename `Unitary_Operator_Gaussian.py → Unitary_Operator_General.py` (§5.1). The `Depentent → Dependent` rename (§5.2) is done.

See `docs/REVIEW.md` §9 for the branch management plan to reach final push.
