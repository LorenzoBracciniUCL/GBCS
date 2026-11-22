# gbcs: Gaussian-branched cat states

Python library for the dynamics and measurements of $N$ qubits coupled to $n$ bosonic modes through operator-valued Gaussian interactions, following

> L. Braccini, S. Bose and A. Serafini, *Superpositions of Quantum Gaussian Processes*, [arXiv:2510.01156](https://arxiv.org/abs/2510.01156).

When the coupling between qubits and modes is linear or quadratic in the mode quadratures and diagonal in the qubit basis, the joint state stays a superposition of Gaussian processes. Every branch $\langle j|\hat\varrho|k\rangle$ is a Gaussian with its own (complex) covariance matrix, first moments, contrast and phase. gbcs follows these quantities in time, so the qubit state, the Wigner function of the modes and the measurement statistics come out without truncating the Fock space.

It covers:

- unitary and open dynamics (Tables I to III of the paper), with a linear solver for the Riccati equations of the general case;
- projective measurements of the qubits, and homodyne and heterodyne measurements of the modes with finite efficiency;
- a perturbation theory for terms that act on the qubits alone, such as $T_1$ decay and qubit drives (Sec. VII);
- Wigner functions, phase-space plots and animations.

## Installation

```bash
pip install git+https://github.com/LorenzoBracciniUCL/GCS.git
```

To run the tutorials and tests, install from a clone of this repository:

```bash
pip install -e ".[test,notebooks]"
pytest
```

The library needs numpy (1.24 or later, 2.x included), scipy and matplotlib. QuTiP is only used by the tests, as an independent reference.

## A first example

A qubit pushes a trapped particle with a state-dependent force (a Stern–Gerlach interferometer). The two branches separate and the qubit loses its coherence; when the branches recombine at $\tau = 2\pi$, the coherence comes back:

```python
import numpy as np
from gbcs import Hamiltonian, Quantum_State

f_q, f_u = 1.5, 0.5                                   # qubit force and an unknown force
H = Hamiltonian(N_qubits=1, n_modes=1)
H.Initialize_Constant_Hamiltonians([np.eye(2)],       # H_m: harmonic trap
                                   [np.array([[f_u], [0.]]), np.array([[f_q], [0.]])],   # r_m, r_q
                                   np.array([0.0]))

state = Quantum_State(N_qubits=1, n_modes=1)
state.Initialize_Gaussian_State(np.zeros((2, 1)), np.eye(2), np.ones((2, 2))/2)   # vacuum, qubit in |+>
tau = np.linspace(0, 2*np.pi, 201)
state.Unitary_Dynamics_Numerical(H, tau)

contrast = np.exp(-state.C_JK_t[:, 1, 0])     # 1 at tau = 0 and 2 pi, ~0 in between
phase = state.phi_JK_t[:, 1, 0]               # 2 f_q f_u (tau - sin tau)
```

## Tutorials

| Notebook | Contents |
|---|---|
| [1. Unitary dynamics](tutorials/01_Unitary_Dynamics.ipynb) | The objects and conventions; a mode alone; linear coupling (Stern–Gerlach); quadratic coupling with the linear solver; Wigner functions |
| [2. Open dynamics](tutorials/02_Open_Dynamics.ipynb) | The `Decoherence` object; loss, drive, diffusion and qubit dephasing; dispersive readout with a parametric amplifier |
| [3. Measurements](tutorials/03_Measurements.ipynb) | Projective qubit measurements and the post-measurement Wigner functions; homodyne and heterodyne readout of a qubit |
| [4. Perturbation theory](tutorials/04_Perturbation_Theory.ipynb) | $T_1$ decay in the two examples of the paper, as quantum-jump trajectories; qubit drives; Monte Carlo sampling |

## Conventions

- **Quadratures:** ordered $\hat r = (\hat x_1, \hat p_1, \dots, \hat x_n, \hat p_n)$, with $\Omega = \bigoplus\begin{pmatrix}0&1\\-1&0\end{pmatrix}$ and the vacuum covariance $\sigma = \mathbb 1$.
- **Hamiltonian:** $\hat H/\hbar\omega = \sum_j \left(\tfrac12\hat r^{\mathsf T}H_j\hat r - r_j^{\mathsf T}\hat r + \tfrac12H^0_j\right)|j\rangle\langle j|$, with $H_j = H_m + \sum_i J_iH_{q,i}$ and $r_j = r_m + \sum_i J_i r_{q,i}$. Time is $\tau = \omega t$.
- **Branch indices:** branch $j$ is read in binary, with bit 1 meaning qubit label $+1$; for one qubit, index 0 is $-1$ and index 1 is $+1$. Arrays are indexed `[time, j, k, ...]`, and the qubit state is `rho_q_t[t, j, k]` $= \langle j|\hat\varrho^q(\tau)|k\rangle$.
- **Wigner normalisation:** as in the paper, each branch integrates to $2^n\varrho^q_{jk}$.

## Status

The constant-Hamiltonian solvers, the measurements and the perturbation theory are tested (`tests/`), against closed forms and against Lindblad simulations in QuTiP. These parts are not reliable yet:

- the solvers for time-dependent Hamiltonians coupled to the qubits (`Initialize_Time_Dep_Hamiltonian` with a qubit coupling) are not validated, and warn when used;
- `Symplectic_Known.Symplectic_QHO_N`, the closed-form symplectic maps for $n$ independent oscillators;
- `Measurements.Negativity_Qubits` splits the qubits into two equal halves, and the 3-D phase-space plots draw the branches of one qubit.

## Citation

If you use gbcs, please cite the paper:

```bibtex
@misc{braccini2025superpositions,
  title         = {Superpositions of Quantum Gaussian Processes},
  author        = {Braccini, Lorenzo and Bose, Sougato and Serafini, Alessio},
  year          = {2025},
  eprint        = {2510.01156},
  archivePrefix = {arXiv}
}
```

## License

MIT, see [LICENSE](LICENSE).
