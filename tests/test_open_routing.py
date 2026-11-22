"""Open dynamics of Hamiltonians without quadratic couplings: every term of Decoherence must reach the solver.

Before the fix, linear couplings ('Force') ignored the drive d and the qubit dephasing Gamma_z, dropped E whenever
one entry of E was zero (the test was E.all() == 0), and crashed for E != 0 otherwise; without qubits the drive was
ignored, and the Ladder basis was converted in single precision.
"""
import numpy as np
import pytest

from gbcs import Gaussian
from gbcs.Objects import Decoherence, Hamiltonian, Quantum_State

OMEGA = np.array([[0., 1.], [-1., 0.]])
LOSS_LADDER = np.array([[0., 1.], [0., 0.]])
LOSS_CANONICAL = 0.5*(np.eye(2) - 1j*OMEGA)


def test_ladder_and_canonical_loss_agree_to_double_precision():
    ladder = Decoherence(1, 1, 'Ladder', 0.7*LOSS_LADDER)
    canonical = Decoherence(1, 1, 'Canonical', 0.7*LOSS_CANONICAL)
    assert np.max(np.abs(ladder.E_mat - canonical.E_mat)) < 1e-15
    assert np.max(np.abs(ladder.D_mat - canonical.D_mat)) < 1e-15


def test_driven_lossy_mode_without_qubit_reaches_its_steady_state():
    """0 = Omega (H + E) r + Omega (d - r_m) at the steady state."""
    kappa, d = 0.8, np.array([[1.2], [0.4]])
    H = Hamiltonian(1, 1)
    H.Initialize_Constant_Hamiltonians([np.eye(2)], [np.array([[0.3], [0.0]])], np.array([0.0]))
    noise = Decoherence(1, 1, 'Ladder', kappa*LOSS_LADDER, d_vec=d)
    G = Quantum_State(1, 1)
    G.Initialize_Gaussian_State(np.zeros((2, 1)), np.eye(2), np.ones((1, 1)))
    G.Open_Dynamics_Numerical(H, noise, np.linspace(0, 40, 401))
    A = OMEGA @ (np.eye(2) + noise.E_mat)
    steady = -np.linalg.solve(A, OMEGA @ (d - np.array([[0.3], [0.0]])))
    assert np.allclose(G.r_t[-1], steady, atol=1e-7)


def test_dephasing_reaches_a_linear_coupling():
    H = Hamiltonian(1, 1)
    H.Initialize_Constant_Hamiltonians([np.eye(2)], [np.array([[0.5], [0.]]), np.array([[1.0], [0.]])], np.array([0.0]))
    t = np.linspace(0, 2*np.pi, 101)
    contrasts = []
    for Gamma_z in (0.0, 0.1):
        G = Quantum_State(1, 1)
        G.Initialize_Gaussian_State(np.zeros((2, 1)), np.eye(2), np.ones((2, 2))/2)
        G.Open_Dynamics_Numerical(H, Decoherence(1, 1, 'Canonical', 0.05*np.diag([1., 0.]), Gamma_z=np.array([Gamma_z])), t)
        contrasts.append(np.real(G.C_JK_t[:, 0, 1]))
    assert np.allclose(contrasts[1] - contrasts[0], 0.1*t, atol=1e-9)


@pytest.mark.parametrize('N_qubits', [1, 2])
def test_linear_coupling_with_loss_drive_and_dephasing_against_qutip(N_qubits):
    pytest.importorskip('qutip')
    import qutip_reference as Q
    rng = np.random.default_rng(N_qubits)
    H_m = np.eye(2)
    r_array = [0.2*rng.normal(size=(2, 1)) for _ in range(N_qubits + 1)]
    H_q_0 = 0.3*rng.normal(size=N_qubits)
    B, d, Gamma_z = 0.3*LOSS_CANONICAL, np.array([[0.4], [-0.2]]), 0.15*np.ones(N_qubits)
    A = rng.normal(size=(2**N_qubits,)*2) + 1j*rng.normal(size=(2**N_qubits,)*2)
    rho_q_0 = A @ A.conj().T
    rho_q_0 /= np.trace(rho_q_0)
    mode_states = [(0.0, 1.0, 0.4, -0.3)]
    t = np.linspace(0, 1.5, 7)

    H = Hamiltonian(N_qubits, 1)
    H.Initialize_Constant_Hamiltonians([H_m], r_array, H_q_0)
    assert H.type == ['Constant', 'Force']
    G = Quantum_State(N_qubits, 1)
    G.Initialize_Gaussian_State(np.array([[0.4], [-0.3]]), np.eye(2), rho_q_0)
    G.Open_Dynamics_Numerical(H, Decoherence(N_qubits, 1, 'Canonical', B, d_vec=d, Gamma_z=Gamma_z), t)

    zero = np.zeros((2, 2))
    rho, r, sigma, edge = Q.Reference_Dynamics(N_qubits, 1, 30, mode_states, [H_m] + [zero]*N_qubits, r_array, H_q_0,
                                               rho_q_0, t, B=B, d=d, Gamma_z=Gamma_z)
    assert edge < 1e-6
    weight = np.abs(rho)[..., None, None]
    assert np.max(np.abs(G.rho_q_t - rho)) < 1e-5
    assert np.max(weight*np.abs(G.r_JK_t - r)) < 1e-5
    assert np.max(weight*np.abs(G.sigma_JK_t - sigma)) < 1e-4
