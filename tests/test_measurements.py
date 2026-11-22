"""Measurements: projective qubit measurements and general-dyne measurements of the modes.

Every reference below is computed directly from the branch Gaussians, without the library's measurement code.
"""
import numpy as np
import pytest

from gbcs import Measurements, Wigner_Functions
from gbcs.Objects import Hamiltonian, Quantum_State

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]])
SIGMA_Z = np.diag([1.0, -1.0]).astype(complex)


def Evolved_State():
    """One qubit and one mode, at a time where the off-diagonal branch has complex sigma and r."""
    H = Hamiltonian(1, 1)
    H.Initialize_Constant_Hamiltonians([np.eye(2), 0.2*SIGMA_X.real], [np.array([[0.3], [0.]]), np.array([[1.0], [0.4]])],
                                       np.array([0.2]))
    G = Quantum_State(1, 1)
    G.Initialize_Gaussian_State(np.array([[0.2], [-0.1]]), np.diag([1.5, 0.8]),
                                np.array([[0.4, 0.3 - 0.2j], [0.3 + 0.2j, 0.6]]))
    G.Unitary_Dynamics_Numerical(H, np.linspace(0, 1.2, 41), solver='linear')
    return G


def Branch_Density(G, u, m, j, k):
    """rho_jk times the density of the quadrature u.r at m, from the branch Gaussian (variance u^T sigma u / 2)."""
    sigma, r = G.sigma_JK_t[-1, j, k], G.r_JK_t[-1, j, k, :, 0]
    mean, variance = u @ r, u @ sigma @ u/2
    return G.rho_q_t[-1, j, k]*np.exp(-(m - mean)**2/(2*variance))/np.sqrt(2*np.pi*variance)


################################## Projective qubit measurements ##################################

def test_sigma_y_on_its_eigenstate_is_certain():
    """Projectors used to be v v^T, which gave (0, 0) here."""
    v = np.array([1, 1j])/np.sqrt(2)
    outcomes, prob, projectors, post = Measurements.Qubit_Ideal_Measurament(1, SIGMA_Y, np.outer(v, v.conj()))
    assert np.allclose(outcomes, [-1, 1])
    assert np.allclose(prob, [0, 1])
    assert np.allclose(post[1], np.outer(v, v.conj()))
    assert np.allclose(post[0], 0)


def test_post_measurement_state_is_projected_on_both_sides():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2))
    rho = A @ A.conj().T
    rho /= np.trace(rho)
    _, prob, projectors, post = Measurements.Qubit_Ideal_Measurament(1, SIGMA_X, rho)
    for a in range(2):
        expected = projectors[a] @ rho @ projectors[a]/prob[a]
        assert np.allclose(post[a], expected)
        assert np.allclose(post[a], post[a].conj().T)
    assert np.isclose(prob.sum(), 1)


def test_degenerate_observable_has_one_projector_per_eigenvalue():
    """sigma_z on the first of two qubits: two outcomes with rank-2 projectors, not four rank-1 ones."""
    observable = np.kron(SIGMA_Z, np.eye(2))
    rho = np.eye(4)/4
    outcomes, prob, projectors, _ = Measurements.Qubit_Ideal_Measurament(2, observable, rho)
    assert np.allclose(outcomes, [-1, 1])
    assert np.allclose(prob, [0.5, 0.5])
    assert [int(round(np.trace(P).real)) for P in projectors] == [2, 2]


def test_time_resolved_measurement_matches_the_single_time_one():
    G = Evolved_State()
    G.Qubit_Ideal_Measurament(SIGMA_Y, 'Time')
    _, prob, _, post = Measurements.Qubit_Ideal_Measurament(1, SIGMA_Y, G.rho_q_t[-1])
    assert np.allclose(G.prob_t[:, -1], prob)
    assert np.allclose(G.rho_q_t_PMS_array[:, -1], post)


################################## Homodyne and heterodyne ##################################

@pytest.mark.parametrize('angle', [0.0, np.pi/2, 0.7, 2.3])
def test_homodyne_at_any_angle(angle):
    """
    Homodyne of x cos(angle) + p sin(angle).  Before the fix only angles 0 and pi/2 were right.
    The measurement covariance models homodyne with variances 1e-10 and 1e10; rotated off the axes that costs
    about 1e-7 relative, hence the tolerance.
    """
    G = Evolved_State()
    m = np.linspace(-8, 8, 801)
    G.Mode_Measurament(['Final', 'Homodyne'], 0, angle, 1.0, m)
    u = np.array([np.cos(angle), np.sin(angle)])
    density = sum(Branch_Density(G, u, m, j, j) for j in range(2)).real
    assert np.max(np.abs(np.real(G.Prob_Dis)/(m[1] - m[0]) - density)) < 1e-6
    k = 450
    expected = np.array([[Branch_Density(G, u, m[k], j, l) for l in range(2)] for j in range(2)])
    assert np.allclose(G.Qubit_PMS[:, :, k]/G.Prob_Dis[k], expected/np.trace(expected), atol=1e-6)


def test_homodyne_efficiency_adds_the_loss_noise():
    """Efficiency eta adds (1 - eta)/eta to the measured variance u^T sigma u / 2 ... times 2: sigma -> sigma + tan^2."""
    G = Evolved_State()
    m = np.linspace(-10, 10, 1001)
    eta = 0.6
    G.Mode_Measurament(['Final', 'Homodyne'], 0, 0.0, eta, m)
    u = np.array([1.0, 0.0])
    density = 0
    for j in range(2):
        sigma = G.sigma_JK_t[-1, j, j] + (1 - eta)/eta*np.eye(2)
        mean, variance = u @ G.r_JK_t[-1, j, j, :, 0], u @ sigma @ u/2
        density = density + G.rho_q_t[-1, j, j]*np.exp(-(m - mean)**2/(2*variance))/np.sqrt(2*np.pi*variance)
    assert np.max(np.abs(np.real(G.Prob_Dis)/(m[1] - m[0]) - density.real)) < 1e-8


def test_heterodyne_takes_two_dimensional_outcomes():
    """Heterodyne outcomes are (x, p) pairs, with density N(r, (sigma + 1)/2) for an ideal detector."""
    G = Evolved_State()
    x = np.linspace(-7, 7, 71)
    X, P = np.meshgrid(x, x)
    outcomes = np.stack([X.ravel(), P.ravel()], axis=1)
    G.Mode_Measurament(['Final', 'Heterodyne'], 0, 0.0, 1.0, outcomes)
    density = 0
    for j in range(2):
        covariance = (G.sigma_JK_t[-1, j, j].real + np.eye(2))/2
        delta = outcomes - G.r_JK_t[-1, j, j, :, 0].real
        exponent = np.einsum('ka,ab,kb->k', delta, np.linalg.inv(covariance), delta)
        density = density + G.rho_q_t[-1, j, j].real*np.exp(-exponent/2)/(2*np.pi*np.sqrt(np.linalg.det(covariance)))
    area = (x[1] - x[0])**2
    assert np.max(np.abs(np.real(G.Prob_Dis)/area - density)) < 1e-4
    with pytest.raises(ValueError, match=r'\(K, 2\)'):
        G.Mode_Measurament(['Final', 'Heterodyne'], 0, 0.0, 1.0, x)


def test_quadrature_shortcut_is_the_general_homodyne():
    G = Evolved_State()
    m = np.linspace(-6, 6, 121)
    G.Homdyne_Mode_Measurament('Final', 0, 0.0, 'Momentum', 0.8, m)
    shortcut = G.Prob_Dis.copy()
    G.Mode_Measurament(['Final', 'Homodyne'], 0, np.pi/2, 0.8, m)
    assert np.allclose(shortcut, G.Prob_Dis)


def test_time_resolved_mode_measurement_keeps_the_complex_off_diagonal():
    G = Evolved_State()
    m = np.linspace(-6, 6, 61)
    G.Mode_Measurament(['Time', 'Homodyne'], 0, 0.0, 1.0, m)
    assert np.iscomplexobj(G.Qubit_PMS_t)
    assert np.max(np.abs(np.imag(G.Qubit_PMS_t[-1, 0, 1]))) > 1e-6
    G.Mode_Measurament(['Final', 'Homodyne'], 0, 0.0, 1.0, m)
    assert np.allclose(G.Qubit_PMS_t[-1], G.Qubit_PMS)


################################## Post-measurement Wigner functions ##################################

def test_post_measurement_wigner_is_normalised_for_a_complex_projector():
    """Tr_q[Pi W] / p integrates to 2^n (the Wigner_Functions normalisation) for sigma_y, whose projectors are complex."""
    G = Evolved_State()
    G.Qubit_Ideal_Measurament(SIGMA_Y, 'Time')
    x = np.linspace(-9, 9, 121)
    X, P = np.meshgrid(x, x)
    grid = np.stack([X, P], -1)[..., None]
    W = Wigner_Functions.Wigner_PMS_func(G, len(G.t_array) - 1, grid)
    area = (x[1] - x[0])**2
    assert np.allclose(W.sum(axis=(1, 2))*area, 2, atol=1e-6)


def test_root_det_is_continuous_for_two_modes():
    """Four eigenvalues near arg 0.9 each: np.sqrt(det) takes the other branch, the product of roots does not."""
    sigma = np.diag(np.exp(0.9j*np.ones(4)))
    assert np.isclose(Wigner_Functions.Root_Det(sigma), np.exp(1.8j))
    assert np.isclose(np.sqrt(np.linalg.det(sigma)), -np.exp(1.8j))
