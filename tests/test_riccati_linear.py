"""Riccati_Linear against the ODE solvers, the paper's closed forms and QuTiP.

Where a paper formula disagrees with QuTiP the test keeps the corrected form, says what
differs, and test_*_as_typed_is_wrong records that the typed form really does fail.
Line numbers refer to manuscript/apssamp-3.tex.

The ODE tolerances are not arbitrary: on a fine grid the ODE solvers saturate at about
3e-10 in sigma and r, so that is what linear-vs-ODE can resolve.
"""
import re

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import Gaussian
import Open_Operator_Gaussian
import Riccati_Linear
import Unitary_Operator_Gaussian
from Objects import Decoherence, Hamiltonian, Quantum_State
from Riccati_Linear import Doubling_Matrix, Dynamics_Linear

SIGMA_X = np.array([[0., 1.], [1., 0.]])
ZERO_FORCE = np.zeros((2, 1))


def Symmetric(rng, dim, scale):
    M = rng.normal(size=(dim, dim))
    return scale*(M + M.T)/2


def Problem(N_qubits, n_modes, seed=1, open_system=False):
    """Generic parameters with every term of Tab. 3 switched on."""
    rng = np.random.default_rng(seed)
    dim = 2*n_modes
    H_m = np.eye(dim) + Symmetric(rng, dim, 0.15)
    H_array = [H_m] + [Symmetric(rng, dim, 0.2) for _ in range(N_qubits)]
    r_array = [0.3*rng.normal(size=(dim, 1)) for _ in range(N_qubits + 1)]
    H_q_0 = 0.4*rng.normal(size=N_qubits)

    S = np.eye(dim) + Symmetric(rng, dim, 0.15)
    sigma_0 = (S @ S.T).astype(complex)
    r_0 = 0.5*rng.normal(size=(dim, 1)).astype(complex)
    psi = rng.normal(size=2**N_qubits) + 1j*rng.normal(size=2**N_qubits)
    psi /= np.linalg.norm(psi)
    rho_q_0 = np.outer(psi, psi.conj())

    kwargs = {}
    if open_system:
        L = rng.normal(size=(dim, dim)) + 1j*rng.normal(size=(dim, dim))
        E, D = Gaussian.Get_Decoherence_Rates(n_modes, 0.05*L @ L.conj().T)
        kwargs = dict(E=E, D=D, d=0.3*rng.normal(size=(dim, 1)), Gamma_z=0.1*rng.uniform(size=N_qubits))
    return (r_0, sigma_0, H_array, r_array, H_q_0, rho_q_0), kwargs


def ODE_Solver(N_qubits, n_modes, problem, kwargs, t):
    r_0, sigma_0, H_array, r_array, H_q_0, rho_q_0 = problem
    if kwargs:
        return Open_Operator_Gaussian.Dynamics_Numerical(N_qubits, n_modes, r_0, sigma_0, H_array, r_array, H_q_0, rho_q_0,
                                                         kwargs['E'], kwargs['D'], t, d=kwargs['d'], Gamma_z=kwargs['Gamma_z'])
    return Unitary_Operator_Gaussian.Dynamics_Numerical(N_qubits, n_modes, r_0, sigma_0, H_array, r_array, H_q_0, rho_q_0, t)


def Max_Errors(a, b):
    return [np.max(np.abs(x - y)) for x, y in zip(a, b)]


################################## Algebra ##################################

def test_doubling_blocks():
    (_, _, H_array, _, _, _), kwargs = Problem(1, 2, open_system=True)
    H_J, H_K = H_array[0] + H_array[1], H_array[0] - H_array[1]
    E, D = kwargs['E'], kwargs['D']
    Omega = Gaussian.Omega_N(2)
    A = Doubling_Matrix(2, H_J, H_K, E, D)
    a, b, c, dd = A[:4, :4], A[:4, 4:], A[4:, :4], A[4:, 4:]
    assert np.max(np.abs(a.T + dd)) < 1e-14
    assert np.max(np.abs(b - b.T)) < 1e-14
    assert np.max(np.abs(c - c.T)) < 1e-14
    assert np.max(np.abs(c - (0.5j*Omega @ (H_J - H_K) @ Omega.T + D))) < 1e-14


def test_D_sits_outside_the_half():
    """App. app:doubling (l. 1584) as typed puts D inside the 1/2 of the lower-left block."""
    (r_0, sigma_0, H_array, r_array, H_q_0, rho_q_0), kwargs = Problem(1, 1, open_system=True)
    sigma_ode = ODE_Solver(1, 1, (r_0, sigma_0, H_array, r_array, H_q_0, rho_q_0), kwargs, np.linspace(0, 1.5, 301))[0][-1, 0, 1]
    H_J, H_K = H_array[0] - H_array[1], H_array[0] + H_array[1]

    A = Doubling_Matrix(1, H_J, H_K, kwargs['E'], kwargs['D'])
    P = expm(1.5*A)
    sigma = (P[2:, :2] + P[2:, 2:] @ sigma_0) @ np.linalg.inv(P[:2, :2] + P[:2, 2:] @ sigma_0)
    assert np.max(np.abs(sigma - sigma_ode)) < 1e-8

    A_typed = A.copy()
    A_typed[2:, :2] -= 0.5*kwargs['D']
    P = expm(1.5*A_typed)
    sigma_typed = (P[2:, :2] + P[2:, 2:] @ sigma_0) @ np.linalg.inv(P[:2, :2] + P[:2, 2:] @ sigma_0)
    assert np.max(np.abs(sigma_typed - sigma_ode)) > 1e-3


################################## Linear against the ODEs ##################################

@pytest.mark.parametrize('open_system', [False, True], ids=['unitary', 'open'])
@pytest.mark.parametrize('N_qubits, n_modes', [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1)])
def test_linear_matches_ode(N_qubits, n_modes, open_system):
    problem, kwargs = Problem(N_qubits, n_modes, open_system=open_system)
    t = np.linspace(0, 3, 601)
    ode = ODE_Solver(N_qubits, n_modes, problem, kwargs, t)
    linear = Dynamics_Linear(N_qubits, n_modes, *problem, t, **kwargs)
    for quantity_ode, quantity_linear in zip(ode, linear):
        assert quantity_ode.shape == quantity_linear.shape
    assert max(Max_Errors(ode, linear)) < 2e-8


def Joint_ODE(N_qubits, n_modes, problem, kwargs, t, rtol=1e-13, atol=1e-14):
    """
    sigma, r and s integrated together, one branch pair at a time.  The library's nested ODEs
    feed sigma and r through interp1d on t_eval, which on this grid (largest gap 0.05) limits
    them to 2.4e-7, so they cannot referee a solver that is good to 1e-12.
    """
    r_0, sigma_0, H_array, r_array, H_q_0, _ = problem
    dim = 2*n_modes
    Omega = Gaussian.Omega_N(n_modes)
    E, D = kwargs.get('E', np.zeros((dim, dim))), kwargs.get('D', np.zeros((dim, dim)))
    d = np.ravel(kwargs.get('d', np.zeros(dim)))
    Gamma_z = kwargs.get('Gamma_z', np.zeros(N_qubits))
    N2 = 2**N_qubits
    sigma_t = np.zeros((len(t), N2, N2, dim, dim), dtype=complex)
    r_t = np.zeros((len(t), N2, N2, dim, 1), dtype=complex)
    s_t = np.zeros((len(t), N2, N2), dtype=complex)

    for j in range(N2):
        J = Riccati_Linear.QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, j)
        H_J, r_J = Riccati_Linear.Compute_H_J(N_qubits, J, H_array), np.ravel(Riccati_Linear.Compute_r_J(N_qubits, J, r_array))
        for k in range(j, N2):
            K = Riccati_Linear.QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, k)
            H_K, r_K = Riccati_Linear.Compute_H_J(N_qubits, K, H_array), np.ravel(Riccati_Linear.Compute_r_J(N_qubits, K, r_array))
            H_sum, H_diff, r_sum, r_diff = H_J + H_K, H_J - H_K, r_J + r_K, r_J - r_K
            c_0 = -0.5j*(J - K) @ H_q_0 + 0.5*np.sum(Gamma_z*(J*K - 1))

            def Derivative(_, z):
                sigma, r = z[:dim**2].reshape(dim, dim), z[dim**2:-1]
                sigma_dot = (0.5*(Omega @ (H_sum + 2*E) @ sigma - sigma @ (H_sum + 2*E.T) @ Omega
                                  - 1j*sigma @ H_diff @ sigma - 1j*Omega @ H_diff @ Omega) + D)
                r_dot = 0.5*((Omega @ (H_sum + 2*E) - 1j*sigma @ H_diff) @ r + Omega @ (2*d - r_sum) + 1j*sigma @ r_diff)
                s_dot = -0.5j*r @ H_diff @ r + 1j*r_diff @ r - 0.25j*np.trace(H_diff @ sigma) + c_0
                return np.concatenate((sigma_dot.ravel(), r_dot, [s_dot]))

            y_0 = np.concatenate((sigma_0.ravel(), np.ravel(r_0), [0]))
            y = solve_ivp(Derivative, (t[0], t[-1]), y_0, t_eval=t, method='DOP853', rtol=rtol, atol=atol).y.T
            sigma_t[:, j, k] = y[:, :dim**2].reshape(-1, dim, dim)
            r_t[:, j, k, :, 0] = y[:, dim**2:-1]
            s_t[:, j, k] = y[:, -1]
            sigma_t[:, k, j], r_t[:, k, j], s_t[:, k, j] = sigma_t[:, j, k].conj(), r_t[:, j, k].conj(), s_t[:, j, k].conj()

    return sigma_t, r_t, -np.real(s_t), np.imag(s_t)


@pytest.mark.parametrize('open_system', [False, True], ids=['unitary', 'open'])
def test_non_uniform_grid(open_system):
    problem, kwargs = Problem(2, 1, seed=4, open_system=open_system)
    t = 3*np.sort(np.random.default_rng(0).uniform(size=400))**2
    t[0] = 0
    linear = Dynamics_Linear(2, 1, *problem, t, **kwargs)
    assert max(Max_Errors(Joint_ODE(2, 1, problem, kwargs, t), linear)) < 1e-10


def test_grid_independence():
    """The linear result at a given time must not depend on the grid used to reach it.
    Relative, because this noise has an unstable mode and sigma reaches 190.
    """
    problem, kwargs = Problem(2, 2, seed=2, open_system=True)
    t_fine = np.linspace(0, 10, 2001)
    fine = Dynamics_Linear(2, 2, *problem, t_fine, **kwargs)
    coarse = Dynamics_Linear(2, 2, *problem, t_fine[::500], **kwargs)
    for quantity_fine, quantity_coarse in zip(fine, coarse):
        assert np.max(np.abs(quantity_fine[::500] - quantity_coarse)) < 1e-12*np.max(np.abs(quantity_fine))


def test_per_branch_initial_conditions_compose():
    """[0, 1] then [1, 2] from the per-branch state at t = 1 must equal [0, 2] in one go."""
    problem, kwargs = Problem(2, 1, seed=3, open_system=True)
    r_0, sigma_0, H_array, r_array, H_q_0, rho_q_0 = problem
    t = np.linspace(0, 2, 201)
    sigma, r, C, phi = Dynamics_Linear(2, 1, *problem, t, **kwargs)

    restart = (r[100], sigma[100], H_array, r_array, H_q_0, rho_q_0)
    sigma_2, r_2, C_2, phi_2 = Dynamics_Linear(2, 1, *restart, t[100:], **kwargs)
    assert np.max(np.abs(sigma_2 - sigma[100:])) < 1e-11
    assert np.max(np.abs(r_2 - r[100:])) < 1e-11
    assert np.max(np.abs(C[100] + C_2 - C[100:])) < 1e-11
    assert np.max(np.abs(phi[100] + phi_2 - phi[100:])) < 1e-11


def test_per_branch_initial_conditions_match_ode():
    problem, kwargs = Problem(1, 1, seed=5, open_system=True)
    r_0, sigma_0, H_array, r_array, H_q_0, rho_q_0 = problem
    t = np.linspace(0, 1, 101)
    sigma, r, _, _ = Dynamics_Linear(1, 1, *problem, t, **kwargs)
    restart = (r[-1], sigma[-1], H_array, r_array, H_q_0, rho_q_0)
    t_2 = np.linspace(0, 2, 401)
    assert max(Max_Errors(ODE_Solver(1, 1, restart, kwargs, t_2), Dynamics_Linear(1, 1, *restart, t_2, **kwargs))) < 2e-8


def test_conjugation_convention():
    problem, kwargs = Problem(2, 1, open_system=True)
    sigma, r, C, phi = Dynamics_Linear(2, 1, *problem, np.linspace(0, 1, 11), **kwargs)
    assert np.array_equal(sigma, np.conj(np.swapaxes(sigma, 1, 2)))
    assert np.array_equal(r, np.conj(np.swapaxes(r, 1, 2)))
    assert np.array_equal(C, np.swapaxes(C, 1, 2))
    # Not array_equal: phi_jj is round-off, and phi_kj = -phi_jk flips its sign as in the ODE solvers
    assert np.max(np.abs(phi + np.swapaxes(phi, 1, 2))) < 1e-14
    assert np.array_equal(sigma, np.swapaxes(sigma, 3, 4))


def test_diagonal_branches_keep_their_norm():
    problem, kwargs = Problem(2, 1, open_system=True)
    _, _, C, phi = Dynamics_Linear(2, 1, *problem, np.linspace(0, 5, 51), **kwargs)
    diagonal = np.arange(4)
    assert np.max(np.abs(C[:, diagonal, diagonal])) < 1e-12
    assert np.max(np.abs(phi[:, diagonal, diagonal])) < 1e-12


def test_zero_qrdm_elements_are_allowed():
    """The ODE solvers need log(rho_q_0) and raise here; C and phi do not depend on rho_q_0."""
    problem, _ = Problem(1, 1)
    r_0, sigma_0, H_array, r_array, H_q_0, _ = problem
    t = np.linspace(0, 1, 11)
    ground = np.array([[0, 0], [0, 1]], dtype=complex)
    pure = np.ones((2, 2), dtype=complex)/2
    a = Dynamics_Linear(1, 1, r_0, sigma_0, H_array, r_array, H_q_0, ground, t)
    b = Dynamics_Linear(1, 1, r_0, sigma_0, H_array, r_array, H_q_0, pure, t)
    assert max(Max_Errors(a, b)) == 0


################################## Numerics of the step map ##################################

def Count_Exponentials(monkeypatch):
    calls = []
    original = Riccati_Linear.linalg.expm
    monkeypatch.setattr(Riccati_Linear.linalg, 'expm', lambda M: calls.append(1) or original(M))
    return calls


def test_uniform_grid_needs_one_propagator(monkeypatch):
    problem, _ = Problem(1, 1)
    calls = Count_Exponentials(monkeypatch)
    Dynamics_Linear(1, 1, *problem, np.linspace(0, 1, 1001))
    # Three branch pairs, each with the step and the midpoint of Accept_Step
    assert len(calls) == 2*3


def test_non_uniform_grid_reuses_repeated_steps(monkeypatch):
    problem, _ = Problem(1, 1)
    calls = Count_Exponentials(monkeypatch)
    t = np.concatenate((np.linspace(0, 1, 101), 1 + 0.5*np.arange(1, 5)))
    Dynamics_Linear(1, 1, *problem, t)
    assert len(calls) == 2*2*3


def test_steps_are_subdivided():
    H_array = [50*np.eye(2), 5*np.eye(2)]
    assert Riccati_Linear.Base_Level(Doubling_Matrix(1, H_array[0] + H_array[1], H_array[0] - H_array[1]), 1.0) == 7


def Continuous_Log(z):
    return np.log(np.abs(z)) + 1j*np.unwrap(np.angle(z))


def Rotation_Exponent(theta, s, x_0, p_0):
    """App. l. 1692 with the two logarithms kept separate and continuous along theta."""
    z_1, z_2 = s*np.cos(theta) + 1j*np.sin(theta), np.cos(theta) + 1j*s*np.sin(theta)
    return -1j*np.sin(theta)*(p_0**2/z_2 + s*x_0**2/z_1) - 0.5*(Continuous_Log(z_1) + Continuous_Log(z_2) - np.log(s))


@pytest.mark.parametrize('stride', [1, 40, 400], ids=['fine', 'coarse', 'very-coarse'])
def test_log_det_stays_continuous_through_many_windings(stride):
    """theta_Delta reaches 60, about 19 pi: a sign error in rho_q would show up here."""
    s, x_0, p_0, theta_delta = 2.5, 0.6, -0.4, 1.0
    t = np.linspace(0, 60, 24001)
    sigma_0 = np.diag([1/s, s]).astype(complex)
    r_0 = np.array([[x_0], [p_0]], dtype=complex)
    H_array = [np.zeros((2, 2)), theta_delta*np.eye(2)]
    _, _, C, phi = Dynamics_Linear(1, 1, r_0, sigma_0, H_array, [ZERO_FORCE, ZERO_FORCE], np.zeros(1), np.ones((2, 2)), t[::stride])
    exact = Rotation_Exponent(theta_delta*t, s, x_0, p_0)[::stride]
    assert np.max(np.abs(-C[:, 1, 0] + 1j*phi[:, 1, 0] - exact)) < 1e-10


@pytest.mark.parametrize('n_modes', [1, 2, 3, 10])
def test_near_pole_is_resolved(n_modes):
    """X = (cos(t) + i (eps + i) sin(t)) 1 passes within about eps of zero at t = pi/4, off the
    real axis.  Stepping over it would lose a phase of pi per eigenvalue of X.

    With enough modes the phases of the eigenvalues add up past pi within one accepted step, even
    with each of them small, so the log of det X would be wrong: the logs must be summed one by one.
    """
    eps = 0.02
    dim = 2*n_modes
    t = np.linspace(0, 3, 13)
    fine = np.linspace(0, 3, 1200001)
    exact = -n_modes*Continuous_Log(np.cos(fine) + 1j*(eps + 1j)*np.sin(fine))[::100000]
    sigma_0_jk = np.broadcast_to(np.eye(dim, dtype=complex), (2, 2, dim, dim)).copy()
    sigma_0_jk[0, 1] = (eps + 1j)*np.eye(dim)
    r_0_jk = np.zeros((2, 2, dim, 1), dtype=complex)
    force = np.zeros((dim, 1))
    _, _, C, phi = Dynamics_Linear(1, n_modes, r_0_jk, sigma_0_jk, [np.zeros((dim, dim)), -np.eye(dim)], [force, force], np.zeros(1),
                                   np.ones((2, 2)), t)
    assert np.max(np.abs(-C[:, 0, 1] + 1j*phi[:, 0, 1] - exact)) < 1e-8


def test_phases_of_many_eigenvalues_are_added_one_by_one():
    """X = exp(i t) 1 for vacuum, so s = -i n t.  With 12 modes each step of 0.135 turns det X by
    3.24 rad, past pi, while every one of the 24 eigenvalues is well within the ball.  The step is
    tuned: it has to be one Accept_Step takes without halving, and few are that large.
    """
    n_modes, dt = 12, 0.135
    dim = 2*n_modes
    A = Doubling_Matrix(n_modes, np.eye(dim), -np.eye(dim))
    M = Riccati_Linear.Step_Generator(A, np.zeros(dim, dtype=complex), np.zeros(dim, dtype=complex))
    propagators = Riccati_Linear.Step_Propagators(M, dt)
    sigma_1 = np.eye(dim, dtype=complex)
    assert Riccati_Linear.Accept_Step(propagators, 0, sigma_1, propagators(0)[1] + propagators(0)[2])

    force = np.zeros((dim, 1))
    t = dt*np.arange(11)
    _, _, C, phi = Dynamics_Linear(1, n_modes, np.zeros((dim, 1), dtype=complex), np.eye(dim, dtype=complex), [np.zeros((dim, dim)), np.eye(dim)],
                                   [force, force], np.zeros(1), np.ones((2, 2)), t)
    assert np.max(np.abs(C[:, 1, 0])) < 1e-12
    assert np.max(np.abs(phi[:, 1, 0] + n_modes*t)) < 1e-11


def test_eigenvalues_exchanging_around_zero():
    """Reproducer from review.  Over this one step X is non-normal and its two eigenvalues swap
    around 0 while det X winds by 2 pi, yet both end with |arg| < pi/2.  Checking those end points
    gave exp(s) with the wrong sign on the 2-point grid and no warning.
    """
    H_j = np.array([[1.0993, 0.9102], [0.9102, -1.4773]])
    H_k = np.array([[1.2505, 0.3759], [0.3759, 0.9604]])
    H_array = [(H_j + H_k)/2, (H_k - H_j)/2]
    sigma = np.array([[0.009 - 129.494j, -0.001 - 51.433j], [-0.001 - 51.433j, 0.011 - 24.39j]])
    sigma_0_jk = np.broadcast_to(np.eye(2, dtype=complex), (2, 2, 2, 2)).copy()
    sigma_0_jk[0, 1] = sigma
    r_0 = np.zeros((2, 1), dtype=complex)
    r_0_jk = np.broadcast_to(r_0, (2, 2, 2, 1))
    forces = [ZERO_FORCE, ZERO_FORCE]
    tau = 0.3898
    reference = Joint_ODE(1, 1, (r_0, sigma, H_array, forces, np.zeros(1), None), {}, np.array([0, tau]))
    exact = np.exp(-reference[2][-1, 0, 1] + 1j*reference[3][-1, 0, 1])
    assert np.real(exact) < 0
    for points in (2, 3, 5, 9, 17, 65):
        _, _, C, phi = Dynamics_Linear(1, 1, r_0_jk, sigma_0_jk, H_array, forces, np.zeros(1), np.ones((2, 2)), np.linspace(0, tau, points))
        assert abs(np.exp(-C[-1, 0, 1] + 1j*phi[-1, 0, 1]) - exact) < 1e-9


@pytest.mark.parametrize('points', [3, 13, 121], ids=['coarse', 'medium', 'fine'])
def test_non_normal_X_near_poles(points):
    """Two modes rotate at different rates, each passing near a pole of its own, so X is not a
    multiple of 1.  A two-mode squeezer S then makes it non-normal: sigma -> S sigma S^T and
    H -> S^-T H S^-1 give X -> S^-T X S^T, which has the same det, so s stays in closed form.
    """
    eps, rates, squeezing = 0.03, np.array([1.0, 1.7]), 0.8
    ch, sh = np.cosh(squeezing), np.sinh(squeezing)
    S = np.array([[ch, 0, sh, 0], [0, ch, 0, -sh], [sh, 0, ch, 0], [0, -sh, 0, ch]])
    S_inverse = np.linalg.inv(S)
    H_q = S_inverse.T @ np.diag(np.repeat(-rates, 2)) @ S_inverse
    sigma = S @ np.diag(np.repeat((eps + 1j)*np.ones(2), 2)) @ S.T
    assert np.max(np.abs(H_q @ sigma - (H_q @ sigma).conj().T*0 - sigma @ H_q)) > 0.1

    sigma_0_jk = np.broadcast_to(np.eye(4, dtype=complex), (2, 2, 4, 4)).copy()
    sigma_0_jk[0, 1] = sigma
    force = np.zeros((4, 1))
    t = np.linspace(0, 3, points)
    _, _, C, phi = Dynamics_Linear(1, 2, np.zeros((2, 2, 4, 1), dtype=complex), sigma_0_jk, [np.zeros((4, 4)), H_q], [force, force],
                                   np.zeros(1), np.ones((2, 2)), t)

    fine = np.linspace(0, 3, 1200001)
    exact = -sum(Continuous_Log(np.cos(rate*fine) + 1j*(eps + 1j)*np.sin(rate*fine)) for rate in rates)
    assert np.max(np.abs(-C[:, 0, 1] + 1j*phi[:, 0, 1] - exact[::1200000//(points - 1)])) < 1e-7


@pytest.mark.parametrize('seed', range(40))
def test_accepted_steps_stay_near_one_on_the_whole_path(seed):
    """The guarantee behind Accept_Step: if a step is accepted, ||X(t) - 1|| < 1 for all t in it."""
    rng = np.random.default_rng(seed)
    (_, _, H_array, _, _, _), _ = Problem(1, 2, seed=seed)
    A = Doubling_Matrix(2, H_array[0] + H_array[1], H_array[0] - H_array[1])
    M = Riccati_Linear.Step_Generator(A, np.zeros(4, dtype=complex), np.zeros(4, dtype=complex))
    sigma = Symmetric(rng, 4, 10.0**rng.uniform(-1, 4)) + 1j*Symmetric(rng, 4, 10.0**rng.uniform(-1, 4))
    propagators = Riccati_Linear.Step_Propagators(M, rng.uniform(0.2, 2))
    level = next(level for level in range(propagators.base_level, propagators.base_level + 60)
                 if Riccati_Linear.Accept_Step(propagators, level, sigma, propagators(level)[1] + propagators(level)[2] @ sigma))
    for tau in np.linspace(0, 1, 201)*propagators(level)[0]:
        P = expm(A*tau)
        assert np.linalg.norm(P[:4, :4] + P[:4, 4:] @ sigma - np.eye(4)) < 1


def Rotation_Propagators(tau, sigma):
    """X(t) = cos(t) 1 + i sin(t) sigma, which returns to 1 at t = 2 pi k while leaving the ball on the way."""
    A = Doubling_Matrix(1, np.eye(2), -np.eye(2))
    M = Riccati_Linear.Step_Generator(A, np.zeros(2, dtype=complex), np.zeros(2, dtype=complex))
    propagators = Riccati_Linear.Step_Propagators(M, tau)
    return propagators, propagators(0)[1] + propagators(0)[2] @ sigma


def test_remainder_guard_alone_rejects_a_path_that_leaves_and_returns():
    """At t = 4 pi both samples, h/2 and h, sit exactly on X = 1: only the remainder can object."""
    sigma = np.eye(2, dtype=complex)*0.5
    propagators, X = Rotation_Propagators(4*np.pi, sigma)
    assert np.linalg.norm(X - np.eye(2)) < 1e-12
    assert np.linalg.norm(propagators(1)[1] + propagators(1)[2] @ sigma - np.eye(2)) < 1e-12
    assert not Riccati_Linear.Accept_Step(propagators, 0, sigma, X)


def test_midpoint_guard_alone_rejects_a_path_that_returns_at_the_end(monkeypatch):
    """At t = 2 pi the end point is X = 1 and only the midpoint X = -1 shows the excursion."""
    monkeypatch.setattr(Riccati_Linear, 'MAX_INTERPOLATION_ERROR', np.inf)
    sigma = np.eye(2, dtype=complex)*0.5
    propagators, X = Rotation_Propagators(2*np.pi, sigma)
    assert np.linalg.norm(X - np.eye(2)) < 1e-12
    assert not Riccati_Linear.Accept_Step(propagators, 0, sigma, X)


def test_remainder_guard_counts_sigma():
    """A tiny coupling and a sigma of about 1e3 tuned so that X(h) = 1 exactly.  The samples look
    perfect, but with sigma that large they say nothing about the path between them.
    """
    A = Doubling_Matrix(1, np.eye(2) + 1e-3*np.diag([1.0, 2.0]), np.eye(2) - 1e-3*np.diag([1.0, 2.0]))
    M = Riccati_Linear.Step_Generator(A, np.zeros(2, dtype=complex), np.zeros(2, dtype=complex))
    propagators = Riccati_Linear.Step_Propagators(M, 0.4)
    _, S_11, S_12, *_ = propagators(0)
    sigma = np.linalg.solve(S_12, np.eye(2) - S_11)
    X = S_11 + S_12 @ sigma
    _, S_11_half, S_12_half, *_ = propagators(1)
    assert np.linalg.norm(sigma) > 100
    assert np.linalg.norm(X - np.eye(2)) < 1e-9
    assert np.linalg.norm(S_11_half + S_12_half @ sigma - np.eye(2)) < Riccati_Linear.MAX_X_DISTANCE
    assert not Riccati_Linear.Accept_Step(propagators, 0, sigma, X)


def test_observed_rows_follow_the_coupling():
    """H_q = diag(1, 0) only touches x directly, but H_m rotates p into it, so both are observed."""
    H_m, H_q = np.eye(2), np.diag([1.0, 0.0])
    assert Riccati_Linear.Observed_Rows(Doubling_Matrix(1, H_m + H_q, H_m - H_q))[0].shape == (2, 2)
    assert Riccati_Linear.Observed_Rows(Doubling_Matrix(1, H_q, -H_q))[0].shape == (1, 2)
    assert Riccati_Linear.Observed_Rows(Doubling_Matrix(1, H_m, H_m))[0].shape == (0, 2)


def test_small_genuine_coupling_is_observed():
    """A coupling of 1e-8 relative to H_m is physics, and must not be taken for round-off."""
    H_m, H_q = np.eye(2), 1e-8*np.eye(2)
    rows, dropped = Riccati_Linear.Observed_Rows(Doubling_Matrix(1, H_m + H_q, H_m - H_q))
    assert rows.shape == (2, 2)
    assert dropped == 0


def test_round_off_rows_are_not_taken_for_coupling():
    """A rotated frame leaves singular values of ~1e-16 for the uncoupled mode: they are not coupling,
    but neither are they zero, so they are reported to be counted against sigma.
    """
    H_m, H_q, _ = Rotated_Problem(np.eye(2))
    rows, dropped = Riccati_Linear.Observed_Rows(Doubling_Matrix(2, H_m + H_q, H_m - H_q))
    assert rows.shape == (2, 4)
    assert 0 < dropped < Riccati_Linear.OBSERVED_TOLERANCE


def test_remainder_counts_the_identity():
    """No coupling, so Q is empty, but X = exp(a t) still rotates.  At h = 4 pi both samples sit on X = 1
    and the whole objection comes from the 1 in ||(1, Q sigma)||.
    """
    A = Doubling_Matrix(1, np.eye(2), np.eye(2))
    M = Riccati_Linear.Step_Generator(A, np.zeros(2, dtype=complex), np.zeros(2, dtype=complex))
    propagators = Riccati_Linear.Step_Propagators(M, 4*np.pi)
    assert propagators.observed.shape == (0, 2)
    assert not Riccati_Linear.Accept_Step(propagators, 0, np.eye(2, dtype=complex), propagators(0)[1])


def test_dropped_rows_are_counted_in_quadrature():
    """size = sqrt(dim + ||Q sigma||^2 + (dropped ||sigma||)^2): with the dropped term unsquared this step passes."""
    H_q = np.diag([0.0, 0.0, 1.0, 1.0])
    A = Doubling_Matrix(2, np.eye(4) + H_q, np.eye(4) - H_q)
    M = Riccati_Linear.Step_Generator(A, np.zeros(4, dtype=complex), np.zeros(4, dtype=complex))
    propagators = Riccati_Linear.Step_Propagators(M, 0.05)
    sigma = np.diag([1e12, 1e12, 1.0, 1.0]).astype(complex)
    X = propagators(0)[1] + propagators(0)[2] @ sigma
    remainder = propagators(0)[9]
    propagators.dropped = 2*Riccati_Linear.MAX_INTERPOLATION_ERROR/remainder/np.linalg.norm(sigma)
    assert not Riccati_Linear.Accept_Step(propagators, 0, sigma, X)


def test_dropped_rows_are_counted_in_the_remainder():
    """A sigma that Q does not see still counts through the rows dropped from Q."""
    H_q = np.diag([0.0, 0.0, 1.0, 1.0])
    A = Doubling_Matrix(2, np.eye(4) + H_q, np.eye(4) - H_q)
    M = Riccati_Linear.Step_Generator(A, np.zeros(4, dtype=complex), np.zeros(4, dtype=complex))
    propagators = Riccati_Linear.Step_Propagators(M, 0.05)
    sigma = np.diag([1e12, 1e12, 1.0, 1.0]).astype(complex)
    X = propagators(0)[1] + propagators(0)[2] @ sigma
    assert Riccati_Linear.Accept_Step(propagators, 0, sigma, X)
    propagators.dropped = 1e-3
    assert not Riccati_Linear.Accept_Step(propagators, 0, sigma, X)


def test_acceptance_constants_keep_the_proof():
    """The guarantee of Accept_Step rests on this inequality and on the two numbers below."""
    assert Riccati_Linear.MAX_X_DISTANCE == 0.75
    assert Riccati_Linear.MAX_INTERPOLATION_ERROR == 0.15
    assert 9/8*Riccati_Linear.MAX_X_DISTANCE + Riccati_Linear.MAX_INTERPOLATION_ERROR < 1

    tau = np.linspace(0, 1, 200001)
    lebesgue = np.abs(4*tau*(1 - tau)) + np.abs(tau*(2*tau - 1))
    assert abs(np.max(lebesgue) - 9/8) < 1e-9
    assert abs(np.max(np.abs(tau*(tau - 0.5)*(tau - 1)))/6 - Riccati_Linear.INTERPOLATION_BOUND) < 1e-11


def test_remainder_formula():
    A = Doubling_Matrix(1, np.eye(2), -np.eye(2))
    M = Riccati_Linear.Step_Generator(A, np.zeros(2, dtype=complex), np.zeros(2, dtype=complex))
    propagators = Riccati_Linear.Step_Propagators(M, 0.8)
    norm = np.linalg.norm(A)*0.8/2
    assert abs(propagators(1)[9] - np.sqrt(3)/216*norm**3*np.exp(norm)) < 1e-14


@pytest.mark.parametrize('seed', range(10))
def test_remainder_bounds_the_interpolation_error(seed):
    """The bound itself, tested against the real path rather than against its own formula."""
    rng = np.random.default_rng(seed)
    (_, _, H_array, _, _, _), _ = Problem(1, 2, seed=seed)
    A = Doubling_Matrix(2, H_array[0] + H_array[1], H_array[0] - H_array[1])
    M = Riccati_Linear.Step_Generator(A, np.zeros(4, dtype=complex), np.zeros(4, dtype=complex))
    sigma = Symmetric(rng, 4, 3.0) + 1j*Symmetric(rng, 4, 3.0)
    h = rng.uniform(0.3, 1.5)
    propagators = Riccati_Linear.Step_Propagators(M, h)
    size = np.sqrt(4 + np.linalg.norm(propagators.observed @ sigma)**2)

    def Z(tau):
        P = expm(A*tau)
        return P[:4, :4] + P[:4, 4:] @ sigma - np.eye(4)

    middle, end = Z(h/2), Z(h)
    for tau in np.linspace(0, 1, 101):
        interpolant = 4*tau*(1 - tau)*middle + tau*(2*tau - 1)*end
        assert np.linalg.norm(Z(tau*h) - interpolant) <= propagators(0)[9]*size


@pytest.mark.parametrize('seed', range(5))
def test_X_only_sees_the_observed_part_of_sigma(seed):
    """What makes ||(1, Q sigma)|| a valid bound: X(t) is the same from sigma and from Q^H Q sigma."""
    rng = np.random.default_rng(seed)
    H_q = np.zeros((4, 4))
    H_q[2:, 2:] = Symmetric(rng, 2, 1.0)
    H_m = np.block([[SIGMA_X, np.zeros((2, 2))], [np.zeros((2, 2)), np.eye(2)]])
    A = Doubling_Matrix(2, H_m + H_q, H_m - H_q)
    Q, _ = Riccati_Linear.Observed_Rows(A)
    assert Q.shape == (2, 4)
    sigma = Symmetric(rng, 4, 1.0)
    P = expm(0.7*A)
    assert np.max(np.abs(P[:4, 4:] @ sigma - P[:4, 4:] @ Q.conj().T @ Q @ sigma)) < 1e-11


def test_end_point_guard_rejects_a_step_that_ends_far_from_one():
    sigma = np.eye(2, dtype=complex)*2.0
    propagators, X = Rotation_Propagators(0.5, sigma)
    size = np.sqrt(2 + np.linalg.norm(sigma)**2)
    assert propagators(0)[9]*size < Riccati_Linear.MAX_INTERPOLATION_ERROR
    assert np.linalg.norm(X - np.eye(2)) > Riccati_Linear.MAX_X_DISTANCE
    assert not Riccati_Linear.Accept_Step(propagators, 0, sigma, X)


def test_unstable_bystander_mode_does_not_slow_the_solver(monkeypatch):
    """Mode 1 is unstable and the qubit does not touch it, so sigma ~ exp(2 lam t) reaches 1e86 there
    but never enters X.  A bound on all of sigma would need of order 1e28 substeps per grid step.
    """
    lam, tau = 1.0, 100.0
    t = np.linspace(0, tau, 21)
    force = np.zeros((4, 1))
    H_m = np.zeros((4, 4))
    H_m[:2, :2], H_m[2:, 2:] = lam*SIGMA_X, np.eye(2)
    H_q = np.zeros((4, 4))
    H_q[2:, 2:] = 0.3*np.eye(2)
    r_0 = np.array([[0.1], [0.2], [0.3], [-0.4]], dtype=complex)
    steps = []
    original = Riccati_Linear.Advance

    def Budgeted_Advance(*arguments):
        steps.append(1)
        assert len(steps) < 100000, 'an unobserved sigma is forcing small steps'
        return original(*arguments)

    monkeypatch.setattr(Riccati_Linear, 'Advance', Budgeted_Advance)
    sigma, _, C, phi = Dynamics_Linear(1, 2, r_0, np.eye(4, dtype=complex), [H_m, H_q], [force, force], np.zeros(1), np.ones((2, 2)), t)
    assert np.max(np.abs(sigma[-1, 0, 1])) > 1e80

    _, _, C_alone, phi_alone = Dynamics_Linear(1, 1, r_0[2:], np.eye(2, dtype=complex), [np.eye(2), 0.3*np.eye(2)], [ZERO_FORCE, ZERO_FORCE],
                                               np.zeros(1), np.ones((2, 2)), t)
    assert np.max(np.abs(C[:, 1, 0] - C_alone[:, 1, 0])) < 1e-11
    assert np.max(np.abs(phi[:, 1, 0] - phi_alone[:, 1, 0])) < 1e-11


def test_strong_squeezing_keeps_its_digits():
    """X is kept near 1 here.  Inverting X directly, with condition number ~1e8 on this coarse
    grid, gives 3e-9 on r.
    """
    s, x_0, p_0, tau = 1e8, 0.3, 0.2, 0.9
    _, r, _, _ = Dynamics_Linear(1, 1, np.array([[x_0], [p_0]], dtype=complex), np.diag([1/s, s]).astype(complex),
                                 [np.zeros((2, 2)), np.eye(2)], [ZERO_FORCE, ZERO_FORCE], np.zeros(1), np.ones((2, 2)), np.linspace(0, tau, 3))
    z_1, z_2 = s*np.cos(tau) + 1j*np.sin(tau), np.cos(tau) + 1j*s*np.sin(tau)
    assert np.max(np.abs(r[-1, 1, 0, :, 0]/np.array([s*x_0/z_1, p_0/z_2]) - 1)) < 1e-12


def Rotated_Problem(block):
    """Mode 1 evolves under block and the qubit only touches mode 2: H_m, H_q in a beam-splitter frame, and R."""
    c, s = np.cos(0.6), np.sin(0.6)
    R = np.array([[c, 0, s, 0], [0, c, 0, s], [-s, 0, c, 0], [0, -s, 0, c]])
    H_m, H_q = np.zeros((4, 4)), np.zeros((4, 4))
    H_m[:2, :2], H_m[2:, 2:] = block, np.eye(2)
    H_q[2:, 2:] = 0.3*np.eye(2)
    H_m_rotated, H_q_rotated = R @ H_m @ R.T, R @ H_q @ R.T
    return 0.5*(H_m_rotated + H_m_rotated.T), 0.5*(H_q_rotated + H_q_rotated.T), R


def Rotated_Run(block, sigma_aligned, tau):
    """
    The same physics twice: aligned, where it is exact to round-off, and rotated.
    Returns the error of the rotated run, relative, on rho_q off-diagonal and on r off-diagonal.
    """
    H_m, H_q, R = Rotated_Problem(block)
    H_m_aligned, H_q_aligned = R.T @ H_m @ R, R.T @ H_q @ R
    H_m_aligned, H_q_aligned = 0.5*(H_m_aligned + H_m_aligned.T), 0.5*(H_q_aligned + H_q_aligned.T)
    H_m_aligned[np.abs(H_m_aligned) < 1e-12], H_q_aligned[np.abs(H_q_aligned) < 1e-12] = 0, 0
    r_0 = np.array([[0.1], [0.2], [0.3], [-0.4]], dtype=complex)
    t = np.linspace(0, tau, 21)
    force, rho_q_0 = np.zeros((4, 1)), np.ones((2, 2))
    # The reference: aligned, with mode 1 decoupled exactly.  A rotating squeezed mode 1 is tilted in any
    # frame, so the guard on sigma would stop it, though nothing here can go wrong in this aligned frame
    limit = Riccati_Linear.MAX_SIGMA_CONDITIONING
    Riccati_Linear.MAX_SIGMA_CONDITIONING = np.inf
    try:
        _, r_true, C_true, phi_true = Dynamics_Linear(1, 2, r_0, sigma_aligned, [H_m_aligned, H_q_aligned], [force, force], np.zeros(1), rho_q_0, t)
    finally:
        Riccati_Linear.MAX_SIGMA_CONDITIONING = limit
    _, r, C, phi = Dynamics_Linear(1, 2, R @ r_0, R @ sigma_aligned @ R.T, [H_m, H_q], [force, force], np.zeros(1), rho_q_0, t)
    rho_error = np.max(np.abs(np.exp(-(C - C_true)[:, 1, 0] + 1j*(phi - phi_true)[:, 1, 0]) - 1))
    difference = np.abs(np.einsum('ab,tb->ta', R.T, r[:, 1, 0, :, 0]) - r_true[:, 1, 0, :, 0])
    return rho_error, np.max(np.max(difference, axis=1)/(1 + np.max(np.abs(r_true[:, 1, 0, :, 0]), axis=1)))


def Unstable_Run(tau):
    return Rotated_Run(SIGMA_X, np.eye(4, dtype=complex), tau)


def Squeezed_Run(s):
    return Rotated_Run(np.eye(2), np.diag([s, 1/s, 1.0, 1.0]).astype(complex), 10.0)


TOO_ILL_CONDITIONED = 'does not resolve, to working precision|too ill-conditioned in this frame'


@pytest.mark.parametrize('run, value', [(Unstable_Run, 6.0), (Unstable_Run, 8.0), (Unstable_Run, 10.0),
                                        (Squeezed_Run, 1e4), (Squeezed_Run, 1e5), (Squeezed_Run, 1e6)])
def test_rotated_frame_answers_within_its_regime(run, value):
    """Guards against a limit set too tight.  Older commits pass this: they were right here."""
    rho_error, r_error = run(value)
    assert rho_error < 1e-8
    assert r_error < 1e-6


@pytest.mark.parametrize('run, value', [(Unstable_Run, 11.0), (Unstable_Run, 12.0), (Unstable_Run, 13.0), (Unstable_Run, 14.0),
                                        (Squeezed_Run, 7e6), (Squeezed_Run, 1e7), (Squeezed_Run, 1e8), (Squeezed_Run, 1e9), (Squeezed_Run, 1e10)])
def test_rotated_frame_is_right_or_raises(run, value):
    """Never silently wrong.  bde56b2 returns r wrong by up to 21 here with no error, and rho_q by 2e-4.
    s = 7e6 sits just past the limit on e_r: 10 times looser and r comes back wrong by 2.7e-5.
    """
    try:
        rho_error, r_error = run(value)
    except np.linalg.LinAlgError as error:
        assert re.search(TOO_ILL_CONDITIONED, str(error))
        return
    assert rho_error < 1e-5
    assert r_error < 1e-5


@pytest.mark.parametrize('run, value', [(Unstable_Run, 14.0), (Unstable_Run, 16.0), (Unstable_Run, 20.0), (Unstable_Run, 100.0),
                                        (Squeezed_Run, 1e10), (Squeezed_Run, 1e12), (Squeezed_Run, 1e14)])
def test_rotated_frame_raises_and_names_the_cause(run, value):
    """bde56b2 returned 1.0 on rho_q after 146 s at T = 16, and did not finish at T = 20."""
    with pytest.raises(np.linalg.LinAlgError, match=TOO_ILL_CONDITIONED):
        run(value)


def Tilted(s, tilt=np.pi/4):
    """sigma squeezed by s along a tilted axis, its thin direction, and the frame."""
    c, n = np.cos(tilt), np.sin(tilt)
    Q = np.array([[c, n], [-n, c]])
    sigma = Q @ np.diag([s, 1/s]) @ Q.T
    return 0.5*(sigma + sigma.T).astype(complex), (Q @ np.array([0.0, 1.0])).reshape(2, 1), Q


def Tilted_Run(s, H_m, H_q, force, tau):
    sigma, thin, _ = Tilted(s)
    return Dynamics_Linear(1, 1, (0.2*thin).astype(complex), sigma, [H_m*np.eye(2), H_q*np.eye(2)], [ZERO_FORCE, force*thin],
                           np.zeros(1), np.ones((2, 2)), np.array([0, tau]))


@pytest.mark.parametrize('s', [1e10, 1e12, 1e14, 1e16, 1e18])
@pytest.mark.parametrize('H_m, H_q, force, tau', [(0.0, 0.0, 0.3, 3.0), (0.0, 1e-3, 0.3, 3.0), (1.0, 0.3, 0.0, 1.0)],
                         ids=['force-only', 'weak-coupling', 'no-force'])
def test_tilted_state_raises(s, H_m, H_q, force, tau):
    """e_s and e_r scale with S_12, but r = y - sigma x and -x^T r cancel against sigma regardless.  3cf159d
    returned these with no error, rho_q wrong by 0.21 in the force-only row and sigma by 0.23 in the last.
    s = 1e14 in the last row is also what a limit 10 times looser would let through, wrong by 1.5e-4.
    """
    with pytest.raises(np.linalg.LinAlgError, match='too ill-conditioned in this frame'):
        Tilted_Run(s, H_m, H_q, force, tau)


def test_force_only_pair_is_guarded():
    """The case from review: S_12 = 0 exactly, so e_s = e_r = 0, and |rho| ~ 1 hides nothing."""
    sigma, thin, _ = Tilted(1e16)
    with pytest.raises(np.linalg.LinAlgError, match='too ill-conditioned in this frame'):
        Dynamics_Linear(1, 1, (0.2*thin).astype(complex), sigma, [np.zeros((2, 2)), np.zeros((2, 2))], [ZERO_FORCE, 0.3*thin],
                        np.zeros(1), np.ones((2, 2)), np.array([0, 3.0]))


def test_branch_with_nothing_to_cancel_is_left_alone():
    """No coupling and no force: X = S_11 and x = 0, so a tilted sigma is only rotated and stays exact."""
    sigma, thin, _ = Tilted(1e12)
    result, _, C, phi = Dynamics_Linear(1, 1, (0.2*thin).astype(complex), sigma, [np.eye(2), np.zeros((2, 2))], [ZERO_FORCE, ZERO_FORCE],
                                        np.zeros(1), np.ones((2, 2)), np.array([0, 2*np.pi]))
    assert np.max(np.abs(result[-1, 1, 0]/np.max(np.abs(sigma)) - sigma/np.max(np.abs(sigma)))) < 1e-12
    assert np.max(np.abs(C)) + np.max(np.abs(phi)) < 1e-12


@pytest.mark.parametrize('s', [1e4, 1e6, 1e7])
def test_tilted_state_answers_below_the_limit(s):
    """H proportional to 1 is the same in the tilted frame, which gives the same physics with sigma diagonal."""
    sigma, thin, Q = Tilted(s)
    H_array = [np.eye(2), 0.3*np.eye(2)]
    r_0 = (0.2*thin).astype(complex)
    force = 0.1*thin
    t = np.array([0, 1.0])
    rho_q_0 = np.ones((2, 2))
    answer = Dynamics_Linear(1, 1, r_0, sigma, H_array, [ZERO_FORCE, force], np.zeros(1), rho_q_0, t)
    exact = Dynamics_Linear(1, 1, Q.T @ r_0, Q.T @ sigma @ Q, H_array, [ZERO_FORCE, Q.T @ force], np.zeros(1), rho_q_0, t)
    assert abs(answer[2][-1, 1, 0] - exact[2][-1, 1, 0]) + abs(answer[3][-1, 1, 0] - exact[3][-1, 1, 0]) < 1e-7
    assert np.max(np.abs(Q.T @ answer[1][-1, 1, 0] - exact[1][-1, 1, 0])) < 1e-7
    assert np.max(np.abs(Q.T @ answer[0][-1, 1, 0] @ Q - exact[0][-1, 1, 0]))/np.max(np.abs(sigma)) < 1e-7


@pytest.mark.parametrize('n_modes', [1, 2, 4])
def test_conditioning_measure_saturates_above_its_limit(n_modes):
    """Past s ~ 1e8 the inverse is capped by what the stored matrix can hold, so the measure stops growing,
    near 1 at worst.  MAX_SIGMA_CONDITIONING has to stay under that floor for a large s to raise at all.
    """
    rng = np.random.default_rng(n_modes)
    dim = 2*n_modes
    for s in (1e9, 1e12, 1e16):
        for _ in range(300):
            Q = np.linalg.qr(rng.normal(size=(dim, dim)))[0]
            eigenvalues = np.ones(dim)
            eigenvalues[:2] = s, 1/s
            sigma = Q @ np.diag(eigenvalues) @ Q.T
            sigma = (0.5*(sigma + sigma.T)).astype(complex)
            assert Riccati_Linear.Sigma_Conditioning(sigma, np.abs(sigma)) > Riccati_Linear.MAX_SIGMA_CONDITIONING


def test_conditioning_measure_is_quiet_where_it_should_be():
    measure = Riccati_Linear.Sigma_Conditioning
    for sigma in (np.diag([1e18, 1e-18]), 2e8*np.eye(2), np.diag([1e8, 4e8])):
        sigma = sigma.astype(complex)
        assert measure(sigma, np.abs(sigma)) < 1e-15
    assert measure(np.ones((2, 2), dtype=complex), np.ones((2, 2))) == np.inf


def test_hot_mode_must_run():
    """Hot and isotropic, nbar = 1e8: large entries with nothing thin."""
    sigma = (1 + 2e8)*np.eye(2, dtype=complex)
    result = Dynamics_Linear(1, 1, np.array([[0.3], [0.2]], dtype=complex), sigma, [np.eye(2), 0.3*np.eye(2)], [ZERO_FORCE, ZERO_FORCE],
                             np.zeros(1), np.ones((2, 2)), np.linspace(0, 10, 21))
    assert np.all(np.isfinite(result[2]))


def test_forty_decibels_at_long_times_must_run():
    """s = 1e4 rotates through every tilt and returns, thousands of times."""
    s = 1e4
    tau = 1e4
    _, r, C, phi = Dynamics_Linear(1, 1, np.array([[0.3], [0.2]], dtype=complex), np.diag([s, 1/s]).astype(complex), [np.eye(2), 0.3*np.eye(2)],
                                   [ZERO_FORCE, ZERO_FORCE], np.zeros(1), np.ones((2, 2)), np.linspace(0, tau, 11))
    assert np.all(np.isfinite(C)) and np.all(np.isfinite(phi)) and np.all(np.isfinite(r))


def test_rounding_limit_guards_zero_r_too():
    """With r = 0, x = 0, so e_r stays 0, and a beam splitter keeps x and p apart, so the guard on the
    conditioning of sigma only sees eps s.  That leaves e_s to stop the error adding up over a run:
    3.7e-7 at T = 10, past its limit by T = 100.  A limit 10 times looser would not raise.
    """
    H_m, H_q, R = Rotated_Problem(np.zeros((2, 2)))
    force = np.zeros((4, 1))
    sigma = (R @ np.diag([1e9, 1e-9, 1.0, 1.0]) @ R.T).astype(complex)
    arguments = (np.zeros((4, 1), dtype=complex), 0.5*(sigma + sigma.T), [H_m, H_q], [force, force], np.zeros(1), np.ones((2, 2)))
    Dynamics_Linear(1, 2, *arguments, np.linspace(0, 10, 21))
    with pytest.raises(np.linalg.LinAlgError, match='does not resolve'):
        Dynamics_Linear(1, 2, *arguments, np.linspace(0, 100, 21))


def test_limits_are_pinned():
    assert Riccati_Linear.MAX_ROUNDING_ERROR_S == 1e-6
    assert Riccati_Linear.MAX_ROUNDING_ERROR_R == 3e-5
    assert Riccati_Linear.MAX_SIGMA_CONDITIONING == 0.3
    assert Riccati_Linear.NEGLIGIBLE_SIGMA == 1e3
    assert Riccati_Linear.SUBSTEP_ALLOWANCE == 100000
    assert Riccati_Linear.SUBSTEPS_PER_BASE_STEP == 1000


def test_rounding_estimate_does_not_depend_on_the_grid():
    """It is meant to be a sum over the path, so that halving cannot dodge it."""
    H_m, H_q, R = Rotated_Problem(np.eye(2))
    force = np.zeros((4, 1))
    sigma = R @ np.diag([1e5, 1e-5, 1.0, 1.0]) @ R.T
    estimates = []
    for points in (21, 81, 321):
        t = np.linspace(0, 10, points)
        A = Doubling_Matrix(2, H_m + H_q, H_m - H_q)
        M = Riccati_Linear.Step_Generator(A, np.zeros(4, dtype=complex), np.zeros(4, dtype=complex))
        propagators = Riccati_Linear.Step_Propagators(M, t[1])
        ledger, state = Riccati_Linear.Step_Ledger(10**7), sigma.astype(complex)
        for _ in range(points - 1):
            for _ in range(2**propagators.base_level):
                state, _, _ = Riccati_Linear.Advance(propagators, propagators.base_level, state, np.zeros(4, dtype=complex), 0.0, ledger)
        estimates.append(ledger.rounding)
    assert max(estimates) < 1.5*min(estimates)
    assert min(estimates) > 1e-13


def Rough_Budget(monkeypatch):
    monkeypatch.setattr(Riccati_Linear, 'SUBSTEP_ALLOWANCE', 10)
    monkeypatch.setattr(Riccati_Linear, 'SUBSTEPS_PER_BASE_STEP', 1)


def test_substep_budget_raises(monkeypatch):
    """Near a pole a run needs more substeps than the grid asks for, which a tight budget refuses."""
    Rough_Budget(monkeypatch)
    sigma_0_jk = np.broadcast_to(np.eye(2, dtype=complex), (2, 2, 2, 2)).copy()
    sigma_0_jk[0, 1] = (0.02 + 1j)*np.eye(2)
    with pytest.raises(np.linalg.LinAlgError, match='gave up after 22 substeps'):
        Dynamics_Linear(1, 1, np.zeros((2, 2, 2, 1), dtype=complex), sigma_0_jk, [np.zeros((2, 2)), -np.eye(2)],
                        [ZERO_FORCE, ZERO_FORCE], np.zeros(1), np.ones((2, 2)), np.linspace(0, 3, 13))


def test_substep_budget_is_far_from_legitimate_work():
    """Long and strongly squeezed runs must stay a factor 20 under their budget.  The worst here, the
    repeated collapse at s = 1e-8, sits a factor 58 under it.
    """
    ledgers = []
    original = Riccati_Linear.Step_Ledger

    def Recording(limit):
        ledgers.append(original(limit))
        return ledgers[-1]

    Riccati_Linear.Step_Ledger = Recording
    try:
        for s, tau in ((1e16, 0.9), (1e-8, 30.0), (2.5, 300.0)):
            Dynamics_Linear(1, 1, np.array([[0.3], [0.2]], dtype=complex), np.diag([1/s, s]).astype(complex), [np.zeros((2, 2)), np.eye(2)],
                            [ZERO_FORCE, ZERO_FORCE], np.zeros(1), np.ones((2, 2)), np.linspace(0, tau, 61))
    finally:
        Riccati_Linear.Step_Ledger = original
    assert max(ledger.substeps/ledger.limit for ledger in ledgers) < 0.05


def test_singular_X_raises():
    """X = (cos t + i (i) sin t) 1 vanishes at t = pi/4, between two grid points."""
    sigma_0 = 1j*np.eye(2, dtype=complex)
    H_array = [np.zeros((2, 2)), -np.eye(2)]
    t = np.linspace(0, 1.5, 4)
    with pytest.raises(np.linalg.LinAlgError, match='between t = 0.5 and t = 1.0.*singular'):
        Dynamics_Linear(1, 1, np.zeros((2, 1), dtype=complex), sigma_0, H_array, [ZERO_FORCE, ZERO_FORCE], np.zeros(1), np.ones((2, 2)), t)


@pytest.mark.parametrize('s', [1e10, 1e14, 1e16, 1e18])
def test_extreme_squeezing_matches_closed_form(s):
    x_0, p_0, tau = 0.3, 0.2, 0.9
    _, r, C, phi = Dynamics_Linear(1, 1, np.array([[x_0], [p_0]], dtype=complex), np.diag([1/s, s]).astype(complex), [np.zeros((2, 2)), np.eye(2)],
                                   [ZERO_FORCE, ZERO_FORCE], np.zeros(1), np.ones((2, 2)), np.linspace(0, tau, 3))
    z_1, z_2 = s*np.cos(tau) + 1j*np.sin(tau), np.cos(tau) + 1j*s*np.sin(tau)
    exact = -1j*np.sin(tau)*(p_0**2/z_2 + s*x_0**2/z_1) - 0.5*(np.log(z_1) + np.log(z_2) - np.log(s))
    assert np.max(np.abs(r[-1, 1, 0, :, 0]/np.array([s*x_0/z_1, p_0/z_2]) - 1)) < 1e-13
    assert abs(-C[-1, 1, 0] + 1j*phi[-1, 1, 0] - exact) < 1e-13


def test_squeezing_beyond_resolution_says_so():
    s = 1e19
    with pytest.raises(np.linalg.LinAlgError, match='too large, to working precision'):
        Dynamics_Linear(1, 1, np.zeros((2, 1), dtype=complex), np.diag([1/s, s]).astype(complex), [np.zeros((2, 2)), np.eye(2)],
                        [ZERO_FORCE, ZERO_FORCE], np.zeros(1), np.ones((2, 2)), np.linspace(0, 0.9, 3))


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_sigma_overflow_says_so():
    """A valid input: sigma_jj grows like exp(2 lam t) and leaves double precision near t = 355."""
    t = np.linspace(0, 400, 11)
    arguments = (np.zeros((2, 1), dtype=complex), np.eye(2, dtype=complex), [SIGMA_X, np.zeros((2, 2))],
                 [ZERO_FORCE, ZERO_FORCE], np.zeros(1), np.ones((2, 2)))
    with pytest.raises(np.linalg.LinAlgError, match='overflowed double precision'):
        Dynamics_Linear(1, 1, *arguments, t)
    sigma = Dynamics_Linear(1, 1, *arguments, t[:9])[0]
    assert np.max(np.abs(sigma[-1, 0, 0]/np.exp(2*t[8]) - np.diag([1.0, 0.0]))) < 1e-9
    # Here sigma is still finite on the last substep and overflows on reaching the last grid point,
    # so no later step can trip on it.  The check at the grid point has to name sigma: r is nan by then too
    with pytest.raises(np.linalg.LinAlgError, match='sigma overflowed double precision'):
        Dynamics_Linear(1, 1, *arguments, np.linspace(0, 355, 11))


def test_exceptional_point():
    """kappa = 0, lambda = mu makes omega_+- = 0 and A defective, so diagonalising it fails."""
    lam = mu = 0.7
    A = Doubling_Matrix(1, lam*SIGMA_X + mu*np.eye(2), lam*SIGMA_X - mu*np.eye(2))
    assert np.max(np.abs(np.linalg.matrix_power(A, 2))) < 1e-14
    t = np.linspace(0, 5, 51)
    sigma, _, _, _ = Dynamics_Linear(1, 1, np.zeros((2, 1), dtype=complex), np.eye(2, dtype=complex),
                                     [lam*SIGMA_X, mu*np.eye(2)], [ZERO_FORCE, ZERO_FORCE], np.zeros(1), np.ones((2, 2)), t)
    # Limit omega -> 0 of the App. l. 1843 form, where sinh(w t)/w -> t
    upper = 2 + (2*lam + 2j*mu)*t
    lower = 2 + (-2*lam + 2j*mu)*t
    assert np.max(np.abs(sigma[:, 1, 0, 0, 0] - upper/lower)) < 1e-12
    assert np.max(np.abs(sigma[:, 1, 0, 1, 1] - lower/upper)) < 1e-12


def test_long_time_stability():
    """||exp(A t)|| reaches ~1e120 here, so a single exponential from t = 0 has no digits left."""
    kappa, lam, mu, alpha, phi = EXAMPLE_1
    t = np.linspace(0, 600, 601)
    sigma, r, _, _ = Example_1(t)
    closed_sigma, closed_r = Example_1_Off_Diagonal(t[-1], kappa, lam, mu, alpha, phi)
    assert np.max(np.abs(np.diag(sigma[-1, 1, 0]) - closed_sigma)) < 1e-12
    assert np.max(np.abs(r[-1, 1, 0, :, 0] - closed_r)) < 1e-12


def test_unstable_branch_does_not_trip_the_singularity_check():
    """Parametric instability: sigma_jj grows like exp(2 lam t) with X perfectly regular."""
    lam = 1.5
    t = np.linspace(0, 12, 121)
    H_array = [lam*SIGMA_X, 0.1*np.eye(2)]
    sigma, _, _, _ = Dynamics_Linear(1, 1, np.zeros((2, 1), dtype=complex), np.eye(2, dtype=complex),
                                     H_array, [ZERO_FORCE, ZERO_FORCE], np.zeros(1), np.ones((2, 2)), t)
    assert np.all(np.isfinite(sigma))
    assert np.max(np.abs(sigma[-1, 0, 0])) > 1e7


################################## Dispatch ##################################

def General_System(open_system=False):
    (r_0, sigma_0, H_array, r_array, H_q_0, rho_q_0), kwargs = Problem(1, 1, open_system=open_system)
    H = Hamiltonian(1, 1)
    H.Initialize_Constant_Hamiltonians(H_array, r_array, H_q_0)
    state = Quantum_State(1, 1)
    state.Initialize_Gaussian_State(r_0, sigma_0, rho_q_0)
    return H, state, kwargs


def Noise(kwargs):
    decoherence = Decoherence(1, 1, 'Canonical', np.zeros((2, 2)), d_vec=kwargs['d'], Gamma_z=kwargs['Gamma_z'])
    decoherence.E_mat, decoherence.D_mat = kwargs['E'], kwargs['D']
    return decoherence


def test_default_solver_is_the_ode():
    H, state, _ = General_System()
    t = np.linspace(0, 1, 21)
    state.Unitary_Dynamics_Numerical(H, t)
    direct = Unitary_Operator_Gaussian.Dynamics_Numerical(1, 1, state.r_0, state.sigma_0, H.H_array, H.r_array, H.H_q_0_array, state.rho_q_0, t)
    assert np.array_equal(state.sigma_JK_t, direct[0])
    assert np.array_equal(state.r_JK_t, direct[1])


def test_default_open_solver_is_the_ode():
    H, state, kwargs = General_System(open_system=True)
    t = np.linspace(0, 1, 21)
    state.Open_Dynamics_Numerical(H, Noise(kwargs), t)
    direct = Open_Operator_Gaussian.Dynamics_Numerical(1, 1, state.r_0, state.sigma_0, H.H_array, H.r_array, H.H_q_0_array, state.rho_q_0,
                                                       kwargs['E'], kwargs['D'], t, d=kwargs['d'], Gamma_z=kwargs['Gamma_z'])
    assert np.array_equal(state.sigma_JK_t, direct[0])
    assert np.array_equal(state.r_JK_t, direct[1])


def test_unitary_dispatch_to_linear():
    H, state, _ = General_System()
    t = np.linspace(0, 1, 21)
    state.Unitary_Dynamics_Numerical(H, t, solver='linear')
    direct = Dynamics_Linear(1, 1, state.r_0, state.sigma_0, H.H_array, H.r_array, H.H_q_0_array, state.rho_q_0, t)
    assert np.array_equal(state.sigma_JK_t, direct[0])
    assert state.type == 'GCS'
    assert state.rho_q_t.shape == (21, 2, 2)
    assert np.max(np.abs(state.rho_q_t[0] - state.rho_q_0)) < 1e-14


def test_open_dispatch_to_linear_forwards_all_noise_terms():
    H, state, kwargs = General_System(open_system=True)
    t = np.linspace(0, 1, 21)
    state.Open_Dynamics_Numerical(H, Noise(kwargs), t, solver='linear')
    direct = Dynamics_Linear(1, 1, state.r_0, state.sigma_0, H.H_array, H.r_array, H.H_q_0_array, state.rho_q_0, t, **kwargs)
    for computed, expected in zip((state.sigma_JK_t, state.r_JK_t, state.C_JK_t, state.phi_JK_t), direct):
        assert np.array_equal(computed, expected)


def Force_Hamiltonian():
    H = Hamiltonian(1, 1)
    H.Initialize_Constant_Hamiltonians([np.eye(2)], [ZERO_FORCE, np.ones((2, 1))], np.zeros(1))
    return H


def Gaussian_Hamiltonian():
    H = Hamiltonian(1, 1)
    H.Initialize_Constant_Hamiltonians([np.eye(2)], [ZERO_FORCE], np.zeros(1))
    return H


def Time_Dependent_Hamiltonian():
    H = Hamiltonian(1, 1)
    H.Initialize_Time_Dep_Hamiltonian([lambda t: np.eye(2), lambda t: np.eye(2)],
                                      [lambda t: ZERO_FORCE, lambda t: ZERO_FORCE], [lambda t: 0.0], np.linspace(0, 1, 3))
    return H


@pytest.mark.parametrize('build', [Force_Hamiltonian, Gaussian_Hamiltonian, Time_Dependent_Hamiltonian])
def test_linear_refuses_other_cases(build):
    _, state, kwargs = General_System(open_system=True)
    t = np.linspace(0, 1, 3)
    with pytest.raises(NotImplementedError):
        state.Unitary_Dynamics_Numerical(build(), t, solver='linear')
    with pytest.raises(NotImplementedError):
        state.Open_Dynamics_Numerical(build(), Noise(kwargs), t, solver='linear')


def test_unknown_solver_is_rejected():
    H, state, _ = General_System()
    with pytest.raises(ValueError, match='solver'):
        state.Unitary_Dynamics_Numerical(H, np.linspace(0, 1, 3), solver='riccati')


def Invalid_Input(change):
    """Keyword arguments of Dynamics_Linear with one entry replaced."""
    (r_0, sigma_0, H_array, r_array, H_q_0, rho_q_0), noise = Problem(1, 1, open_system=True)
    arguments = dict(N_qubits=1, n_modes=1, r_0=r_0, sigma_0=sigma_0, H_array=H_array, r_ham_array=r_array, H_q_0_array=H_q_0,
                     rho_q_0=rho_q_0, t_array=np.linspace(0, 1, 3), **noise)
    return arguments | change(arguments)


SKEW = np.array([[0.0, 1e-3], [0.0, 0.0]])


@pytest.mark.parametrize('name, change', [
    ('H_array', lambda a: dict(H_array=[a['H_array'][0] + SKEW, a['H_array'][1]])),
    ('H_array', lambda a: dict(H_array=[a['H_array'][0], a['H_array'][1] + SKEW])),
    ('sigma_0', lambda a: dict(sigma_0=a['sigma_0'] + SKEW)),
    ('sigma_0', lambda a: dict(sigma_0=np.broadcast_to(a['sigma_0'], (2, 2, 2, 2)) + SKEW*np.array([[[[0.0]], [[1.0]]], [[[0.0]], [[0.0]]]]))),
    ('E', lambda a: dict(E=a['E'] + np.eye(2)*1e-3)),
    ('D', lambda a: dict(D=a['D'] + SKEW)),
    ('H_array', lambda a: dict(H_array=a['H_array'][:1])),
    ('H_array', lambda a: dict(H_array=a['H_array'] + a['H_array'][:1])),
], ids=['H_m', 'H_q', 'sigma_0', 'sigma_0-per-branch', 'E', 'D', 'too-few-H', 'too-many-H'])
def test_invalid_inputs_are_rejected(name, change):
    with pytest.raises(ValueError, match=name):
        Dynamics_Linear(**Invalid_Input(change))


def test_symmetry_is_judged_per_matrix():
    """An asymmetry of 2 on one branch must not hide beside 1e8 on another."""
    def Change(arguments):
        sigma = np.broadcast_to(arguments['sigma_0'], (2, 2, 2, 2)).copy()
        sigma[1, 1] = 1e8*np.eye(2)
        sigma[0, 1, 0, 1] += 2.0
        return dict(sigma_0=sigma, r_0=np.broadcast_to(arguments['r_0'], (2, 2, 2, 1)).copy())
    with pytest.raises(ValueError, match='sigma_0'):
        Dynamics_Linear(**Invalid_Input(Change))


def test_small_asymmetric_H_q_is_rejected():
    """Entries of 1e-7 are real coupling here, wholly asymmetric, and no floor at 1 may let them pass."""
    with pytest.raises(ValueError, match='H_array'):
        Dynamics_Linear(**Invalid_Input(lambda a: dict(H_array=[a['H_array'][0], np.array([[0.0, 1e-7], [0.0, 0.0]])])))


def test_noise_is_judged_against_the_larger_of_E_and_D():
    """E and D share a B: Ladder round-off in a zero E must not count against that E's own size."""
    Dynamics_Linear(**Invalid_Input(lambda a: dict(E=np.array([[1e-9, 0.0], [0.0, 0.0]]), D=np.eye(2))))


@pytest.mark.parametrize('name, change', [
    ('D must be positive', lambda a: dict(D=np.diag([1.0, -0.1]))),
    ('d must be real', lambda a: dict(d=np.array([[0.1 + 0.1j], [0.2]]))),
    ('r_ham_array', lambda a: dict(r_ham_array=a['r_ham_array'] + a['r_ham_array'][:1])),
    ('r_ham_array', lambda a: dict(r_ham_array=a['r_ham_array'][:1])),
], ids=['indefinite-D', 'complex-d', 'long-r', 'short-r'])
def test_noise_and_force_inputs_are_rejected(name, change):
    with pytest.raises(ValueError, match=name):
        Dynamics_Linear(**Invalid_Input(change))


@pytest.mark.parametrize('name', ['r_0', 'sigma_0', 'H_q_0_array', 'rho_q_0', 't_array', 'E', 'D', 'd', 'Gamma_z', 'H_array[1]', 'r_ham_array[0]'])
@pytest.mark.parametrize('bad', [np.nan, np.inf], ids=['nan', 'inf'])
def test_non_finite_input_fails_the_same_way_everywhere(name, bad):
    def Change(arguments):
        key = name.split('[')[0]
        value = [np.array(M, dtype=complex) for M in arguments[key]] if '[' in name else np.array(arguments[key], dtype=complex)
        target = value[int(name[-2])] if '[' in name else value
        target.flat[0] = bad
        return {key: value}
    with pytest.raises(ValueError, match='non-finite'):
        Dynamics_Linear(**Invalid_Input(Change))


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_first_moment_overflow_raises():
    """r_0 = 1e160 gives s ~ 1e320: this used to come back as non-finite C and phi with no error."""
    (_, sigma_0, H_array, r_array, H_q_0, rho_q_0), _ = Problem(1, 1)
    with pytest.raises(np.linalg.LinAlgError, match='r or s overflowed'):
        Dynamics_Linear(1, 1, np.array([[1e160], [1e160]], dtype=complex), sigma_0, H_array, r_array, H_q_0, rho_q_0, np.linspace(0, 1, 5))


def test_validation_passes_the_complex64_floor():
    """Ladder noise goes through U_N, which is complex64: asymmetries of ~1e-8 are round-off."""
    noise = 3e-8*SKEW/1e-3
    Dynamics_Linear(**Invalid_Input(lambda a: dict(D=a['D'] + noise, sigma_0=a['sigma_0'] + noise, H_array=[a['H_array'][0] + noise, a['H_array'][1]])))


################################## Ladder basis ##################################

def Ladder_State(kappa, H_array, t, solver):
    """Through the public API with the input the notebooks use.  B_a = [[0, k], [0, 0]] is k D[a]."""
    H = Hamiltonian(1, 1)
    H.Initialize_Constant_Hamiltonians(H_array, [ZERO_FORCE, ZERO_FORCE], np.zeros(1))
    state = Quantum_State(1, 1)
    state.Initialize_Gaussian_State(np.array([[0.8], [-0.3]]), np.eye(2), np.ones((2, 2))/2)
    state.Open_Dynamics_Numerical(H, Decoherence(1, 1, 'Ladder', kappa*np.array([[0.0, 1.0], [0.0, 0.0]])), t, solver=solver)
    return state


def test_ladder_noise_is_cavity_loss():
    """Pins the current Decoherence behaviour, whose U^T B U differs from the paper's U^dag B U."""
    decoherence = Decoherence(1, 1, 'Ladder', 0.3*np.array([[0.0, 1.0], [0.0, 0.0]]))
    E, D = Gaussian.Get_Decoherence_Rates(1, 0.3*LOSS)
    assert np.max(np.abs(decoherence.E_mat - E)) < 1e-7
    assert np.max(np.abs(decoherence.D_mat - D)) < 1e-7


def test_ladder_linear_matches_ode():
    H_array = [np.eye(2) + 0.1*SIGMA_X, 0.2*np.eye(2)]
    t = np.linspace(0, 4, 401)
    linear = Ladder_State(0.3, H_array, t, 'linear')
    ode = Ladder_State(0.3, H_array, t, 'ode')
    assert np.max(np.abs(linear.sigma_JK_t - ode.sigma_JK_t)) < 1e-8
    assert np.max(np.abs(linear.r_JK_t - ode.r_JK_t)) < 1e-8
    assert np.max(np.abs(linear.rho_q_t - ode.rho_q_t)) < 1e-8


def test_ladder_linear_matches_qutip():
    """Ladder noise is only good to the complex64 floor of U_N, so ~1e-7 and not 1e-10 here."""
    Q = Reference()
    kappa = 0.3
    H_array = [np.eye(2) + 0.1*SIGMA_X, 0.2*np.eye(2)]
    t = np.linspace(0, 4, 5)
    state = Ladder_State(kappa, H_array, t, 'linear')
    rho, r, _, edge = Q.Reference_Dynamics(1, 1, 30, [(0, 1.0, 0.8, -0.3)], H_array, [ZERO_FORCE, ZERO_FORCE], np.zeros(1), np.ones((2, 2))/2, t, B=kappa*LOSS)
    assert edge < 1e-10
    assert np.max(np.abs(state.rho_q_t - rho)) < 1e-6
    assert np.max(np.abs(rho[..., None, None]*(state.r_JK_t - r))) < 1e-6


################################## Conditional rotations (App. l. 1653, main text l. 780) ##################################

ROTATIONS = [(0.7, 0.4, 1.8), (0.3, 1.2, 0.4), (2.2, 2.6, 2.5), (-0.4, 4.4, 3.0)]


def Rotation(theta_bar, theta_delta, s, x_0, p_0, points=2001):
    H_array = [theta_bar*np.eye(2), theta_delta*np.eye(2)]
    sigma_0 = np.diag([1/s, s]).astype(complex)
    r_0 = np.array([[x_0], [p_0]], dtype=complex)
    return Dynamics_Linear(1, 1, r_0, sigma_0, H_array, [ZERO_FORCE, ZERO_FORCE], np.zeros(1), np.ones((2, 2)), np.linspace(0, 1, points))


@pytest.mark.parametrize('theta_bar, theta_delta, s', ROTATIONS)
def test_rotation_X_and_Y(theta_bar, theta_delta, s):
    A = Doubling_Matrix(1, (theta_bar + theta_delta)*np.eye(2), (theta_bar - theta_delta)*np.eye(2))
    P = expm(A)
    sigma_0 = np.diag([1/s, s])
    X, Y = P[:2, :2] + P[:2, 2:] @ sigma_0, P[2:, :2] + P[2:, 2:] @ sigma_0
    c, t = np.cos(theta_delta), np.sin(theta_delta)
    cb, sb = np.cos(theta_bar), np.sin(theta_bar)
    X_paper = np.array([[cb*(c + 1j/s*t), sb*(c + 1j*s*t)], [-sb/s*(s*c + 1j*t), cb*(c + 1j*s*t)]])
    Y_paper = np.array([[cb/s*(c + 1j*s*t), sb*(s*c + 1j*t)], [-sb/s*(c + 1j*s*t), cb*(s*c + 1j*t)]])
    assert np.max(np.abs(X - X_paper)) < 1e-13
    assert np.max(np.abs(Y - Y_paper)) < 1e-13


@pytest.mark.parametrize('theta_bar, theta_delta, s', ROTATIONS)
def test_rotation_sigma_r_and_rho(theta_bar, theta_delta, s):
    x_0, p_0 = 0.8, -0.5
    sigma, r, C, phi = Rotation(theta_bar, theta_delta, s, x_0, p_0)
    c, t = np.cos(theta_delta), np.sin(theta_delta)
    cb, sb = np.cos(theta_bar), np.sin(theta_bar)
    z_1, z_2 = s*c + 1j*t, c + 1j*s*t

    sigma_paper = np.array([[cb**2*z_2/z_1 + sb**2*z_1/z_2, (s**2 - 1)*sb*cb/(z_1*z_2)],
                            [(s**2 - 1)*sb*cb/(z_1*z_2), cb**2*z_1/z_2 + sb**2*z_2/z_1]])
    assert np.max(np.abs(sigma[-1, 1, 0] - sigma_paper)) < 1e-12

    rotation = np.array([[cb, sb], [-sb, cb]])
    assert np.max(np.abs(r[-1, 1, 0, :, 0] - rotation @ np.array([s*x_0/z_1, p_0/z_2]))) < 1e-12

    rho_paper = np.sqrt(s)*np.exp(-1j*t*(p_0**2/z_2 + s*x_0**2/z_1))/(np.sqrt(z_1)*np.sqrt(z_2))
    assert abs(np.exp(-C[-1, 1, 0] + 1j*phi[-1, 1, 0]) - rho_paper) < 1e-12


@pytest.mark.parametrize('theta_bar, theta_delta, s', ROTATIONS)
def test_rotation_squeezing_and_phase(theta_bar, theta_delta, s):
    """Main text, Eq. cov_rotation and the zeta, psi below it."""
    sigma, _, _, _ = Rotation(theta_bar, theta_delta, s, 0, 0)
    c, t = np.cos(theta_delta), np.sin(theta_delta)
    rotation = np.array([[np.cos(theta_bar), np.sin(theta_bar)], [-np.sin(theta_bar), np.cos(theta_bar)]])
    diagonal = rotation.T @ sigma[-1, 1, 0] @ rotation
    zeta_2 = (c**2 + s**2*t**2)/(t**2 + s**2*c**2)
    psi = np.arctan(0.5*(s - 1/s)*np.sin(2*theta_delta))
    assert abs(diagonal[0, 0] - np.exp(1j*psi)*np.sqrt(zeta_2)) < 1e-12
    assert abs(diagonal[1, 1] - np.exp(-1j*psi)/np.sqrt(zeta_2)) < 1e-12


def test_principal_log_form_of_s_off_as_typed_is_wrong():
    """App. l. 1692 writes log(z_1 z_2 / s) for the whole product.  The phase of the product runs
    with 2 theta_Delta, so the principal log is off by 2 pi i, and exp(s) by a sign, for theta_Delta
    in (pi/2, 3 pi/2) mod 2 pi, and is right elsewhere.  l. 1696 (sqrt z_1 sqrt z_2) and the
    continuous form in Rotation_Exponent are right everywhere.
    """
    s, x_0, p_0 = 2.5, 0.8, -0.5

    def Error(theta):
        z_1, z_2 = s*np.cos(theta) + 1j*np.sin(theta), np.cos(theta) + 1j*s*np.sin(theta)
        _, _, C, phi = Rotation(0.0, theta, s, x_0, p_0)
        typed = -1j*np.sin(theta)*(p_0**2/z_2 + s*x_0**2/z_1) - 0.5*np.log(z_1*z_2/s)
        return abs(np.exp(typed) - np.exp(-C[-1, 1, 0] + 1j*phi[-1, 1, 0]))

    assert Error(2.6) > 0.1
    assert Error(4.4) > 0.1
    assert Error(1.2) < 1e-12
    assert Error(5.2) < 1e-12
    assert Error(1.2 + 2*np.pi) < 1e-12


def test_rotation_of_coherent_state():
    """Main text l. 787."""
    theta = 0.9
    r_0 = np.array([[1.1], [0.4]], dtype=complex)
    sigma, r, C, phi = Dynamics_Linear(1, 1, r_0, np.eye(2, dtype=complex), [np.zeros((2, 2)), theta*np.eye(2)],
                                       [ZERO_FORCE, ZERO_FORCE], np.zeros(1), np.ones((2, 2)), np.linspace(0, 1, 201))
    norm = np.sum(np.abs(r_0)**2)
    assert np.max(np.abs(sigma[:, 1, 0] - np.eye(2))) < 1e-13
    assert np.max(np.abs(r[-1, 1, 0] - np.exp(-1j*theta)*r_0)) < 1e-13
    assert abs(np.exp(-C[-1, 1, 0] + 1j*phi[-1, 1, 0]) - np.exp(-norm*(1 - np.exp(-2j*theta))/2 - 1j*theta)) < 1e-13


################################## Dispersive coupling and parametric amplifier (App. l. 1811) ##################################

EXAMPLE_1 = (0.6, 0.2, 0.5, 0.7, 0.4)
LOSS = np.array([[1, -1j], [1j, 1]])/2


def Example_1(t, parameters=EXAMPLE_1):
    """H_j = lam sigma_x + j mu 1 (l. 1034), D = kappa 1, E = kappa/2 Omega, vacuum, rho_q = |+><+|."""
    kappa, lam, mu, alpha, phi = parameters
    E, D = Gaussian.Get_Decoherence_Rates(1, kappa*LOSS)
    d = alpha*np.sqrt(2*kappa)*np.array([[np.cos(phi)], [np.sin(phi)]])
    return Dynamics_Linear(1, 1, np.zeros((2, 1), dtype=complex), np.eye(2, dtype=complex), [lam*SIGMA_X, mu*np.eye(2)],
                           [ZERO_FORCE, ZERO_FORCE], np.zeros(1), np.ones((2, 2))/2, t, E=E, D=D, d=d)


def Omega_Pm(kappa, lam, mu):
    return (0.25*((kappa + 2*lam)**2 + 4j*kappa*mu - 4*mu**2),
            0.25*((kappa - 2*lam)**2 + 4j*kappa*mu - 4*mu**2))


def Example_1_Blocks(tau, kappa, lam, mu):
    omega_p, omega_m = Omega_Pm(kappa, lam, mu)

    def Denominator(sign, omega):
        root = np.sqrt(omega + 0j)
        return (kappa + sign*2*lam + 2j*mu)*np.sinh(tau*root) + 2*root*np.cosh(tau*root)

    def Numerator(sign, omega):
        root = np.sqrt(omega + 0j)
        return (kappa + sign*2*lam + 2j*mu)/(2*root)*(np.cosh(tau*root) - 1) + np.sinh(tau*root)

    return omega_p, omega_m, Denominator, Numerator


def Example_1_Off_Diagonal(tau, kappa, lam, mu, alpha, phi):
    """l. 1843 and l. 1859."""
    omega_p, omega_m, Den, Num = Example_1_Blocks(tau, kappa, lam, mu)
    sigma = np.array([Den(1, omega_m)/Den(-1, omega_m), Den(-1, omega_p)/Den(1, omega_p)])
    r = 2*np.sqrt(2)*alpha*np.sqrt(kappa)*np.array([np.sin(phi)*Num(-1, omega_m)/Den(-1, omega_m),
                                                    -np.cos(phi)*Num(1, omega_p)/Den(1, omega_p)])
    return sigma, r


def Example_1_Exponent_Rates(tau, kappa, lam, mu, alpha, phi):
    """l. 1877: the first two lines are the r part of ds/dt, the third the sigma part."""
    omega_p, omega_m, Den, Num = Example_1_Blocks(tau, kappa, lam, mu)
    r_part = -8j*alpha**2*kappa*mu*(np.cos(phi)**2*(Num(1, omega_p)/Den(1, omega_p))**2
                                    + np.sin(phi)**2*(Num(-1, omega_m)/Den(-1, omega_m))**2)
    sigma_part = -0.5j*mu*(Den(-1, omega_p)/Den(1, omega_p) + Den(1, omega_m)/Den(-1, omega_m))
    return r_part, sigma_part


@pytest.mark.parametrize('parameters', [EXAMPLE_1, (1.0, 0.3, 0.2, 1.0, 0.0), (0.4, 0.1, 0.7, 0.5, 1.0)])
def test_example_1_off_diagonal(parameters):
    tau = np.linspace(0, 40, 2001)
    sigma, r, C, phi = Example_1(tau, parameters)
    for i in (0, 5, 200, 1000, 2000):
        closed_sigma, closed_r = Example_1_Off_Diagonal(tau[i], *parameters)
        assert np.max(np.abs(sigma[i, 1, 0] - np.diag(closed_sigma))) < 1e-12
        assert np.max(np.abs(r[i, 1, 0, :, 0] - closed_r)) < 1e-12


def test_example_1_exponent_rate():
    tau = np.linspace(0, 20, 20001)
    _, _, C, phi = Example_1(tau)
    s = -C[:, 1, 0] + 1j*phi[:, 1, 0]
    numerical = np.gradient(s, tau, edge_order=2)
    r_part, sigma_part = Example_1_Exponent_Rates(tau, *EXAMPLE_1[:3], *EXAMPLE_1[3:])
    assert np.max(np.abs(numerical[1:-1] - (r_part + sigma_part)[1:-1])) < 1e-6


@pytest.mark.parametrize('parameters', [EXAMPLE_1, (1.0, 0.3, 0.2, 1.0, 0.0), (1.0, 0.3, 0.2, 1.0, np.pi/2), (0.4, 0.1, 0.7, 0.5, 1.0)])
def test_example_1_steady_contrast_rates(parameters):
    """l. 1904: the first limit is labelled C_rbar but is C_sigma, as l. 1901 shows."""
    kappa, lam, mu, alpha, phi = parameters
    sigma, r, _, _ = Example_1(np.linspace(0, 300, 301), parameters)
    H_diff = 2*mu*np.eye(2)
    r_off, sigma_off = r[-1, 1, 0, :, 0], sigma[-1, 1, 0]
    omega_p, omega_m = Omega_Pm(kappa, lam, mu)
    C_dot_sigma = -np.real(-0.25j*np.trace(H_diff @ sigma_off))
    C_dot_r = -np.real(-0.5j*r_off @ H_diff @ r_off)
    assert abs(C_dot_sigma - (np.real(np.sqrt(omega_p) + np.sqrt(omega_m))/2 - kappa/2)) < 1e-10
    limit = 2*mu**2*kappa**2*alpha**2*(np.cos(phi)**2/abs(omega_p)**2 + np.sin(phi)**2/abs(omega_m)**2)
    assert abs(C_dot_r - limit) < 1e-10


def test_example_1_S_on_needs_lambda_squared_minus_mu_squared():
    """l. 1822 types omega_mu,lam = sqrt(mu^2 - lam^2).  cosh and sinh in l. 1817 need the
    opposite sign: exp(t Omega (H + E)) has eigenvalues -kappa/2 +- sqrt(lam^2 - mu^2).
    """
    kappa, lam, mu, _, _ = EXAMPLE_1
    Omega = Gaussian.Omega_N(1)
    E, _ = Gaussian.Get_Decoherence_Rates(1, kappa*LOSS)
    tau = 3.0

    def S_On(sign, w):
        c, s = np.cosh(w*tau), np.sinh(w*tau)
        return np.exp(-kappa*tau/2)*np.array([[c + lam/w*s, sign*mu/w*s], [-sign*mu/w*s, c - lam/w*s]])

    for sign in (1, -1):
        exact = expm(tau*Omega @ (lam*SIGMA_X + sign*mu*np.eye(2) + E))
        assert np.max(np.abs(S_On(sign, np.sqrt(lam**2 - mu**2 + 0j)) - exact)) < 1e-13
        assert np.max(np.abs(S_On(sign, np.sqrt(mu**2 - lam**2 + 0j)) - exact)) > 0.1


def test_example_1_steady_sigma_on():
    """l. 1824."""
    kappa, lam, mu, _, _ = EXAMPLE_1
    sigma, _, _, _ = Example_1(np.linspace(0, 300, 301))
    denominator = 4*mu**2 + kappa**2 - 4*lam**2
    for index, sign in ((1, 1), (0, -1)):
        paper = np.array([[4*mu**2 + kappa**2 + 2*kappa*lam, -sign*4*lam*mu],
                          [-sign*4*lam*mu, 4*mu**2 + kappa**2 - 2*kappa*lam]])/denominator
        assert np.max(np.abs(sigma[-1, index, index] - paper)) < 1e-11


def Steady_r_On(sign, kappa, lam, mu, alpha, phi):
    """l. 1829 as typed."""
    return (2*alpha*np.sqrt(2*kappa)/(4*mu**2 + kappa**2 - 4*lam**2)
            * np.array([sign*2*mu*np.cos(phi) - (kappa + 2*lam)*np.sin(phi),
                        (kappa - 2*lam)*np.cos(phi) + sign*2*mu*np.sin(phi)]))


def test_example_1_steady_r_on_has_the_opposite_sign():
    """l. 1829 has the wrong overall sign.  It also contradicts l. 1859, and with lam = mu = 0 it
    gives x = -2 sqrt(2) alpha sin(phi) / sqrt(kappa), where the drive of Eq. example_1_master
    gives a = -2 i alpha / sqrt(kappa) and so a positive x.  QuTiP confirms this sign in
    test_example_1_on_diagonal_sign_against_qutip.
    """
    _, r, _, _ = Example_1(np.linspace(0, 300, 301))
    for index, sign in ((1, 1), (0, -1)):
        typed = Steady_r_On(sign, *EXAMPLE_1)
        assert np.max(np.abs(r[-1, index, index, :, 0] + typed)) < 1e-11
        assert np.max(np.abs(r[-1, index, index, :, 0] - typed)) > 0.1


def test_example_1_steady_state_equation():
    """l. 1919 as typed has lam sigma_x - kappa 1 acting from both sides with a minus sign.  The
    fixed point of Tab. 3 uses Omega(lam sigma_x + E) = lam Omega sigma_x - kappa/2 1, with
    A sigma + sigma A^T, and the opposite sign for the Riccati term.
    """
    kappa, lam, mu, _, _ = EXAMPLE_1
    sigma = Example_1(np.linspace(0, 300, 301))[0][-1, 1, 0]
    identity = np.eye(2)
    A_typed = lam*SIGMA_X - kappa*identity
    typed = A_typed @ sigma - sigma @ A_typed + kappa*identity + 1j*mu*(sigma @ sigma - identity)
    assert np.max(np.abs(typed)) > 0.1

    A = lam*Gaussian.Omega_N(1) @ SIGMA_X - kappa/2*identity
    corrected = A @ sigma + sigma @ A.T + kappa*identity - 1j*mu*(sigma @ sigma - identity)
    assert np.max(np.abs(corrected)) < 1e-10


def test_example_1_rates_have_the_opposite_sign():
    """Eq. rate_sigma and rate_r: with sigma^off = sigma_{+1,-1} (l. 364), the physical rates are
    minus the typed expressions, which come out negative while C grows.
    """
    kappa, lam, mu, alpha, phi = EXAMPLE_1
    tau = np.linspace(0, 40, 4001)
    sigma, r, C, _ = Example_1(tau)
    C_dot = np.real(np.gradient(C[:, 1, 0], tau, edge_order=2))
    typed = (mu/2*np.real(np.trace(sigma[:, 1, 0].imag, axis1=1, axis2=2))
             + 2*mu*np.sum(r[:, 1, 0, :, 0].real*r[:, 1, 0, :, 0].imag, axis=1))
    assert np.max(np.abs(C_dot[1:-1] + typed[1:-1])) < 1e-5
    assert typed[-1] < 0 < C_dot[-1]


################################## QuTiP as the referee ##################################

def Reference():
    pytest.importorskip('qutip')
    import qutip_reference
    return qutip_reference


def Weighted_Errors(reference, linear, rho_q_0):
    """
    r and sigma are normalised by Tr[rho_jk], so where that is tiny their QuTiP values only
    keep |rho_jk| ~ 1e-11 / Tr[rho_jk].  Weighting by |rho_jk| compares what is actually
    resolved, namely Tr[r_hat rho_jk] and Tr[r_hat r_hat^T rho_jk].
    """
    rho_ref, r_ref, sigma_ref, _ = reference
    sigma, r, C, phi = linear
    rho = np.exp(-C + 1j*phi)*rho_q_0
    weight = np.abs(rho)
    return (np.max(np.abs(rho - rho_ref)),
            np.max(weight[..., None, None]*np.abs(r - r_ref)),
            np.max(weight[..., None, None]*np.abs(sigma - sigma_ref)))


def Mixed_Qubits(N_qubits, seed=0):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(2**N_qubits, 2**N_qubits)) + 1j*rng.normal(size=(2**N_qubits, 2**N_qubits))
    rho = G @ G.conj().T
    return rho/np.trace(rho)


def QuTiP_Problem(N_qubits, n_modes, seed=0):
    rng = np.random.default_rng(seed)
    dim = 2*n_modes
    H_array = [np.eye(dim) + Symmetric(rng, dim, 0.1)] + [Symmetric(rng, dim, 0.15) for _ in range(N_qubits)]
    r_array = [0.2*rng.normal(size=(dim, 1)) for _ in range(N_qubits + 1)]
    H_q_0 = 0.3*rng.normal(size=N_qubits)
    mode_states = [(0.1*rng.uniform(), 0.8 + 0.4*rng.uniform(), 0.4*rng.normal(), 0.4*rng.normal()) for _ in range(n_modes)]
    sigma_0 = np.zeros((dim, dim), dtype=complex)
    r_0 = np.zeros((dim, 1), dtype=complex)
    for m, (nbar, s, x_0, p_0) in enumerate(mode_states):
        sigma_0[2*m, 2*m], sigma_0[2*m + 1, 2*m + 1] = (1 + 2*nbar)*s, (1 + 2*nbar)/s
        r_0[2*m, 0], r_0[2*m + 1, 0] = x_0, p_0
    return mode_states, r_0, sigma_0, H_array, r_array, H_q_0


def Open_Noise(n_modes, seed=0):
    """Loss, gain and x-measurement, so B is complex with E != 0, plus drive and dephasing."""
    rng = np.random.default_rng(seed + 10)
    L = rng.normal(size=(2*n_modes, 2*n_modes)) + 1j*rng.normal(size=(2*n_modes, 2*n_modes))
    B = 0.03*L @ L.conj().T
    return B, 0.2*rng.normal(size=(2*n_modes, 1))


@pytest.mark.parametrize('N_qubits, n_modes, cutoff', [(1, 1, 30), (2, 1, 30), (1, 2, 20)])
@pytest.mark.parametrize('open_system', [False, True], ids=['unitary', 'open'])
def test_against_qutip(N_qubits, n_modes, cutoff, open_system):
    Q = Reference()
    mode_states, r_0, sigma_0, H_array, r_array, H_q_0 = QuTiP_Problem(N_qubits, n_modes)
    rho_q_0 = Mixed_Qubits(N_qubits)
    t = np.linspace(0, 1.5, 7)
    noise, kwargs = {}, {}
    if open_system:
        B, d = Open_Noise(n_modes)
        Gamma_z = 0.2*np.ones(N_qubits)
        E, D = Gaussian.Get_Decoherence_Rates(n_modes, B)
        noise = dict(B=B, d=d, Gamma_z=Gamma_z)
        kwargs = dict(E=E, D=D, d=d, Gamma_z=Gamma_z)
    linear = Dynamics_Linear(N_qubits, n_modes, r_0, sigma_0, H_array, r_array, H_q_0, rho_q_0, t, **kwargs)
    reference = Q.Reference_Dynamics(N_qubits, n_modes, cutoff, mode_states, H_array, r_array, H_q_0, rho_q_0, t, **noise)
    assert reference[3] < 1e-6
    rho_error, r_error, sigma_error = Weighted_Errors(reference, linear, rho_q_0)
    assert rho_error < 1e-5
    assert r_error < 1e-5
    assert sigma_error < 1e-4


def test_qutip_converges_to_the_linear_solution():
    """Errors must fall as the cutoff grows, down to the mesolve tolerance."""
    Q = Reference()
    mode_states, r_0, sigma_0, H_array, r_array, H_q_0 = QuTiP_Problem(1, 1)
    rho_q_0 = Mixed_Qubits(1)
    B, d = Open_Noise(1)
    E, D = Gaussian.Get_Decoherence_Rates(1, B)
    Gamma_z = np.array([0.2])
    t = np.linspace(0, 2, 5)
    linear = Dynamics_Linear(1, 1, r_0, sigma_0, H_array, r_array, H_q_0, rho_q_0, t, E=E, D=D, d=d, Gamma_z=Gamma_z)
    errors = [Weighted_Errors(Q.Reference_Dynamics(1, 1, cutoff, mode_states, H_array, r_array, H_q_0, rho_q_0, t, B=B, d=d, Gamma_z=Gamma_z),
                              linear, rho_q_0)[0] for cutoff in (15, 25, 40)]
    assert errors[0] > 10*errors[2]
    assert errors[2] < 1e-8


def test_tab_1_as_typed_is_wrong():
    """Tab. 1 (l. 413, 417) has +i Omega Dl Omega and -i sigma rm.  Tab. 3 has the opposite signs,
    and so does Eq. ODE_conditional_rot, which needs -i Omega Dl Omega for sigma = 1 to be stationary.
    QuTiP sides with Tab. 3.
    """
    Q = Reference()
    Omega = Gaussian.Omega_N(1)
    mode_states, r_0, sigma_0, H_array, r_array, _ = QuTiP_Problem(1, 1, seed=2)
    mode_states = [(0, *mode_states[0][1:])]
    sigma_0 = np.diag([mode_states[0][1], 1/mode_states[0][1]]).astype(complex)
    H_sum, H_diff, r_sum, r_diff = 2*H_array[0], 2*H_array[1], np.ravel(2*r_array[0]), np.ravel(2*r_array[1])
    t = np.linspace(0, 2, 5)
    rho_q_0 = np.ones((2, 2), dtype=complex)/2
    _, r_ref, sigma_ref, _ = Q.Reference_Dynamics(1, 1, 40, mode_states, H_array, r_array, np.zeros(1), rho_q_0, t)

    def Solve(sign_omega, sign_force):
        def Derivative(_, z):
            sigma, r = z[:4].reshape(2, 2), z[4:]
            sigma_dot = 0.5*(Omega @ H_sum @ sigma - sigma @ H_sum @ Omega - 1j*sigma @ H_diff @ sigma + sign_omega*1j*Omega @ H_diff @ Omega)
            r_dot = 0.5*((Omega @ H_sum - 1j*sigma @ H_diff) @ r - Omega @ r_sum + sign_force*1j*sigma @ r_diff)
            return np.concatenate((sigma_dot.ravel(), r_dot))
        y = solve_ivp(Derivative, (0, 2), np.concatenate((sigma_0.ravel(), np.ravel(r_0))), t_eval=t, **Gaussian.ODE_KWARGS).y.T
        return np.max(np.abs(y[:, :4].reshape(-1, 2, 2) - sigma_ref[:, 1, 0])), np.max(np.abs(y[:, 4:] - r_ref[:, 1, 0, :, 0]))

    assert max(Solve(-1, 1)) < 1e-6
    assert Solve(1, 1)[0] > 0.1
    assert Solve(-1, -1)[1] > 0.1


def test_eq_lindo_order_as_typed_is_not_trace_preserving():
    """l. 631 has {rho, r^a r^b}.  Only {rho, r^b r^a} preserves the trace for a complex B."""
    qutip = pytest.importorskip('qutip')
    Q = Reference()
    x, p = Q.Mode_Operators(1, 30)
    r_ops = [x, p]
    rho = qutip.coherent_dm(30, 0.5)
    B = np.array([[0.5, -0.5j], [0.5j, 0.5]])

    def Trace(first, second):
        return sum(B[a, b]*(r_ops[a]*rho*r_ops[b] - 0.5*(rho*r_ops[first(a, b)]*r_ops[second(a, b)]
                                                         + r_ops[first(a, b)]*r_ops[second(a, b)]*rho)).tr()
                   for a in range(2) for b in range(2))

    assert abs(Trace(lambda a, b: b, lambda a, b: a)) < 1e-12
    assert abs(Trace(lambda a, b: a, lambda a, b: b) + 1) < 1e-10


def test_example_1_on_diagonal_sign_against_qutip():
    """Referee for test_example_1_steady_r_on_has_the_opposite_sign."""
    Q = Reference()
    kappa, lam, mu, alpha, phi = EXAMPLE_1
    t = np.linspace(0, 14, 8)
    d = alpha*np.sqrt(2*kappa)*np.array([[np.cos(phi)], [np.sin(phi)]])
    rho_ref, r_ref, _, edge = Q.Reference_Dynamics(1, 1, 40, [(0, 1.0, 0.0, 0.0)], [lam*SIGMA_X, mu*np.eye(2)], [ZERO_FORCE, ZERO_FORCE],
                                                   np.zeros(1), np.ones((2, 2))/2, t, B=kappa*LOSS, d=d)
    assert edge < 1e-12
    sigma, r, C, phi_t = Example_1(t)
    assert np.max(np.abs(rho_ref - np.exp(-C + 1j*phi_t)/2)) < 1e-8
    assert np.max(np.abs(r_ref[:, 1, 1] - r[:, 1, 1])) < 1e-7
    assert np.max(np.abs(r_ref[:, 0, 0] - r[:, 0, 0])) < 1e-7
    assert np.sign(r_ref[-1, 1, 1, 0, 0].real) == -np.sign(Steady_r_On(1, *EXAMPLE_1)[0])


def test_eq_example_1_hamiltonian_disagrees_with_its_matrices():
    """l. 1026 has (mu/2)(a^dag a + 1/2) sigma_z, which is H_q = mu/2 1.  l. 1034 and all of the
    closed forms use H_j = lam sigma_x + j mu 1, so H_q = mu 1.  One of the two is off by 2.
    """
    qutip = pytest.importorskip('qutip')
    kappa, lam, mu, alpha, phi = EXAMPLE_1
    cutoff = 40
    a = qutip.tensor(qutip.destroy(cutoff), qutip.qeye(2))
    sigma_z = qutip.tensor(qutip.qeye(cutoff), qutip.Qobj(np.diag([-1.0, 1.0])))
    H_1 = 0.5*mu*(a.dag()*a + 0.5)*sigma_z + 0.5j*lam*(a.dag()**2 - a**2)
    alpha_in = alpha*np.exp(1j*phi)
    drive = np.sqrt(kappa)*(alpha_in*a.dag() + np.conj(alpha_in)*a)
    psi_0 = qutip.tensor(qutip.basis(cutoff, 0), (qutip.basis(2, 0) + qutip.basis(2, 1)).unit())
    t = np.linspace(0, 6, 7)
    states = qutip.mesolve(H_1 + drive, psi_0, t, c_ops=[np.sqrt(kappa)*a],
                           options={'rtol': 1e-9, 'atol': 1e-11, 'nsteps': 10**6, 'store_states': True}).states
    rho_off = np.array([state.ptrace(1).full()[1, 0] for state in states])

    E, D = Gaussian.Get_Decoherence_Rates(1, kappa*LOSS)
    d = alpha*np.sqrt(2*kappa)*np.array([[np.cos(phi)], [np.sin(phi)]])

    def Error(H_q):
        _, _, C, phi_t = Dynamics_Linear(1, 1, np.zeros((2, 1), dtype=complex), np.eye(2, dtype=complex), [lam*SIGMA_X, H_q],
                                         [ZERO_FORCE, ZERO_FORCE], np.zeros(1), np.ones((2, 2))/2, t, E=E, D=D, d=d)
        return np.max(np.abs(np.exp(-C[:, 1, 0] + 1j*phi_t[:, 1, 0])/2 - rho_off))

    assert Error(0.5*mu*np.eye(2)) < 1e-8
    assert Error(mu*np.eye(2)) > 0.1
