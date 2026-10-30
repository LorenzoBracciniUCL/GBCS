"""Riccati_Linear against the ODE solvers, the paper's closed forms and QuTiP.

Where a paper formula disagrees with QuTiP the test keeps the corrected form, says what
differs, and test_*_as_typed_is_wrong records that the typed form really does fail.
Line numbers refer to manuscript/apssamp-3.tex.

The ODE tolerances are not arbitrary: on a fine grid the ODE solvers saturate at about
3e-10 in sigma and r, so that is what linear-vs-ODE can resolve.
"""
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


def Joint_ODE(N_qubits, n_modes, problem, kwargs, t):
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
            y = solve_ivp(Derivative, (t[0], t[-1]), y_0, t_eval=t, method='DOP853', rtol=1e-13, atol=1e-14).y.T
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
    assert np.max(np.abs(sigma - np.swapaxes(sigma, 3, 4))) < 1e-13


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
    assert len(calls) == 3


def test_non_uniform_grid_reuses_repeated_steps(monkeypatch):
    problem, _ = Problem(1, 1)
    calls = Count_Exponentials(monkeypatch)
    t = np.concatenate((np.linspace(0, 1, 101), 1 + 0.5*np.arange(1, 5)))
    Dynamics_Linear(1, 1, *problem, t)
    assert len(calls) == 2*3


def test_steps_are_subdivided():
    H_array = [50*np.eye(2), 5*np.eye(2)]
    assert Riccati_Linear.Base_Level(Doubling_Matrix(1, H_array[0] + H_array[1], H_array[0] - H_array[1]), 1.0) == 6


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


def test_near_pole_is_resolved():
    """X = cos(t) + i (eps + i) sin(t) passes within about eps of zero at t = pi/4, off the
    real axis.  Stepping over it would lose a phase of pi per eigenvalue of X.
    """
    eps = 0.02
    sigma_0 = (eps + 1j)*np.eye(2, dtype=complex)
    H_array = [np.zeros((2, 2)), np.eye(2)]
    t = np.linspace(0, 3, 13)
    fine = np.linspace(0, 3, 1200001)
    x = np.cos(fine) + 1j*(eps + 1j)*np.sin(fine)
    exact = -Continuous_Log(x)[::100000]
    sigma_0_jk = np.broadcast_to(sigma_0, (2, 2, 2, 2)).copy()
    sigma_0_jk[1, 1] = np.eye(2)
    sigma_0_jk[0, 0] = np.eye(2)
    r_0_jk = np.zeros((2, 2, 2, 1), dtype=complex)
    H_flipped = [H_array[0], -H_array[1]]
    _, _, C, phi = Dynamics_Linear(1, 1, r_0_jk, sigma_0_jk, H_flipped, [ZERO_FORCE, ZERO_FORCE], np.zeros(1), np.ones((2, 2)), t)
    assert np.max(np.abs(-C[:, 0, 1] + 1j*phi[:, 0, 1] - exact)) < 1e-8


def test_singular_X_raises():
    """X = (cos t + i (i) sin t) 1 vanishes at t = pi/4, between two grid points."""
    sigma_0 = 1j*np.eye(2, dtype=complex)
    H_array = [np.zeros((2, 2)), -np.eye(2)]
    t = np.linspace(0, 1.5, 4)
    with pytest.raises(np.linalg.LinAlgError, match='singular between t = 0.5 and t = 1.0'):
        Dynamics_Linear(1, 1, np.zeros((2, 1), dtype=complex), sigma_0, H_array, [ZERO_FORCE, ZERO_FORCE], np.zeros(1), np.ones((2, 2)), t)


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
    """App. l. 1692 writes log(z_1 z_2 / s) for the whole product.  With the principal log that
    leaves the correct branch once theta_Delta > pi/2.  l. 1696 (sqrt z_1 sqrt z_2) and the
    continuous form in Rotation_Exponent are right.
    """
    theta, s, x_0, p_0 = 2.6, 2.5, 0.8, -0.5
    z_1, z_2 = s*np.cos(theta) + 1j*np.sin(theta), np.cos(theta) + 1j*s*np.sin(theta)
    _, _, C, phi = Rotation(0.0, theta, s, x_0, p_0)
    exact = np.exp(-C[-1, 1, 0] + 1j*phi[-1, 1, 0])
    typed = -1j*np.sin(theta)*(p_0**2/z_2 + s*x_0**2/z_1) - 0.5*np.log(z_1*z_2/s)
    assert abs(np.exp(typed) - exact) > 0.1


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
    """Tab. 1 (l. 408, 412) has +i Omega Dl Omega and -i sigma rm.  Tab. 3 has the opposite signs,
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
