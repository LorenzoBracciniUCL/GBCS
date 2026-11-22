"""Perturbation_Theory: Dyson series in DV-only terms, against closed forms, Riccati_Linear and QuTiP."""
import numpy as np
import pytest

from gbcs import Gaussian, Perturbation, QIT_Functions
from gbcs import Perturbation_Theory as PT
from gbcs.Objects import Decoherence, Hamiltonian, Quantum_State
from gbcs.Riccati_Linear import Dynamics_Linear

SIGMA_X = np.array([[0., 1.], [1., 0.]])


def Symmetric(rng, dim, scale):
    M = rng.normal(size=(dim, dim))
    return scale*(M + M.T)/2


def Density(rng, N):
    G = rng.normal(size=(N, N)) + 1j*rng.normal(size=(N, N))
    rho = G @ G.conj().T
    return rho/np.trace(rho)


def Register_Levels(N_qubits, H_array, r_array, H_q_0):
    """Per-level H_j, r_j, H^0_j of a qubit register, as Perturbation_Theory.Level_Structure builds them."""
    labels = [QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, j) for j in range(2**N_qubits)]
    H = [H_array[0] + sum(l[i]*H_array[i + 1] for i in range(N_qubits)) for l in labels]
    r = [r_array[0] + sum(l[i]*r_array[i + 1] for i in range(N_qubits)) for l in labels]
    return H, r, np.array([l @ H_q_0 for l in labels])


def Mode_Initial(mode_states):
    dim = 2*len(mode_states)
    sigma_0, r_0 = np.zeros((dim, dim), complex), np.zeros((dim, 1), complex)
    for m, (nbar, s, x_0, p_0) in enumerate(mode_states):
        sigma_0[2*m, 2*m], sigma_0[2*m + 1, 2*m + 1] = (1 + 2*nbar)*s, (1 + 2*nbar)/s
        r_0[2*m, 0], r_0[2*m + 1, 0] = x_0, p_0
    return sigma_0, r_0


def One_Qubit_Problem(seed=3, linear=False):
    rng = np.random.default_rng(seed)
    H_m = np.eye(2) + Symmetric(rng, 2, 0.1)
    H_q = np.zeros((2, 2)) if linear else Symmetric(rng, 2, 0.2)
    r_m, r_q = 0.2*rng.normal(size=(2, 1)), 0.3*rng.normal(size=(2, 1))
    H_levels, r_levels = [H_m - H_q, H_m + H_q], [r_m - r_q, r_m + r_q]   # index 0 is label -1
    return H_levels, r_levels, np.array([-0.2, 0.2])


################################## The DV superoperator ##################################

def test_liouvillian_is_the_master_equation():
    rng = np.random.default_rng(0)
    P = Perturbation(N_levels=3)
    H_dv = rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3))
    P.Add_Hamiltonian(H_dv + H_dv.conj().T)
    P.Add_Ladder_Decay([0.3, 0.7])
    rho = Density(rng, 3)
    H = P.H_dv
    expected = -1j*(H @ rho - rho @ H)
    for L in P.jumps:
        expected += L @ rho @ L.conj().T - 0.5*(L.conj().T @ L @ rho + rho @ L.conj().T @ L)
    assert np.allclose(P.Liouvillian() @ rho.reshape(-1), expected.reshape(-1), atol=1e-13)
    c, V = P.Split()
    assert np.allclose(np.diag(V), 0)
    assert np.allclose(np.diag(c.reshape(-1)) + V, P.Liouvillian())


def test_lowering_operator_takes_label_plus_to_minus():
    for N_qubits in (1, 2, 3):
        for i in range(N_qubits):
            L = PT.Lowering_Operator(N_qubits, i)
            for j in range(2**N_qubits):
                before = QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, j)
                column = L[:, j]
                if before[i] < 0:
                    assert not np.any(column)
                    continue
                after = QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, int(np.flatnonzero(column)[0]))
                expected = before.copy()
                expected[i] = -1
                assert np.array_equal(after, expected)


def test_input_errors():
    with pytest.raises(ValueError, match='either'):
        Perturbation()
    with pytest.raises(ValueError, match='Add_Ladder_Decay'):
        Perturbation(N_levels=3).Add_T1([0.1])
    with pytest.raises(ValueError, match='Add_T1'):
        Perturbation(N_qubits=1).Add_Ladder_Decay([0.1])
    with pytest.raises(ValueError, match='Hermitian'):
        Perturbation(N_qubits=1).Add_Hamiltonian(np.array([[0, 1], [0, 0]]))
    with pytest.raises(ValueError, match='one rate per qubit'):
        Perturbation(N_qubits=2).Add_T1([0.1])
    H_levels, r_levels, H_0 = One_Qubit_Problem()
    P = Perturbation(N_qubits=1)
    P.Add_Hamiltonian(0.3*SIGMA_X)
    args = (1, np.zeros((2, 1)), np.eye(2), np.eye(2)/2, H_levels, r_levels, H_0, np.zeros((2, 2)), P)
    with pytest.raises(ValueError, match='order'):
        PT.Perturbation_Dynamics(*args, np.linspace(0, 1, 5), order=-1)
    with pytest.raises(ValueError, match='method'):
        PT.Perturbation_Dynamics(*args, np.linspace(0, 1, 5), method='exact')
    with pytest.raises(MemoryError, match='max_paths'):
        PT.Perturbation_Dynamics(*args, np.linspace(0, 1, 21), order=2, max_paths=100)


################################## Order 0 and closed forms ##################################

@pytest.mark.parametrize('N_qubits', [1, 2])
def test_no_terms_reproduce_the_linear_solver(N_qubits):
    """With nothing to perturb, order 0 is the GBCS of Riccati_Linear, branch by branch, and nothing else."""
    rng = np.random.default_rng(1)
    H_array = [np.eye(2) + Symmetric(rng, 2, 0.15)] + [Symmetric(rng, 2, 0.2) for _ in range(N_qubits)]
    r_array = [0.3*rng.normal(size=(2, 1)) for _ in range(N_qubits + 1)]
    H_q_0 = 0.4*rng.normal(size=N_qubits)
    rho_q_0 = Density(rng, 2**N_qubits)
    sigma_0, r_0 = np.diag([1.3, 0.8]).astype(complex), np.array([[0.4], [-0.2]], dtype=complex)
    L = rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2))
    E, D = Gaussian.Get_Decoherence_Rates(1, 0.05*L @ L.conj().T)
    d, Gamma_z = 0.2*rng.normal(size=(2, 1)), 0.1*np.ones(N_qubits)
    t = np.linspace(0, 2, 41)

    H = Hamiltonian(N_qubits, 1)
    H.Initialize_Constant_Hamiltonians(H_array, r_array, H_q_0)
    noise = Decoherence(N_qubits, 1, 'Canonical', 0.05*L @ L.conj().T, d_vec=d, Gamma_z=Gamma_z)
    G = Quantum_State(N_qubits, 1)
    G.Initialize_Gaussian_State(r_0, sigma_0, rho_q_0)
    result = G.Perturbed_Dynamics(H, Perturbation(N_qubits=N_qubits), t, Decoherence=noise, order=2)

    sigma, r, C, phi = Dynamics_Linear(N_qubits, 1, r_0, sigma_0, H_array, r_array, H_q_0, rho_q_0, t,
                                       E=E, D=D, d=d, Gamma_z=Gamma_z)
    assert np.max(np.abs(G.sigma_JK_t - sigma)) < 1e-12
    assert np.max(np.abs(G.r_JK_t - r)) < 1e-12
    assert np.max(np.abs(G.C_JK_t - C)) < 1e-12
    assert np.max(np.abs(G.phi_JK_t - phi)) < 1e-12
    assert np.max(np.abs(result.rho_q_t_orders[1:])) == 0
    assert np.allclose(G.rho_q_t_total, G.rho_q_t)


def test_T1_of_one_qubit_against_its_closed_form():
    """
    rho_++ = p_+ e^(-Gamma tau) is order 0 and exact; rho_-- = p_- + p_+ (1 - e^(-Gamma tau)) needs the one jump,
    with the trapezoid error, O(dt^2).  Orders >= 2 vanish: one qubit has one excitation to lose.
    """
    H_levels, r_levels, H_0 = One_Qubit_Problem()
    rho_q_0 = Density(np.random.default_rng(2), 2)
    Gamma = 0.3
    P = Perturbation(N_qubits=1)
    P.Add_T1([Gamma])
    errors = []
    for T in (41, 81):
        t = np.linspace(0, 2, T)
        result = PT.Perturbation_Dynamics(1, np.array([[0.3], [0.1]]), np.diag([1.2, 0.9]), rho_q_0,
                                          H_levels, r_levels, H_0, np.zeros((2, 2)), P, t, order=3)
        rho = result.rho_q_t_total
        assert np.max(np.abs(rho[:, 1, 1] - rho_q_0[1, 1]*np.exp(-Gamma*t))) < 1e-13
        errors.append(np.max(np.abs(rho[:, 0, 0] - rho_q_0[0, 0] - rho_q_0[1, 1]*(1 - np.exp(-Gamma*t)))))
        assert np.max(np.abs(result.rho_q_t_orders[2:])) == 0
    assert errors[0] < 1e-5
    assert 3.8 < errors[0]/errors[1] < 4.2


def test_series_stops_after_one_jump_per_qubit():
    rng = np.random.default_rng(4)
    H_array = [np.eye(2)] + [Symmetric(rng, 2, 0.2) for _ in range(2)]
    r_array = [0.2*rng.normal(size=(2, 1)) for _ in range(3)]
    H_levels, r_levels, H_0 = Register_Levels(2, H_array, r_array, np.array([0.3, -0.1]))
    P = Perturbation(N_qubits=2)
    P.Add_T1([0.4, 0.25])
    result = PT.Perturbation_Dynamics(1, np.zeros((2, 1)), np.eye(2), Density(rng, 4), H_levels, r_levels, H_0,
                                      np.zeros((4, 4)), P, np.linspace(0, 1, 11), order=4)
    assert np.max(np.abs(result.rho_q_t_orders[2])) > 1e-3
    assert np.max(np.abs(result.rho_q_t_orders[3:])) == 0


def test_total_state_is_hermitian_with_unit_trace_under_decay():
    H_levels, r_levels, H_0 = One_Qubit_Problem()
    P = Perturbation(N_qubits=1)
    P.Add_T1([0.5])
    P.Add_Hamiltonian(0.2*SIGMA_X)
    result = PT.Perturbation_Dynamics(1, np.zeros((2, 1)), np.eye(2), Density(np.random.default_rng(5), 2),
                                      H_levels, r_levels, H_0, np.zeros((2, 2)), P, np.linspace(0, 1, 21), order=2)
    rho = result.rho_q_t_total
    assert np.max(np.abs(rho - np.conj(np.swapaxes(rho, -1, -2)))) < 1e-12
    second = result.second_moments_t
    assert np.max(np.abs(second - np.conj(np.swapaxes(np.swapaxes(second, 1, 2), -1, -2)))) < 1e-12


################################## QuTiP as the referee ##################################

def Reference():
    pytest.importorskip('qutip')
    import qutip_reference
    return qutip_reference


MODES = [(0.05, 1.2, 0.3, -0.2)]


def QuTiP_Case(name):
    """N_levels, H_levels, r_levels, H_0, Perturbation, DV Hamiltonian and jumps for QuTiP, rho_q_0, B, order."""
    rng = np.random.default_rng(7)
    if name in ('drive_linear', 'drive_quadratic_open'):
        H_levels, r_levels, H_0 = One_Qubit_Problem(seed=3, linear=(name == 'drive_linear'))
        P = Perturbation(N_qubits=1)
        P.Add_Hamiltonian(0.05*SIGMA_X)
        B = None
        if name == 'drive_quadratic_open':
            L = rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2))
            B = 0.03*L @ L.conj().T
        return 2, H_levels, r_levels, H_0, P, 0.05*SIGMA_X, [], Density(rng, 2), B, 3
    if name == 'two_qubit_T1_open':
        H_array = [np.eye(2) + Symmetric(rng, 2, 0.1)] + [Symmetric(rng, 2, 0.15) for _ in range(2)]
        r_array = [0.25*rng.normal(size=(2, 1)) for _ in range(3)]
        H_levels, r_levels, H_0 = Register_Levels(2, H_array, r_array, np.array([0.3, -0.2]))
        P = Perturbation(N_qubits=2)
        P.Add_T1([0.4, 0.25])
        L = rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2))
        return 4, H_levels, r_levels, H_0, P, None, P.jumps, Density(rng, 4), 0.03*L @ L.conj().T, 2
    if name == 'qutrit_ladder':
        H_levels = [np.eye(2) + Symmetric(rng, 2, 0.15) for _ in range(3)]
        r_levels = [0.25*rng.normal(size=(2, 1)) for _ in range(3)]
        P = Perturbation(N_levels=3)
        P.Add_Ladder_Decay([0.5, 0.3])
        return 3, H_levels, r_levels, np.array([0.0, 0.7, 1.5]), P, None, P.jumps, Density(rng, 3), None, 2
    raise ValueError(name)


@pytest.mark.parametrize('name', ['drive_linear', 'drive_quadratic_open', 'two_qubit_T1_open', 'qutrit_ladder'])
def test_against_qutip(name):
    """
    Unnormalised branch moments Tr[rho_jk], Tr[r rho_jk], Tr[sym(r r^T) rho_jk] of the full state.  The drive is
    weak enough (2 Omega tau = 0.15) for order 3 to be within 2e-6 of all orders; T1 and ladder decay stop at 2.
    What remains is the trapezoid error of the jump times, about 4e-5 on this grid.
    """
    Q = Reference()
    N, H_levels, r_levels, H_0, P, H_dv, jumps, rho_q_0, B, order = QuTiP_Case(name)
    sigma_0, r_0 = Mode_Initial(MODES)
    E = D = None
    if B is not None:
        E, D = Gaussian.Get_Decoherence_Rates(1, B)
    t = np.linspace(0, 1.5, 31)
    check = np.arange(0, 31, 10)
    result = PT.Perturbation_Dynamics(1, r_0, sigma_0, rho_q_0, H_levels, r_levels, H_0, np.zeros((N, N)), P, t,
                                      E=E, D=D, order=order)
    rho, first, second, edge = Q.Reference_Dynamics_Levels(N, 1, 30, MODES, H_levels, r_levels, H_0, rho_q_0,
                                                           t[check], B=B, H_dv=H_dv, dv_jumps=jumps)
    assert edge < 1e-6
    assert np.max(np.abs(result.rho_q_t_total[check] - rho)) < 1e-4
    assert np.max(np.abs(result.first_moments_t[check] - first)) < 2e-4
    assert np.max(np.abs(result.second_moments_t[check] - second)) < 4e-4


def test_truncation_error_is_the_next_order():
    """A strong drive (2 Omega tau = 0.9): what order K misses against QuTiP is order K + 1, to within order K + 2."""
    Q = Reference()
    H_levels, r_levels, H_0 = One_Qubit_Problem(seed=3)
    P = Perturbation(N_qubits=1)
    P.Add_Hamiltonian(0.3*SIGMA_X)
    rho_q_0 = Density(np.random.default_rng(8), 2)
    sigma_0, r_0 = Mode_Initial(MODES)
    t = np.linspace(0, 1.5, 31)
    result = PT.Perturbation_Dynamics(1, r_0, sigma_0, rho_q_0, H_levels, r_levels, H_0, np.zeros((2, 2)), P, t,
                                      order=3)
    rho = Q.Reference_Dynamics_Levels(2, 1, 30, MODES, H_levels, r_levels, H_0, rho_q_0, t[[0, -1]],
                                      H_dv=0.3*SIGMA_X)[0][1]
    orders = result.rho_q_t_orders[:, -1]
    for K in range(2):
        missing = np.max(np.abs(orders[:K + 1].sum(0) - rho))
        assert abs(missing - np.max(np.abs(orders[K + 1]))) < 1.5*np.max(np.abs(orders[K + 2])) + 1e-3
    assert np.max(np.abs(orders.sum(0) - rho)) < 1.5*np.max(np.abs(orders[3]))
    assert result.last_order_size == pytest.approx(np.max(np.abs(result.rho_q_t_orders[3])))


################################## Monte Carlo ##################################

def Mixed_Problem():
    H_levels, r_levels, H_0 = One_Qubit_Problem(seed=3)
    P = Perturbation(N_qubits=1)
    P.Add_Hamiltonian(0.3*SIGMA_X)
    P.Add_T1([0.2])
    rho_q_0 = np.array([[0.7, 0.3 - 0.2j], [0.3 + 0.2j, 0.3]])
    return (1, np.array([[0.3], [-0.2]]), np.diag([1.2, 0.9]), rho_q_0, H_levels, r_levels, H_0, np.zeros((2, 2)), P)


def test_monte_carlo_agrees_with_quadrature():
    """
    Within 5 standard errors plus the trapezoid error, from t = 0.5 on.  Before that few samples have all their
    jumps in [0, t] (a fraction (t / t_end)^k), and with none the standard error is 0.
    """
    args = Mixed_Problem()
    t = np.linspace(0, 1.5, 41)
    later = t >= 0.5
    quadrature = PT.Perturbation_Dynamics(*args, t, order=2)
    sampled = PT.Perturbation_Dynamics(*args, t, order=2, method='monte_carlo', n_samples=600, seed=1)
    assert np.allclose(sampled.rho_q_t_orders[0], quadrature.rho_q_t_orders[0])
    for k in (1, 2):
        difference = np.abs(sampled.rho_q_t_orders[k] - quadrature.rho_q_t_orders[k])[later]
        stderr = sampled.rho_q_t_stderr[k][later]
        assert np.all(stderr > 0)
        assert np.all(difference <= 5*stderr + 2e-4)
        assert np.mean(difference/stderr) < 1.5
    assert np.all(quadrature.rho_q_t_stderr == 0)


def test_monte_carlo_is_reproducible_with_a_seed():
    args = Mixed_Problem()
    t = np.linspace(0, 1, 11)
    a = PT.Perturbation_Dynamics(*args, t, order=1, method='monte_carlo', n_samples=50, seed=4)
    b = PT.Perturbation_Dynamics(*args, t, order=1, method='monte_carlo', n_samples=50, seed=4)
    assert np.array_equal(a.rho_q_t_total, b.rho_q_t_total)


################################## Phase space and the Quantum_State API ##################################

def test_wigner_of_the_components_integrates_to_their_trace():
    """Each W_jk integrates to 2^n rho_jk (the Wigner_Functions normalisation), off-diagonal ones included."""
    result = PT.Perturbation_Dynamics(*Mixed_Problem(), np.linspace(0, 1.5, 21), order=2)
    x = np.linspace(-7, 7, 81)
    X, P = np.meshgrid(x, x)
    grid = np.stack([X, P], -1)[..., None]
    area = (x[1] - x[0])**2
    perturbative = result.rho_q_t_orders[1:, -1].sum(0)
    W = PT.Wigner_Components(result.components[20], grid, 1, part='diagonal')
    assert W.sum()*area/2 == pytest.approx(np.trace(perturbative).real, abs=1e-9)
    Pi = np.array([[1, 1], [1, 1]])/2
    W = PT.Wigner_Components(result.components[20], grid, 1, projector=Pi)
    assert W.sum()*area/2 == pytest.approx(np.trace(Pi @ perturbative).real, abs=1e-9)
    W_all = PT.Wigner_Components(result.components[20], grid, 1, part='all')
    W_split = sum(PT.Wigner_Components(result.components[20], grid, 1, part=p) for p in ('diagonal', 'off_diagonal'))
    assert np.allclose(W_all, W_split)


def test_quantum_state_api_for_a_qudit():
    rng = np.random.default_rng(9)
    H = Hamiltonian(None, 1)
    H.Initialize_Qudit_Hamiltonians([np.eye(2) + Symmetric(rng, 2, 0.1) for _ in range(3)],
                                    [0.2*rng.normal(size=(2, 1)) for _ in range(3)], [0.0, 0.5, 1.2])
    P = Perturbation(N_levels=3)
    P.Add_Ladder_Decay([0.4, 0.2])
    G = Quantum_State(None, 1)
    G.Initialize_Gaussian_State(np.zeros((2, 1)), np.eye(2), Density(rng, 3))
    t = np.linspace(0, 1, 21)
    G.Perturbed_Dynamics(H, P, t, order=2, store_times=[10, -1])
    assert sorted(G.perturbation_components) == [10, 20]
    assert np.allclose(G.rho_q_t_total, G.rho_q_t_orders.sum(0))
    assert np.allclose(G.rho_q_t, G.rho_q_t_orders[0])
    assert np.trace(G.rho_q_t_total[-1]).real == pytest.approx(1, abs=1e-4)
    W, x, p = G.Wigner_Function_Perturbation(41, -1, 0, 3, part='diagonal')
    assert W.shape == (len(p), len(x))
    with pytest.raises(ValueError, match='store_times'):
        G.Wigner_Function_Perturbation(41, 5, 0, 3)


def test_quantum_state_api_for_a_register():
    H_levels, r_levels, H_0 = One_Qubit_Problem(seed=3, linear=True)
    H = Hamiltonian(1, 1)
    H.Initialize_Constant_Hamiltonians([H_levels[0]], [0.5*(r_levels[0] + r_levels[1]), 0.5*(r_levels[1] - r_levels[0])],
                                       np.array([0.2]))
    assert H.type == ['Constant', 'Force']
    P = Perturbation(N_qubits=1)
    P.Add_T1([0.3])
    G = Quantum_State(1, 1)
    G.Initialize_Gaussian_State(np.zeros((2, 1)), np.eye(2), np.ones((2, 2))/2)
    result = G.Perturbed_Dynamics(H, P, np.linspace(0, 1, 21), order=1)
    direct = PT.Perturbation_Dynamics(1, np.zeros((2, 1)), np.eye(2), np.ones((2, 2))/2, H_levels, r_levels, H_0,
                                      np.zeros((2, 2)), P, np.linspace(0, 1, 21), order=1)
    assert np.allclose(result.rho_q_t_total, direct.rho_q_t_total, atol=1e-13)
    assert np.allclose(result.second_moments_t, direct.second_moments_t, atol=1e-13)
