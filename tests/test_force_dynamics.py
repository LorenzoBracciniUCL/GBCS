"""Closed-form regression tests for the operator-valued-force dynamics.

Every reference value is the analytical Stern-Gerlach solution of Appendix I B, written
in the paper's own convention Omega_1 = [[0,1],[-1,0]] (Sec. II).  Where the printed
appendix disagrees with that convention the correction is noted; see the errata table in
notebooks/Wigner_SG_Decay.ipynb.

These guard the four fixes to Open_Operator_Force / Unitary_Operator_Force.  Before them:
  * np.exp instead of linalg.expm  -> diffusive sigma diverges, max|num-ana| = 2.9e+04 at 2pi
  * missing 1/4 on the C integral  -> C(2pi) too large by exactly 4x
  * complex64 storage              -> a 2e-07 floor on every first moment
"""
import numpy as np
import pytest

from Objects import Quantum_State, Hamiltonian, Decoherence

F_U, F_Q, N_P, S, GAMMA_X = 0.5, 2.0, 0.8, 2.0, 0.1
TOL = 1e-12


def _state(gamma_x, t_array):
    H = Hamiltonian(1, 1)
    H.Initialize_Constant_Hamiltonians(
        [np.eye(2)], [np.array([[F_U], [0.0]]), np.array([[F_Q], [0.0]])],
        np.array([0.0 + 0j]))
    G = Quantum_State(1, 1)
    G.Initialize_Gaussian_State(np.zeros((2, 1)),
                                (1 + 2*N_P)*np.diag([S, 1/S]), np.ones((2, 2))/2)
    if gamma_x == 0.0:
        G.Unitary_Dynamics_Numerical(H, t_array)
    else:
        G.Open_Dynamics_Numerical(
            H, Decoherence(1, 1, 'Canonical', gamma_x*np.array([[1., 0.], [0., 0.]])),
            t_array)
    return G


def sigma_ana(tau, gx):
    c, sn = np.cos(tau), np.sin(tau)
    return ((1 + 2*N_P)*np.array([[S*c**2 + sn**2/S, (1/S - S)*sn*c],
                                  [(1/S - S)*sn*c,   S*sn**2 + c**2/S]])
            + gx*np.array([[tau - sn*c, sn**2], [sn**2, tau + sn*c]]))


def r_on_ana(tau, J):
    """(f_u + J f_q)(1 - cos tau, +sin tau).  The appendix prints (f_u -+ f_q) and
    -sin tau; both follow from its transposed S_qho and are errata."""
    c, sn = np.cos(tau), np.sin(tau)
    return (F_U + J*F_Q)*np.array([1 - c, sn])


def r_off_ana(tau, gx):
    """Correct as printed; it is the (j,k) = (-1,+1) element, i.e. r_JK_t[:,0,1]."""
    c, sn = np.cos(tau), np.sin(tau)
    s2 = np.sin(tau/2)**2
    return (F_U*np.array([1 - c, sn])
            + 1j*F_Q*((1 + 2*N_P)*np.array([sn*((1/S - S)*c - 1/S),
                                            2*s2*((S - 1/S)*c + S)])
                      + gx*np.array([-4*s2**2, tau + sn*(c - 2)])))


def C_ana(tau, gx):
    c, sn = np.cos(tau), np.sin(tau)
    s2 = np.sin(tau/2)**2
    return F_Q**2*(2*(1 + 2*N_P)*s2*((S - 1/S)*c + S + 1/S)
                   + gx*(3*tau + sn*(c - 4)))


def phi_ana(tau):
    """2 f_q f_u (tau - sin tau), the (+1,-1) element.  The appendix prints (tau + sin
    tau); the sign follows its transposed S_qho."""
    return 2*F_Q*F_U*(tau - np.sin(tau))


@pytest.mark.parametrize('gx', [0.0, GAMMA_X], ids=['unitary', 'diffusive'])
def test_sg_closed_form(gx):
    t = np.linspace(0.15, 2*np.pi, 25)
    G = _state(gx, t)
    for i, tau in enumerate(t):
        assert np.max(np.abs(np.real(G.sigma_JK_t[i, 0, 0]) - sigma_ana(tau, gx))) < TOL
        assert np.max(np.abs(np.real(G.r_JK_t[i, 1, 1, :, 0]) - r_on_ana(tau, +1))) < TOL
        assert np.max(np.abs(np.real(G.r_JK_t[i, 0, 0, :, 0]) - r_on_ana(tau, -1))) < TOL
        assert np.max(np.abs(G.r_JK_t[i, 0, 1, :, 0] - r_off_ana(tau, gx))) < TOL
        assert abs(G.C_JK_t[i, 0, 1] - C_ana(tau, gx)) < 1e-11
        assert abs(G.phi_JK_t[i, 1, 0] - phi_ana(tau)) < TOL


def test_C_quarter_factor_on_the_diffusive_integral():
    """C(2pi) = 6 pi f_q^2 Gamma_x exactly (Sec. IX B).  Without the 1/4 on the
    diffusive integral this is 4x too large."""
    t = np.array([2*np.pi])
    G = _state(GAMMA_X, t)
    assert abs(G.C_JK_t[0, 0, 1] - 6*np.pi*F_Q**2*GAMMA_X) < 1e-11


def test_first_moments_are_double_precision():
    """complex64 storage put a ~2e-07 floor on r_JK_t."""
    G = _state(GAMMA_X, np.array([1.0]))
    assert G.r_JK_t.dtype == np.complex128


def test_branch_labels():
    """Extract_Qubit_Labels_Array maps 2*bit-1: index 0 -> J=-1, index 1 -> J=+1."""
    G = _state(GAMMA_X, np.array([np.pi]))
    assert np.real(G.r_JK_t[0, 0, 0, 0, 0]) == pytest.approx(2*(F_U - F_Q), abs=1e-10)
    assert np.real(G.r_JK_t[0, 1, 1, 0, 0]) == pytest.approx(2*(F_U + F_Q), abs=1e-10)


def test_hermitian_conjugate_symmetry_two_qubits():
    """sigma_KJ = sigma_JK*, r_KJ = r_JK*, C_KJ = C_JK, phi_KJ = -phi_JK."""
    H = Hamiltonian(2, 1)
    H.Initialize_Constant_Hamiltonians(
        [np.eye(2)], [np.array([[0.3], [0.]]), np.array([[1.], [0.]]),
                      np.array([[0.], [0.7]])], np.zeros(2, dtype=complex))
    G = Quantum_State(2, 1)
    G.Initialize_Gaussian_State(np.zeros((2, 1)), np.eye(2), np.ones((4, 4))/4)
    dec = Decoherence(2, 1, 'Canonical', 0.05*np.eye(2))
    G.Open_Dynamics_Numerical(H, dec, np.linspace(0.1, 3, 8))
    r, C, ph = G.r_JK_t, G.C_JK_t, G.phi_JK_t
    assert np.max(np.abs(r - np.conj(np.swapaxes(r, 1, 2)))) < 1e-12
    assert np.max(np.abs(C - np.swapaxes(C, 1, 2))) < 1e-12
    assert np.max(np.abs(ph + np.swapaxes(ph, 1, 2))) < 1e-12


def test_no_superposition_limit():
    """f_q = 0: both branches coincide, C = 0, phi = 0, r^off is real."""
    H = Hamiltonian(1, 1)
    H.Initialize_Constant_Hamiltonians(
        [np.eye(2)], [np.array([[F_U], [0.0]]), np.zeros((2, 1))], np.array([0.0 + 0j]))
    G = Quantum_State(1, 1)
    G.Initialize_Gaussian_State(np.zeros((2, 1)), (1 + 2*N_P)*np.diag([S, 1/S]),
                                np.ones((2, 2))/2)
    t = np.linspace(0.15, 2*np.pi, 9)
    G.Open_Dynamics_Numerical(
        H, Decoherence(1, 1, 'Canonical', GAMMA_X*np.array([[1., 0.], [0., 0.]])), t)
    assert np.max(np.abs(G.C_JK_t)) < 1e-12
    assert np.max(np.abs(G.phi_JK_t)) < 1e-12
    assert np.max(np.abs(G.r_JK_t.imag)) < 1e-12
    assert np.max(np.abs(G.r_JK_t[:, 0, 0] - G.r_JK_t[:, 1, 1])) < 1e-12


def test_no_squeezing_no_diffusion_limit():
    """s = 1, Gamma_x = 0, N_p = 0: sigma stays the identity for all time."""
    H = Hamiltonian(1, 1)
    H.Initialize_Constant_Hamiltonians(
        [np.eye(2)], [np.array([[F_U], [0.0]]), np.array([[F_Q], [0.0]])],
        np.array([0.0 + 0j]))
    G = Quantum_State(1, 1)
    G.Initialize_Gaussian_State(np.zeros((2, 1)), np.eye(2), np.ones((2, 2))/2)
    t = np.linspace(0.15, 2*np.pi, 9)
    G.Unitary_Dynamics_Numerical(H, t)
    for i in range(len(t)):
        assert np.max(np.abs(np.real(G.sigma_JK_t[i, 0, 0]) - np.eye(2))) < TOL
