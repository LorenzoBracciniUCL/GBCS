"""The library's ODE settings must be tight enough that the numerical solutions can be
compared against the paper's closed forms.

scipy's solve_ivp defaults (RK45, rtol=1e-3, atol=1e-6) left max|sigma_num - sigma_ana|
= 3.65e-03 at tau = pi on the General path, independent of the t_eval grid, which is
larger than most of the quantities these solutions are used to check.
"""
import numpy as np
import pytest

import Gaussian
from Objects import Quantum_State, Hamiltonian, Decoherence

F_U, F_Q, N_P, S, GAMMA_X = 0.5, 2.0, 0.8, 0.5, 0.05


def test_ode_settings_are_declared_once():
    assert Gaussian.ODE_KWARGS['method'] == 'DOP853'
    assert Gaussian.ODE_KWARGS['rtol'] <= 1e-10


def test_every_solve_ivp_uses_them():
    """A new call site that forgets the settings silently reintroduces rtol=1e-3."""
    import os
    import re
    src = os.path.join(os.path.dirname(__file__), '..', 'src')
    missing = []
    for fn in os.listdir(src):
        if not fn.endswith('.py'):
            continue
        text = open(os.path.join(src, fn)).read()
        for m in re.finditer(r'solve_ivp\(', text):
            i = m.end()
            depth, j = 1, i
            while depth:
                depth += (text[j] == '(') - (text[j] == ')')
                j += 1
            if 'ODE_KWARGS' not in text[i:j]:
                missing.append(f"{fn}:{text[:i].count(chr(10)) + 1}")
    assert not missing, f"solve_ivp without ODE_KWARGS at {missing}"


@pytest.mark.parametrize('H_array_len', [1, 2], ids=['force-path', 'general-path'])
def test_general_path_reaches_the_closed_form(H_array_len):
    """Both dispatch paths must reproduce sigma(pi), r^on and C of Appendix I B.

    Before the fix the General path (len(H_array) > 1) gave 3.65e-03 here.
    """
    H = Hamiltonian(1, 1)
    H_list = [np.eye(2)] + ([np.zeros((2, 2))] if H_array_len == 2 else [])
    H.Initialize_Constant_Hamiltonians(
        H_list, [F_U*np.array([[1.], [0.]]), F_Q*np.array([[1.], [0.]])],
        np.array([0.0 + 0j]))
    G = Quantum_State(1, 1)
    G.Initialize_Gaussian_State(np.zeros((2, 1)),
                                (1 + 2*N_P)*np.diag([S, 1/S]), np.ones((2, 2))/2)
    t = np.linspace(0, 2*np.pi, 401)
    G.Open_Dynamics_Numerical(
        H, Decoherence(1, 1, 'Canonical', GAMMA_X*np.array([[1., 0.], [0., 0.]])), t)
    i = int(np.argmin(np.abs(t - np.pi)))

    sigma_ana = np.diag([S*(1 + 2*N_P) + np.pi*GAMMA_X,
                         (1 + 2*N_P)/S + np.pi*GAMMA_X])
    assert np.max(np.abs(np.real(G.sigma_JK_t[i, 0, 0]) - sigma_ana)) < 1e-8
    assert np.max(np.abs(np.real(G.r_JK_t[i, 1, 1, :, 0])
                         - np.array([2*(F_U + F_Q), 0.0]))) < 1e-8
    C_ana = F_Q**2*(4*(1 + 2*N_P)/S + 3*np.pi*GAMMA_X)
    assert abs(G.C_JK_t[i, 0, 1] - C_ana) < 1e-6
