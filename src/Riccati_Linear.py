"""Linear solver for the General case (qubit-dependent H, constant in time).

Replaces the nested ODEs sigma -> r -> s of Tab. 1 and Tab. 3 by one linear system per
branch pair (j, k), generalising the doubling of Sec. subsec:riccati and App. app:doubling
to the first moments and the QRDM exponent:

    (X, Y)' = A (X, Y)                      X(0) = 1,  Y(0) = sigma_0
    (x, y)' = A (x, y) + (f1, f2)           x(0) = 0,  y(0) = r_0
    w'      = f2^T x - f1^T y               w(0) = 0

    sigma = Y X^-1
    r     = y - sigma x
    s(t) - s(0) = -x^T r + w - 1/2 ln det X + t (1/2 Tr[a] + c0)

with a the upper-left block of A, c0 = -i/2 (H^0_j - H^0_k) + dephasing, f1 = -i/2 (r_j - r_k)
and f2 = 1/2 Omega (2d - r_j - r_k).  All transposes are unconjugated.
"""
import numpy as np
from scipy import linalg as linalg

import Gaussian as Gaussian
import QIT_Functions as QIT_Functions
from Open_Operator_Gaussian import Compute_H_J, Compute_r_J

# Largest |arg| accepted for an eigenvalue of X over one step.  The eigenvalues of X must
# trace a path that stays on one branch of the log, otherwise ln det X, and so the sign of
# the QRDM element, is lost.  A step that does not satisfy this is halved.
MAX_EIGENVALUE_ANGLE = np.pi/2

# Largest |lambda_max/lambda_min| of X accepted over one step.  X is inverted, so this
# bounds the digits lost per step.
MAX_EIGENVALUE_RATIO = 1e3

# Halving deeper than this means X is singular to within the resolution of the grid:
# sigma has a pole, which is where Tr[rho_jk] vanishes and s_jk is undefined.
MAX_HALVINGS = 40


def Doubling_Matrix(n_modes, H_J, H_K, E=None, D=None):
    """
    4n x 4n matrix A generating (X, Y)' = A (X, Y), with sigma_jk = Y X^-1.

    D sits outside the factor 1/2 of the lower-left block: this is what Tab. 3 and direct
    differentiation of Y X^-1 require.  App. app:doubling as typed puts it inside.
    """
    dim = 2*n_modes
    Omega = Gaussian.Omega_N(n_modes)
    E = np.zeros((dim, dim)) if E is None else E
    D = np.zeros((dim, dim)) if D is None else D
    H_sum, H_diff = H_J + H_K, H_J - H_K

    a = 0.5*(H_sum + 2*E.T) @ Omega
    b = 0.5j*H_diff
    c = 0.5j*Omega @ H_diff @ Omega.T + D
    dd = 0.5*Omega @ (H_sum + 2*E)
    return np.block([[a, b], [c, dd]]).astype(np.complex128)


def Linear_Sources(n_modes, r_J, r_K, d=None):
    """Sources (f1, f2) of the affine system for (x, y), each of shape (2n,)."""
    Omega = Gaussian.Omega_N(n_modes)
    d = np.zeros((2*n_modes, 1)) if d is None else d
    f1 = -0.5j*(r_J - r_K)
    f2 = 0.5*Omega @ (2*d - r_J - r_K)
    return np.ravel(f1).astype(np.complex128), np.ravel(f2).astype(np.complex128)


def Step_Generator(A, f1, f2):
    """
    Generator of (x, y, w, 1) as a (4n + 2)-dimensional homogeneous linear system.
    Its upper-left block is A, so its exponential also propagates (X, Y).
    """
    dim = len(f1)
    M = np.zeros((2*dim + 2, 2*dim + 2), dtype=np.complex128)
    M[:2*dim, :2*dim] = A
    M[:2*dim, 2*dim + 1] = np.concatenate((f1, f2))
    M[2*dim, :2*dim] = np.concatenate((f2, -f1))
    return M


def Base_Level(A, h):
    """Halvings needed to bring ||A h|| to order one, so that Y X^-1 keeps its digits."""
    norm = np.linalg.norm(A*h, 1)
    return 0 if norm <= 1 else int(np.ceil(np.log2(norm)))


class Step_Propagators:
    """Blocks of exp(M h / 2^level), computed on first use for each level."""

    def __init__(self, M, h):
        self.M, self.h = M, h
        self.dim = (M.shape[0] - 2)//2
        self.base_level = Base_Level(M[:2*self.dim, :2*self.dim], h)
        self.blocks = {}

    def __call__(self, level):
        if level not in self.blocks:
            dim = self.dim
            h_level = self.h/2**level
            P = linalg.expm(self.M*h_level)
            self.blocks[level] = (h_level,
                                  P[:dim, :dim], P[:dim, dim:2*dim],
                                  P[dim:2*dim, :dim], P[dim:2*dim, dim:2*dim],
                                  P[:dim, 2*dim + 1], P[dim:2*dim, 2*dim + 1],
                                  P[2*dim, dim:2*dim], P[2*dim, 2*dim + 1])
        return self.blocks[level]


def Accept_X(eigenvalues):
    magnitude = np.abs(eigenvalues)
    return (np.all(np.isfinite(eigenvalues))
            and np.min(magnitude) > 0
            and np.max(magnitude) < MAX_EIGENVALUE_RATIO*np.min(magnitude)
            and np.max(np.abs(np.angle(eigenvalues))) < MAX_EIGENVALUE_ANGLE)


def Advance(propagators, level, sigma, r, rate):
    """
    Restart the linear system from X = 1, Y = sigma, x = 0, y = r and step it once,
    halving the step until X is safe to invert.
    Returns the updated sigma and r, and the increment of s.
    """
    (h_level, S_11, S_12, S_21, S_22,
     p_x, p_y, w_y, w_0) = propagators(level)

    X = S_11 + S_12 @ sigma
    eigenvalues = np.linalg.eigvals(X)

    if not Accept_X(eigenvalues):
        if level - propagators.base_level >= MAX_HALVINGS:
            raise np.linalg.LinAlgError('X is numerically singular')
        sigma, r, ds_1 = Advance(propagators, level + 1, sigma, r, rate)
        sigma, r, ds_2 = Advance(propagators, level + 1, sigma, r, rate)
        return sigma, r, ds_1 + ds_2

    Y = S_21 + S_22 @ sigma
    sigma_new = np.linalg.solve(X.T, Y.T).T
    sigma_new = 0.5*(sigma_new + sigma_new.T)

    x = S_12 @ r + p_x
    r_new = S_22 @ r + p_y - sigma_new @ x
    w = w_y @ r + w_0

    ds = -x @ r_new + w - 0.5*np.sum(np.log(eigenvalues)) + h_level*rate
    return sigma_new, r_new, ds


def Branch_Trajectory(n_modes, sigma_0, r_0, A, f1, f2, c_0, steps, step_index, t_array):
    """
    sigma, r and s - s(0) of one branch pair on t_array.

    steps      : distinct step sizes of the grid
    step_index : for each grid interval, its position in steps
    """
    T = len(t_array)
    dim = 2*n_modes
    sigma_t = np.zeros((T, dim, dim), dtype=np.complex128)
    r_t = np.zeros((T, dim), dtype=np.complex128)
    s_t = np.zeros(T, dtype=np.complex128)

    M = Step_Generator(A, f1, f2)
    rate = 0.5*np.trace(A[:dim, :dim]) + c_0

    propagators = [Step_Propagators(M, h) for h in steps]

    sigma = 0.5*(sigma_0 + sigma_0.T).astype(np.complex128)
    r = np.ravel(r_0).astype(np.complex128)
    s = 0j
    sigma_t[0], r_t[0] = sigma, r

    for i, index in enumerate(step_index):
        propagator = propagators[index]
        level = propagator.base_level
        try:
            for _ in range(2**level):
                sigma, r, ds = Advance(propagator, level, sigma, r, rate)
                s += ds
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                f'X_jk is singular between t = {t_array[i]} and t = {t_array[i+1]}: sigma_jk has '
                'a pole there, which means Tr[rho_jk] vanishes and s_jk is undefined.') from error
        sigma_t[i + 1], r_t[i + 1], s_t[i + 1] = sigma, r, s

    return sigma_t, r_t, s_t


def Grid_Steps(t_array):
    """
    Group the intervals of t_array by size, so that a uniform grid needs one propagator.
    Group means are used: they keep the sum of the steps, and so the final time, exact.
    """
    dt = np.diff(t_array)
    if len(dt) == 0:
        return np.zeros(0), np.zeros(0, dtype=int)
    scale = np.max(np.abs(dt))
    if scale == 0:
        return np.zeros(1), np.zeros(len(dt), dtype=int)
    _, step_index = np.unique(np.round(dt/scale, 12), return_inverse=True)
    steps = np.bincount(step_index, weights=dt)/np.bincount(step_index)
    return steps, step_index


def Dynamics_Linear(N_qubits, n_modes, r_0, sigma_0, H_array, r_ham_array, H_q_0_array, rho_q_0, t_array,
                    E=None, D=None, d=None, Gamma_z=None):
    """
    Same inputs, outputs and conventions as Open_Operator_Gaussian.Dynamics_Numerical, with
    E = D = d = Gamma_z = None giving the unitary case.

    r_0     : (2n, 1)                   - single IC for all branches, OR
              (2^N, 2^N, 2n, 1)         - per-branch IC
    sigma_0 : (2n, 2n)                  - single IC for all branches, OR
              (2^N, 2^N, 2n, 2n)        - per-branch IC
    rho_q_0 : (2^N, 2^N)                - initial QRDM.  C and phi are increments from t_array[0],
              so they do not depend on it, and zero elements are allowed.
    E, D    : (2n, 2n) drift and diffusion (Gaussian.Get_Decoherence_Rates)
    d       : (2n, 1) Lindblad linear driving vector
    Gamma_z : (N,) qubit dephasing rates

    Returns sigma_JK_t, r_JK_t, C_JK_t, phi_JK_t.
    Raises numpy.linalg.LinAlgError if X_jk becomes singular.
    """
    N2 = 2**N_qubits
    dim = 2*n_modes
    t_array = np.asarray(t_array, dtype=float)
    T = len(t_array)
    H_q_0_array = np.ravel(H_q_0_array)
    Gamma_z = np.zeros(N_qubits) if Gamma_z is None else np.ravel(Gamma_z)

    if sigma_0.ndim == 4:
        sigma_0_jk = sigma_0
        r_0_jk = r_0
    else:
        sigma_0_jk = np.broadcast_to(sigma_0[np.newaxis, np.newaxis], (N2, N2, dim, dim))
        r_0_jk = np.broadcast_to(r_0[np.newaxis, np.newaxis], (N2, N2, dim, 1))

    sigma_JK_t = np.zeros((T, N2, N2, dim, dim), dtype=np.complex128)
    r_JK_t = np.zeros((T, N2, N2, dim, 1), dtype=np.complex128)
    C_JK_t = np.zeros((T, N2, N2), dtype=np.complex128)
    phi_JK_t = np.zeros((T, N2, N2), dtype=np.complex128)

    steps, step_index = Grid_Steps(t_array)

    for j in range(N2):
        J_labels = QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, j)
        r_J = Compute_r_J(N_qubits, J_labels, r_ham_array)
        H_J = Compute_H_J(N_qubits, J_labels, H_array)

        for k in range(j, N2):
            K_labels = QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, k)
            r_K = Compute_r_J(N_qubits, K_labels, r_ham_array)
            H_K = Compute_H_J(N_qubits, K_labels, H_array)

            A = Doubling_Matrix(n_modes, H_J, H_K, E, D)
            f1, f2 = Linear_Sources(n_modes, r_J, r_K, d)
            c_0 = -0.5j*(J_labels - K_labels) @ H_q_0_array + 0.5*np.real(np.sum(Gamma_z*(J_labels*K_labels - 1)))

            sigma_t, r_t, s_t = Branch_Trajectory(n_modes, sigma_0_jk[j, k], r_0_jk[j, k], A, f1, f2, c_0,
                                                  steps, step_index, t_array)

            sigma_JK_t[:, j, k] = sigma_t
            r_JK_t[:, j, k, :, 0] = r_t
            C_JK_t[:, j, k] = -np.real(s_t)
            phi_JK_t[:, j, k] = np.imag(s_t)

            sigma_JK_t[:, k, j] = np.conjugate(sigma_JK_t[:, j, k])
            r_JK_t[:, k, j] = np.conjugate(r_JK_t[:, j, k])
            C_JK_t[:, k, j] = C_JK_t[:, j, k]
            phi_JK_t[:, k, j] = -phi_JK_t[:, j, k]

    return sigma_JK_t, r_JK_t, C_JK_t, phi_JK_t
