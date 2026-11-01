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

# ln det X is summed from the principal logs of the eigenvalues of X, which gives the continuous
# value, and so the right sign of the QRDM element, only if X(t) stays on one branch of the log for
# the whole step.  A step is accepted only if ||X(t) - 1|| < 1 for every t in it: X(t) is then
# invertible with a continuous Log, Tr Log X(t) is the continuous ln det X(t), and X(h) has
# condition number at most 7.  Eigenvalues at the end of the step alone do not show this: for a
# non-normal X they can exchange around 0 while det X winds by 2 pi.
#
# The condition is enforced on all of [0, h] as follows.  Z(t) = X(t) - 1 has Z(0) = 0.  Its quadratic
# interpolant through t = 0, h/2, h is bounded by 9/8 of the larger sample, and differs from Z by at
# most INTERPOLATION_BOUND ||A h||^3 exp(||A h||) ||(1, Q sigma)||, all Frobenius norms, with Q from
# Observed_Rows.  So samples below MAX_X_DISTANCE and a remainder below MAX_INTERPOLATION_ERROR give
# 9/8 * 0.75 + 0.15 < 1.
MAX_X_DISTANCE = 0.75
MAX_INTERPOLATION_ERROR = 0.15
INTERPOLATION_BOUND = np.sqrt(3)/216

# Singular values below this, relative to ||A||, are round-off in H_J - H_K rather than coupling.
OBSERVED_TOLERANCE = 1e-13

# Halvings before giving up.  This means X is singular or sigma is too large to resolve, to working
# precision.  sigma ~ 1e16 still passes, 1e20 does not.  A singular X is a pole of sigma, where ln det X
# in s diverges too: with r = 0, exp(s) ~ (det X)^(-1/2) grows without bound.
MAX_HALVINGS = 60

# Relative to the largest entry.  Loose enough for the complex64 U_N floor of ~1e-8 in Ladder runs.
SYMMETRY_TOLERANCE = 1e-6


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
    """Halvings needed to bring ||A h|| to 1/2.  Most steps with a moderate sigma then pass at once."""
    norm = np.linalg.norm(A*h, 1)
    return 0 if norm <= 0.5 else int(np.ceil(np.log2(2*norm)))


def Observed_Rows(A):
    """
    Orthonormal rows Q spanning the rows of the upper-right block of every power of A.  X(t) only sees
    sigma through Q sigma, so a large sigma elsewhere, such as an unstable mode the qubits do not
    couple to, must not force small steps: it does not move X at all.
    """
    dim = A.shape[0]//2
    scale = np.linalg.norm(A)
    if scale == 0:
        return np.zeros((0, dim), dtype=np.complex128)
    scaled = A/scale
    power, blocks = scaled, []
    for _ in range(2*dim):
        blocks.append(power[:dim, dim:])
        power = scaled @ power
    _, values, rows = np.linalg.svd(np.vstack(blocks), full_matrices=False)
    return rows[values > OBSERVED_TOLERANCE]


class Step_Propagators:
    """Blocks of exp(M h / 2^level), computed on first use for each level."""

    def __init__(self, M, h):
        self.M, self.h = M, h
        self.dim = (M.shape[0] - 2)//2
        A = M[:2*self.dim, :2*self.dim]
        self.base_level = Base_Level(A, h)
        self.norm_A = np.linalg.norm(A)
        self.observed = Observed_Rows(A)
        self.identity = np.eye(self.dim)
        self.blocks = {}

    def __call__(self, level):
        if level not in self.blocks:
            dim = self.dim
            h_level = self.h/2**level
            P = linalg.expm(self.M*h_level)
            norm = self.norm_A*abs(h_level)
            self.blocks[level] = (h_level,
                                  P[:dim, :dim], P[:dim, dim:2*dim],
                                  P[dim:2*dim, :dim], P[dim:2*dim, dim:2*dim],
                                  P[:dim, 2*dim + 1], P[dim:2*dim, 2*dim + 1],
                                  P[2*dim, dim:2*dim], P[2*dim, 2*dim + 1],
                                  INTERPOLATION_BOUND*norm**3*np.exp(norm))
        return self.blocks[level]


def Accept_Step(propagators, level, sigma, X):
    """True if ||X(t) - 1|| < 1 on the whole step, see MAX_X_DISTANCE."""
    remainder = propagators(level)[-1]
    size = np.sqrt(propagators.dim + np.linalg.norm(propagators.observed @ sigma)**2)
    if not remainder*size <= MAX_INTERPOLATION_ERROR:
        return False
    if not np.linalg.norm(X - propagators.identity) <= MAX_X_DISTANCE:
        return False
    _, S_11_half, S_12_half, *_ = propagators(level + 1)
    return bool(np.linalg.norm(S_11_half + S_12_half @ sigma - propagators.identity) <= MAX_X_DISTANCE)


def Advance(propagators, level, sigma, r, rate):
    """
    Restart the linear system from X = 1, Y = sigma, x = 0, y = r and step it once,
    halving the step until X stays on one branch of the log.
    Returns the updated sigma and r, and the increment of s.
    """
    (h_level, S_11, S_12, S_21, S_22,
     p_x, p_y, w_y, w_0, _) = propagators(level)

    X = S_11 + S_12 @ sigma

    if not Accept_Step(propagators, level, sigma, X):
        if level - propagators.base_level >= MAX_HALVINGS:
            raise np.linalg.LinAlgError(f'X could not be kept near 1 with {MAX_HALVINGS} halvings: '
                                        'it is singular, or sigma is too large, to working precision')
        sigma, r, ds_1 = Advance(propagators, level + 1, sigma, r, rate)
        sigma, r, ds_2 = Advance(propagators, level + 1, sigma, r, rate)
        return sigma, r, ds_1 + ds_2

    Y = S_21 + S_22 @ sigma
    sigma_new = np.linalg.solve(X.T, Y.T).T
    sigma_new = 0.5*sigma_new + 0.5*sigma_new.T
    if not np.all(np.isfinite(sigma_new)):
        raise np.linalg.LinAlgError('sigma overflowed double precision')

    x = S_12 @ r + p_x
    r_new = S_22 @ r + p_y - sigma_new @ x
    w = w_y @ r + w_0

    ds = -x @ r_new + w - 0.5*np.sum(np.log(np.linalg.eigvals(X))) + h_level*rate
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
            raise np.linalg.LinAlgError(f'Stopped between t = {t_array[i]} and t = {t_array[i+1]}: {error}. '
                                        'sigma_jk and s_jk cannot be followed past this point.') from error
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


def Check_Symmetry(name, M, sign):
    """Raise unless M = sign M^T, on its last two axes, to SYMMETRY_TOLERANCE."""
    M = np.asarray(M)
    if np.max(np.abs(M - sign*np.swapaxes(M, -1, -2))) > SYMMETRY_TOLERANCE*max(1, np.max(np.abs(M))):
        raise ValueError(f"{name} must be {'symmetric' if sign > 0 else 'antisymmetric'}")


def Check_Inputs(N_qubits, sigma_0, H_array, E, D):
    if len(H_array) != N_qubits + 1:
        raise ValueError(f'H_array needs N_qubits + 1 = {N_qubits + 1} entries, not {len(H_array)}')
    for i, H in enumerate(H_array):
        Check_Symmetry(f'H_array[{i}]', H, 1)
    Check_Symmetry('sigma_0', sigma_0, 1)
    if E is not None:
        Check_Symmetry('E', E, -1)
    if D is not None:
        Check_Symmetry('D', D, 1)


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
    Raises ValueError for inputs the doubling cannot represent: it needs sigma symmetric and
    a^T = -dd, which symmetric H_array entries and an antisymmetric E provide.
    Raises numpy.linalg.LinAlgError if X_jk becomes singular or sigma_jk overflows, to working precision.
    """
    Check_Inputs(N_qubits, sigma_0, H_array, E, D)

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
