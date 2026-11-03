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
# Observed_Rows, and with the rows it dropped counted through their size.  So samples below
# MAX_X_DISTANCE and a remainder below MAX_INTERPOLATION_ERROR give 9/8 * 0.75 + 0.15 < 1.
# All of this holds for X as defined; rounding of X is the subject of the next block.
MAX_X_DISTANCE = 0.75
MAX_INTERPOLATION_ERROR = 0.15
INTERPOLATION_BOUND = np.sqrt(3)/216

# Singular values below this, relative to ||A||, are round-off in H_J - H_K rather than coupling.
OBSERVED_TOLERANCE = 1e-13

# X only depends on sigma through Q sigma, but it is computed entry by entry as S_12 @ sigma, and r_new =
# y - sigma_new x and -x^T r cancel against sigma the same way.  If sigma has large entries around a
# thin direction, these cancellations are lost to rounding, and three guards detect it:
#
# 1. eps max |sigma^-1| |sigma|, on sigma alone.  It is eps for a diagonal sigma however large, and for a hot
#    isotropic one, and grows as eps s^2 for a state squeezed by s in a tilted frame.  It catches what the
#    two below cannot: they scale with S_12, and are 0 for a pair that differs only in its force.
#    It only grows that way when the fat and thin directions share entries.  A beam splitter keeps x and p
#    apart, so a state squeezed in x and p stays at eps s under it, and there 2. and 3. do the work.
#    It does not keep growing.  Once the thin eigenvalue is below eps ||sigma|| the stored matrix no longer
#    holds it, the inverse is capped, and the measure saturates: over random frames, phases and up to 4 modes
#    it stayed at 0.67 or above for every s >= 1e9.  MAX_SIGMA_CONDITIONING = 0.3 sits under that floor, and
#    on the rising branch near s = 5e7.  A value above it says nothing about how bad sigma is, and an evolving
#    sigma cannot jump over the limit, as ||A h|| <= 1/2 bounds its growth per step.
#
# 2. and 3. Two running estimates, which do not change when a step is halved:
#   of the relative error in s and sigma:  e_s = sum eps max |S_12| |sigma|
#   of the error in r:                     e_r = sum e |sigma| |x| / (1 + |r|), with e the term above.
# e_r is the larger effect.  X feeds sigma_new = Y X^-1 with eps sigma^2 in a block that is really zero,
# and r_new = y - sigma_new x picks it up, so r can be wrong by 1 while rho_q is good to 1e-6.  These two keep
# working inside limit 1: sigma ~ 1e6 to 1e7 passes it, and a long run accumulates error all the same.
# The componentwise product stays at 0 on exact zeros in S_12, where a norm product would not, so the
# aligned case is quiet.  A large sigma that the coupling does see makes these estimates small, because
# it collapses within a step whose S_12 is small: that is a statement about the estimate.  Whether the result
# is accurate there is up to limit 1, as a tilted state of that kind can be wrong with both estimates small.
#
# The limits are measured, not derived.  On the bystander modes they were fitted to, at beam-splitter
# angles 0.1, 0.6, 1.2, e_r exceeds the true error in r by 9 to 20 near its limit, and e_s tracks the error in
# rho_q up to about 1e-6 and then falls short by up to 8.  Those ratios hold on that family only: other
# structures have shown e_s and e_r low by 1e3 or more, which is why limit 1 has to carry the general case.
#
# Neither estimate, nor limit 1, is computed while |sigma| stays below NEGLIGIBLE_SIGMA, where eps s^2 ~ 2e-10
# and the sum of eps |S_12| |sigma| would need a coupled evolution of order 1e6 ||H_J - H_K||^-1 to reach its
# limit.  This keeps the bookkeeping to one abs and max per step in the ordinary case.
UNIT_ROUNDOFF = np.finfo(float).eps
NEGLIGIBLE_SIGMA = 1e3
MAX_SIGMA_CONDITIONING = 0.3
MAX_ROUNDING_ERROR_S = 1e-6
MAX_ROUNDING_ERROR_R = 3e-5

# Substeps one branch pair may take, as a fixed allowance plus a factor on the steps the grid and ||A||
# ask for.  Each pole of sigma costs about 100 extra substeps, so a legitimate run stays far below this.
SUBSTEP_ALLOWANCE = 100000
SUBSTEPS_PER_BASE_STEP = 1000

# Halvings before giving up.  This means X is singular or sigma is too large to resolve, to working
# precision.  Measured on a diagonal sigma the coupling sees, s ~ 1e18 passes and 1e19 does not.
# A singular X is a pole of sigma, where ln det X in s diverges too: with r = 0, exp(s) ~ (det X)^(-1/2)
# grows without bound.
MAX_HALVINGS = 60

# Relative to the size of each matrix on its own.  Loose enough for the complex64 U_N floor of ~1e-8 in
# Ladder runs.  E and D come from one B, so there they are relative to the larger of the two.
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
    Also returns the largest singular value it dropped, which is 0 when the zeros are exact.
    """
    dim = A.shape[0]//2
    scale = np.linalg.norm(A)
    if scale == 0:
        return np.zeros((0, dim), dtype=np.complex128), 0.0
    scaled = A/scale
    power, blocks = scaled, []
    for _ in range(2*dim):
        blocks.append(power[:dim, dim:])
        power = scaled @ power
    _, values, rows = np.linalg.svd(np.vstack(blocks), full_matrices=False)
    kept = values > OBSERVED_TOLERANCE
    return rows[kept], np.max(values[~kept], initial=0.0)


class Step_Propagators:
    """Blocks of exp(M h / 2^level), computed on first use for each level."""

    def __init__(self, M, h):
        self.M, self.h = M, h
        self.dim = (M.shape[0] - 2)//2
        A = M[:2*self.dim, :2*self.dim]
        self.base_level = Base_Level(A, h)
        self.norm_A = np.linalg.norm(A)
        self.observed, self.dropped = Observed_Rows(A)
        self.coupled = np.any(A[:self.dim, self.dim:] != 0)
        self.sensitive = self.coupled or np.any(M[:self.dim, 2*self.dim + 1] != 0)
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
                                  INTERPOLATION_BOUND*norm**3*np.exp(norm),
                                  np.abs(P[:dim, dim:2*dim]))
        return self.blocks[level]


def Accept_Step(propagators, level, sigma, X):
    """True if ||X(t) - 1|| < 1 on the whole step, see MAX_X_DISTANCE."""
    remainder = propagators(level)[-2]
    size = propagators.dim + np.linalg.norm(propagators.observed @ sigma)**2
    if propagators.dropped > 0:
        size += (propagators.dropped*np.linalg.norm(sigma))**2
    size = np.sqrt(size)
    if not remainder*size <= MAX_INTERPOLATION_ERROR:
        return False
    if not np.linalg.norm(X - propagators.identity) <= MAX_X_DISTANCE:
        return False
    _, S_11_half, S_12_half, *_ = propagators(level + 1)
    return bool(np.linalg.norm(S_11_half + S_12_half @ sigma - propagators.identity) <= MAX_X_DISTANCE)


def Sigma_Conditioning(sigma, abs_sigma):
    """eps max |sigma^-1| |sigma|, which is eps for any diagonal sigma, however large."""
    try:
        inverse = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        return np.inf
    conditioning = UNIT_ROUNDOFF*(np.abs(inverse) @ abs_sigma).max()
    return conditioning if np.isfinite(conditioning) else np.inf


def Check_Sigma_Conditioning(sigma, abs_sigma):
    conditioning = Sigma_Conditioning(sigma, abs_sigma)
    if conditioning > MAX_SIGMA_CONDITIONING:
        raise np.linalg.LinAlgError(f'sigma is too ill-conditioned in this frame, to working precision: eps max |sigma^-1| '
                                    f'|sigma| = {conditioning:.1e}.  A frame in which sigma is diagonal avoids this')


class Step_Ledger:
    """Running totals for one branch pair: work done, and the rounding error of X accumulated into s."""

    def __init__(self, limit):
        self.limit = limit
        self.substeps = 0
        self.rounding = 0.0
        self.rounding_r = 0.0


def Advance(propagators, level, sigma, r, rate, ledger):
    """
    Restart the linear system from X = 1, Y = sigma, x = 0, y = r and step it once,
    halving the step until X stays on one branch of the log.
    Returns the updated sigma and r, and the increment of s.
    """
    (h_level, S_11, S_12, S_21, S_22,
     p_x, p_y, w_y, w_0, _, abs_S_12) = propagators(level)

    ledger.substeps += 1
    if ledger.substeps > ledger.limit:
        raise np.linalg.LinAlgError(f'gave up after {ledger.limit} substeps: sigma_jk changes too fast to follow')

    X = S_11 + S_12 @ sigma

    if not Accept_Step(propagators, level, sigma, X):
        if level - propagators.base_level >= MAX_HALVINGS:
            if not np.all(np.isfinite(sigma)):
                raise np.linalg.LinAlgError('sigma overflowed double precision')
            raise np.linalg.LinAlgError(f'X could not be kept near 1 with {MAX_HALVINGS} halvings: '
                                        'it is singular, or sigma is too large, to working precision')
        sigma, r, ds_1 = Advance(propagators, level + 1, sigma, r, rate, ledger)
        sigma, r, ds_2 = Advance(propagators, level + 1, sigma, r, rate, ledger)
        return sigma, r, ds_1 + ds_2

    rounding = 0.0
    if propagators.sensitive:
        abs_sigma = np.abs(sigma)
        if abs_sigma.max() > NEGLIGIBLE_SIGMA:
            Check_Sigma_Conditioning(sigma, abs_sigma)
            if propagators.coupled:
                rounding = UNIT_ROUNDOFF*(abs_S_12 @ abs_sigma).max()
                ledger.rounding += rounding

    Y = S_21 + S_22 @ sigma
    sigma_new = np.linalg.solve(X.T, Y.T).T
    sigma_new = 0.5*sigma_new + 0.5*sigma_new.T

    x = S_12 @ r + p_x
    r_new = S_22 @ r + p_y - sigma_new @ x
    w = w_y @ r + w_0

    ds = -x @ r_new + w - 0.5*np.sum(np.log(np.linalg.eigvals(X))) + h_level*rate
    if rounding:
        ledger.rounding_r += rounding*(np.abs(sigma_new) @ np.abs(x)).max()/(1 + np.abs(r_new).max())
        if ledger.rounding > MAX_ROUNDING_ERROR_S or ledger.rounding_r > MAX_ROUNDING_ERROR_R:
            raise np.linalg.LinAlgError('sigma is too large in a direction the coupling does not resolve, to working '
                                        f'precision: estimated rounding error {ledger.rounding:.1e} in s and '
                                        f'{ledger.rounding_r:.1e} in r.  A frame aligned with the uncoupled modes avoids this')
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
    base_substeps = sum(2**propagators[index].base_level for index in step_index)
    ledger = Step_Ledger(SUBSTEP_ALLOWANCE + SUBSTEPS_PER_BASE_STEP*base_substeps)

    sigma = 0.5*(sigma_0 + sigma_0.T).astype(np.complex128)
    r = np.ravel(r_0).astype(np.complex128)
    s = 0j
    sigma_t[0], r_t[0] = sigma, r

    for i, index in enumerate(step_index):
        propagator = propagators[index]
        level = propagator.base_level
        try:
            for _ in range(2**level):
                sigma, r, ds = Advance(propagator, level, sigma, r, rate, ledger)
                s += ds
            if not np.all(np.isfinite(sigma)):
                raise np.linalg.LinAlgError('sigma overflowed double precision')
            if not (np.all(np.isfinite(r)) and np.isfinite(s)):
                raise np.linalg.LinAlgError('r or s overflowed double precision')
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


def Check_Symmetry(name, M, sign, scale=None):
    """Raise unless M = sign M^T, for each matrix on the last two axes, to SYMMETRY_TOLERANCE."""
    M = np.asarray(M)
    residual = np.max(np.abs(M - sign*np.swapaxes(M, -1, -2)), axis=(-2, -1))
    if scale is None:
        scale = np.max(np.abs(M), axis=(-2, -1))
    if np.any(residual > SYMMETRY_TOLERANCE*scale):
        raise ValueError(f"{name} must be {'symmetric' if sign > 0 else 'antisymmetric'}")


def Check_Inputs(N_qubits, r_0, sigma_0, H_array, r_ham_array, H_q_0_array, rho_q_0, t_array, E, D, d, Gamma_z):
    for name, value in (('H_array', H_array), ('r_ham_array', r_ham_array)):
        if len(value) != N_qubits + 1:
            raise ValueError(f'{name} needs N_qubits + 1 = {N_qubits + 1} entries, not {len(value)}')

    named = [('r_0', r_0), ('sigma_0', sigma_0), ('H_q_0_array', H_q_0_array), ('rho_q_0', rho_q_0), ('t_array', t_array),
             ('E', E), ('D', D), ('d', d), ('Gamma_z', Gamma_z)]
    named += [(f'H_array[{i}]', H) for i, H in enumerate(H_array)]
    named += [(f'r_ham_array[{i}]', r) for i, r in enumerate(r_ham_array)]
    for name, value in named:
        if value is not None and not np.all(np.isfinite(value)):
            raise ValueError(f'{name} has non-finite entries')

    for i, H in enumerate(H_array):
        Check_Symmetry(f'H_array[{i}]', H, 1)
    Check_Symmetry('sigma_0', sigma_0, 1)

    scale = max(np.max(np.abs(M)) for M in (E, D, np.zeros(1)) if M is not None)
    if E is not None:
        Check_Symmetry('E', E, -1, scale)
    if D is not None:
        Check_Symmetry('D', D, 1, scale)
        if np.min(np.linalg.eigvalsh(0.5*(D + D.T))) < -SYMMETRY_TOLERANCE*scale:
            raise ValueError('D must be positive semidefinite')
    if d is not None and np.max(np.abs(np.imag(d))) > SYMMETRY_TOLERANCE*np.max(np.abs(d)):
        raise ValueError('d must be real')


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

    Regimes, as measured on families of test problems and not proved.  The measure is eps max |sigma^-1| |sigma|.
    Aligned, with sigma diagonal in the modes: the measure stays at eps, and states squeezed up to s ~ 1e18 match
    their closed forms where the coupling sees them.  On a mode the qubits do not touch, with exact zeros in
    H_array there, sigma may grow up to overflow.
    Not aligned: the run raises once the measure passes 0.3, near s ~ 5e7, or earlier if the running estimates of
    rounding in s and r pass their limits, which first happens for r near s ~ 3e6 on a bystander mode.  What returns
    was measured at no more than about 1e-4 on rho, r/(1 + |r|) and sigma.
    Such a raise is a limit of this algorithm in double precision, and not always of the input.  For rho it
    is close to one: rounding the input alone costs 1e-6 at s = 1e10.  For r it is not: the same stored doubles
    give r to 1e-7 in higher precision at s = 1e10, where this solver would be wrong by 20.

    Raises ValueError for inputs the doubling cannot represent, or non-finite ones: it needs sigma symmetric
    and a^T = -dd, which symmetric H_array entries and an antisymmetric E provide.
    Raises numpy.linalg.LinAlgError, naming the cause and the time, if X_jk becomes singular, sigma_jk, r_jk or s_jk
    overflow, the limits above are passed, or the substep budget runs out.
    """
    Check_Inputs(N_qubits, r_0, sigma_0, H_array, r_ham_array, H_q_0_array, rho_q_0, t_array, E, D, d, Gamma_z)

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
