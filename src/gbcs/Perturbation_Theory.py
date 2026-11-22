"""
Dyson series for perturbations that act on the DV only (Sec. sec:perturbation, App. app:perturbation).

The master equation is split as  d rho / d tau = L_0 rho + V rho.

  L_0  is the operator-valued Gaussian dynamics of Tab. 1 and Tab. 3.  It does not mix the branches
       rho_jk = <j|rho|k>.  The diagonal part of the DV terms below is added to it, as a constant c_jk in ds_jk/dtau.
  V    is the rest of the DV terms, V[(j,k), (l,m)] rho_lm, and is the identity on the modes.

The DV terms are a DV Hamiltonian, -i[H_dv, rho], and DV jumps, L rho L^dag - 1/2 {L^dag L, rho}: T1 decay of
each qubit of a register, or the ladder decay |j-1><j| of a single N-level system.

Each term of the Dyson series is a path of branches alpha_0 -> alpha_1 -> ... -> alpha_K, with jump times
0 < t_1 < ... < t_K < tau.  Along a path the state stays a single Gaussian.  Between jumps, L_0 evolves it in one
branch, from whatever Gaussian it starts in, by the linear map of Riccati_Linear (Riccati term included).  A jump
only multiplies its amplitude by V[alpha_m, alpha_m-1].  So the phase-space integrals of App. app:perturbation are
done exactly, whether or not the Green's function is a delta function.  Only the integrals over the jump times are
numerical:

  'quadrature'   jump times on the grid t_array, trapezoid rule in every nested time, so the error is O(dt^2).
                 All paths in one branch are advanced together, as one batch per grid step.
  'monte_carlo'  ordered jump times drawn uniformly in [t_0, t_end], branches drawn in proportion to |V|.  Each
                 sample is an unbiased estimate, and the standard error of the QRDM is returned.

The series is cut after order K.  T1 and ladder decay stop by themselves once the DV has no excitations left (one
qubit: K = 1), and the higher orders then come out exactly 0.  A DV Hamiltonian does not stop, and the largest QRDM
element of the last order computed is returned so that the cut can be judged.

Conventions are those of Riccati_Linear and the QuTiP reference of the tests: rho_jk = <j|rho|k>, and the
characteristic function of a component is weight * exp(-1/4 xi^T sigma xi + i xi^T r).
"""
import math

import numpy as np

from . import QIT_Functions
from . import Riccati_Linear


class Perturbation:
    """
    DV-only terms to add to an operator-valued Gaussian dynamics.

    Perturbation(N_qubits=N)   a register of N qubits, levels indexed as everywhere in gbcs
                               (QIT_Functions.Extract_Qubit_Labels_Array: index 0 has every qubit at -1);
    Perturbation(N_levels=d)   one d-level system, level 0 the ground state (Hamiltonian.Initialize_Qudit_Hamiltonians).

    Units are those of the Hamiltonian, H / (hbar omega), with time tau = omega t.
    """

    def __init__(self, N_qubits=None, N_levels=None):
        if (N_qubits is None) == (N_levels is None):
            raise ValueError('give either N_qubits (a qubit register) or N_levels (one N-level system)')
        self.N_qubits = N_qubits
        self.N_levels = 2**N_qubits if N_qubits is not None else int(N_levels)
        self.H_dv = np.zeros((self.N_levels, self.N_levels), dtype=complex)
        self.jumps = []

    def Add_Hamiltonian(self, H_dv):
        """
        A DV Hamiltonian H_dv (N_levels x N_levels, Hermitian), entering as -i[H_dv, rho].
        Its diagonal works like H^0: a diagonal entry h_jj is the same as H^0_j = 2 h_jj.
        """
        H_dv = np.asarray(H_dv, dtype=complex)
        if H_dv.shape != self.H_dv.shape:
            raise ValueError(f'H_dv must be {self.H_dv.shape}, not {H_dv.shape}')
        if not np.allclose(H_dv, H_dv.conj().T, atol=1e-12*max(1.0, np.max(np.abs(H_dv)))):
            raise ValueError('H_dv must be Hermitian')
        self.H_dv = self.H_dv + H_dv

    def Add_T1(self, Gamma_1):
        """T1 decay of each qubit: jump operator sqrt(Gamma_1[i]) sigma_-^(i), which takes label +1 to -1."""
        if self.N_qubits is None:
            raise ValueError('Add_T1 acts on the qubits of a register; for one N-level system use Add_Ladder_Decay')
        Gamma_1 = np.ravel(np.asarray(Gamma_1, dtype=float))
        if len(Gamma_1) != self.N_qubits:
            raise ValueError(f'Gamma_1 needs one rate per qubit, {self.N_qubits}, not {len(Gamma_1)}')
        if np.any(Gamma_1 < 0):
            raise ValueError('decay rates must be non-negative')
        for i, rate in enumerate(Gamma_1):
            if rate > 0:
                self.jumps.append(np.sqrt(rate)*Lowering_Operator(self.N_qubits, i))

    def Add_Ladder_Decay(self, Gamma):
        """Decay of one N-level system: jump operators sqrt(Gamma[j-1]) |j-1><j| for j = 1, ..., N_levels - 1."""
        if self.N_qubits is not None:
            raise ValueError('Add_Ladder_Decay is for one N-level system; for a qubit register use Add_T1')
        Gamma = np.ravel(np.asarray(Gamma, dtype=float))
        if len(Gamma) != self.N_levels - 1:
            raise ValueError(f'Gamma needs N_levels - 1 = {self.N_levels - 1} rates, not {len(Gamma)}')
        if np.any(Gamma < 0):
            raise ValueError('decay rates must be non-negative')
        for j in range(1, self.N_levels):
            if Gamma[j - 1] > 0:
                L = np.zeros((self.N_levels, self.N_levels))
                L[j - 1, j] = np.sqrt(Gamma[j - 1])
                self.jumps.append(L)

    def Liouvillian(self):
        """
        DV superoperator M with (d rho/d tau)_jk = sum_lm M[j*N + k, l*N + m] rho_lm, for the terms added so far.
        """
        d = self.N_levels
        identity = np.eye(d)
        M = -1j*np.kron(self.H_dv, identity) + 1j*np.kron(identity, self.H_dv.T)
        for L in self.jumps:
            P = L.conj().T @ L
            M = M + np.kron(L, L.conj()) - 0.5*np.kron(P, identity) - 0.5*np.kron(identity, P.T)
        return M

    def Split(self):
        """(c, V): c[j, k] is the constant added to ds_jk/dtau, and V the off-diagonal part of the Liouvillian."""
        M = self.Liouvillian()
        diagonal = np.diag(M).copy()
        return diagonal.reshape(self.N_levels, self.N_levels), M - np.diag(diagonal)


def Lowering_Operator(N_qubits, i):
    """sigma_- of qubit i in the gbcs index order: clears the bit of qubit i, label +1 -> -1."""
    d = 2**N_qubits
    L = np.zeros((d, d))
    for j in range(d):
        if QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, j)[i] > 0:
            L[j - 2**(N_qubits - 1 - i), j] = 1.0
    return L


################################## The levels and their branches ##################################

def Level_Structure(Hamiltonian, Decoherence, N_levels):
    """
    H_j, r_j and H^0_j of each DV level, as Tab. 1 uses them, and the dephasing constant of each branch.
    """
    if Hamiltonian.type[0] != 'Constant':
        raise NotImplementedError('the perturbation theory is implemented for constant Hamiltonians only')

    Gamma_z = None if Decoherence is None else np.ravel(Decoherence.Gamma_z)
    dephasing = np.zeros((N_levels, N_levels))

    if Hamiltonian.type[1] == 'Qudit':
        if Hamiltonian.N_levels != N_levels:
            raise ValueError(f'the Hamiltonian has {Hamiltonian.N_levels} levels and the perturbation {N_levels}')
        if Gamma_z is not None and np.any(Gamma_z != 0):
            raise ValueError('Gamma_z is a dephasing of qubits; for one N-level system add it as a DV jump')
        H_levels = [np.asarray(H) for H in Hamiltonian.H_levels]
        r_levels = [np.asarray(r).reshape(-1, 1) for r in Hamiltonian.r_levels]
        H_0_levels = np.ravel(Hamiltonian.H_0_levels)
        return H_levels, r_levels, H_0_levels, dephasing

    N_qubits = Hamiltonian.N_qubits
    if 2**N_qubits != N_levels:
        raise ValueError(f'the Hamiltonian has {N_qubits} qubits, {2**N_qubits} levels, and the perturbation {N_levels}')
    H_array, r_array = Hamiltonian.H_array, Hamiltonian.r_array
    H_q_0 = np.ravel(Hamiltonian.H_q_0_array)
    for name, array in (('H_array', H_array), ('r_array', r_array)):
        if len(array) not in (1, N_qubits + 1):
            raise ValueError(f'{name} needs 1 or N_qubits + 1 = {N_qubits + 1} entries, not {len(array)}')
    if len(H_q_0) != N_qubits and np.any(H_q_0 != 0):
        raise ValueError(f'H_q_0_array needs N_qubits = {N_qubits} entries, not {len(H_q_0)}')
    if Gamma_z is not None and len(Gamma_z) not in (0, N_qubits):
        raise ValueError(f'Gamma_z needs N_qubits = {N_qubits} entries, not {len(Gamma_z)}')

    labels = [QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, j) for j in range(N_levels)]
    H_levels, r_levels, H_0_levels = [], [], np.zeros(N_levels, dtype=complex)
    for j in range(N_levels):
        H_j = np.array(H_array[0], dtype=float)
        r_j = np.array(r_array[0], dtype=float).reshape(-1, 1)
        for i in range(N_qubits):
            if len(H_array) > 1:
                H_j = H_j + labels[j][i]*H_array[i + 1]
            if len(r_array) > 1:
                r_j = r_j + labels[j][i]*np.reshape(r_array[i + 1], (-1, 1))
        H_levels.append(H_j)
        r_levels.append(r_j)
        if len(H_q_0) == N_qubits:
            H_0_levels[j] = labels[j] @ H_q_0
    if Gamma_z is not None and len(Gamma_z) == N_qubits:
        for j in range(N_levels):
            for k in range(N_levels):
                dephasing[j, k] = 0.5*np.real(np.sum(Gamma_z*(labels[j]*labels[k] - 1)))
    return H_levels, r_levels, H_0_levels, dephasing


class Branch:
    """
    L_0 on one branch pair (j, k): the affine generator of Riccati_Linear and its step propagators.
    Propagators for the steps of the time grid are kept; those of other steps (Monte Carlo) are not.
    """

    def __init__(self, n_modes, H_j, H_k, r_j, r_k, c_0, E, D, d, grid_steps):
        A = Riccati_Linear.Doubling_Matrix(n_modes, H_j, H_k, E, D)
        f1, f2 = Riccati_Linear.Linear_Sources(n_modes, r_j, r_k, d)
        self.dim = 2*n_modes
        self.M = Riccati_Linear.Step_Generator(A, f1, f2)
        self.rate = 0.5*np.trace(A[:self.dim, :self.dim]) + c_0
        self.grid_keys = {Step_Key(h) for h in grid_steps}
        self.cache = {}

    def Propagator(self, h):
        key = Step_Key(h)
        if key in self.cache:
            return self.cache[key]
        propagator = Riccati_Linear.Step_Propagators(self.M, h)
        if key in self.grid_keys:
            self.cache[key] = propagator
        return propagator

    def Evolve(self, sigma, r, times):
        """
        One Gaussian from (sigma, r) at times[0]: its sigma, r and s - s(times[0]) at every entry of times.
        This is Riccati_Linear.Branch_Trajectory with a cache shared across calls.
        """
        T = len(times)
        sigma_t = np.zeros((T, self.dim, self.dim), dtype=np.complex128)
        r_t = np.zeros((T, self.dim), dtype=np.complex128)
        s_t = np.zeros(T, dtype=np.complex128)
        sigma = 0.5*(sigma + sigma.T).astype(np.complex128)
        r = np.ravel(r).astype(np.complex128)
        s = 0j
        sigma_t[0], r_t[0] = sigma, r
        ledger = Riccati_Linear.Step_Ledger(Riccati_Linear.SUBSTEP_ALLOWANCE)
        for i in range(T - 1):
            h = times[i + 1] - times[i]
            if h > 0:
                propagator = self.Propagator(h)
                level = propagator.base_level
                ledger.limit += Riccati_Linear.SUBSTEPS_PER_BASE_STEP*2**level
                try:
                    for _ in range(2**level):
                        sigma, r, ds = Riccati_Linear.Advance(propagator, level, sigma, r, self.rate, ledger)
                        s += ds
                except np.linalg.LinAlgError as error:
                    raise np.linalg.LinAlgError(f'Stopped between t = {times[i]} and t = {times[i + 1]}: {error}') from error
            sigma_t[i + 1], r_t[i + 1], s_t[i + 1] = sigma, r, s
        return sigma_t, r_t, s_t


def Step_Key(h):
    return round(float(h), 12)


def Build_Branches(n_modes, H_levels, r_levels, H_0_levels, dephasing, c, E, D, d, t_array):
    N = len(H_levels)
    grid_steps = np.diff(t_array)
    branches = [[None]*N for _ in range(N)]
    for j in range(N):
        for k in range(N):
            c_0 = -0.5j*(H_0_levels[j] - H_0_levels[k]) + dephasing[j, k] + c[j, k]
            branches[j][k] = Branch(n_modes, H_levels[j], H_levels[k], r_levels[j], r_levels[k], c_0, E, D, d, grid_steps)
    return branches


################################## Batched steps ##################################

# Terms of the series tr log(1 + Z) = sum_k (-1)^(k+1) tr(Z^k) / k that pick the branch of ln det X.  An accepted
# step has ||Z|| <= MAX_X_DISTANCE = 0.75, so the remainder is below dim 0.75^9 / (9 (1 - 0.75)) = 0.033 dim,
# far from the pi that would pick the wrong branch for any dim < 90.
BRANCH_SERIES_TERMS = 8


def Inverse_Log_Det(X):
    """
    X^-1 and ln det X for a stack of X with ||X - 1|| <= 0.75, ln det X on the branch continuous from X = 1:
    the same value as Riccati_Linear.Advance's sum of the principal logs of the eigenvalues.
    """
    dim = X.shape[-1]
    if dim == 2:
        a, b, c, d = X[..., 0, 0], X[..., 0, 1], X[..., 1, 0], X[..., 1, 1]
        det = a*d - b*c
        inverse = np.stack((np.stack((d, -b), axis=-1), np.stack((-c, a), axis=-1)), axis=-2)/det[..., None, None]
        # |lambda - 1| <= 0.75 gives |arg lambda| <= asin(0.75) = 0.85 for each eigenvalue, so the two arguments
        # add up to less than pi and the principal log of det X is already the continuous branch.
        return inverse, np.log(det)
    else:
        inverse = np.linalg.inv(X)
        sign, log_abs = np.linalg.slogdet(X)
        log_det = log_abs + np.log(sign)
    Z = X - np.eye(dim)
    power, series = Z, np.trace(Z, axis1=-2, axis2=-1)
    for k in range(2, BRANCH_SERIES_TERMS + 1):
        power = power @ Z
        series = series + (-1)**(k + 1)*np.trace(power, axis1=-2, axis2=-1)/k
    log_det = log_det + 2j*np.pi*np.round((series.imag - log_det.imag)/(2*np.pi))
    return inverse, log_det


def Advance_Batch(propagator, level, sigma, r, rate):
    """
    Riccati_Linear.Advance for many Gaussians of one branch at once.  Those whose step it would halve, or whose
    sigma is large enough for its rounding guards, go through Riccati_Linear.Advance itself, one by one.
    """
    (h_level, S_11, S_12, S_21, S_22, p_x, p_y, w_y, w_0, remainder, _) = propagator(level)
    identity = propagator.identity

    if not propagator.coupled:
        # H_j = H_k: S_12 = 0, so X = S_11 for every Gaussian, and nothing observes sigma.
        return Advance_Uncoupled(propagator, level, sigma, r, rate)

    X = S_11 + S_12 @ sigma
    size = propagator.dim + np.sum(np.abs(propagator.observed @ sigma)**2, axis=(-2, -1))
    if propagator.dropped > 0:
        size = size + (propagator.dropped*np.linalg.norm(sigma, axis=(-2, -1)))**2
    accept = remainder*np.sqrt(size) <= Riccati_Linear.MAX_INTERPOLATION_ERROR
    accept &= np.linalg.norm(X - identity, axis=(-2, -1)) <= Riccati_Linear.MAX_X_DISTANCE
    _, S_11_half, S_12_half, *_ = propagator(level + 1)
    accept &= np.linalg.norm(S_11_half + S_12_half @ sigma - identity, axis=(-2, -1)) <= Riccati_Linear.MAX_X_DISTANCE
    if propagator.sensitive:
        accept &= np.max(np.abs(sigma), axis=(-2, -1)) <= Riccati_Linear.NEGLIGIBLE_SIGMA

    sigma_new = np.empty_like(sigma)
    r_new = np.empty_like(r)
    ds = np.empty(len(sigma), dtype=np.complex128)

    a = np.flatnonzero(accept)
    if len(a):
        Xa, sa, ra = X[a], sigma[a], r[a]
        inverse, log_det = Inverse_Log_Det(Xa)
        sn = (S_21 + S_22 @ sa) @ inverse
        sn = 0.5*(sn + np.swapaxes(sn, -1, -2))
        x = ra @ S_12.T + p_x
        rn = ra @ S_22.T + p_y - np.einsum('nab,nb->na', sn, x)
        w = ra @ w_y + w_0
        ds[a] = -np.sum(x*rn, axis=-1) + w - 0.5*log_det + h_level*rate
        sigma_new[a], r_new[a] = sn, rn

    for n in np.flatnonzero(~accept):
        ledger = Riccati_Linear.Step_Ledger(Riccati_Linear.SUBSTEP_ALLOWANCE)
        sigma_new[n], r_new[n], ds[n] = Riccati_Linear.Advance(propagator, level, sigma[n], r[n], rate, ledger)
    return sigma_new, r_new, ds


def Advance_Uncoupled(propagator, level, sigma, r, rate):
    """Advance_Batch for a branch with H_j = H_k, where one X = S_11 serves every Gaussian."""
    (h_level, S_11, S_12, S_21, S_22, p_x, p_y, w_y, w_0, remainder, _) = propagator(level)
    identity = propagator.identity
    _, S_11_half, *_ = propagator(level + 1)
    accept = (remainder*np.sqrt(propagator.dim) <= Riccati_Linear.MAX_INTERPOLATION_ERROR
              and np.linalg.norm(S_11 - identity) <= Riccati_Linear.MAX_X_DISTANCE
              and np.linalg.norm(S_11_half - identity) <= Riccati_Linear.MAX_X_DISTANCE)
    if not accept:
        # Not expected at the base level, where ||A h|| <= 1/2; Riccati_Linear.Advance halves the step.
        sigma_new, r_new, ds = np.empty_like(sigma), np.empty_like(r), np.empty(len(sigma), dtype=np.complex128)
        for n in range(len(sigma)):
            ledger = Riccati_Linear.Step_Ledger(Riccati_Linear.SUBSTEP_ALLOWANCE)
            sigma_new[n], r_new[n], ds[n] = Riccati_Linear.Advance(propagator, level, sigma[n], r[n], rate, ledger)
        return sigma_new, r_new, ds
    if propagator.sensitive:
        # The one guard of Riccati_Linear.Advance that applies without coupling: r_new = y - sigma_new x loses
        # digits when sigma is large and ill-conditioned.
        abs_sigma = np.abs(sigma)
        for n in np.flatnonzero(np.max(abs_sigma, axis=(-2, -1)) > Riccati_Linear.NEGLIGIBLE_SIGMA):
            Riccati_Linear.Check_Sigma_Conditioning(sigma[n], abs_sigma[n])
    inverse, log_det = Inverse_Log_Det(S_11[None])
    sigma_new = (S_21 + S_22 @ sigma) @ inverse[0]
    sigma_new = 0.5*(sigma_new + np.swapaxes(sigma_new, -1, -2))
    r_new = r @ S_22.T + p_y - sigma_new @ p_x
    ds = -(r_new @ p_x) + (r @ w_y + w_0) - 0.5*log_det[0] + h_level*rate
    return sigma_new, r_new, ds


class Paths:
    """
    The Gaussians of one branch, one per path, advanced together.
      a0     amplitude at creation, with the V of every jump and the trapezoid weights of all but the last jump time
      s      increment of s since creation
      m      grid index of the last jump (0 and order 0 for the unperturbed branch itself)
      order  number of jumps
    The contribution at grid index i is a0 exp(s) w(m, i), with w = 1 at order 0.
    """

    def __init__(self, dim):
        self.sigma = np.zeros((0, dim, dim), dtype=np.complex128)
        self.r = np.zeros((0, dim), dtype=np.complex128)
        self.a0 = np.zeros(0, dtype=np.complex128)
        self.s = np.zeros(0, dtype=np.complex128)
        self.m = np.zeros(0, dtype=int)
        self.order = np.zeros(0, dtype=int)

    def __len__(self):
        return len(self.a0)

    def Add(self, sigma, r, a0, m, order):
        n = len(a0)
        self.sigma = np.concatenate((self.sigma, sigma))
        self.r = np.concatenate((self.r, r))
        self.a0 = np.concatenate((self.a0, a0))
        self.s = np.concatenate((self.s, np.zeros(n, dtype=np.complex128)))
        self.m = np.concatenate((self.m, np.full(n, m, dtype=int)))
        self.order = np.concatenate((self.order, np.full(n, order, dtype=int)))


def Trapezoid_Weight(t_array, m, i):
    """Weight of grid point m in the trapezoid rule for an integral over [t_0, t_i] (m <= i)."""
    m = np.asarray(m)
    return 0.5*(t_array[np.minimum(m + 1, i)] - t_array[np.maximum(m - 1, 0)])


################################## Results ##################################

class Perturbation_Result:
    """
    Order 0 (the GBCS of L_0, with the diagonal DV terms in it):
      sigma_JK_t (T, N, N, 2n, 2n), r_JK_t (T, N, N, 2n, 1), C_JK_t, phi_JK_t (T, N, N), rho_q_t (T, N, N)
    All orders:
      rho_q_t_orders   (K+1, T, N, N)   QRDM contribution of each order, order 0 included
      rho_q_t_total    (T, N, N)        their sum
      rho_q_t_stderr   (K+1, T, N, N)   standard error of each order (Monte Carlo; zeros for quadrature)
      first_moments_t  (T, N, N, 2n)    Tr[r_hat rho_jk], all orders
      second_moments_t (T, N, N, 2n, 2n) Tr[sym(r_hat r_hat^T) rho_jk] = sum weight (sigma/2 + r r^T), all orders
      components       {time index: dict(order, j, k, sigma, r, weight)} for orders >= 1, at store_times
      last_order_size  max over t, j, k of |rho_q_t_orders[K]|
    """


def Accumulate(result, i, j, k, sigma, r, weight, order):
    np.add.at(result.rho_q_t_orders[:, i, j, k], order, weight)
    result.first_moments_t[i, j, k] += weight @ r
    result.second_moments_t[i, j, k] += (np.einsum('n,nab->ab', weight, 0.5*sigma)
                                         + np.einsum('n,na,nb->ab', weight, r, r))


def Store(result, i, j, k, sigma, r, weight, order):
    keep = (order >= 1) & (weight != 0)
    if not np.any(keep):
        return
    store = result.components.setdefault(i, dict(order=[], j=[], k=[], sigma=[], r=[], weight=[]))
    n = int(np.sum(keep))
    store['order'].append(order[keep])
    store['j'].append(np.full(n, j))
    store['k'].append(np.full(n, k))
    store['sigma'].append(sigma[keep])
    store['r'].append(r[keep])
    store['weight'].append(weight[keep])


def Finish_Components(result):
    for i, store in result.components.items():
        result.components[i] = {key: np.concatenate(value) for key, value in store.items()}


################################## Drivers ##################################

def Check_Arguments(order, method, t_array, rho_q_0, N):
    if not (isinstance(order, (int, np.integer)) and order >= 0):
        raise ValueError(f'order must be a non-negative integer, not {order!r}')
    if method not in ('quadrature', 'monte_carlo'):
        raise ValueError(f"method must be 'quadrature' or 'monte_carlo', not {method!r}")
    if len(t_array) < 2 or np.any(np.diff(t_array) <= 0):
        raise ValueError('t_array must be increasing, with at least two points')
    if np.shape(rho_q_0) != (N, N):
        raise ValueError(f'rho_q_0 must be {N} x {N}, not {np.shape(rho_q_0)}')


def Initial_Conditions(sigma_0, r_0, N, dim):
    sigma_0 = np.asarray(sigma_0, dtype=complex)
    r_0 = np.asarray(r_0, dtype=complex)
    if sigma_0.ndim == 4:
        return sigma_0, r_0.reshape(N, N, dim)
    return (np.broadcast_to(sigma_0, (N, N, dim, dim)),
            np.broadcast_to(r_0.reshape(dim), (N, N, dim)))


def Perturbation_Dynamics(n_modes, r_0, sigma_0, rho_q_0, H_levels, r_levels, H_0_levels, dephasing, perturbation,
                          t_array, E=None, D=None, d=None, order=1, method='quadrature', n_samples=1000, seed=None,
                          store_times=None, max_paths=2_000_000):
    """
    Dyson series of perturbation (a Perturbation) around the operator-valued Gaussian dynamics of the levels
    (H_levels, r_levels, H_0_levels, dephasing from Level_Structure), with noise E, D, d as in Riccati_Linear.

    r_0, sigma_0 : (2n, 1), (2n, 2n) for every branch, or (N, N, 2n, 1), (N, N, 2n, 2n) per branch
    rho_q_0      : (N, N)
    store_times  : grid indices at which the components of orders >= 1 are kept (default: the last one)
    max_paths    : quadrature raises once more paths than this are alive at once; Monte Carlo then scales better

    Returns a Perturbation_Result.
    """
    N = perturbation.N_levels
    dim = 2*n_modes
    t_array = np.asarray(t_array, dtype=float)
    T = len(t_array)
    Check_Arguments(order, method, t_array, rho_q_0, N)
    store = {T - 1} if store_times is None else {int(i) % T for i in np.ravel(store_times)}

    c, V = perturbation.Split()
    branches = Build_Branches(n_modes, H_levels, r_levels, H_0_levels, dephasing, c, E, D, d, t_array)
    sigma_0_jk, r_0_jk = Initial_Conditions(sigma_0, r_0, N, dim)
    rho_q_0 = np.asarray(rho_q_0, dtype=complex)

    result = Perturbation_Result()
    result.t_array, result.order, result.method = t_array, order, method
    result.sigma_JK_t = np.zeros((T, N, N, dim, dim), dtype=np.complex128)
    result.r_JK_t = np.zeros((T, N, N, dim, 1), dtype=np.complex128)
    result.s_JK_t = np.zeros((T, N, N), dtype=np.complex128)
    result.rho_q_t_orders = np.zeros((order + 1, T, N, N), dtype=np.complex128)
    result.rho_q_t_stderr = np.zeros((order + 1, T, N, N))
    result.first_moments_t = np.zeros((T, N, N, dim), dtype=np.complex128)
    result.second_moments_t = np.zeros((T, N, N, dim, dim), dtype=np.complex128)
    result.components = {}

    quadrature_order = order if method == 'quadrature' else 0
    Sweep(result, branches, V, t_array, sigma_0_jk, r_0_jk, rho_q_0, quadrature_order, store, max_paths)
    if method == 'monte_carlo' and order > 0:
        Monte_Carlo(result, branches, V, t_array, rho_q_0, order, n_samples, np.random.default_rng(seed), store)
    Finish_Components(result)

    result.C_JK_t = -np.real(result.s_JK_t)
    result.phi_JK_t = np.imag(result.s_JK_t)
    result.rho_q_t = result.rho_q_t_orders[0].copy()
    result.rho_q_t_total = np.sum(result.rho_q_t_orders, axis=0)
    result.last_order_size = float(np.max(np.abs(result.rho_q_t_orders[order]))) if order > 0 else 0.0
    return result


def Sweep(result, branches, V, t_array, sigma_0_jk, r_0_jk, rho_q_0, K, store, max_paths):
    """
    One pass over the grid with every path of order <= K alive at once (quadrature), or only order 0 (K = 0).
    At each grid index: create the jumps at that time, add up the contributions, then step every path.
    """
    N = len(branches)
    dim = branches[0][0].dim
    T = len(t_array)
    paths = [[Paths(dim) for _ in range(N)] for _ in range(N)]
    for j in range(N):
        for k in range(N):
            paths[j][k].Add(0.5*(sigma_0_jk[j, k] + sigma_0_jk[j, k].T)[None], r_0_jk[j, k][None],
                            np.array([rho_q_0[j, k]]), 0, 0)
    targets = [[[(b // N, b % N) for b in np.flatnonzero(V[:, j*N + k])] for k in range(N)] for j in range(N)]

    for i in range(T):
        for order in range(K):
            for j in range(N):
                for k in range(N):
                    if not targets[j][k]:
                        continue
                    P = paths[j][k]
                    parents = np.flatnonzero(P.order == order)
                    if not len(parents):
                        continue
                    base = P.a0[parents]*np.exp(P.s[parents])
                    if order > 0:
                        base = base*Trapezoid_Weight(t_array, P.m[parents], i)
                    alive = parents[base != 0]
                    base = base[base != 0]
                    if not len(alive):
                        continue
                    for (jb, kb) in targets[j][k]:
                        paths[jb][kb].Add(P.sigma[alive], P.r[alive], base*V[jb*N + kb, j*N + k], i, order + 1)

        alive = sum(len(paths[j][k]) for j in range(N) for k in range(N))
        if alive > max_paths:
            raise MemoryError(f'{alive} paths alive at t = {t_array[i]}, more than max_paths = {max_paths}. '
                              "Use a coarser grid, a lower order, or method='monte_carlo'")

        for j in range(N):
            for k in range(N):
                P = paths[j][k]
                weight = P.a0*np.exp(P.s)
                later = P.order > 0
                weight[later] = weight[later]*Trapezoid_Weight(t_array, P.m[later], i)
                Accumulate(result, i, j, k, P.sigma, P.r, weight, P.order)
                if i in store:
                    Store(result, i, j, k, P.sigma, P.r, weight, P.order)
                zeroth = np.flatnonzero(P.order == 0)[0]
                result.sigma_JK_t[i, j, k] = P.sigma[zeroth]
                result.r_JK_t[i, j, k, :, 0] = P.r[zeroth]
                result.s_JK_t[i, j, k] = P.s[zeroth]

        if i == T - 1:
            break
        h = t_array[i + 1] - t_array[i]
        for j in range(N):
            for k in range(N):
                P = paths[j][k]
                propagator = branches[j][k].Propagator(h)
                level = propagator.base_level
                try:
                    for _ in range(2**level):
                        P.sigma, P.r, ds = Advance_Batch(propagator, level, P.sigma, P.r, branches[j][k].rate)
                        P.s = P.s + ds
                except np.linalg.LinAlgError as error:
                    raise np.linalg.LinAlgError(f'Stopped between t = {t_array[i]} and t = {t_array[i + 1]} in '
                                                f'branch ({j}, {k}): {error}') from error


def Monte_Carlo(result, branches, V, t_array, rho_q_0, K, n_samples, rng, store):
    """
    Orders 1..K by sampling.  For order k: k ordered times uniform in [t_0, t_end] (volume (t_end - t_0)^k / k!),
    a starting branch in proportion to |rho_q_0| among those V leaves, and each jump in proportion to |V|.
    Each sample is an unbiased estimate of the order-k term at every grid time after its last jump.
    """
    N = len(branches)
    T = len(t_array)
    t_0, t_end = t_array[0], t_array[-1]
    out = np.sum(np.abs(V), axis=0)
    start = np.abs(rho_q_0).reshape(-1)*(out > 0)
    if not np.any(start > 0):
        return
    p_start = start/np.sum(start)

    for k in range(1, K + 1):
        volume = (t_end - t_0)**k/math.factorial(k)
        total = np.zeros((T, N, N), dtype=np.complex128)
        total_sq = np.zeros((T, N, N))
        for _ in range(n_samples):
            times = np.sort(rng.uniform(t_0, t_end, size=k))
            alpha = rng.choice(N*N, p=p_start)
            factor = volume/p_start[alpha]

            j, kk = divmod(alpha, N)
            g = np.searchsorted(t_array, times[0], side='right') - 1
            sigma = result.sigma_JK_t[g, j, kk]
            r = result.r_JK_t[g, j, kk, :, 0]
            amplitude = rho_q_0[j, kk]*np.exp(result.s_JK_t[g, j, kk])
            sigma_t, r_t, s_t = branches[j][kk].Evolve(sigma, r, np.array([t_array[g], times[0]]))
            sigma, r, amplitude = sigma_t[-1], r_t[-1], amplitude*np.exp(s_t[-1])

            dead = False
            for m in range(k):
                column = np.abs(V[:, alpha])
                if column.sum() == 0:
                    dead = True
                    break
                p_jump = column/column.sum()
                beta = rng.choice(N*N, p=p_jump)
                amplitude = amplitude*V[beta, alpha]
                factor = factor/p_jump[beta]
                alpha = beta
                j, kk = divmod(alpha, N)
                if m < k - 1:
                    inside = t_array[(t_array > times[m]) & (t_array < times[m + 1])]
                    segment = np.concatenate(([times[m]], inside, [times[m + 1]]))
                    sigma_t, r_t, s_t = branches[j][kk].Evolve(sigma, r, segment)
                    sigma, r, amplitude = sigma_t[-1], r_t[-1], amplitude*np.exp(s_t[-1])
            if dead or amplitude == 0:
                continue

            first = np.searchsorted(t_array, times[-1], side='right')
            if first >= T:
                continue
            segment = np.concatenate(([times[-1]], t_array[first:]))
            sigma_t, r_t, s_t = branches[j][kk].Evolve(sigma, r, segment)
            y = factor*amplitude*np.exp(s_t[1:])
            weight = y/n_samples
            index = np.arange(first, T)
            total[index, j, kk] += y
            total_sq[index, j, kk] += np.abs(y)**2
            result.rho_q_t_orders[k, index, j, kk] += weight
            result.first_moments_t[index, j, kk] += weight[:, None]*r_t[1:]
            result.second_moments_t[index, j, kk] += weight[:, None, None]*(0.5*sigma_t[1:]
                                                                            + np.einsum('na,nb->nab', r_t[1:], r_t[1:]))
            for i in store:
                if i >= first:
                    n = i - first + 1
                    Store(result, i, j, kk, sigma_t[n][None], r_t[n][None], weight[n - 1:n], np.array([k]))

        mean = total/n_samples
        result.rho_q_t_stderr[k] = np.sqrt(np.maximum(total_sq/n_samples - np.abs(mean)**2, 0)/n_samples)


################################## Phase space ##################################

def Wigner_Components(components, r_tilde_array, n_modes, part='all', projector=None):
    """
    Wigner function of a set of components, in the normalisation of Wigner_Functions
    (each branch W_jk integrates to 2^n rho_jk), on the grid r_tilde_array (steps, steps, 2n, 1).

    part      : 'all' (sum over every branch), 'diagonal' (j = k: the reduced state of the modes) or
                'off_diagonal' (j != k: the interference term)
    projector : (N, N) DV operator Pi; if given, part is ignored and the result is Tr_q[Pi W] = sum Pi_kj W_jk
    """
    W = np.zeros(r_tilde_array.shape[:2], dtype=np.complex128)
    if components is None or len(components['weight']) == 0:
        return np.real(W)
    weight = components['weight'].copy()
    j, k = components['j'], components['k']
    if projector is not None:
        weight = weight*np.asarray(projector)[k, j]
    elif part == 'diagonal':
        weight = np.where(j == k, weight, 0)
    elif part == 'off_diagonal':
        weight = np.where(j != k, weight, 0)
    elif part != 'all':
        raise ValueError(f"part must be 'all', 'diagonal' or 'off_diagonal', not {part!r}")

    used = np.flatnonzero(weight != 0)
    if not len(used):
        return np.real(W)
    sigma, r, weight = components['sigma'][used], components['r'][used], weight[used]
    inverse = np.linalg.inv(sigma)
    inverse = 0.5*(inverse + np.swapaxes(inverse, -1, -2))
    # sqrt(det sigma) on the branch continuous from positive-definite sigma: Re sigma > 0 puts every eigenvalue in
    # the right half-plane, where the principal square root is that branch.
    root_det = np.prod(np.sqrt(np.linalg.eigvals(sigma)), axis=-1)
    coefficient = weight*(2**n_modes)/(np.pi**n_modes*root_det)

    # (g - r)^T S (g - r) = g^T S g - 2 g^T S r + r^T S r, as matrix products over chunks of components.
    grid = r_tilde_array[..., 0].reshape(-1, r_tilde_array.shape[2])
    outer = np.einsum('pa,pb->pab', grid, grid).reshape(len(grid), -1)
    S_r = np.einsum('nab,nb->na', inverse, r)
    r_S_r = np.sum(r*S_r, axis=-1)
    total = np.zeros(len(grid), dtype=np.complex128)
    chunk = max(1, int(2**22 // max(len(grid), 1)))
    for start in range(0, len(weight), chunk):
        part_ = slice(start, start + chunk)
        exponent = (outer @ inverse[part_].reshape(len(inverse[part_]), -1).T
                    - 2*grid @ S_r[part_].T + r_S_r[part_])
        total += np.exp(-exponent) @ coefficient[part_]
    return np.real(total.reshape(W.shape))
