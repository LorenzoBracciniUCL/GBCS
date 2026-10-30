"""Independent reference: the Lindblad equation of the paper in a truncated Fock space.

Nothing here uses the phase-space ODEs.  The state is evolved with qutip.mesolve under

    d rho/d tau = i[rho, H] + i[rho, d^T r] + D[r](rho) + D[J_z](rho)

    H = sum_j ( 1/2 r^T H_j r - r_j^T r + 1/2 H^0_j ) |j><j|

and sigma_jk, r_jk and s_jk are read off the branch rho_jk = <j|rho|k> through the
characteristic function Tr[D_{-xi} rho_jk] = exp(-1/4 xi^T sigma xi + i xi^T r + s):

    exp(s) = Tr[rho_jk]
    r      = Tr[r_hat rho_jk] / Tr[rho_jk]
    sigma  = 2 Tr[sym(r_hat r_hat^T) rho_jk] / Tr[rho_jk] - 2 r r^T

Branch index j runs as in QIT_Functions.Extract_Qubit_Labels_Array, so the qubit operators
are built from those labels and never from qutip.sigmaz.
"""
import numpy as np
import qutip
from scipy import linalg

import QIT_Functions


def Mode_Operators(n_modes, cutoff):
    """[x_1, p_1, ..., x_n, p_n] on the tensor product of n truncated modes."""
    r_ops = []
    for m in range(n_modes):
        a = qutip.destroy(cutoff)
        x = (a + a.dag())/np.sqrt(2)
        p = -1j*(a - a.dag())/np.sqrt(2)
        for op in (x, p):
            factors = [qutip.qeye(cutoff) for _ in range(n_modes)]
            factors[m] = op
            r_ops.append(qutip.tensor(factors) if n_modes > 1 else op)
    return r_ops


def Single_Mode_State(cutoff, nbar, s, x0, p0, padding=60):
    """
    Displaced squeezed thermal state with covariance (1 + 2 nbar) diag(s, 1/s).
    Built with padding levels, because the truncated squeezing and displacement
    generators are not unitary near the cutoff, and then cut.
    """
    size = cutoff + padding
    a = qutip.destroy(size).full()
    xi = -0.5*np.log(s)
    alpha = (x0 + 1j*p0)/np.sqrt(2)
    squeeze = linalg.expm(0.5*xi*(a @ a - a.T.conj() @ a.T.conj()))
    displace = linalg.expm(alpha*a.T.conj() - np.conj(alpha)*a)
    U = displace @ squeeze
    if nbar == 0:
        psi = U[:cutoff, 0]
        return qutip.Qobj(psi/np.linalg.norm(psi))
    populations = (nbar/(1 + nbar))**np.arange(size)/(1 + nbar)
    rho = U @ np.diag(populations) @ U.T.conj()
    rho = rho[:cutoff, :cutoff]
    return qutip.Qobj(rho/np.trace(rho))


def Initial_State(N_qubits, n_modes, cutoff, mode_states, rho_q_0):
    """mode_states: one (nbar, s, x0, p0) per mode, independent modes."""
    states = [Single_Mode_State(cutoff, *state) for state in mode_states]
    pure = all(state.isket for state in states)
    if pure and np.isclose(np.linalg.matrix_rank(rho_q_0, tol=1e-12), 1):
        w, v = np.linalg.eigh(rho_q_0)
        qubit = qutip.Qobj(v[:, -1]*np.sqrt(w[-1]), dims=[[2]*N_qubits, [1]*N_qubits])
    else:
        pure = False
        states = [state if state.isoper else qutip.ket2dm(state) for state in states]
        qubit = qutip.Qobj(rho_q_0, dims=[[2]*N_qubits, [2]*N_qubits])
    return qutip.tensor(*states, qubit)


def Branch_Projectors(N_qubits):
    """|j><k| as Qobj on the qubits, indexed [j][k]."""
    N2 = 2**N_qubits
    dims = [[2]*N_qubits, [2]*N_qubits]
    return [[qutip.Qobj(np.outer(np.eye(N2)[j], np.eye(N2)[k]), dims=dims) for k in range(N2)]
            for j in range(N2)]


def Lindblad_Operators(B, r_ops):
    """
    Jump operators of D[r](rho) = sum_ab B^ab (r^a rho r^b - 1/2 {rho, r^b r^a}).

    B = sum_m v_m v_m^dag gives L_m = sum_a (v_m)_a r^a, with jump L rho L^dag and
    L^dag L = sum_ab B^ab r^b r^a.  Note the order r^b r^a, which the paper's Eq. lindo
    writes as r^a r^b: only r^b r^a preserves the trace when Im(B) != 0.
    """
    w, v = np.linalg.eigh(B)
    if np.min(w) < -1e-12:
        raise ValueError('B is not positive semidefinite')
    ops = []
    for m in range(len(w)):
        if w[m] > 1e-14:
            ops.append(np.sqrt(w[m])*sum(v[a, m]*r_ops[a] for a in range(len(r_ops))))
    return ops


def Reference_Dynamics(N_qubits, n_modes, cutoff, mode_states, H_array, r_array, H_q_0_array, rho_q_0, t_array,
                       B=None, d=None, Gamma_z=None, rtol=1e-9, atol=1e-11, moments=True):
    """
    Returns rho_q_t (T, 2^N, 2^N), r_JK_t (T, 2^N, 2^N, 2n, 1), sigma_JK_t (T, 2^N, 2^N, 2n, 2n)
    and the largest population on the last Fock level of any mode, which is the convergence
    diagnostic for the cutoff.  With moments=False the last two moments are None.
    """
    N2 = 2**N_qubits
    dim = 2*n_modes
    r_ops = Mode_Operators(n_modes, cutoff)
    qubit_identity = qutip.qeye([2]*N_qubits)
    projectors = Branch_Projectors(N_qubits)

    H_total = 0
    for j in range(N2):
        labels = QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, j)
        H_j = H_array[0] + sum(labels[i]*H_array[i + 1] for i in range(N_qubits))
        r_j = np.ravel(r_array[0] + sum(labels[i]*r_array[i + 1] for i in range(N_qubits)))
        H_0_j = np.real(labels @ np.ravel(H_q_0_array))
        block = 0.5*H_0_j*qutip.tensor(*[qutip.qeye(cutoff)]*n_modes)
        for a in range(dim):
            block -= r_j[a]*r_ops[a]
            for b in range(dim):
                block += 0.5*H_j[a, b]*r_ops[a]*r_ops[b]
        H_total += qutip.tensor(block, projectors[j][j])

    if d is not None:
        drive = sum(np.ravel(d)[a]*r_ops[a] for a in range(dim))
        H_total += qutip.tensor(drive, qubit_identity)

    c_ops = []
    if B is not None:
        c_ops += [qutip.tensor(L, qubit_identity) for L in Lindblad_Operators(B, r_ops)]
    if Gamma_z is not None:
        modes_identity = qutip.tensor(*[qutip.qeye(cutoff)]*n_modes)
        for i in range(N_qubits):
            labels = np.array([QIT_Functions.Extract_Qubit_Labels_Array(N_qubits, j)[i] for j in range(N2)])
            sigma_z_i = qutip.Qobj(np.diag(labels), dims=[[2]*N_qubits, [2]*N_qubits])
            c_ops.append(np.sqrt(Gamma_z[i]/2)*qutip.tensor(modes_identity, sigma_z_i))

    psi_0 = Initial_State(N_qubits, n_modes, cutoff, mode_states, rho_q_0)
    result = qutip.mesolve(H_total, psi_0, t_array, c_ops=c_ops,
                           options={'store_states': True, 'rtol': rtol, 'atol': atol, 'nsteps': 10**7,
                                    'progress_bar': False})

    T = len(t_array)
    D_modes = cutoff**n_modes
    rho_q_t = np.zeros((T, N2, N2), dtype=complex)
    r_JK_t = np.zeros((T, N2, N2, dim, 1), dtype=complex) if moments else None
    sigma_JK_t = np.zeros((T, N2, N2, dim, dim), dtype=complex) if moments else None
    if moments:
        r_mats = [op.full() for op in r_ops]
        sym_mats = [[0.5*(r_mats[a] @ r_mats[b] + r_mats[b] @ r_mats[a]) for b in range(dim)] for a in range(dim)]
    edge = 0.0

    for t, state in enumerate(result.states):
        if state.isoper:
            branch = state.full().reshape(D_modes, N2, D_modes, N2).transpose(1, 3, 0, 2)
            rho_q_t[t] = np.trace(branch, axis1=2, axis2=3)
            diagonal = np.real(np.einsum('jjaa->a', branch))
        else:
            psi = state.full().reshape(D_modes, N2)
            rho_q_t[t] = psi.T @ psi.conj()
            diagonal = np.sum(np.abs(psi)**2, axis=1)
        diagonal = diagonal.reshape([cutoff]*n_modes)
        for m in range(n_modes):
            marginal = np.sum(diagonal, axis=tuple(i for i in range(n_modes) if i != m))
            edge = max(edge, marginal[-1])
        if not moments:
            continue
        if not state.isoper:
            branch = np.einsum('aj,bk->jkab', psi, psi.conj())
        for j in range(N2):
            for k in range(N2):
                rho_jk = branch[j, k]
                norm = rho_q_t[t, j, k]
                r = np.array([np.sum(r_mats[a].T*rho_jk) for a in range(dim)])/norm
                r_JK_t[t, j, k, :, 0] = r
                for a in range(dim):
                    for b in range(a, dim):
                        value = 2*np.sum(sym_mats[a][b].T*rho_jk)/norm - 2*r[a]*r[b]
                        sigma_JK_t[t, j, k, a, b] = sigma_JK_t[t, j, k, b, a] = value

    return rho_q_t, r_JK_t, sigma_JK_t, edge
