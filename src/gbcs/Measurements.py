import numpy as np 
import matplotlib.pyplot as plt
import math as math 
import cmath as cmath
from scipy import linalg as linalg
from scipy import integrate as integ
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from . import QIT_Functions


########################################## Expectation Values ##########################################

def Qubit_Expecation_Value(operator, rho_q):
    return np.trace(operator@rho_q)

def Qubit_Expecation_Value_Time_Evolution(operator, rho_q_t, t_array):
    exp_t = np.zeros(len(t_array))
    for i in range(len(t_array)):
        exp_t[i] = Qubit_Expecation_Value(operator, rho_q_t[i])
    return exp_t

########################################## Ideal Measurements ##########################################

def Projectors(operator, tol=1e-9):
    """
    Outcomes and projectors of a projective measurement of a Hermitian qubit observable: one projector per
    distinct eigenvalue, so a degenerate observable such as sigma_z on one qubit of two gives rank-2 projectors.
    Outcomes are in increasing order.
    """
    eigenvalues, vectors = linalg.eigh(operator)
    outcomes, projectors = [], []
    for value, vector in zip(eigenvalues, vectors.T):
        projector = np.outer(vector, vector.conj())
        if outcomes and abs(value - outcomes[-1]) < tol:
            projectors[-1] = projectors[-1] + projector
        else:
            outcomes.append(value)
            projectors.append(projector)
    return np.array(outcomes), np.array(projectors)


def Post_Measurement(POVM_array, rho_q):
    """Probabilities Tr[Pi rho] and normalised post-measurement states Pi rho Pi / p (zero where p = 0)."""
    prob = np.real(np.einsum('aij,ji->a', POVM_array, rho_q))
    rho_post = np.einsum('aij,jk,akl->ail', POVM_array, rho_q, POVM_array)
    nonzero = prob > 0
    rho_post[nonzero] = rho_post[nonzero]/prob[nonzero, None, None]
    rho_post[~nonzero] = 0
    return prob, rho_post


def Qubit_Ideal_Measurament(N_qubits, operator, rho_q):
    outcomes, POVM_array = Projectors(operator)
    prob, rho_q_PMS_array = Post_Measurement(POVM_array, rho_q)
    return outcomes, prob, POVM_array, rho_q_PMS_array


def Qubit_Ideal_Measurament_Time_Evolution(N_qubits, operator, rho_q_t, t_array):
    outcomes, POVM_array = Projectors(operator)
    prob_t = np.zeros((len(outcomes), len(t_array)))
    rho_q_t_PMS_array = np.zeros((len(outcomes), len(t_array)) + np.shape(rho_q_t)[1:], dtype=np.complex128)
    for j in range(len(t_array)):
        prob_t[:, j], rho_q_t_PMS_array[:, j] = Post_Measurement(POVM_array, rho_q_t[j])
    return outcomes, prob_t, POVM_array, rho_q_t_PMS_array

########################################## Noisy Measurements ##########################################

def Outcome_Vectors(n_modes, r_measure_array):
    """
    Measurement outcomes as phase-space vectors, shape (K, 2n).  An array of shape (K, 2n) is used as it is.
    A 1-D array of K numbers m is the legacy form, read as the vectors (m, ..., m); Mode_Measurament no longer
    uses it, as it is only right for homodyne along x or p.
    """
    r_measure_array = np.asarray(r_measure_array, dtype=float)
    if r_measure_array.ndim == 1:
        return np.repeat(r_measure_array[:, None], 2*n_modes, axis=1)
    if r_measure_array.ndim == 2 and r_measure_array.shape[1] == 2*n_modes:
        return r_measure_array
    raise ValueError(f'outcomes must have shape (K, {2*n_modes}) or (K,), not {r_measure_array.shape}')


def Generadyne_Qubit_Post_Measurement_State(N_qubits, n_modes, sigma_jk, sigma_measure, r_jk, rho_q, r_measure_array):
    """
    General-dyne measurement of the modes with covariance sigma_measure, on the outcomes r_measure_array
    (see Outcome_Vectors).  For each outcome m,

        P_jk(m) = rho_jk exp(-(r_jk - m)^T (sigma_jk + sigma_m)^-1 (r_jk - m)) / (pi^n sqrt det(sigma_jk + sigma_m)),

    Returns P_jk / sum_m Prob(m), shape (2^N, 2^N, K), whose trace over the qubit at fixed m is the probability of
    m, and Prob, shape (K,), normalised to sum to 1 over the given outcomes.  The post-measurement QRDM at outcome
    k is P_jk[:, :, k] / Prob[k].
    """
    vectors = Outcome_Vectors(n_modes, r_measure_array)
    N = np.shape(rho_q)[0]
    P_jk = np.zeros((N, N, len(vectors)), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            total = sigma_jk[i, j] + sigma_measure
            inverse = linalg.inv(total)
            # sqrt(det) on the branch continuous from a positive-definite matrix: the product of principal roots
            root_det = np.prod(np.sqrt(linalg.eigvals(total)))
            delta = np.ravel(r_jk[i, j])[None, :] - vectors
            exponent = np.einsum('ka,ab,kb->k', delta, inverse, delta)
            P_jk[i, j] = rho_q[i, j]*np.exp(-exponent)/(np.pi**n_modes*root_det)
    Prob_dis = np.einsum('iik->k', P_jk)
    norm = np.sum(Prob_dis)
    return P_jk/norm, Prob_dis/norm


def Generadyne_Qubit_Post_Measurement_State_Time_Evolution(N_qubits, n_modes, sigma_jk_t, sigma_measure, r_jk_t, rho_q_t, r_measure_array, t_array):
    K = len(Outcome_Vectors(n_modes, r_measure_array))
    N = np.shape(rho_q_t)[1]
    P_jk_t = np.zeros((len(t_array), N, N, K), dtype=np.complex128)
    Prob_dis_t = np.zeros((len(t_array), K), dtype=np.complex128)
    for i in range(len(t_array)):
        P_jk_t[i], Prob_dis_t[i] = Generadyne_Qubit_Post_Measurement_State(N_qubits, n_modes, sigma_jk_t[i], sigma_measure, r_jk_t[i], rho_q_t[i], r_measure_array)
    return P_jk_t, Prob_dis_t


def Homodyne_Direction(n_modes, mode_number, axis_angle):
    """Unit phase-space vector of the quadrature x cos(angle) + p sin(angle) of mode mode_number."""
    u = np.zeros(2*n_modes)
    u[2*mode_number], u[2*mode_number + 1] = np.cos(axis_angle), np.sin(axis_angle)
    return u


def CM_Homodyne_Noisy_1Mode(n_modes, mode_number, axis_angle, efficency):
    """
    Measurement covariance of homodyne detection of x cos(angle) + p sin(angle) on one mode with efficiency eta:
    variance ~0 along that quadrature and ~infinite across it, plus (1 - eta)/eta = tan^2(arccos sqrt(eta)) from
    the loss.  Unmeasured modes get a ~infinite variance.
    """
    z = 10**(-5)
    efficency_angle = np.arccos(np.sqrt(efficency))
    sigma_m = 1/z**2*np.eye(2*n_modes)
    rotation = np.array([[np.cos(axis_angle), -np.sin(axis_angle)],
                         [np.sin(axis_angle), np.cos(axis_angle)]])
    sigma_homo = np.diag([z**2, 1/z**2])
    i = 2*mode_number
    sigma_m[i:i + 2, i:i + 2] = rotation @ sigma_homo @ rotation.T + np.tan(efficency_angle)**2*np.eye(2)
    return sigma_m


def CM_Heterodyne_Noisy_1Mode(n_modes, mode_number, axis_angle, efficency):
    """
    Measurement covariance of heterodyne detection of one mode with efficiency eta: 1 + 2 (1 - eta)/eta, the
    vacuum noise of the ideal measurement plus the loss.  Unmeasured modes get a ~infinite variance.
    """
    z = 10**(-5)
    efficency_angle = np.arccos(np.sqrt(efficency))
    sigma_m = 1/z**2*np.eye(2*n_modes)
    i = 2*mode_number
    sigma_m[i:i + 2, i:i + 2] = (1 + 2*np.tan(efficency_angle)**2)*np.eye(2)
    return sigma_m


def Homodyne_Outcomes(n_modes, mode_number, axis_angle, r_measure_array):
    """Outcome vectors m u for the homodyne outcomes m along the quadrature of Homodyne_Direction."""
    return np.outer(np.ravel(r_measure_array), Homodyne_Direction(n_modes, mode_number, axis_angle))


def Heterodyne_Outcomes(n_modes, mode_number, r_measure_array):
    """Outcome vectors for heterodyne outcomes given as (x, p) pairs of the measured mode, shape (K, 2)."""
    r_measure_array = np.asarray(r_measure_array, dtype=float)
    if r_measure_array.ndim != 2 or r_measure_array.shape[1] != 2:
        raise ValueError('heterodyne outcomes are (x, p) pairs of the measured mode: pass an array of shape (K, 2), '
                         f'not {r_measure_array.shape}')
    vectors = np.zeros((len(r_measure_array), 2*n_modes))
    vectors[:, 2*mode_number:2*mode_number + 2] = r_measure_array
    return vectors


def Homodyne_Qubit_Post_Measurement_State(N_qubits, n_modes, sigma_jk, sigma_measure, r_jk, rho_q, r_measure_array, mode_number, quadrature):
    """Homodyne of x ('Position') or p ('Momentum') of one mode; sigma_measure from CM_Homodyne_Noisy_1Mode."""
    angle = {'Position': 0.0, 'Momentum': np.pi/2}[quadrature]
    vectors = Homodyne_Outcomes(n_modes, mode_number, angle, r_measure_array)
    return Generadyne_Qubit_Post_Measurement_State(N_qubits, n_modes, sigma_jk, sigma_measure, r_jk, rho_q, vectors)


def Homodyne_Qubit_Post_Measurement_State_Time_Evolution(N_qubits, n_modes, sigma_jk_t, sigma_measure, r_jk_t, rho_q_t, r_measure_array, t_array, mode_number, quadrature):
    angle = {'Position': 0.0, 'Momentum': np.pi/2}[quadrature]
    vectors = Homodyne_Outcomes(n_modes, mode_number, angle, r_measure_array)
    return Generadyne_Qubit_Post_Measurement_State_Time_Evolution(N_qubits, n_modes, sigma_jk_t, sigma_measure, r_jk_t, rho_q_t, vectors, t_array)

########################################## Entanlgement Measurements ##########################################


def Partial_Transpose_Dense(rho, mask, dimensions_list):
        """Computes the Partial Transposition"""

        dims = [dimensions_list, dimensions_list]
        nsys = len(mask)
        pt_dims = np.arange(2 * nsys).reshape(2, nsys).T
        pt_idx = np.concatenate([[pt_dims[n, mask[n]] for n in range(nsys)],
                                [pt_dims[n, 1 - mask[n]] for n in range(nsys)]])

        partial_transpose = rho.reshape(
            np.array(dims).flatten()).transpose(pt_idx).reshape(rho.shape)

        return partial_transpose

def Negativity_Qubits(rho):
        """Compute the witness (PPT) both as value (given the state) and as operator """
        J = (len(rho)-1)/4 
        dim = int(2*J+1)
        dimensions_list = [dim,dim]
        mask = [0,1]
        
        rho_pt = Partial_Transpose_Dense(rho, mask, dimensions_list)
        
        eva, eve = linalg.eig(rho_pt)
        eve = np.transpose(eve)
        
        nega = 0
        witness_matrix = np.zeros((dim**2, dim**2),dtype =np.complex128)
        
        for i in range(len(eva)):
            num = 0
            if eva[i] < 0:
                nega += eva[i]
                witness_matrix += np.outer(eve[i], np.conj(eve[i].T))
                num += 1
        
        witness_matrix = Partial_Transpose_Dense(witness_matrix, mask, dimensions_list)
        return nega, witness_matrix


def Negativity_Qubits_Time(rho_t, t_array):
        """Compute the witness (PPT) both as value (given the state) and as operator """
        J = (len(rho_t[0])-1)/4 
        dim = int(2*J+1)
        dimensions_list = [dim,dim]
        mask = [0,1]
        witness_matrix = np.zeros((len(t_array), dim**2, dim**2),dtype =np.complex128)
        nega_array = np.zeros(len(t_array))
        
        for j in range(len(t_array)):
            
            rho_pt = Partial_Transpose_Dense(rho_t[j], mask, dimensions_list)
            
            eva, eve = linalg.eig(rho_pt)
            eve = np.transpose(eve)
            
            nega = 0
            
            
            for i in range(len(eva)):
                num = 0
                if eva[i] < 0:
                    nega_array[j] += eva[i]
                    witness_matrix[j] += np.outer(eve[i], np.conj(eve[i].T))
                    num += 1
            
            witness_matrix[j] = Partial_Transpose_Dense(witness_matrix[j], mask, dimensions_list)
        return nega_array, witness_matrix
