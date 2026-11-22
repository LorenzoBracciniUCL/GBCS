import numpy as np 
import matplotlib.pyplot as plt
import math as math 
import cmath as cmath
from scipy import linalg as linalg


from matplotlib import cm
from matplotlib import rcParams
from matplotlib import colors


from . import Gaussian
from . import Unitary_Operator_Force
from . import Unitary_Operator_Gaussian
from . import Unitary_Operator_Time_Dependent

from . import Open_Operator_Force
from . import Open_Operator_Gaussian
from . import Open_Operator_Time_Dependent

from . import Riccati_Linear
from . import Symplectic_Known
from . import Measurements
from . import Plots_Functions
from . import Wigner_Functions
from . import Operations
from . import Perturbation_Theory


class Hamiltonian:
    
    def __init__(self, N_qubits, n_modes):

        # Parameters for Initiation
        self.N_qubits = N_qubits
        self.n_modes = n_modes
        #self.evolution_type = evolution_type # Gaussian, Force, General
        #self.sim_type = sim_type # Numerical or Analytical

    def Initialize_Constant_Hamiltonians(self, H_array, r_array, H_q_0_array):
        """
        H_array       Array of 1 to N+1 Matrix Hamiltonians (2nx2n)
        r_array       Array of 0 to N+1 Force Vectors (2nx1)
        H_q_0_array   Array of free evolution of the qubit (2Nx1)
        """
        
        self.H_array = H_array
        self.r_array = r_array
        self.H_q_0_array = H_q_0_array
        
        if len(H_array) == 1:
            
            if len(r_array) == 1:
                self.type = ['Constant','Gaussian']
            else:
                self.type = ['Constant','Force']           
        else:
            self.type = ['Constant','General']
        return

    def Initialize_Qudit_Hamiltonians(self, H_levels, r_levels, H_0_levels):
        """
        One N-level DV system instead of a qubit register: H = sum_j (1/2 r^T H_j r - r_j^T r + 1/2 H^0_j) |j><j|.
        Only Quantum_State.Perturbed_Dynamics uses this form.

        H_levels    N matrices (2n x 2n), one per level
        r_levels    N force vectors (2n x 1)
        H_0_levels  N energies H^0_j
        """
        if not (len(H_levels) == len(r_levels) == len(np.ravel(H_0_levels))):
            raise ValueError('H_levels, r_levels and H_0_levels need one entry per level')
        self.H_levels = [np.asarray(H) for H in H_levels]
        self.r_levels = [np.asarray(r).reshape(-1, 1) for r in r_levels]
        self.H_0_levels = np.ravel(H_0_levels)
        self.N_levels = len(H_levels)
        self.type = ['Constant', 'Qudit']
        return

    def Initialize_Time_Dep_Hamiltonian(self, H_array_func, r_array_func, H_q_0_array_func, t_array):
        """
        H_array_func       Function which returns an array of 1 to N+1 Matrix Hamiltonians (2nx2n) at each t
        r_array_func       Function which returns an array of 0 to N+1 Force Vectors (2nx1) at each t 
        H_q_0_array_func   Function which returns an array of free evolution of the qubit (2Nx1) at each t
        """

        self.H_array_func = H_array_func
        self.r_array_func = r_array_func
        self.H_q_0_array_func = H_q_0_array_func
        
        if len(H_array_func) == 1:
            
            if len(r_array_func) == 1:
                self.type = ['Time','Gaussian']
            else:
                self.type = ['Time','Force']    
        else:
            self.type = ['Time','General']

        H_array_t = np.zeros((len(H_array_func), len(t_array), 2*self.n_modes, 2*self.n_modes))
        r_array_t = np.zeros((len(r_array_func), len(t_array), 2*self.n_modes, 1))
        H_q_0_array_t = np.zeros((len(H_q_0_array_func), len(t_array)))
        
        for j in range(len(H_array_func)):
            for i in range(len(t_array)):
                H_array_t[j,i] = H_array_func[j](t_array[i])
                r_array_t[j,i] = r_array_func[j](t_array[i])
        
        for j in range(len(H_q_0_array_t)):
            for i in range(len(t_array)):
                H_q_0_array_t[j,i] = H_q_0_array_func[j](t_array[i])

        self.H_array_t = H_array_t
        self.r_array_t = r_array_t
        self.H_q_0_array_t = H_q_0_array_t
        
        return
    
class Symplectic:
    
    def __init__(self, N_qubits, n_modes):

        # Parameters for Initiation
        self.N_qubits = N_qubits
        self.n_modes = n_modes
        #self.evolution_type = evolution_type # Gaussian, Force, General
        #self.sim_type = sim_type # Numerical or Analytical

    def Initialize_Constant_Symplectic(self, symplectic_name = False,  symplectic_parameters = False, symplectic_func = False):
        
        self.type = 'Constant'

        if symplectic_name == False:
            self.symplectic_transformation = symplectic_func
        else:
            symplectic_transformation_known = {
                "QHO_1": Symplectic_Known.Symplectic_QHO_1,
                "QHO_n": Symplectic_Known.Symplectic_QHO_N,
                "QHO_2_XX": Symplectic_Known.Symplectic_2QHO_XX}
            self.symplectic_transformation = symplectic_transformation_known[symplectic_name](self.n_modes, symplectic_parameters)
        
        return 

class Decoherence:

    def __init__(self, N_qubits, n_modes, basis, B_matrix, d_vec=None, Gamma_z=None):

        # Parameters for Initiation
        self.N_qubits = N_qubits
        self.n_modes = n_modes
        self.d_vec = d_vec if d_vec is not None else np.zeros((2*n_modes, 1))
        self.Gamma_z = Gamma_z if Gamma_z is not None else np.zeros(N_qubits if N_qubits is not None else 0)

        if basis == 'Canonical':
            self.E_mat, self.D_mat = Gaussian.Get_Decoherence_Rates(n_modes, B_matrix)

        elif basis == 'Ladder':
            U_N = Gaussian.U_N(n_modes)
            self.E_mat, self.D_mat = Gaussian.Get_Decoherence_Rates(n_modes,  U_N.T@B_matrix@U_N)

        else:
            raise ValueError(f"basis must be 'Canonical' or 'Ladder', not {basis!r}")

class Quantum_State:

################################## Initialization ################################## 

    def __init__(self, N_qubits, n_modes):
        self.N_qubits = N_qubits
        self.n_modes = n_modes
        self.sigma_JK_t_PMS = None
        self.Measurament = None

    def Initialize_Gaussian_State(self, r_0, sigma_0,  rho_q_0):


        self.sigma_0 = sigma_0.astype(complex)
        self.r_0 = r_0.astype(complex)
        self.rho_q_0 = rho_q_0.astype(complex)
        self.type = 'Gaussian' # Gaussian or GCS

        return

    def Initialize_GCS(self, sigma_JK_0, r_JK_0, rho_q_0):
        """
        Initialize a general GCS with per-branch initial conditions.
        Typical use: feed the final time-step of a previous evolution,
        or the post-measurement state produced by Mode_Measurament.

        sigma_JK_0 : (2^N, 2^N, 2n, 2n)  — per-branch covariance matrices
        r_JK_0     : (2^N, 2^N, 2n, 1)   — per-branch first moments
        rho_q_0    : (2^N, 2^N)           — per-branch QRDM elements
        """

        self.sigma_0 = sigma_JK_0.astype(complex)
        self.r_0 = r_JK_0.astype(complex)
        self.rho_q_0 = rho_q_0.astype(complex)
        self.type = 'GCS'

        return

    def Start_Dynamics_Unitary(self):
        
        self.r_t = 0 
        self.sigma_t = 0 
        self.sigma_JK_t = 0
        self.r_JK_t = 0 
        self.rho_q_t = 0
        self.r_JK_0_t = 0
        self.C_JK_t = 0
        self.phi_JK_t = 0 
        
        return 

    def Start_Dynamics_Noise(self):
        
        self.r_t_D = 0 
        self.sigma_t_D = 0 
        self.sigma_JK_t_D = 0
        self.r_JK_t_D = 0 
        self.rho_q_t_D = 0
        self.r_JK_0_t_D = 0
        self.C_JK_t_D = 0
        self.phi_JK_t_D = 0 
        
        return 

################################## Dynamics ################################## 

    def Check_Solver(self, solver, Hamiltonian):
        if solver not in ('ode', 'linear'):
            raise ValueError(f"solver must be 'ode' or 'linear', not {solver!r}")
        if solver == 'linear' and list(Hamiltonian.type) != ['Constant', 'General']:
            raise NotImplementedError(f"solver='linear' only covers the Constant + General case, not {Hamiltonian.type}")

    def Unitary_Dynamics_Numerical(self, Hamiltonian, t_array, solver='ode'):
        """Given an Hamiltonian, Computes the unitary Dyanmics Numerically:
        1. Operator Valued Gaussian -> Solves the ODEs of Table 1
        2. Operator Valued Force -> Numerically Exponentiate the Symplectic Transformation for Table 2

        solver='linear' replaces the ODEs of case 1 by the linear map of Riccati_Linear.
        """

        self.Check_Solver(solver, Hamiltonian)

        self.t_array = t_array

        self.Start_Dynamics_Unitary()
        
        if Hamiltonian.type[0] == 'Constant':
            if Hamiltonian.type[1] == 'Gaussian':
                if self.type == 'GCS':
                    N2 = 2**self.N_qubits
                    self.sigma_JK_t = np.zeros((len(t_array), N2, N2, 2*self.n_modes, 2*self.n_modes), dtype=complex)
                    self.r_JK_t     = np.zeros((len(t_array), N2, N2, 2*self.n_modes, 1), dtype=complex)
                    for j in range(N2):
                        for k in range(j, N2):
                            r_jk, sigma_jk = Gaussian.Unitary_Numerical(self.n_modes, self.r_0[j,k], self.sigma_0[j,k],
                                                                         Hamiltonian.H_array[0], Hamiltonian.r_array[0], t_array)
                            self.r_JK_t[:,j,k]     = r_jk
                            self.sigma_JK_t[:,j,k] = sigma_jk
                            self.r_JK_t[:,k,j]     = np.conjugate(r_jk)
                            self.sigma_JK_t[:,k,j] = np.conjugate(sigma_jk)
                    self.C_JK_t   = np.zeros((len(t_array), N2, N2))
                    self.phi_JK_t = np.zeros((len(t_array), N2, N2))
                    self.rho_q_t  = np.tile(self.rho_q_0, (len(t_array), 1, 1))
                    self.type = 'GCS'
                else:
                    self.r_t, self.sigma_t = Gaussian.Unitary_Numerical(self.n_modes, self.r_0, self.sigma_0,
                                                                         Hamiltonian.H_array[0], Hamiltonian.r_array[0], t_array)

            elif Hamiltonian.type[1] == 'Force':
                self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t  = Unitary_Operator_Force.Dynamics_Numerical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0,  Hamiltonian.H_array[0], Hamiltonian.r_array, Hamiltonian.H_q_0_array, t_array)
                self.r_JK_0_t =  - self.C_JK_t + 1j*self.phi_JK_t
                self.rho_q_t =  np.exp(- self.C_JK_t + 1j*self.phi_JK_t)*self.rho_q_0
                self.type = 'GCS'

            elif Hamiltonian.type[1] == 'General':
                if solver == 'linear':
                    self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t = Riccati_Linear.Dynamics_Linear(self.N_qubits, self.n_modes, self.r_0, self.sigma_0,
                                                                                                    Hamiltonian.H_array, Hamiltonian.r_array, Hamiltonian.H_q_0_array, self.rho_q_0, t_array)
                else:
                    self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t = Unitary_Operator_Gaussian.Dynamics_Numerical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0,
                                                                                                    Hamiltonian.H_array, Hamiltonian.r_array, Hamiltonian.H_q_0_array, self.rho_q_0, t_array)
                self.r_JK_0_t =  - self.C_JK_t + 1j*self.phi_JK_t
                self.rho_q_t =  np.exp(- self.C_JK_t + 1j*self.phi_JK_t)*self.rho_q_0
                self.type = 'GCS'

        elif Hamiltonian.type[0] == 'Time':
            if Hamiltonian.type[1] == 'Gaussian' and self.type != 'GCS':
                self.r_t, self.sigma_t = Gaussian.Unitary_Numerical_Time_Dep(
                    self.n_modes, self.r_0, self.sigma_0,
                    Hamiltonian.H_array_func[0], Hamiltonian.r_array_func[0], t_array)
            else:
                self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t  = Unitary_Operator_Time_Dependent.Dynamics_Numerical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0, Hamiltonian.H_array_func,
                                                                                                                Hamiltonian.r_array_func, Hamiltonian.H_q_0_array_func, self.rho_q_0, self.t_array)
                self.r_JK_0_t =  - self.C_JK_t + 1j*self.phi_JK_t
                self.rho_q_t =  np.exp(- self.C_JK_t + 1j*self.phi_JK_t)*self.rho_q_0
                self.type = 'GCS'

        return
    
    def Unitary_Dynamics_Analytical(self, Hamiltonian, Symplectic, t_array):
        
        self.t_array = t_array
        self.Start_Dynamics_Unitary()
        
        if Symplectic.type == 'Constant':
            if Hamiltonian.type[1] == 'Gaussian':
                if self.type == 'GCS':
                    N2 = 2**self.N_qubits
                    self.sigma_JK_t = np.zeros((len(t_array), N2, N2, 2*self.n_modes, 2*self.n_modes), dtype=complex)
                    self.r_JK_t     = np.zeros((len(t_array), N2, N2, 2*self.n_modes, 1), dtype=complex)
                    for j in range(N2):
                        for k in range(j, N2):
                            r_jk, sigma_jk = Gaussian.Unitary_Analytical(self.n_modes, self.r_0[j,k], self.sigma_0[j,k],
                                                                          Symplectic.symplectic_transformation,
                                                                          Hamiltonian.H_array[0], Hamiltonian.r_array[0], t_array)
                            self.r_JK_t[:,j,k]     = r_jk
                            self.sigma_JK_t[:,j,k] = sigma_jk
                            self.r_JK_t[:,k,j]     = np.conjugate(r_jk)
                            self.sigma_JK_t[:,k,j] = np.conjugate(sigma_jk)
                    self.C_JK_t   = np.zeros((len(t_array), N2, N2))
                    self.phi_JK_t = np.zeros((len(t_array), N2, N2))
                    self.rho_q_t  = np.tile(self.rho_q_0, (len(t_array), 1, 1))
                    self.type = 'GCS'
                else:
                    self.r_t, self.sigma_t = Gaussian.Unitary_Analytical(self.n_modes, self.r_0, self.sigma_0,
                                                                          Symplectic.symplectic_transformation,
                                                                          Hamiltonian.H_array[0], Hamiltonian.r_array[0], t_array)
            elif Hamiltonian.type[1] == 'Force':
                self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t  = Unitary_Operator_Force.Dynamics_Analytical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0,
                                                                                                                    Symplectic.symplectic_transformation,
                                                                                                                    Hamiltonian.H_array[0], Hamiltonian.r_array, Hamiltonian.H_q_0_array, t_array)
                self.r_JK_0_t =  - self.C_JK_t + 1j*self.phi_JK_t
                self.rho_q_t =  np.exp(- self.C_JK_t + 1j*self.phi_JK_t)*self.rho_q_0
                self.type = 'GCS'

        elif Symplectic.type == 'Time':
            self.sigma_JK_t, self.r_JK_t, self.r_JK_0_t = Unitary_Operator_Time_Dependent.Dynamics_Analytical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0, Hamiltonian.H_array_func,
                                                                                                               Hamiltonian.r_array_func, Hamiltonian.H_q_0_array_func, self.rho_q_0, self.t_array)
            self.type = 'GCS'

        return
    
    def Open_Dynamics_Numerical(self, Hamiltonian, Decoherence, t_array, solver='ode'):
        """solver='linear' replaces the ODEs of the Constant + General case by the linear map of Riccati_Linear."""

        self.Check_Solver(solver, Hamiltonian)

        self.t_array = t_array

        E = Decoherence.E_mat
        D = Decoherence.D_mat 

        self.Start_Dynamics_Noise()
        
        if Hamiltonian.type[0] == 'Constant':
            if Hamiltonian.type[1] == 'Gaussian':
                if self.type == 'GCS':
                    N2 = 2**self.N_qubits
                    self.sigma_JK_t = np.zeros((len(t_array), N2, N2, 2*self.n_modes, 2*self.n_modes), dtype=complex)
                    self.r_JK_t     = np.zeros((len(t_array), N2, N2, 2*self.n_modes, 1), dtype=complex)
                    for j in range(N2):
                        for k in range(j, N2):
                            r_jk, sigma_jk = Gaussian.Open_Numerical(self.n_modes, self.r_0[j,k], self.sigma_0[j,k],
                                                                      Hamiltonian.H_array[0], Hamiltonian.r_array[0], E, D, t_array)
                            self.r_JK_t[:,j,k]     = r_jk
                            self.sigma_JK_t[:,j,k] = sigma_jk
                            self.r_JK_t[:,k,j]     = np.conjugate(r_jk)
                            self.sigma_JK_t[:,k,j] = np.conjugate(sigma_jk)
                    self.C_JK_t   = np.zeros((len(t_array), N2, N2))
                    self.phi_JK_t = np.zeros((len(t_array), N2, N2))
                    self.rho_q_t  = np.tile(self.rho_q_0, (len(t_array), 1, 1))
                    self.type = 'GCS'
                else:
                    self.r_t, self.sigma_t = Gaussian.Open_Numerical(self.n_modes, self.r_0, self.sigma_0,
                                                                      Hamiltonian.H_array[0], Hamiltonian.r_array[0], E, D, t_array)

            elif Hamiltonian.type[1] == 'Force':
                if E.all() == 0.0:
                    self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t  = Open_Operator_Force.Dynamics_Symetric_Numerical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0,  Hamiltonian.H_array[0], Hamiltonian.r_array, Hamiltonian.H_q_0_array, D, t_array)
                    self.r_JK_0_t =  - self.C_JK_t + 1j*self.phi_JK_t
                    self.rho_q_t =  np.exp(- self.C_JK_t + 1j*self.phi_JK_t)*self.rho_q_0
                else:
                    self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t  = Open_Operator_Force.Dynamics_General_Numerical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0,  Hamiltonian.H_array[0], Hamiltonian.r_array, Hamiltonian.H_q_0_array, E, D, t_array)
                    self.r_JK_0_t =  - self.C_JK_t + 1j*self.phi_JK_t
                    self.rho_q_t =  np.exp(- self.C_JK_t + 1j*self.phi_JK_t)*self.rho_q_0
                self.type = 'GCS'

            elif Hamiltonian.type[1] == 'General':
                if solver == 'linear':
                    self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t = Riccati_Linear.Dynamics_Linear(self.N_qubits, self.n_modes, self.r_0, self.sigma_0,
                                                                                                    Hamiltonian.H_array, Hamiltonian.r_array, Hamiltonian.H_q_0_array, self.rho_q_0, t_array,
                                                                                                    E=E, D=D, d=Decoherence.d_vec, Gamma_z=Decoherence.Gamma_z)
                else:
                    self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t = Open_Operator_Gaussian.Dynamics_Numerical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0,
                                                                                                    Hamiltonian.H_array, Hamiltonian.r_array, Hamiltonian.H_q_0_array, self.rho_q_0, E, D, t_array,
                                                                                                    d=Decoherence.d_vec, Gamma_z=Decoherence.Gamma_z)
                self.r_JK_0_t =  - self.C_JK_t + 1j*self.phi_JK_t
                self.rho_q_t =  np.exp(- self.C_JK_t + 1j*self.phi_JK_t)*self.rho_q_0
                self.type = 'GCS'

        elif Hamiltonian.type[0] == 'Time':
            if Hamiltonian.type[1] == 'Gaussian' and self.type != 'GCS':
                self.r_t, self.sigma_t = Gaussian.Open_Numerical_Time_Dep(
                    self.n_modes, self.r_0, self.sigma_0,
                    Hamiltonian.H_array_func[0], Hamiltonian.r_array_func[0], E, D, t_array)
            else:
                self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t  = Open_Operator_Time_Dependent.Dynamics_Numerical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0, Hamiltonian.H_array_func,
                                                                                                                Hamiltonian.r_array_func, Hamiltonian.H_q_0_array_func, self.rho_q_0, E, D, self.t_array)
                self.r_JK_0_t =  - self.C_JK_t + 1j*self.phi_JK_t
                self.rho_q_t =  np.exp(- self.C_JK_t + 1j*self.phi_JK_t)*self.rho_q_0
                self.type = 'GCS'

        return

    def Open_Dynamics_Analytical(self, Hamiltonian, Symplectic, t_array):
        self.t_array = t_array

        self.Start_Dynamics_Unitary()
        
        if Symplectic.type == 'Constant':
            if Hamiltonian.type[1] == 'Gaussian':
                if self.type == 'GCS':
                    N2 = 2**self.N_qubits
                    self.sigma_JK_t = np.zeros((len(t_array), N2, N2, 2*self.n_modes, 2*self.n_modes), dtype=complex)
                    self.r_JK_t     = np.zeros((len(t_array), N2, N2, 2*self.n_modes, 1), dtype=complex)
                    for j in range(N2):
                        for k in range(j, N2):
                            r_jk, sigma_jk = Gaussian.Unitary_Analytical(self.n_modes, self.r_0[j,k], self.sigma_0[j,k],
                                                                          Symplectic.symplectic_transformation,
                                                                          Hamiltonian.H_array[0], Hamiltonian.r_array[0], t_array)
                            self.r_JK_t[:,j,k]     = r_jk
                            self.sigma_JK_t[:,j,k] = sigma_jk
                            self.r_JK_t[:,k,j]     = np.conjugate(r_jk)
                            self.sigma_JK_t[:,k,j] = np.conjugate(sigma_jk)
                    self.C_JK_t   = np.zeros((len(t_array), N2, N2))
                    self.phi_JK_t = np.zeros((len(t_array), N2, N2))
                    self.rho_q_t  = np.tile(self.rho_q_0, (len(t_array), 1, 1))
                    self.type = 'GCS'
                else:
                    self.r_t, self.sigma_t = Gaussian.Unitary_Analytical(self.n_modes, self.r_0, self.sigma_0,
                                                                          Symplectic.symplectic_transformation,
                                                                          Hamiltonian.H_array[0], Hamiltonian.r_array[0], t_array)

            elif Hamiltonian.type[1] == 'Force':
                self.sigma_JK_t, self.r_JK_t, self.C_JK_t, self.phi_JK_t  = Unitary_Operator_Force.Unitary_Analytical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0,
                                                                                                                    Symplectic.symplectic_transformation,
                                                                                                                    Hamiltonian.H_array[0], Hamiltonian.r_array, Hamiltonian.H_q_0_array, t_array)
                self.r_JK_0_t =  - self.C_JK_t + 1j*self.phi_JK_t
                self.rho_q_t =  np.exp(- self.C_JK_t + 1j*self.phi_JK_t)*self.rho_q_0
                self.type = 'GCS'

        elif Symplectic.type == 'Time':
            self.sigma_JK_t, self.r_JK_t, self.r_JK_0_t = Unitary_Operator_Time_Dependent.Unitary_Gaussian_General_Time_Dep_Numerical(self.N_qubits, self.n_modes, self.r_0, self.sigma_0, Hamiltonian.H_array_func,
                                                                                                               Hamiltonian.r_array_func, Hamiltonian.H_q_0_array_func, self.rho_q_0, self.t_array)
            self.rho_q_t = np.exp(self.r_JK_0_t)
            self.type = 'GCS'

        return

    def Perturbed_Dynamics(self, Hamiltonian, Perturbation, t_array, Decoherence=None, order=1,
                           method='quadrature', n_samples=1000, seed=None, store_times=None, max_paths=2_000_000):
        """
        Dyson series in the DV-only terms of Perturbation (T1, ladder decay, DV Hamiltonian) around the
        operator-valued Gaussian dynamics of Hamiltonian (constant), with Decoherence if given.
        See gbcs.Perturbation_Theory for the method.

        order       the series is cut after this many jumps
        method      'quadrature' (jump times on t_array, trapezoid rule) or 'monte_carlo' (n_samples per order)
        store_times grid indices at which the Gaussian components of orders >= 1 are kept for Wigner functions
                    (default: the last one)

        Sets the order-0 GBCS in the usual attributes (sigma_JK_t, r_JK_t, C_JK_t, phi_JK_t, r_JK_0_t, rho_q_t), so
        every existing method still describes that part consistently, and adds:
          rho_q_t_total            QRDM to the given order
          rho_q_t_orders           QRDM contribution of each order, (order + 1, T, N, N)
          rho_q_t_stderr           its standard error (Monte Carlo; zeros for quadrature)
          first_moments_t          Tr[r_hat rho_jk], all orders
          second_moments_t         Tr[sym(r_hat r_hat^T) rho_jk], all orders
          perturbation_components  {time index: Gaussian components of orders >= 1}
          perturbation_last_order  largest |rho_jk| of the last order, to judge the cut
        Returns the Perturbation_Theory.Perturbation_Result.
        """
        N = Perturbation.N_levels
        E = D = d = None
        if Decoherence is not None:
            E, D, d = Decoherence.E_mat, Decoherence.D_mat, Decoherence.d_vec
        H_levels, r_levels, H_0_levels, dephasing = Perturbation_Theory.Level_Structure(Hamiltonian, Decoherence, N)

        result = Perturbation_Theory.Perturbation_Dynamics(
            self.n_modes, self.r_0, self.sigma_0, self.rho_q_0, H_levels, r_levels, H_0_levels, dephasing,
            Perturbation, t_array, E=E, D=D, d=d, order=order, method=method, n_samples=n_samples,
            seed=seed, store_times=store_times, max_paths=max_paths)

        self.t_array = result.t_array
        self.N_levels = N
        self.sigma_JK_t, self.r_JK_t = result.sigma_JK_t, result.r_JK_t
        self.C_JK_t, self.phi_JK_t = result.C_JK_t, result.phi_JK_t
        self.r_JK_0_t = -self.C_JK_t + 1j*self.phi_JK_t
        self.rho_q_t = result.rho_q_t
        self.rho_q_t_total = result.rho_q_t_total
        self.rho_q_t_orders = result.rho_q_t_orders
        self.rho_q_t_stderr = result.rho_q_t_stderr
        self.first_moments_t = result.first_moments_t
        self.second_moments_t = result.second_moments_t
        self.perturbation_components = result.components
        self.perturbation_last_order = result.last_order_size
        self.type = 'GCS'
        return result

    def Wigner_Function_Perturbation(self, steps, time_index, mode_number, sigma_para, part='all', projector=None):
        """
        Wigner function of the orders >= 1 of Perturbed_Dynamics at time_index, which must be one of its
        store_times, on the grid of the other Wigner methods.  It is the W_g of Plot_Wigner_Function_GCS_Weighted.

        part      'all', 'diagonal' (the modes' reduced state) or 'off_diagonal' (interference term)
        projector (N, N) DV operator Pi; gives Tr_q[Pi W] instead of part, e.g. for a measurement outcome

        Returns W, x_array, p_array.
        """
        time_index = int(time_index) % len(self.t_array)
        if time_index not in self.perturbation_components:
            stored = sorted(self.perturbation_components)
            raise ValueError(f'no components stored at time index {time_index} (stored: {stored}); '
                             'pass it in store_times to Perturbed_Dynamics')
        r_tilde_array, x_array, p_array, _, _ = Wigner_Functions.Create_r_tilde_array(self, sigma_para, mode_number, steps)
        W = Perturbation_Theory.Wigner_Components(self.perturbation_components[time_index], r_tilde_array,
                                                  self.n_modes, part=part, projector=projector)
        return W, x_array, p_array

################################## Operations ##################################

    def _current_gcs_snapshot(self):
        """Return (sigma, r, rho_q) for the latest GCS snapshot (after dynamics if run)."""
        if (hasattr(self, 'sigma_JK_t')
                and self.sigma_JK_t is not None
                and not np.isscalar(self.sigma_JK_t)):
            return self.sigma_JK_t[-1], self.r_JK_t[-1], self.rho_q_t[-1]
        return self.sigma_0, self.r_0, self.rho_q_0

    def Apply_Conditional_Displacement(self, r_array):
        """
        Apply a conditional displacement to the current GCS state and re-initialise.

        r_array : (2^N, 2n, 1) — displacement vector r_j for each branch J.
        """
        sigma_JK, r_JK, rho_q = self._current_gcs_snapshot()
        sigma_new, r_new, delta_C, delta_phi = Operations.Conditional_Displacement(
            self.N_qubits, self.n_modes, sigma_JK, r_JK, r_array)
        rho_q_new = np.exp(-delta_C + 1j*delta_phi) * rho_q
        self.Initialize_GCS(sigma_new, r_new, rho_q_new)

    def Apply_Gaussian_Operation(self, X, Y, r_c):
        """
        Apply a Gaussian operation (X, Y, r_c) to the current GCS state and re-initialise.

        X, Y : (2n, 2n)
        r_c  : (2n, 1)
        """
        sigma_JK, r_JK, rho_q = self._current_gcs_snapshot()
        sigma_new, r_new, delta_C, delta_phi = Operations.Gaussian_Operation(
            self.N_qubits, self.n_modes, sigma_JK, r_JK, X, Y, r_c)
        rho_q_new = np.exp(-delta_C + 1j*delta_phi) * rho_q
        self.Initialize_GCS(sigma_new, r_new, rho_q_new)

################################## Measuraments ##################################

    def Qubit_Expectation_Values(self, operator,type_exp):
        if type_exp == 'Final':
            self.exp = Measurements.Qubit_Expecation_Value(operator, self.rho_q_t[-1])
        elif type_exp == 'Time':
            self.exp_t = Measurements.Qubit_Expecation_Value_Time_Evolution(operator, self.rho_q_t, self.t_array)
        else:
            raise ValueError(f"type_exp must be 'Final' or 'Time', not {type_exp!r}")
        return 

    def Qubit_Ideal_Measurament(self, operator, type_measure):
        """Compute a qubit measurament
        If type == Ideal, parameter == index time 
        """
        self.Measurament = 'qubit'

        if type_measure == 'Final':
            self.outcomes, self.prob, self.POVM_array, self.rho_q_PMS_array = Measurements.Qubit_Ideal_Measurament(self.N_qubits, operator, self.rho_q_t[-1])
        
        elif type_measure == 'Time':
            self.outcomes, self.prob_t, self.POVM_array, self.rho_q_t_PMS_array =  Measurements.Qubit_Ideal_Measurament_Time_Evolution(self.N_qubits, operator,  self.rho_q_t,self.t_array)
        else:
            raise ValueError(f"type_measure must be 'Final' or 'Time', not {type_measure!r}")
        return 
    
    def Negativity_Qubit(self, type_nega):
        if type_nega == 'Final':
            self.nega, self.witness_matrix = Measurements.Negativity_Qubits(self.rho_q_t[-1])
        elif type_nega == 'Time':
            self.nega_t, self.witness_matrix_t = Measurements.Negativity_Qubits_Time(self.rho_q_t, self.t_array)
        else:
            raise ValueError(f"type_nega must be 'Final' or 'Time', not {type_nega!r}")
        return 
    
    def Mode_Measurament(self, type_measure, mode_number, axis_angle, efficency, r_measure_array):
        """Compute a qubit measurament
        If type == Ideal, parameter == index time 
        """
        if type_measure[0] == 'Final':
            if type_measure[1] == 'Heterodyne':
                sigma_measure = Measurements.CM_Heterodyne_Noisy_1Mode(self.n_modes, mode_number, axis_angle, efficency)
                self.Qubit_PMS, self.Prob_Dis = Measurements.Generadyne_Qubit_Post_Measurement_State(self.N_qubits, self.n_modes, self.sigma_JK_t[-1], sigma_measure, self.r_JK_t[-1], self.rho_q_t[-1], r_measure_array)
            elif type_measure[1] == 'Homodyne':
                sigma_measure = Measurements.CM_Homodyne_Noisy_1Mode(self.n_modes, mode_number, axis_angle, efficency)
                self.Qubit_PMS, self.Prob_Dis = Measurements.Generadyne_Qubit_Post_Measurement_State(self.N_qubits, self.n_modes, self.sigma_JK_t[-1], sigma_measure, self.r_JK_t[-1], self.rho_q_t[-1], r_measure_array)

        elif type_measure[0] == 'Time':
            if type_measure[1] == 'Heterodyne':
                sigma_measure = Measurements.CM_Heterodyne_Noisy_1Mode(self.n_modes, mode_number, axis_angle, efficency)
                self.Qubit_PMS_t, self.Prob_Dis_t = Measurements.Generadyne_Qubit_Post_Measurement_State_Time_Evolution(self.N_qubits, self.n_modes, self.sigma_JK_t, sigma_measure, self.r_JK_t, self.rho_q_t, r_measure_array, self.t_array)
            elif type_measure[1] == 'Homodyne':
                sigma_measure = Measurements.CM_Homodyne_Noisy_1Mode(self.n_modes, mode_number, axis_angle, efficency)
                self.Qubit_PMS_t, self.Prob_Dis_t = Measurements.Generadyne_Qubit_Post_Measurement_State_Time_Evolution(self.N_qubits, self.n_modes, self.sigma_JK_t, sigma_measure, self.r_JK_t, self.rho_q_t, r_measure_array, self.t_array)
        else:
            raise ValueError(f"type_measure[0] must be 'Final' or 'Time', not {type_measure[0]!r}")
        return 
    
    def Homdyne_Mode_Measurament(self, type_measure, mode_number, axis_angle, quadrature, efficency, r_measure_array):
        """Compute a qubit measurament
        If type == Ideal, parameter == index time 
        """
        if type_measure == 'Final':
            sigma_measure = Measurements.CM_Heterodyne_Noisy_1Mode(self.n_modes, mode_number, axis_angle, efficency)
            self.Qubit_PMS, self.Prob_Dis = Measurements.Homodyne_Qubit_Post_Measurement_State(self.N_qubits, self.n_modes, self.sigma_JK_t[-1], sigma_measure, self.r_JK_t[-1], self.rho_q_t[-1], r_measure_array, mode_number, quadrature)
        elif type_measure == 'Time':
            sigma_measure = Measurements.CM_Heterodyne_Noisy_1Mode(self.n_modes, mode_number, axis_angle, efficency)
            self.Qubit_PMS_t, self.Prob_Dis_t = Measurements.Homodyne_Qubit_Post_Measurement_State_Time_Evolution(self.N_qubits, self.n_modes, self.sigma_JK_t, sigma_measure, self.r_JK_t, self.rho_q_t, r_measure_array, self.t_array, mode_number, quadrature)
        else:
            raise ValueError(f"type_measure must be 'Final' or 'Time', not {type_measure!r}")
        return 
    
################################## Plot ################################## 

    def Plot_Phase_Space_First_QRDM(self, leged_lables, array_pi, save, mode=0):
        Plots_Functions.Plot_Phase_Space_First_QRDM_Func(self, leged_lables, array_pi, save, mode=mode)

    def Plot_Wigner_Function_Diag(self, steps, time_index, mode_number, sigma_para, array_tick, title, save):
            
        r_tilde_array, x_array, p_array, XX, PP = Wigner_Functions.Create_r_tilde_array(self,sigma_para, mode_number, steps)
        self.wigner_diag = Wigner_Functions.Wigner_Sum_Diagonal(self, time_index, r_tilde_array)
        bar_limits = [0, np.max(self.wigner_diag)]
        fig = Plots_Functions.Plot_Wigner_Diag(self.wigner_diag,  x_array, p_array, bar_limits, array_tick, title,save) 
        return

    def Plot_Wigner_4_Times(self, steps, time_index_array, mode_number, sigma_para, array_tick, save):
            
        r_tilde_array, x_array, p_array, XX, PP = Wigner_Functions.Create_r_tilde_array(self,sigma_para, mode_number, steps)
        wigner_array = np.zeros((4,len(x_array),len(p_array)))

        for i in range(len(time_index_array)):
            wigner_array[i] = Wigner_Functions.Wigner_Sum_Diagonal(self, time_index_array[i], r_tilde_array)
        
        fig = Plots_Functions.Plot_Wigner_4_Times(wigner_array, x_array, p_array, save)
        return

    def Plot_Wigner_Function_PMS(self, steps, time_index, mode_number, sigma_para, array_tick, title, save):
        r_tilde_array, x_array, p_array, XX, PP = Wigner_Functions.Create_r_tilde_array(self,sigma_para, mode_number,steps)
        self.wigner_PMS = Wigner_Functions.Wigner_PMS_func(self, time_index, r_tilde_array)

        bar_limits = [-np.max(self.wigner_PMS), np.max(self.wigner_PMS)]
        fig = Plots_Functions.Plot_Wigner_Fringes(self.wigner_PMS,  x_array, p_array, bar_limits,array_tick,title, save) 
        return
    
    def Plot_Wigner_Function_Gauss(self, steps, time_index, mode_number, sigma_para, array_tick, title, save):
            
        r_tilde_array, x_array, p_array, XX, PP = Wigner_Functions.Create_r_tilde_array(self,sigma_para, mode_number, steps)
        self.wigner_gauss = Wigner_Functions.Wigner_Gaussian(self, time_index, r_tilde_array)
        bar_limits = [0, np.max(self.wigner_gauss)]
        fig = Plots_Functions.Plot_Wigner_Diag(self.wigner_gauss,  x_array, p_array, bar_limits, array_tick,title, save)
        return

    def Plot_Trajectories_Single(self, mode=0, z_comp=0,
                                 title="Dynamics of the first moments",
                                 legend_handles=None, color_map=None, sym_lim=None,
                                 figsize=(10, 10), elev=22, azim=-55, save=False):
        r_m, _ = Plots_Functions.Select_Mode(mode, self.r_JK_t, self.sigma_JK_t)
        return Plots_Functions.Plot_Trajectories_Single(
            r_m, z_comp=z_comp, title=title,
            legend_handles=legend_handles, color_map=color_map,
            sym_lim=sym_lim, figsize=figsize, elev=elev, azim=azim, save=save)

    def Plot_Vectors_Single(self, t_idx, mode=0, z_comp=0, title="First moments",
                            legend_handles=None, color_map=None, sym_lim=None,
                            figsize=(10, 10), elev=22, azim=-55, save=False):
        r_m, _ = Plots_Functions.Select_Mode(mode, self.r_JK_t, self.sigma_JK_t)
        return Plots_Functions.Plot_Vectors_Single(
            r_m, t_idx, z_comp=z_comp, title=title,
            legend_handles=legend_handles, color_map=color_map,
            sym_lim=sym_lim, figsize=figsize, elev=elev, azim=azim, save=save)

    def Plot_Vectors_Ellipsoids_Single(self, t_idx, mode=0, z_comp=0,
                                       title="First moments with uncertainty",
                                       legend_handles=None, color_map=None, sym_lim=None,
                                       figsize=(10, 10), elev=22, azim=-55, save=False):
        r_m, s_m = Plots_Functions.Select_Mode(mode, self.r_JK_t, self.sigma_JK_t)
        return Plots_Functions.Plot_Vectors_Ellipsoids_Single(
            r_m, s_m, t_idx, z_comp=z_comp, title=title,
            legend_handles=legend_handles, color_map=color_map,
            sym_lim=sym_lim, figsize=figsize, elev=elev, azim=azim, save=save)

    def Animate_Vectors_Single(self, mode=0, z_comp=0,
                               title="First moments",
                               legend_handles=None, color_map=None, sym_lim=None,
                               figsize=(10, 10), elev=22, azim=-55,
                               n_frames=None, interval=50, save=False):
        r_m, _ = Plots_Functions.Select_Mode(mode, self.r_JK_t, self.sigma_JK_t)
        return Plots_Functions.Animate_Vectors_Single(
            r_m, z_comp=z_comp, title=title,
            legend_handles=legend_handles, color_map=color_map,
            sym_lim=sym_lim, figsize=figsize, elev=elev, azim=azim,
            n_frames=n_frames, interval=interval, save=save)

    def Plot_Diagonal_Multi_Time_2D(self, t_idx_array, alpha_array, mode=0,
                                    color_map=None, sym_lim=None,
                                    title="On-diagonal first moments",
                                    figsize=(6, 6), save=False):
        r_m, s_m = Plots_Functions.Select_Mode(mode, self.r_JK_t, self.sigma_JK_t)
        return Plots_Functions.Plot_Diagonal_Multi_Time_2D(
            r_m, s_m, t_idx_array, alpha_array,
            color_map=color_map, sym_lim=sym_lim, title=title, figsize=figsize, save=save)

    def Plot_Phase_Space_Summary(self, t_idx_array, alpha_array, mode=0,
                                 color_map=None, sym_lim_2d=None, sym_lim_3d=None,
                                 title="", figsize=(18, 6), elev=22, azim=-55, save=False):
        r_m, s_m = Plots_Functions.Select_Mode(mode, self.r_JK_t, self.sigma_JK_t)
        return Plots_Functions.Plot_Phase_Space_Summary(
            r_m, s_m, t_idx_array, alpha_array,
            color_map=color_map, sym_lim_2d=sym_lim_2d, sym_lim_3d=sym_lim_3d,
            title=title, figsize=figsize, elev=elev, azim=azim, save=save)

    def Plot_Vectors_Ellipsoids_Multi_Time(self, t_idx_array, alpha_array, mode=0, z_comp=0,
                                           title="First moments with uncertainty",
                                           legend_handles=None, color_map=None, sym_lim=None,
                                           figsize=(10, 10), elev=22, azim=-55, save=False):
        r_m, s_m = Plots_Functions.Select_Mode(mode, self.r_JK_t, self.sigma_JK_t)
        return Plots_Functions.Plot_Vectors_Ellipsoids_Multi_Time(
            r_m, s_m, t_idx_array, alpha_array, z_comp=z_comp, title=title,
            legend_handles=legend_handles, color_map=color_map,
            sym_lim=sym_lim, figsize=figsize, elev=elev, azim=azim, save=save)

    def Animate_Vectors_Ellipsoids_Single(self, mode=0, z_comp=0,
                                          title="First moments with uncertainty",
                                          legend_handles=None, color_map=None, sym_lim=None,
                                          figsize=(10, 10), elev=22, azim=-55,
                                          n_frames=None, interval=50, save=False):
        r_m, s_m = Plots_Functions.Select_Mode(mode, self.r_JK_t, self.sigma_JK_t)
        return Plots_Functions.Animate_Vectors_Ellipsoids_Single(
            r_m, s_m, z_comp=z_comp, title=title,
            legend_handles=legend_handles, color_map=color_map,
            sym_lim=sym_lim, figsize=figsize, elev=elev, azim=azim,
            n_frames=n_frames, interval=interval, save=save)

    def Plot_3D_Diagonal_Sigma(self, t_idx, mode=0, z_comp=0,
                               title="3-D phase space",
                               legend_handles=None, color_map=None, sym_lim=None,
                               pancake_color='tomato',
                               figsize=(10, 10), elev=22, azim=-55, save=False):
        """3-D phase-space snapshot: on-diagonal ellipsoids + off-diagonal tilted pancake.

        For N=1 qubit, n=1 mode only. Decomposes σ^off = U Σ U^T (real orthogonal U,
        complex diagonal Σ) and draws it as a pancake tilted by the polar phases of Σ.
        On-diagonal branches are shown as arrows (first moments) + uncertainty ellipsoids.

        Parameters
        ----------
        t_idx        : int — time index to snapshot
        mode         : int — bosonic mode index (0-based), for multi-mode states
        z_comp       : 0 → Im[x] on z-axis,  1 → Im[p] on z-axis
        title        : figure title
        pancake_color : colour for the off-diagonal pancake (default 'tomato')
        sym_lim      : (lo, hi) — inferred from data if None
        save         : file path to save, or False

        Returns
        -------
        fig, ax : Figure and Axes3D
        """
        r_m, s_m = Plots_Functions.Select_Mode(mode, self.r_JK_t, self.sigma_JK_t)
        return Plots_Functions.Plot_3D_Diagonal_Sigma(
            r_m, s_m, t_idx, z_comp=z_comp, title=title,
            legend_handles=legend_handles, color_map=color_map,
            sym_lim=sym_lim, pancake_color=pancake_color,
            figsize=figsize, elev=elev, azim=azim, save=save)

################################## Animation ##################################

    def Animate_Wigner_Function_Diag(self, steps, mode_number, sigma_para, array_tick, n_frames, title,save):
            
        r_tilde_array, x_array, p_array, XX, PP = Wigner_Functions.Create_r_tilde_array(self,sigma_para, mode_number, steps)
        self.wigner_diag_t = Wigner_Functions.Wigner_t_Sum_Diag(self, r_tilde_array)
        bar_limits = [0, np.max(self.wigner_diag_t)]
        fig = Plots_Functions.Animate_Diagonal(self.wigner_diag_t,  x_array, p_array, bar_limits, array_tick, n_frames,title ,save)
        return

    def Animate_Wigner_Function_PMS(self, steps, mode_number, sigma_para, array_tick, n_frames,title, save):
            
        r_tilde_array, x_array, p_array, XX, PP = Wigner_Functions.Create_r_tilde_array(self,sigma_para, mode_number, steps)
        self.wigner_PMS_t = Wigner_Functions.Wigner_t_PMS(self, r_tilde_array)
        bar_limits = [-np.max(self.wigner_PMS_t), np.max(self.wigner_PMS_t)]
        fig = Plots_Functions.Animate_Fringes(self.wigner_PMS_t,  x_array, p_array, bar_limits, array_tick, n_frames, title, save)
        return
    
    def Plot_Wigner_Function_GCS_Fringes(self, steps, time_index, mode_number, sigma_para, array_tick, title, save):
        """Plot W_c and W_q side by side at a single time index.

        W_c = sum of diagonal Wigner functions (classical mixture).
        W_q = 2 Re[off-diagonal sum] (quantum interference fringes).

        Parameters match Plot_Wigner_Function_Diag; sigma_para sets the grid extent.
        """
        r_tilde_array, x_array, p_array, _, _ = Wigner_Functions.Create_r_tilde_array(
            self, sigma_para, mode_number, steps)
        W_c = Wigner_Functions.Wigner_Sum_Diagonal(self, time_index, r_tilde_array)
        W_q = Wigner_Functions.Wigner_Off_Diagonal(self, time_index, r_tilde_array)
        Plots_Functions.Plot_Wigner_GCS_Fringes(W_c, W_q, x_array, p_array, array_tick, title, save)

    def Animate_Wigner_Function_GCS_Fringes(self, steps, mode_number, sigma_para, array_tick, n_frames, title, save):
        """Animate W_c and W_q over the full time evolution.

        Precomputes both Wigner time series, then passes them to Animate_GCS_Fringes.
        Color limits are fixed to the global extrema so fringes are visually comparable.
        """
        r_tilde_array, x_array, p_array, _, _ = Wigner_Functions.Create_r_tilde_array(
            self, sigma_para, mode_number, steps)
        self.wigner_c_t = Wigner_Functions.Wigner_t_Sum_Diag(self, r_tilde_array)
        self.wigner_q_t = Wigner_Functions.Wigner_t_Off_Diag(self, r_tilde_array)
        Plots_Functions.Animate_GCS_Fringes(
            self.wigner_c_t, self.wigner_q_t,
            x_array, p_array, array_tick, n_frames, title, save)

    def Plot_Wigner_Function_GCS_Total(self, steps, time_index, mode_number, sigma_para, array_tick, title, save):
        """Plot W_total = W_c + W_q at a single time index.

        Reuses cached wigner_c_t / wigner_q_t if already computed, otherwise
        evaluates both at the requested time index on the fly.
        """
        r_tilde_array, x_array, p_array, _, _ = Wigner_Functions.Create_r_tilde_array(
            self, sigma_para, mode_number, steps)
        W_c = Wigner_Functions.Wigner_Sum_Diagonal(self, time_index, r_tilde_array)
        W_q = Wigner_Functions.Wigner_Off_Diagonal(self, time_index, r_tilde_array)
        Plots_Functions.Plot_Wigner_GCS_Total(W_c + W_q, x_array, p_array, array_tick, title, save)

    def Plot_Wigner_Function_GCS_Weighted(self, steps, time_index, mode_number, sigma_para,
                                          array_tick, W_g, p_c=1.0, p_q=1.0, p_g=1.0,
                                          title=False, save=False):
        """Plot p_c*W_c + p_q*W_q + p_g*W_g at a single time index.

        Sibling of Plot_Wigner_Function_GCS_Total that admits an extra user-supplied
        Wigner component W_g (callable W(x, p) or an array on the same grid), e.g. the
        DV-decay term W_Gamma of Appendix I B.

        Returns
        -------
        fig, im, W_sum
        """
        r_tilde_array, x_array, p_array, _, _ = Wigner_Functions.Create_r_tilde_array(
            self, sigma_para, mode_number, steps)
        W_c = Wigner_Functions.Wigner_Sum_Diagonal(self, time_index, r_tilde_array)
        W_q = Wigner_Functions.Wigner_Off_Diagonal(self, time_index, r_tilde_array)
        return Plots_Functions.Plot_Wigner_GCS_Weighted(
            W_c, W_q, W_g, p_c, p_q, p_g, x_array, p_array, array_tick, title, save)

    def Animate_Wigner_Function_GCS_Total(self, steps, mode_number, sigma_para, array_tick, n_frames, title, save):
        """Animate W_total = W_c + W_q over the full time evolution.

        Precomputes both Wigner time series and sums them before animating.
        Color limits are fixed globally so the scale is stable across frames.
        """
        r_tilde_array, x_array, p_array, _, _ = Wigner_Functions.Create_r_tilde_array(
            self, sigma_para, mode_number, steps)
        W_c_t = Wigner_Functions.Wigner_t_Sum_Diag(self, r_tilde_array)
        W_q_t = Wigner_Functions.Wigner_t_Off_Diag(self, r_tilde_array)
        Plots_Functions.Animate_GCS_Total(
            W_c_t + W_q_t, x_array, p_array, array_tick, n_frames, title, save)

    def Animate_Gaussian(self, steps, mode_number, sigma_para, array_tick, n_frames, title, save):

        r_tilde_array, x_array, p_array, XX, PP = Wigner_Functions.Create_r_tilde_array(self, sigma_para, mode_number, steps)
        if isinstance(self.r_t, np.ndarray):
            self.wigner_gauss_t = Wigner_Functions.Wigner_t_Gauss(self, r_tilde_array)
        else:
            self.wigner_gauss_t = Wigner_Functions.Wigner_t_Sum_Diag(self, r_tilde_array)
        bar_limits = [0, np.max(self.wigner_gauss_t)]
        fig = Plots_Functions.Animate_Diagonal(self.wigner_gauss_t, x_array, p_array, bar_limits, array_tick, n_frames, title, save)
        return
        
    def Animate_Gaussian_Trajectory(self, steps, mode_number, sigma_para, array_tick, n_frames, title, save,
                                    trail=True, trail_length=None, trail_color='navy'):
        """Animate_Gaussian with an optional trailing path of the first moment r_t."""
        r_tilde_array, x_array, p_array, XX, PP = Wigner_Functions.Create_r_tilde_array(self, sigma_para, mode_number, steps)
        self.wigner_gauss_t = Wigner_Functions.Wigner_t_Gauss(self, r_tilde_array)
        bar_limits = [0, np.max(self.wigner_gauss_t)]
        R_t = self.r_t[:, 2*mode_number:2*mode_number + 2, 0]
        return Plots_Functions.Animate_Gaussian_Trajectory(
            self.wigner_gauss_t, x_array, p_array, bar_limits, array_tick, R_t, n_frames, title, save,
            trail=trail, trail_length=trail_length, trail_color=trail_color)

################################## Stochastic Monitoring ##################################

    def Initialize_Stochastic_Sample(self, t_array, R_t, Sigma_t, sample=0, stride=1):
        """
        Load one conditional trajectory as a Gaussian time series, so that the Gaussian
        Wigner/animation methods (Animate_Gaussian, Animate_Gaussian_Trajectory) apply.

        t_array : (n_t,)                  time grid from Stochastic.Sample_CQ_Trajectories
        R_t     : (n_samples, n_t, 2n)    stochastic first moments
        Sigma_t : (n_t, 2n, 2n)           deterministic covariance
        sample  : which trajectory to load
        stride  : keep every stride-th time step (animation frames)
        """
        self.t_array = t_array[::stride]
        self.r_t = R_t[sample, ::stride, :, None]
        self.sigma_t = Sigma_t[::stride]
        self.type = 'Gaussian'
        return
