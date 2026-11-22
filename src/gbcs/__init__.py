"""Gaussian-branched cat states: N qubits coupled to n bosonic modes (arXiv:2510.01156)."""

from .Objects import Decoherence, Hamiltonian, Quantum_State, Symplectic
from .Perturbation_Theory import Perturbation

__version__ = "0.9.0"

__all__ = ["Decoherence", "Hamiltonian", "Perturbation", "Quantum_State", "Symplectic"]
