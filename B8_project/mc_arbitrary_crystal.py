"""
MC Arbitrary Crystal
===========

This module contains the MCArbitraryCrystal class, a child class of
DiffractionMonteCarlo, for calculating the diffraction spectra of arbitrary crystals
using Monte Carlo methods.
"""

from typing import Mapping, Callable
import numpy as np
from B8_project.crystal import Atom
from B8_project.diffraction_monte_carlo import DiffractionMonteCarlo
from B8_project.form_factor import FormFactorProtocol

class MCArbitraryCrystal(DiffractionMonteCarlo):
    """
    Child class of DiffractionMonteCarlo for calculating the diffraction spectra of
    arbitrary crystals using Monte Carlo Methods.
    """

    _all_atom_pos: np.ndarray
    _all_atoms: np.ndarray

    def __init__(self,
                 wavelength: float,
                 atoms: list[Atom],
                 pdf: Callable[[np.ndarray], np.ndarray] = None,
                 min_angle_deg: float = 0.,
                 max_angle_deg: float = 180.):
        super().__init__(wavelength, pdf, min_angle_deg, max_angle_deg)
        self.set_atoms(atoms)

    def set_atoms(self, atoms: list[Atom]):
        self._all_atom_pos = np.array([np.array(atom.position) for atom in atoms])
        self._all_atoms = np.array([atom.atomic_number for atom in atoms])

    def set_atoms_pos(self, atoms_pos: np.ndarray):
        self._all_atom_pos = atoms_pos

    def set_atomic_nums(self, atomic_nums: np.ndarray):
        self._all_atoms = atomic_nums

    def compute_intensities(self,
                            scattering_vecs: np.ndarray,
                            form_factors: Mapping[int, FormFactorProtocol]):
        """
        Computes the intensities for each scattering vector.

        Parameters
        ----------
        scattering_vecs : np.ndarray
            List of scattering vectors for which to evaluate the intensity.
        form_factors : Mapping[int, FormFactorProtocol]
            Dictionary mapping atomic number to associated NeutronFormFactor or
            XRayFormFactor.
        """
        # all_atom_pos.shape = (n_atoms, 3)
        # all_scattering_lengths = (n_atoms,)
        # scattering_vec.shape = (batch_trials, 3)
        # structure_factors.shape = (batch_trials, )
        # k•r[i, j] = scattering_vec[i][k] • all_atom_pos[j][k]

        # dot_products.shape = (# trials after filter, n_atoms)
        dot_products = np.einsum("ik,jk", scattering_vecs, self._all_atom_pos)

        # Evaluate form factors for each element
        form_factors_evaluated = {}
        for atomic_number, form_factor in form_factors.items():
            form_factors_evaluated[atomic_number] = (
                form_factor.evaluate_form_factors(scattering_vecs))
        all_form_factors = np.array([form_factors_evaluated[atom] for atom in
                                     self._all_atoms]).T

        # exp_terms.shape = (# trials, n_atoms)
        exps = np.exp(1j * dot_products)
        exp_terms = np.multiply(all_form_factors, exps)

        # structure_factors.shape = (# trials, )
        structure_factors = np.sum(exp_terms, axis=1)

        intensities = np.abs(structure_factors) ** 2
        return intensities
