"""
MC Random Occupation
===========

This module contains the MCRandomOccupation class, a child class of
DiffractionMonteCarlo, for calculating the diffraction spectra of random occupation
crystals using Monte Carlo methods.
"""

from typing import Mapping, Callable
import numpy as np
from B8_project.crystal import UnitCell, UnitCellVarieties, ReplacementProbability
from B8_project.diffraction_monte_carlo import DiffractionMonteCarlo
from B8_project.form_factor import FormFactorProtocol

class MCRandomOccupation(DiffractionMonteCarlo):
    """
    Child class of DiffractionMonteCarlo for calculating the diffraction spectra of
    random occupation crystals using Monte Carlo Methods.
    """

    _atom_from: int
    _atom_to: int
    _probability: float
    _unit_cell_pos: np.ndarray
    _atom_pos_in_uc: np.ndarray
    _atomic_numbers_vars: list[np.ndarray]
    _probs: list[float]
    _rng: np.random.Generator

    def __init__(self,
                 wavelength: float,
                 unit_cell: UnitCell,
                 atom_from: int,
                 atom_to: int,
                 probability: float,
                 pdf: Callable[[np.ndarray], np.ndarray] = None,
                 min_angle_deg: float = 0.,
                 max_angle_deg: float = 180.):
        super().__init__(wavelength, pdf, min_angle_deg, max_angle_deg, unit_cell)
        self._rng = np.random.default_rng()
        self.set_unit_cell(unit_cell)
        self.set_random_occupation_parameters(atom_from, atom_to, probability)

    def set_unit_cell(self, unit_cell: UnitCell):
        self._unit_cell = unit_cell
        atoms_in_uc, atom_pos_in_uc = self._atoms_and_pos_in_uc()
        self._atom_pos_in_uc = atom_pos_in_uc

    def set_random_occupation_parameters(self,
                                         atom_from: int,
                                         atom_to: int,
                                         probability: float):
        self._atom_from = atom_from
        self._atom_to = atom_to
        self._probability = probability
        uc_vars = UnitCellVarieties(self._unit_cell,
                                    ReplacementProbability(self._atom_from,
                                                           self._atom_to,
                                                           self._probability))
        atomic_numbers_vars, probs = uc_vars.atomic_number_lists()
        self._atomic_numbers_vars = atomic_numbers_vars
        self._probs = probs

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
        if self._unit_cell_pos is None:
            raise ValueError("_unit_cell_pos is None: You must call setup_cuboid_crystal"
                             " or setup_spherical_crystal to define the shape of the "
                             "crystal particle.")

        # Compute basis portion of structure factors
        # scattering_vecs.shape = (# trials filtered, 3)
        # atom_pos_in_uc.shape = (# atoms in a unit cell, 3)
        # dot_products_basis.shape = (# trials filtered, # atoms in a unit cell)
        dot_products_basis = np.einsum("ik,jk", scattering_vecs, self._atom_pos_in_uc)

        # form_factors_vars.shape = (# trials, varieties, # atoms in unit cell)
        # atomic_numbers_vars.shape = (varieties, # atoms in unit cell)
        form_factors_evaluated = {}
        for atomic_number, form_factor in form_factors.items():
            form_factors_evaluated[atomic_number] = (
                form_factor.evaluate_form_factors(scattering_vecs))
        form_factors_vars = np.array([[form_factors_evaluated[atom] for atom in
                                       uc] for uc in self._atomic_numbers_vars])
        form_factors_vars = np.transpose(form_factors_vars, axes=(2, 0, 1))

        # exp_terms.shape = (# trials filtered, varieties, # atoms in a unit cell)
        exps = np.exp(1j * dot_products_basis)
        exp_terms_basis = np.einsum("ik,ijk->ijk", exps,
                                    form_factors_vars)

        # structure_factors_basis.shape = (# trials filtered, varieties)
        structure_factors_basis = np.sum(exp_terms_basis, axis=2)

        # Compute lattice portion of structure factors
        # scattering_vecs.shape = (# trials filtered, 3)
        # unit_cell_pos.shape = (# unit cells, 3)
        # dot_products_lattice.shape = (# trials filtered, # unit cells)
        dot_products_lattice = np.einsum("ik,jk", scattering_vecs, self._unit_cell_pos)

        # exp_terms_lattice.shape = (# trials filtered, # unit cells)
        exp_terms_lattice = np.exp(1j * dot_products_lattice)

        # Each term of the lattice structure factor is multiplied with the basis
        # structure factor of one of the unit cells

        # structure_factors_basis.shape = (# trials filtered, varieties)
        # structure_factors_basis_random.shape = (# trials filtered, # unit cells)
        n_unit_cells = self._unit_cell_pos.shape[0]
        n_uc_varieties = len(self._atomic_numbers_vars)
        n_vecs = len(scattering_vecs)
        random_indices = self._rng.choice(np.arange(n_uc_varieties),
                                          size=(n_vecs, n_unit_cells),
                                          p=self._probs)
        structure_factors_basis_random = (
            structure_factors_basis)[np.arange(n_vecs)[:, np.newaxis], random_indices]

        # structure_factors_lattice.shape = (# trials filtered,)
        structure_factors = np.sum(
            np.multiply(exp_terms_lattice, structure_factors_basis_random), axis=1)

        intensities = np.abs(structure_factors) ** 2
        return intensities