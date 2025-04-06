"""
MC Ideal Crystal
===========

This module contains the MCIdealCrystal class, a child class of DiffractionMonteCarlo,
for calculating the diffraction spectra of ideal crystals using Monte Carlo methods.
"""

from typing import Mapping, Callable
import numpy as np
from B8_project.crystal import UnitCell
from B8_project.diffraction_monte_carlo import DiffractionMonteCarlo
from B8_project.form_factor import FormFactorProtocol

class MCIdealCrystal(DiffractionMonteCarlo):
    """
    Child class of DiffractionMonteCarlo for calculating the diffraction spectra of
    ideal crystals using Monte Carlo Methods.

    Attributes
    ----------

    """
    _unit_cell_reps: tuple[int, int, int]
    _unit_cell_pos: np.ndarray
    _atom_pos_in_uc: np.ndarray
    _atoms_in_uc: np.ndarray

    def __init__(self,
                 wavelength: float,
                 unit_cell: UnitCell,
                 unit_cell_reps: tuple[int, int, int],
                 pdf: Callable[[np.ndarray], np.ndarray] = None,
                 min_angle_deg: float = 0.,
                 max_angle_deg: float = 180.):
        super().__init__(wavelength, pdf, min_angle_deg, max_angle_deg, unit_cell)
        self.set_unit_cell(unit_cell)
        self.set_unit_cell_reps(unit_cell_reps)

    def set_unit_cell(self, unit_cell: UnitCell):
        self._unit_cell = unit_cell
        atoms_in_uc, atom_pos_in_uc = self._atoms_and_pos_in_uc()
        self._atom_pos_in_uc = atom_pos_in_uc
        self._atoms_in_uc = atoms_in_uc

    def set_unit_cell_reps(self, unit_cell_reps: tuple[int, int, int]):
        self._unit_cell_reps = unit_cell_reps
        self._unit_cell_pos = self._unit_cell_positions(self._unit_cell_reps)

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
        # scattering_vecs.shape = (# trials filtered, 3)
        # unit_cell_pos.shape = (# unit cells, 3)
        # dot_products_lattice.shape = (# trials filtered, # unit cells)
        dot_products_lattice = np.einsum("ik,jk", scattering_vecs, self._unit_cell_pos)

        # exp_terms.shape = (# trials filtered, # unit cells)
        exp_terms_lattice = np.exp(1j * dot_products_lattice)

        # structure_factors_lattice.shape = (# trials filtered,)
        structure_factors_lattice = np.sum(exp_terms_lattice, axis=1)

        # Compute basis portion of structure factors
        # scattering_vecs.shape = (# trials filtered, 3)
        # atom_pos_in_uc.shape = (# atoms in a unit cell, 3)
        # dot_products_lattice.shape = (# trials filtered, # atoms in a unit cell)
        dot_products_basis = np.einsum("ik,jk", scattering_vecs, self._atom_pos_in_uc)

        # form_factors_basis.shape = (# trials filtered, # atoms in a unit cell)
        form_factors_basis = np.stack(
            [form_factors[atom].evaluate_form_factors(scattering_vecs) for atom in
             self._atoms_in_uc],
            axis=1)

        # exp_terms.shape = (# trials filtered, # atoms in a unit cell)
        exps = np.exp(1j * dot_products_basis)
        exp_terms_basis = np.multiply(form_factors_basis, exps)

        # structure_factors_basis.shape = (# trials filtered,)
        structure_factors_basis = np.sum(exp_terms_basis, axis=1)

        structure_factors = np.multiply(structure_factors_lattice,
                                        structure_factors_basis)
        intensities = np.abs(structure_factors) ** 2
        return intensities


