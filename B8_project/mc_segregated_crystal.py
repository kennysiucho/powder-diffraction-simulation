"""
MC Segregated Crystal
===========

This module contains the MCSegregatedCrystal, for simulating random occupation crystals
with local concentrations that varies depending on its position.
"""

from typing import Mapping, Callable
import numpy as np

from B8_project.crystal import UnitCell
from B8_project.form_factor import FormFactorProtocol
from B8_project.mc_random_occupation import MCRandomOccupation


class MCSegregatedCrystal(MCRandomOccupation):
    """
    Child class of MCRandomOccupation for simulating random occupation crystals
    with local concentrations that varies depending on its position.
    """
    _conc_func: Callable[[np.ndarray, float, float], np.ndarray]
    _uc_var_indices: np.ndarray = None

    def __init__(self,
                 wavelength: float,
                 unit_cell: UnitCell,
                 atom_from: int,
                 atom_to: int,
                 probability: float,
                 conc_func: Callable[[np.ndarray, float, float], np.ndarray],
                 pdf: Callable[[np.ndarray], np.ndarray] = None,
                 min_angle_deg: float = 0.,
                 max_angle_deg: float = 180.):
        super().__init__(wavelength, unit_cell, atom_from, atom_to, probability,
                         pdf, min_angle_deg, max_angle_deg)
        self._conc_func = conc_func

    def generate_crystal(self):
        if self._unit_cell_pos is None:
            raise ValueError("_unit_cell_pos is None: You must call setup_cuboid_crystal"
                             " or setup_spherical_crystal to define the shape of the "
                             "crystal particle.")
        x_coords = self._unit_cell_pos[:, 0]
        concentrations = self._conc_func(x_coords, np.min(x_coords), np.max(x_coords))
        self._uc_var_indices = np.empty(len(self._unit_cell_pos), dtype=int)
        n_uc_varieties = len(self._atomic_numbers_vars)
        for i in range(len(self._unit_cell_pos)):
            probs = self._uc_vars.calculate_probabilities(concentrations[i])
            ind = self._rng.choice(np.arange(n_uc_varieties),
                                          size=1,
                                          p=probs)
            self._uc_var_indices[i] = ind
        print(self._uc_var_indices)

    def compute_intensities(self,
                            scattering_vecs: np.ndarray,
                            form_factors: Mapping[int, FormFactorProtocol]):
        if self._unit_cell_pos is None:
            raise ValueError("_unit_cell_pos is None: You must call setup_cuboid_crystal"
                             " or setup_spherical_crystal to define the shape of the "
                             "crystal particle.")
        if self._uc_var_indices is None:
            raise ValueError("_uc_var_indices is None: You must call generate_crystal.")

        # Compute basis portion of structure factors
        # scattering_vecs.shape = (# trials filtered, 3)
        # atom_pos_in_uc.shape = (# atoms in a unit cell, 3)
        # dot_products_basis.shape = (# trials filtered, # atoms in a unit cell)
        dot_products_basis = np.einsum("ik,jk", scattering_vecs,
                                       self._atom_pos_in_uc)

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
        dot_products_lattice = np.einsum("ik,jk", scattering_vecs,
                                         self._unit_cell_pos)

        # exp_terms_lattice.shape = (# trials filtered, # unit cells)
        exp_terms_lattice = np.exp(1j * dot_products_lattice)

        # Each term of the lattice structure factor is multiplied with the basis
        # structure factor of one of the unit cells

        # structure_factors_basis.shape = (# trials filtered, varieties)
        # structure_factors_basis_random.shape = (# trials filtered, # unit cells)
        n_vecs = len(scattering_vecs)
        structure_factors_basis_random = (
            structure_factors_basis)[
            np.arange(n_vecs)[:, np.newaxis], self._uc_var_indices]

        # structure_factors_lattice.shape = (# trials filtered,)
        structure_factors = np.sum(
            np.multiply(exp_terms_lattice, structure_factors_basis_random), axis=1)

        intensities = np.abs(structure_factors) ** 2
        return intensities