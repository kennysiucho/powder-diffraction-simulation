"""
MC Displacement
===========

This module contains the MCDisplacement class, a child class of MCArbitrary and
MCRandomOccupation. Adds capability to displace atoms by user-defined function.
"""
from typing import Mapping, Callable
import numpy as np
from B8_project.crystal import UnitCell
from B8_project.mc_arbitrary_crystal import MCArbitraryCrystal
from B8_project.mc_random_occupation import MCRandomOccupation
from B8_project.form_factor import FormFactorProtocol


class MCDisplacement(MCArbitraryCrystal, MCRandomOccupation):
    """
    Child class of MCArbitrary and MCRandomOccupation. Adds capability to displace atoms
    by user-defined function: displace_func(positions, atomic_nums) -> modified_positions.
    """

    def __init__(self,
                 wavelength: float,
                 unit_cell: UnitCell,
                 atom_from: int,
                 atom_to: int,
                 probability: float,
                 displace_func: Callable[[np.ndarray, float], np.ndarray] = None,
                 pdf: Callable[[np.ndarray], np.ndarray] = None,
                 min_angle_deg: float = 0.,
                 max_angle_deg: float = 180.):
        MCRandomOccupation.__init__(self, wavelength, unit_cell,
                                    atom_from, atom_to, probability,
                                    pdf, min_angle_deg, max_angle_deg)
        self._displace_func = displace_func or (lambda pos, uc: self.gaussian_displaced(
            pos, uc, sigma=0.05, atoms_to_displace=[atom_from, atom_to]
        ))

    @staticmethod
    def gaussian_displaced(positions: np.ndarray,
                          atoms_in_uc: np.ndarray,
                          sigma: float,
                          atoms_to_displace: list[int]) -> np.ndarray:
        displacement = np.zeros_like(positions)
        for i in range(len(atoms_in_uc)):
            if atoms_in_uc[i] in atoms_to_displace:
                displacement[i] = np.random.normal(scale=sigma, size=3)
        return positions + displacement

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
        # Build crystal once per batch
        n_unit_cells = self._unit_cell_pos.shape[0]
        n_atoms_per_uc = len(self._unit_cell.atoms)
        n_uc_varieties = len(self._atomic_numbers_vars)
        n_atoms = n_unit_cells * n_atoms_per_uc

        atom_pos = np.empty((n_atoms, 3), dtype=np.float64)
        atomic_nums = np.empty(n_atoms, dtype=int)
        random_indices = self._rng.choice(np.arange(n_uc_varieties),
                                          size=n_unit_cells,
                                          p=self._probs)
        for i in range(n_unit_cells):
            uc_pos = self._unit_cell_pos[i]
            uc_var = self._atomic_numbers_vars[random_indices[i]]
            atom_pos[i * n_atoms_per_uc:(i + 1) * n_atoms_per_uc] = \
                uc_pos + self._displace_func(self._atom_pos_in_uc, uc_var)
            atomic_nums[i * n_atoms_per_uc:(i + 1) * n_atoms_per_uc] = uc_var
        self.set_atoms_pos(atom_pos)
        self.set_atomic_nums(atomic_nums)

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

