"""
MC Distributed Concentration
===========

This module contains the MC Distributed Concentration class, for simulating random
occupation crystals each with a concentration drawn from a distribution.
"""
import random
from typing import Mapping, Callable
import numpy as np
import scipy
from B8_project.crystal import UnitCell
from B8_project.form_factor import FormFactorProtocol
from B8_project.mc_random_occupation import MCRandomOccupation


class MCDistributedConcentration(MCRandomOccupation):
    """
    Child class of MCRandomOccupation for simulating random occupation crystals each
    with a concentration drawn from a distribution.
    """
    _conc_pdf: Callable[[np.ndarray], np.ndarray]
    _lattice_parameter_func: Callable[[float], tuple[float, float, float]]
    _conc_inverse_cdf: Callable[[np.ndarray], np.ndarray]

    def __init__(self,
                 wavelength: float,
                 unit_cell: UnitCell,
                 atom_from: int,
                 atom_to: int,
                 probability: float,
                 conc_pdf: Callable[[np.ndarray], np.ndarray],
                 lattice_parameter_func: Callable[[float], tuple[float, float, float]],
                 pdf: Callable[[np.ndarray], np.ndarray] = None,
                 min_angle_deg: float = 0.,
                 max_angle_deg: float = 180.):
        super().__init__(wavelength, unit_cell, atom_from, atom_to, probability,
                         pdf, min_angle_deg, max_angle_deg)
        self._conc_pdf = conc_pdf
        self._lattice_parameter_func = lattice_parameter_func
        x_vals = np.linspace(0., 1., 1000)
        pdf_vals = self._conc_pdf(x_vals)
        cdf_vals = scipy.integrate.cumulative_simpson(pdf_vals, x=x_vals, initial=0.)
        cdf_vals /= cdf_vals[-1]  # Normalize CDF
        try:
            inverse_cdf_func = scipy.interpolate.PchipInterpolator(cdf_vals, x_vals)
        except ValueError as exc:
            raise ValueError("Inverse CDF interpolation failed. Possibly because PDF "
                             "is negative or too close to zero at certain points.") \
                from exc
        self._conc_inverse_cdf = inverse_cdf_func

    def random_conc(self):
        return self._conc_inverse_cdf(random.random())

    def set_lattice_parameter(self, concentration: float):
        a, b, c = self._lattice_parameter_func(concentration)
        self._unit_cell_pos /= np.array(self._unit_cell.lattice_constants)
        self._unit_cell.lattice_constants = (a, b, c)
        self._unit_cell_pos *= np.array(self._unit_cell.lattice_constants).T

    def compute_intensities(self,
                            scattering_vecs: np.ndarray,
                            form_factors: Mapping[int, FormFactorProtocol]):
        if self._unit_cell_pos is None:
            raise ValueError(
                "_unit_cell_pos is None: You must call setup_cuboid_crystal"
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
        conc = self.random_conc()
        self.set_lattice_parameter(conc)
        probs = self._uc_vars.calculate_probabilities(conc)
        random_indices = self._rng.choice(np.arange(n_uc_varieties),
                                          size=(n_vecs, n_unit_cells),
                                          p=probs)
        structure_factors_basis_random = (
            structure_factors_basis)[np.arange(n_vecs)[:, np.newaxis], random_indices]

        # structure_factors_lattice.shape = (# trials filtered,)
        structure_factors = np.sum(
            np.multiply(exp_terms_lattice, structure_factors_basis_random), axis=1)

        intensities = np.abs(structure_factors) ** 2
        return intensities