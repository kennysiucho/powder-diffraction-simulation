"""
Diffraction Monte Carlo
===========

This module contains classes to calculate diffraction spectra using Monte Carlo methods.

Classes
-------
`DiffractionMonteCarloRunStats`
    A class to store statistics associated with each run of
    `calculate_diffraction_pattern`.
`DiffractionMonteCarlo`
    A class to calculate diffraction patterns, with different optimizations
    based on the type of crystal
"""

import time
from dataclasses import dataclass
from typing import Mapping
import numpy as np
from B8_project import utils
from B8_project.crystal import Atom, UnitCell, UnitCellVarieties, ReplacementProbability
from B8_project.form_factor import FormFactorProtocol


@dataclass
class DiffractionMonteCarloRunStats:
    """
    A class to store statistics associated with each run of `calculate_diffraction_
    pattern`.

    Attributes that end with `_` denote attributes that are not returned in `__str__`.

    Attributes
    ----------
    accepted_data_points : int
        Number of accepted trials so far
    total_trials : int
        Total number of trials attempted, regardless of their angles and intensities
    start_time_ : float
        Start time of calculation, in seconds since the Epoch
    prev_print_time_ : float
        Previous time stamp when the run stats are printed; keeps track so that the
        run stats are printed in regular intervals.
    microseconds_per_trial : float
        Microseconds spent to calculate each trial; includes all trials, accepted or
        not.
    """

    accepted_data_points: int = 0
    total_trials: int = 0
    start_time_: float = time.time()
    prev_print_time_: float = 0.0
    microseconds_per_trial: float = 0.0

    def recalculate_microseconds_per_trial(self):
        """
        Recalculate average `microseconds_per_trial`.
        """
        if self.total_trials == 0:
            return
        self.microseconds_per_trial = ((time.time() - self.start_time_) /
                                       self.total_trials * 1000000)

    def __str__(self):
        """
        Returns a formatted string containing all attributes that don't
        end in `_`.
        """
        self.recalculate_microseconds_per_trial()
        return "".join([f"{key}={val:.1f} | " if key[-1] != '_' else ""
                        for key, val in self.__dict__.items()])


class DiffractionMonteCarlo:
    """
    A class to calculate neutron diffraction patterns via Monte Carlo methods.

    Attributes
    ----------
    unit_cell : UnitCell
        The unit cell of the crystal
    wavelength : float
        The wavelength of the incident neutrons (in nm)
    """
    def __init__(self, unit_cell: UnitCell, wavelength: float):
        self.unit_cell = unit_cell
        self.wavelength = wavelength

    def k(self):
        """
        Returns 2pi / wavelength.
        """
        return 2 * np.pi / self.wavelength

    def _unit_cell_positions(self, unit_cell_reps: tuple[int, int, int]):
        """
        Returns a list of positions of the unit cells in the crystal particle.

        Parameters
        ----------
        unit_cell_reps : tuple[int, int, int]
            An input of (a, b, c) specifies the unit cell is repeated a, b, c
            times in the x, y, and z directions respectively.

        Returns
        -------
        unit_cell_pos : np.ndarray
            (a * b * c, 3) array, with each row representing the coordinate of the
            [0, 0, 0] position of each unit cell.
        """
        unit_cell_pos = np.vstack(
            np.mgrid[0:unit_cell_reps[0], 0:unit_cell_reps[1],
            0:unit_cell_reps[2]]).reshape(3, -1).T
        unit_cell_pos = unit_cell_pos.astype(np.float64)
        np.multiply(unit_cell_pos, self.unit_cell.lattice_constants, out=unit_cell_pos)
        return unit_cell_pos

    def _atoms_and_pos_in_uc(self):
        """
        Returns a list of the atomic number and coordinates w.r.t. the unit cell of
        each atom in one unit cell.
        """
        atoms_in_uc = []
        atom_pos_in_uc = []
        for atom in self.unit_cell.atoms:
            atoms_in_uc.append(atom.atomic_number)
            atom_pos_in_uc.append(np.multiply(atom.position,
                                              self.unit_cell.lattice_constants))
        atoms_in_uc = np.array(atoms_in_uc)
        atom_pos_in_uc = np.array(atom_pos_in_uc)
        return atoms_in_uc, atom_pos_in_uc

    def _get_scattering_vecs_and_angles(self,
                                        n: int,
                                        min_angle_deg: float,
                                        max_angle_deg: float):
        """
        Generates random scattering vectors and their angles. Discards those outside
        angle range of interest.

        TODO: Add weighting function

        Parameters
        ----------
        n : int
            Number of random vectors to generate initially. The number of vectors
            returned will be less after filtering.
        min_angle_deg, max_angle_deg : float
            Minimum/maximum angle in degrees.

        Returns
        -------
        scattering_vecs, two_thetas : np.ndarray
            List of scattering vectors and their corresponding scattering angles.
        """
        k_vecs = self.k() * utils.random_uniform_unit_vectors(n, 3)
        k_primes = self.k() * utils.random_uniform_unit_vectors(n, 3)
        scattering_vecs = k_primes - k_vecs

        # Compute scattering angle
        dot_products = np.einsum("ij,ij->i", k_vecs, k_primes)
        two_thetas = np.degrees(np.arccos(dot_products / self.k() ** 2))

        # Discard trials with scattering angle out of range of interest
        angles_accepted = np.where(np.logical_and(two_thetas >= min_angle_deg,
                                                  two_thetas <= max_angle_deg))
        scattering_vecs = scattering_vecs[angles_accepted]
        two_thetas = two_thetas[angles_accepted]

        return scattering_vecs, two_thetas

    # TODO: change this take a list of all atoms
    def calculate_diffraction_pattern(self,
                                      atoms: list[Atom],
                                      form_factors: Mapping[int, FormFactorProtocol],
                                      target_accepted_trials: int = 5000,
                                      trials_per_batch: int = 1000,
                                      min_angle_deg: float = 0.0,
                                      max_angle_deg: float = 180.0,
                                      angle_bins: int = 100):
        """
        Calculates the neutron diffraction spectrum using a Monte Carlo method.

        For each Monte Carlo trial, randomly choose the incident and scattered k-
        vectors. Sum over all atoms to calculate the structure factor and hence
        intensity of this trial. If the scattering angle is within the range
        specified then add this trial to the final result.

        Parameters
        ----------
        atoms : list[Atom]
            List of all atoms in the crystal.
        form_factors : Mapping[int, FormFactorProtocol]
            Dictionary mapping atomic number to associated NeutronFormFactor or
            XRayFormFactor.
        target_accepted_trials : int
            Target number of accepted trials.
        trials_per_batch : int
            Number of trials calculated at once using NumPy methods.
        min_angle_deg, max_angle_deg : float
            Minimum/maximum scattering angle in degrees for a scattering trial to be
            accepted.
        angle_bins : int
            Number of bins for scattering angles.

        Returns
        -------
        two_thetas : (target_accepted_trials,) ndarray
            the left edges of the bins, evenly spaced within angle range specified
        intensities : (target_accepted_trials,) ndarray
            intensity calculated for each bin
        """
        two_thetas = np.linspace(min_angle_deg, max_angle_deg, angle_bins)
        intensities = np.zeros(angle_bins)

        stats = DiffractionMonteCarloRunStats()

        all_atom_pos = np.array([np.array(atom.position) for atom in atoms])
        all_atoms = np.array([atom.atomic_number for atom in atoms])

        while stats.accepted_data_points < target_accepted_trials:

            if time.time() - stats.prev_print_time_ > 5:
                stats.prev_print_time_ = time.time()
                print(stats)

            scattering_vecs, two_thetas_batch = (
                self._get_scattering_vecs_and_angles(trials_per_batch, min_angle_deg,
                                                     max_angle_deg))

            # all_atom_pos.shape = (n_atoms, 3)
            # all_scattering_lengths = (n_atoms,)
            # scattering_vec.shape = (batch_trials, 3)
            # structure_factors.shape = (batch_trials, )
            # k•r[i, j] = scattering_vec[i][k] • all_atom_pos[j][k]

            # dot_products.shape = (# trials after filter, n_atoms)
            dot_products = np.einsum("ik,jk", scattering_vecs, all_atom_pos)

            # Evaluate form factors for each element
            form_factors_evaluated = {}
            for atomic_number, form_factor in form_factors.items():
                form_factors_evaluated[atomic_number] = (
                    form_factor.evaluate_form_factors(scattering_vecs))
            all_form_factors = np.array([form_factors_evaluated[atom] for atom in
                                         all_atoms]).T

            # exp_terms.shape = (# trials, n_atoms)
            exps = np.exp(1j * dot_products)
            exp_terms = np.multiply(all_form_factors, exps)

            # structure_factors.shape = (# trials, )
            structure_factors = np.sum(exp_terms, axis=1)

            intensity_batch = np.abs(structure_factors)**2

            bins = np.searchsorted(two_thetas, two_thetas_batch) - 1
            intensities[bins] += intensity_batch

            stats.total_trials += two_thetas_batch.shape[0]
            stats.accepted_data_points += two_thetas_batch.shape[0]

        intensities /= np.max(intensities)

        return two_thetas, intensities


    def calculate_diffraction_pattern_ideal_crystal(
            self,
            form_factors: Mapping[int, FormFactorProtocol],
            target_accepted_trials: int = 5000,
            trials_per_batch: int = 1000,
            unit_cell_reps: tuple[int, int, int] = (8, 8, 8),
            min_angle_deg: float = 0.0,
            max_angle_deg: float = 180.0,
            angle_bins: int = 100):
        """
        Calculates the neutron diffraction spectrum using a Monte Carlo method,
        assuming the crystal consists of the same unit cell throughout (ideal crystal).

        For each Monte Carlo trial, randomly choose the incident and scattered k-
        vectors. If the scattering angle is within the range specified, compute the
        lattice and basis structure factors and hence intensity of the trial. Add
        intensity to final diffraction pattern.

        Parameters
        ----------
        form_factors : Mapping[int, FormFactorProtocol]
            Dictionary mapping atomic number to associated NeutronFormFactor or
            XRayFormFactor.
        target_accepted_trials : int
            Target number of accepted trials.
        trials_per_batch : int
            Number of trials calculated at once using NumPy methods
        unit_cell_reps : tuple[int, int, int]
            How many times to repeat the unit cell in x, y, z directions, forming the
            crystal powder for diffraction.
        min_angle_deg, max_angle_deg : float
            Minimum/maximum scattering angle in degrees for a scattering trial to be
            accepted
        angle_bins : int
            Number of bins for scattering angles

        Returns
        -------
        two_thetas : (`target_accepted_trials`,) ndarray
            The left edges of the bins, evenly spaced within angle range specified
        intensities : (`target_accepted_trials`,) ndarray
            Intensity calculated for each bin
        """
        two_thetas = np.linspace(min_angle_deg, max_angle_deg, angle_bins)
        intensities = np.zeros(angle_bins)

        unit_cell_pos = self._unit_cell_positions(unit_cell_reps)

        atoms_in_uc, atom_pos_in_uc = self._atoms_and_pos_in_uc()

        stats = DiffractionMonteCarloRunStats()

        while stats.accepted_data_points < target_accepted_trials:

            if time.time() - stats.prev_print_time_ > 5:
                stats.prev_print_time_ = time.time()
                print(stats)

            scattering_vecs, two_thetas_batch = self._get_scattering_vecs_and_angles(
                trials_per_batch, min_angle_deg, max_angle_deg)

            # Compute lattice portion of structure factors
            # scattering_vecs.shape = (# trials filtered, 3)
            # unit_cell_pos.shape = (# unit cells, 3)
            # dot_products_lattice.shape = (# trials filtered, # unit cells)
            dot_products_lattice = np.einsum("ik,jk", scattering_vecs, unit_cell_pos)

            # exp_terms.shape = (# trials filtered, # unit cells)
            exp_terms_lattice = np.exp(1j * dot_products_lattice)

            # structure_factors_lattice.shape = (# trials filtered,)
            structure_factors_lattice = np.sum(exp_terms_lattice, axis=1)

            # Compute basis portion of structure factors
            # scattering_vecs.shape = (# trials filtered, 3)
            # atom_pos_in_uc.shape = (# atoms in a unit cell, 3)
            # dot_products_lattice.shape = (# trials filtered, # atoms in a unit cell)
            dot_products_basis = np.einsum("ik,jk", scattering_vecs, atom_pos_in_uc)

            # form_factors_basis.shape = (# trials filtered, # atoms in a unit cell)
            form_factors_basis = np.stack(
                [form_factors[atom].evaluate_form_factors(scattering_vecs) for atom in
                 atoms_in_uc],
                axis=1)

            # exp_terms.shape = (# trials filtered, # atoms in a unit cell)
            exps = np.exp(1j * dot_products_basis)
            exp_terms_basis = np.multiply(form_factors_basis, exps)

            # structure_factors_basis.shape = (# trials filtered,)
            structure_factors_basis = np.sum(exp_terms_basis, axis=1)

            structure_factors = np.multiply(structure_factors_lattice,
                                            structure_factors_basis)
            intensity_batch = np.abs(structure_factors) ** 2

            bins = np.searchsorted(two_thetas, two_thetas_batch) - 1
            intensities[bins] += intensity_batch

            stats.total_trials += two_thetas_batch.shape[0]
            stats.accepted_data_points += two_thetas_batch.shape[0]

        intensities /= np.max(intensities)

        return two_thetas, intensities

    def calculate_diffraction_pattern_random_occupation(
            self,
            atom_from: int,
            atom_to: int,
            probability: float,
            form_factors: Mapping[int, FormFactorProtocol],
            target_accepted_trials: int = 5000,
            trials_per_batch: int = 1000,
            unit_cell_reps: tuple[int, int, int] = (8, 8, 8),
            min_angle_deg: float = 0.0,
            max_angle_deg: float = 180.0,
            angle_bins: int = 100):
        """
        TODO: update docstring, add tests
        """
        two_thetas = np.linspace(min_angle_deg, max_angle_deg, angle_bins)
        intensities = np.zeros(angle_bins)

        unit_cell_pos = self._unit_cell_positions(unit_cell_reps)

        _, atom_pos_in_uc = self._atoms_and_pos_in_uc()

        uc_vars = UnitCellVarieties(self.unit_cell,
                                    ReplacementProbability(atom_from, atom_to,
                                                           probability))
        atomic_numbers_vars, probs \
            = uc_vars.atomic_number_lists()
        print(probs)

        stats = DiffractionMonteCarloRunStats()

        rng = np.random.default_rng() # A NumPy Random Generator

        while stats.accepted_data_points < target_accepted_trials:

            if time.time() - stats.prev_print_time_ > 5:
                stats.prev_print_time_ = time.time()
                print(stats)

            scattering_vecs, two_thetas_batch = self._get_scattering_vecs_and_angles(
                trials_per_batch, min_angle_deg, max_angle_deg
            )

            # Compute basis portion of structure factors
            # scattering_vecs.shape = (# trials filtered, 3)
            # atom_pos_in_uc.shape = (# atoms in a unit cell, 3)
            # dot_products_basis.shape = (# trials filtered, # atoms in a unit cell)
            dot_products_basis = np.einsum("ik,jk", scattering_vecs, atom_pos_in_uc)

            # form_factors_vars.shape = (# trials, varieties, # atoms in unit cell)
            # atomic_numbers_vars.shape = (varieties, # atoms in unit cell)
            form_factors_evaluated = {}
            for atomic_number, form_factor in form_factors.items():
                form_factors_evaluated[atomic_number] = (
                    form_factor.evaluate_form_factors(scattering_vecs))
            form_factors_vars = np.array([[form_factors_evaluated[atom] for atom in
                                           uc] for uc in atomic_numbers_vars])
            form_factors_vars = np.transpose(form_factors_vars, axes=(2,0,1))

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
            dot_products_lattice = np.einsum("ik,jk", scattering_vecs, unit_cell_pos)

            # exp_terms_lattice.shape = (# trials filtered, # unit cells)
            exp_terms_lattice = np.exp(1j * dot_products_lattice)

            # Each term of the lattice structure factor is multiplied with the basis
            # structure factor of one of the unit cells

            # structure_factors_basis.shape = (# trials filtered, varieties)
            # structure_factors_basis_random.shape = (# trials filtered, # unit cells)
            n_unit_cells = unit_cell_pos.shape[0]
            n_uc_varieties = len(atomic_numbers_vars)
            random_indices = rng.choice(np.arange(n_uc_varieties), size=n_unit_cells,
                                        p=probs)
            structure_factors_basis_random = structure_factors_basis[:, random_indices]

            # structure_factors_lattice.shape = (# trials filtered,)
            structure_factors = np.sum(
                np.multiply(exp_terms_lattice, structure_factors_basis_random), axis=1)

            # TODO: test to Sanity check to ensure concentration in alloy is as expected

            intensity_batch = np.abs(structure_factors) ** 2

            bins = np.searchsorted(two_thetas, two_thetas_batch) - 1
            intensities[bins] += intensity_batch

            stats.total_trials += two_thetas_batch.shape[0]
            stats.accepted_data_points += two_thetas_batch.shape[0]

        intensities /= np.max(intensities)

        return two_thetas, intensities
