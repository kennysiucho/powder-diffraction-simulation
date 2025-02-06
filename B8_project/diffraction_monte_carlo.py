"""
Diffraction Monte Carlo
===========

This module contains classes to calculate diffraction spectra using Monte Carlo methods.

Classes
-------
    - `NeutronDiffractionMonteCarloRunStats`: A class to store statistics associated
    with each run of `calculate_diffraction_pattern`.
    - `NeutronDiffractionMonteCarlo`: A class to calculate neutron diffraction
    patterns, with different optimizations based on the type of crystal
"""

import time
from dataclasses import dataclass
import numpy as np
from B8_project.file_reading import read_neutron_scattering_lengths
from B8_project import utils
from B8_project.crystal import UnitCell

@dataclass
class NeutronDiffractionMonteCarloRunStats:
    """
    Neutron Diffraction Monte Carlo Run Stats
    =========================================

    A class to store statistics associated with each run of `calculate_diffraction_
    pattern`.

    Attributes that end with `_` denote attributes that are not returned in `__str__`.

    Attributes
    ----------
        - `accepted_data_points` (`int`): Number of accepted trials so far
        - `total_trials` (`int`): Total number of trials attempted, regardless of their
        angles and intensities
        - `start_time_` (`float`): Start time of calculation, in seconds since the Epoch
        - `prev_print_time_` (`float`): Previous time stamp when the run stats are
        printed; keeps track so that the run stats are printed in regular intervals.
        - `microseconds_per_trial` (`float`): Microseconds spent to calculate each
        trial; includes all trials, accepted or not.

    Methods
    -------
        - `recalculate_microseconds_per_trial`: Recalculate average `microseconds_per
        _trial`.
        - `__str__`: Returns a formatted string containing all attributes that don't
        end in `_`.
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
        self.recalculate_microseconds_per_trial()
        return "".join([f"{key}={val:.1f} | " if key[-1] != '_' else ""
                        for key, val in self.__dict__.items()])


class NeutronDiffractionMonteCarlo:
    """
    Neutron Diffraction Monte Carlo
    ===============================

    A class to calculate neutron diffraction patterns via Monte Carlo methods.

    Attributes
    ----------
        - `unit_cell` (`UnitCell`): The unit cell of the crystal
        - `wavelength` (`float`): The wavelength of the incident neutrons (in nm)

    Methods
    -------
        - `calculate_diffraction_pattern`: Calculate spectrum given list of atoms in
        a crystal, without further assumptions about the structure.
        - `calculate_diffraction_pattern_ideal_crystal`: Calculate spectrum of a
        crystal consisting of perfectly repeating unit cells.
    """
    def __init__(self, unit_cell: UnitCell, wavelength: float):
        self.unit_cell = unit_cell
        self.wavelength = wavelength

    # TODO: change this take a list of all atoms
    def calculate_diffraction_pattern(self,
                                      target_accepted_trials: int = 5000,
                                      trials_per_batch: int = 1000,
                                      unit_cells_in_crystal: tuple[int, int, int] = (
                                      8, 8, 8),
                                      min_angle_deg: float = 0.0,
                                      max_angle_deg: float = 180.0,
                                      angle_bins: int = 100):
        """
        Calculate diffraction pattern
        =============================

        Calculates the neutron diffraction spectrum using a Monte Carlo method.

        For each Monte Carlo trial, randomly choose the incident and scattered k-
        vectors. Sum over all atoms to calculate the structure factor and hence
        intensity of this trial. If the scattering angle is within the range
        specified then add this trial to the final result.

        Parameters
        ----------
            - `target_accepted_trials` (`int`): Target number of accepted trials.
            - `trials_per_batch` (`int`): Number of trials calculated at once using
            NumPy methods
            - `unit_cells_in_crystal` (`tuple[int, int, int]`): How many times to
            repeat the unit cell in x, y, z directions, forming the crystal powder
            for diffraction.
            - `min_angle_rad`, `max_angle_rad` (`float`): Minimum/maximum scattering
            angle in radians for a scattering trial to be accepted
            - `angle_bins` (`int`): Number of bins for scattering angles

        Returns
        -------
            - `two_thetas` ((`target_accepted_trials`,) ndarray): representing the
            left edges of the bins, evenly spaced within angle range specified
            - `intensities` ((`target_accepted_trials`,) ndarray): intensity
            calculated for each bin
        """
        k = 2 * np.pi / self.wavelength
        two_thetas = np.linspace(min_angle_deg, max_angle_deg, angle_bins)
        intensities = np.zeros(angle_bins)

        # read relevant neutron scattering lengths
        all_scattering_lengths = read_neutron_scattering_lengths(
            "data/neutron_scattering_lengths.csv")
        scattering_lengths = {}
        for atom in self.unit_cell.atoms:
            scattering_lengths[atom.atomic_number] = all_scattering_lengths[
                atom.atomic_number].neutron_scattering_length

        unit_cell_pos = np.vstack(
            np.mgrid[0:unit_cells_in_crystal[0], 0:unit_cells_in_crystal[1],
            0:unit_cells_in_crystal[2]]).reshape(3, -1).T
        unit_cell_pos = unit_cell_pos.astype(np.float64)
        np.multiply(unit_cell_pos, self.unit_cell.lattice_constants, out=unit_cell_pos)

        atom_pos_in_uc = []
        scattering_lengths_in_uc = []
        for atom in self.unit_cell.atoms:
            atom_pos_in_uc.append(np.multiply(atom.position,
                                              self.unit_cell.lattice_constants))
            scattering_lengths_in_uc.append(scattering_lengths[atom.atomic_number])
        atom_pos_in_uc = np.array(atom_pos_in_uc)
        scattering_lengths_in_uc = np.array(scattering_lengths_in_uc)

        # Compute list of positions and scattering lengths of all atoms in the crystal
        n_unit_cells = unit_cell_pos.shape[0]
        n_atoms_per_uc = atom_pos_in_uc.shape[0]
        all_atom_pos = np.repeat(unit_cell_pos, n_atoms_per_uc, axis=0) + np.tile(
            atom_pos_in_uc, (n_unit_cells, 1))
        all_scattering_lengths = np.tile(scattering_lengths_in_uc, n_unit_cells)

        stats = NeutronDiffractionMonteCarloRunStats()

        while stats.accepted_data_points < target_accepted_trials:

            if time.time() - stats.prev_print_time_ > 5:
                stats.prev_print_time_ = time.time()
                print(stats)

            k_vecs = k * utils.random_uniform_unit_vectors(trials_per_batch, 3)
            k_primes = k * utils.random_uniform_unit_vectors(trials_per_batch, 3)
            scattering_vecs = k_primes - k_vecs

            # all_atom_pos.shape = (n_atoms, 3)
            # all_scattering_lengths = (n_atoms,)
            # scattering_vec.shape = (batch_trials, 3)
            # structure_factors.shape = (batch_trials, )
            # k•r[i, j] = scattering_vec[i][k] • all_atom_pos[j][k]

            # dot_products.shape = (batch_trials, n_atoms)
            dot_products = np.einsum("ik,jk", scattering_vecs, all_atom_pos)

            # exp_terms.shape = (batch_trials, n_atoms)
            exps = np.exp(1j * dot_products)
            exp_terms = all_scattering_lengths * exps

            # structure_factors.shape = (batch_trials, )
            structure_factors = np.sum(exp_terms, axis=1)

            dot_products = np.einsum("ij,ij->i", k_vecs, k_primes)
            two_theta_batch = np.degrees(np.arccos(dot_products / k**2))
            intensity_batch = np.abs(structure_factors)**2

            stats.total_trials += trials_per_batch

            angles_accepted = np.where(np.logical_and(two_theta_batch > min_angle_deg,
                                                      two_theta_batch < max_angle_deg))
            two_theta_batch = two_theta_batch[angles_accepted]
            intensity_batch = intensity_batch[angles_accepted]

            bins = np.searchsorted(two_thetas, two_theta_batch)
            intensities[bins] += intensity_batch

            stats.accepted_data_points += two_theta_batch.shape[0]

        intensities /= np.max(intensities)

        return two_thetas, intensities


    def calculate_diffraction_pattern_ideal_crystal(
            self,
            target_accepted_trials: int = 5000,
            trials_per_batch: int = 1000,
            unit_cells_in_crystal: tuple[int, int, int] = (8, 8, 8),
            min_angle_deg: float = 0.0,
            max_angle_deg: float = 180.0,
            angle_bins: int = 100):
        """
        Calculate diffraction pattern ideal crystal
        =============================

        Calculates the neutron diffraction spectrum using a Monte Carlo method,
        assuming the crystal consists of the same unit cell throughout (ideal crystal).

        For each Monte Carlo trial, randomly choose the incident and scattered k-
        vectors. If the scattering angle is within the range specified, compute the
        lattice and basis structure factors and hence intensity of the trial. Add
        intensity to final diffraction pattern.

        Parameters
        ----------
            - `target_accepted_trials` (`int`): Target number of accepted trials.
            - `trials_per_batch` (`int`): Number of trials calculated at once using
            NumPy methods
            - `unit_cells_in_crystal` (`tuple[int, int, int]`): How many times to
            repeat the unit cell in x, y, z directions, forming the crystal powder
            for diffraction.
            - `min_angle_rad`, `max_angle_rad` (`float`): Minimum/maximum scattering
            angle in radians for a scattering trial to be accepted
            - `angle_bins` (`int`): Number of bins for scattering angles

        Returns
        -------
            - `two_thetas` ((`target_accepted_trials`,) ndarray): representing the
            left edges of the bins, evenly spaced within angle range specified
            - `intensities` ((`target_accepted_trials`,) ndarray): intensity
            calculated for each bin
        """
        k = 2 * np.pi / self.wavelength
        two_thetas = np.linspace(min_angle_deg, max_angle_deg, angle_bins)
        intensities = np.zeros(angle_bins)

        # TODO: get scattering lengths as parameter instead
        # read relevant neutron scattering lengths
        all_scattering_lengths = read_neutron_scattering_lengths(
            "data/neutron_scattering_lengths.csv")
        scattering_lengths = {}
        for atom in self.unit_cell.atoms:
            scattering_lengths[atom.atomic_number] = all_scattering_lengths[
                atom.atomic_number].neutron_scattering_length

        # Compute positions of the unit cells in the crystal
        unit_cell_pos = np.vstack(
            np.mgrid[0:unit_cells_in_crystal[0], 0:unit_cells_in_crystal[1],
            0:unit_cells_in_crystal[2]]).reshape(3, -1).T
        unit_cell_pos = unit_cell_pos.astype(np.float64)
        np.multiply(unit_cell_pos, self.unit_cell.lattice_constants, out=unit_cell_pos)

        # Prepare the positions and scattering lengths for each atom in a single unit
        # cell
        atom_pos_in_uc = []
        scattering_lengths_in_uc = []
        for atom in self.unit_cell.atoms:
            atom_pos_in_uc.append(np.multiply(atom.position,
                                              self.unit_cell.lattice_constants))
            scattering_lengths_in_uc.append(scattering_lengths[atom.atomic_number])
        atom_pos_in_uc = np.array(atom_pos_in_uc)
        scattering_lengths_in_uc = np.array(scattering_lengths_in_uc)

        stats = NeutronDiffractionMonteCarloRunStats()

        while stats.accepted_data_points < target_accepted_trials:

            if time.time() - stats.prev_print_time_ > 5:
                stats.prev_print_time_ = time.time()
                print(stats)

            k_vecs = k * utils.random_uniform_unit_vectors(trials_per_batch, 3)
            k_primes = k * utils.random_uniform_unit_vectors(trials_per_batch, 3)
            scattering_vecs = k_primes - k_vecs

            # Compute scattering angle
            dot_products = np.einsum("ij,ij->i", k_vecs, k_primes)
            two_theta_batch = np.degrees(np.arccos(dot_products / k ** 2))

            # Discard trials with scattering angle out of range of interest
            angles_accepted = np.where(np.logical_and(two_theta_batch > min_angle_deg,
                                                      two_theta_batch < max_angle_deg))
            two_theta_batch = two_theta_batch[angles_accepted]
            scattering_vecs = scattering_vecs[angles_accepted]

            # Compute lattice portion of structure factors
            # scattering_vecs.shape = (batch_trials, 3)
            # unit_cell_pos.shape = (# unit cells, 3)
            # dot_products_lattice.shape = (batch_trials, # unit cells)
            dot_products_lattice = np.einsum("ik,jk", scattering_vecs, unit_cell_pos)

            # exp_terms.shape = (batch_trials, # unit cells)
            exp_terms_lattice = np.exp(1j * dot_products_lattice)

            # structure_factors_lattice.shape = (batch_trials,)
            structure_factors_lattice = np.sum(exp_terms_lattice, axis=1)

            # Compute basis portion of structure factors
            # scattering_vecs.shape = (batch_trials, 3)
            # atom_pos_in_uc.shape = (# atoms in a unit cell, 3)
            # dot_products_lattice.shape = (batch_trials, # atoms in a unit cell)
            dot_products_basis = np.einsum("ik,jk", scattering_vecs, atom_pos_in_uc)

            # exp_terms.shape = (batch_trials, # atoms in a unit cell)
            exps = np.exp(1j * dot_products_basis)
            exp_terms_basis = scattering_lengths_in_uc * exps

            # structure_factors_basis.shape = (batch_trials,)
            structure_factors_basis = np.sum(exp_terms_basis, axis=1)

            structure_factors = np.multiply(structure_factors_lattice,
                                            structure_factors_basis)
            intensity_batch = np.abs(structure_factors) ** 2

            bins = np.searchsorted(two_thetas, two_theta_batch)
            intensities[bins] += intensity_batch

            stats.total_trials += two_theta_batch.shape[0]
            stats.accepted_data_points += two_theta_batch.shape[0]

        intensities /= np.max(intensities)

        return two_thetas, intensities
