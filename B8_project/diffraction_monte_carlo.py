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
import heapq
import time
from dataclasses import dataclass
from typing import Mapping, Callable
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from B8_project import utils
from B8_project.crystal import Atom, UnitCell, UnitCellVarieties, ReplacementProbability
from B8_project.form_factor import FormFactorProtocol

class WeightingFunction:
    """
    A class containing predefined weighting functions for sampling scattering angles.
    Weighting functions used for Monte Carlo calculations must be from this class.
    """
    @staticmethod
    def uniform(two_theta: [float, np.ndarray]) -> [float, np.ndarray]:
        """
        The uniform distribution. Not normalized.
        """
        return np.ones_like(two_theta)

    @staticmethod
    def natural_distribution(two_theta: [float, np.ndarray]) -> [float, np.ndarray]:
        """
        The "natural distribution" of scattering angles, i.e. the angle between k and
        k', if k and k' are each sampled randomly and uniformly from a sphere.
        """
        return np.pi / 360. * np.sin(np.radians(two_theta))

    @staticmethod
    def get_gaussians_at_peaks(locations: list[float], constant: float=0.1,
                               sigma: float=3):
        """
        Returns a weighting function consisting of Gaussians at the specified locations
        on top of a constant function
        """
        def gaussians(two_theta: [float, np.ndarray]) -> [float, np.ndarray]:
            result = constant * np.ones_like(two_theta)
            for location in locations:
                result += np.exp(-0.5 * ((two_theta - location) / sigma) ** 2)
            return result
        return gaussians


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
    _min_angle_deg, _max_angle_deg : float
        Defines angle range of interest. Needed to compute inverse CDF for weighting
        function.
    _pdf: Callable[[np.ndarray], ndarray]
        A probability density function on the angle domain. Can be not normalized.
    _inverse_cdf: Callable[[np.ndarray], ndarray]
        The inverse CDF for given pdf. Numerically computed.
    """
    def __init__(self,
                 unit_cell: UnitCell,
                 wavelength: float,
                 pdf: Callable[[np.ndarray], np.ndarray]=None,
                 min_angle_deg: float=0.,
                 max_angle_deg: float=180.):
        self.unit_cell = unit_cell
        self.wavelength = wavelength
        self._min_angle_deg = min_angle_deg
        self._max_angle_deg = max_angle_deg
        self._pdf = None
        self._inverse_cdf = None
        if pdf is not None:
            self.set_pdf(pdf)
        else:
            self.set_pdf(WeightingFunction.natural_distribution)

    def k(self):
        """
        Returns 2pi / wavelength.
        """
        return 2 * np.pi / self.wavelength

    def set_pdf(self, pdf: Callable):
        """
        Updates weighting function and recomputes inverse CDF.
        """
        self._pdf = pdf
        self._compute_inverse_cdf()

    def set_angle_range(self, min_angle_deg: float=None, max_angle_deg: float=None):
        """
        Sets angle range of interest and recomputes inverse CDF.
        """
        self._min_angle_deg = min_angle_deg
        self._max_angle_deg = max_angle_deg
        self._compute_inverse_cdf()

    def _compute_inverse_cdf(self):
        x_vals = np.linspace(self._min_angle_deg, self._max_angle_deg, 1000)
        pdf_vals = self._pdf(x_vals)

        # Compute CDF by integrating
        cdf_vals = scipy.integrate.cumulative_simpson(pdf_vals, x=x_vals, initial=0.)
        cdf_vals /= cdf_vals[-1]  # Normalize CDF

        # Return the interpolation of the inverse of the CDF
        try:
            inverse_cdf_func = scipy.interpolate.PchipInterpolator(cdf_vals, x_vals)
        except ValueError as exc:
            raise ValueError("Inverse CDF interpolation failed. Possibly because PDF "
                             "is negative or too close to zero at certain points.") \
                from exc
        self._inverse_cdf = inverse_cdf_func

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

    def _get_scattering_vecs_and_angles(self, n: int):
        """
        Generates random scattering vectors and their angles by sampling k and k'
        from a sphere uniformly. Discards those outside angle range of interest.

        Parameters
        ----------
        n : int
            Number of random vectors to generate initially. The number of vectors
            returned will be less after filtering.

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
        angles_accepted = np.where(np.logical_and(two_thetas >= self._min_angle_deg,
                                                  two_thetas <= self._max_angle_deg))
        scattering_vecs = scattering_vecs[angles_accepted]
        two_thetas = two_thetas[angles_accepted]

        return scattering_vecs, two_thetas

    def _get_scattering_vecs_and_angles_weighted(self, n: int):
        """
        Generates random scattering vectors according to the weighting function for
        the scattering angle. The scattering angle uniquely corresponds to the
        magnitude, and the scattering vectors' directions are spherically uniform.

        Parameters
        ----------
        n : int
            Number of random vectors to generate.

        Returns
        -------
        scattering_vecs, two_thetas : np.ndarray
            List of scattering vectors and their corresponding scattering angles.
        """
        two_thetas = self._inverse_cdf(np.random.uniform(size=n))
        magnitudes = 2 * self.k() * np.sin(np.radians(two_thetas) / 2)
        unit_vecs = utils.random_uniform_unit_vectors(n, 3)
        scattering_vecs = magnitudes[:, np.newaxis] * unit_vecs
        return scattering_vecs, two_thetas

    def _plot_diagnostics(
            self,
            two_thetas: np.ndarray,
            two_theta_cnts: np.ndarray,
            spectrum: np.ndarray,
            top_trials: np.ndarray | None
    ):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6),
                                            gridspec_kw={'width_ratios': [1, 1, 1.5]})
        # brute-force diffraction spectrum
        ax1.plot(two_thetas, spectrum)
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel("Scattering angle (2θ) (deg)")
        ax1.set_ylabel("Intensity")
        ax1.set_title("Brute-force diffraction pattern")
        ax1.grid(linestyle=":")

        # Intensity of the top trials - tests for enough decay
        if top_trials is not None:
            ax2.plot(top_trials[:, 3])
        ax2.set_ylim(bottom=0)
        ax2.set_xlabel("Top trial #")
        ax2.set_ylabel("Intensity")
        ax2.set_title(
            "Intensity should decay enough that subsequent trials are negligible")

        if top_trials is not None:
            # Create a second y-axis for the hexbin
            ax3_2 = ax3.twinx()
            ax3_2.zorder = 1
            ax3.zorder = 2
            ax3.patch.set_visible(False)
            ks = np.linalg.norm(top_trials[:, 0:3], axis=1)
            two_thetas_top = np.degrees(np.arcsin(ks / 2 / self.k()) * 2)
            bins = np.searchsorted(two_thetas, two_thetas_top) - 1
            top_distribution = np.bincount(bins, minlength=len(two_thetas)) / len(
                two_thetas_top)
            # Hexbin plot
            hb = ax3_2.hexbin(two_thetas_top, top_trials[:, 3], gridsize=80, cmap='YlOrRd',
                            norm=LogNorm())
            cb = plt.colorbar(hb, ax=ax3_2, pad=0.15)
            cb.set_label('Count')
            ax3_2.set_ylabel('Intensity')
            # Histogram of two_thetas for the top trials
            ax3.plot(two_thetas, top_distribution, label='Top trials',
                     color='C1')

        # Histogram of two_thetas for all trials
        ax3.plot(two_thetas, two_theta_cnts / np.sum(two_theta_cnts),
                   label='All trials', color='C0')
        ax3.set_ylim(bottom=0)
        ax3.set_xlabel('Two Theta')
        ax3.set_ylabel('Frequency')
        ax3.legend()

        if top_trials is None:
            ax3.set_title('Histogram of angles of all trials')
        else:
            ax3.set_title('Histograms of angles of all/top trials w/ scatter plot')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compute_intensities(
            scattering_vecs: np.ndarray,
            all_atom_pos: np.ndarray,
            all_atoms: np.ndarray,
            form_factors: Mapping[int, FormFactorProtocol]
    ):
        """
        Computes the intensities for each scattering vector.
        """
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

        intensities = np.abs(structure_factors) ** 2
        return intensities

    def calculate_diffraction_pattern(self,
                                      atoms: list[Atom],
                                      form_factors: Mapping[int, FormFactorProtocol],
                                      target_accepted_trials: int = 5000,
                                      trials_per_batch: int = 1000,
                                      angle_bins: int = 100,
                                      weighted: bool = True,
                                      num_top: int = 40000):
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
        angle_bins : int
            Number of bins for scattering angles.
        weighted : bool
            Whether to draw scattering vectors from a sphere or via inverse transform
            sampling using pdf.
        num_top : int
            The top num_top scattering trials in intensity will be returned in stream.

        Returns
        -------
        two_thetas : (angle_bins,) ndarray
            the left edges of the bins, evenly spaced within angle range specified
        intensities : (angle_bins,) ndarray
            intensity calculated for each bin (not normalized)
        stream : ndarray
            Top num_top intensity data points
        counts : ndarray
            Number of trials in each angle bin
        """
        two_thetas = np.linspace(self._min_angle_deg, self._max_angle_deg,
                                 angle_bins + 1)[:-1]
        intensities = np.zeros(angle_bins)
        counts = np.zeros(angle_bins)

        stats = DiffractionMonteCarloRunStats()
        stream = TopIntensityStream(num_top)

        all_atom_pos = np.array([np.array(atom.position) for atom in atoms])
        all_atoms = np.array([atom.atomic_number for atom in atoms])

        while stats.accepted_data_points < target_accepted_trials:

            if time.time() - stats.prev_print_time_ > 5:
                stats.prev_print_time_ = time.time()
                print(stats)

            if weighted:
                scattering_vecs, two_thetas_batch = (
                    self._get_scattering_vecs_and_angles_weighted(trials_per_batch))
            else:
                scattering_vecs, two_thetas_batch = (
                    self._get_scattering_vecs_and_angles(trials_per_batch))

            intensity_batch = DiffractionMonteCarlo.compute_intensities(
                scattering_vecs, all_atom_pos, all_atoms, form_factors
            )

            bins = np.searchsorted(two_thetas, two_thetas_batch) - 1
            intensities[bins] += intensity_batch
            counts += np.bincount(bins, minlength=counts.shape[0])

            stats.total_trials += two_thetas_batch.shape[0]
            stats.accepted_data_points += two_thetas_batch.shape[0]

            # Add to stream
            for i, inten in enumerate(intensity_batch):
                stream.add(scattering_vecs[i][0], scattering_vecs[i][1],
                           scattering_vecs[i][2], inten)

        if weighted:
            # Re-normalize intensity distribution
            renormalization = np.ones_like(intensities)
            renormalization /= self._pdf(two_thetas)
            renormalization *= WeightingFunction.natural_distribution(two_thetas)
            intensities *= renormalization

        return two_thetas, intensities, np.array(stream.get_top_n()), counts

    def calculate_neighborhood_diffraction_pattern(
            self,
            atoms: list[Atom],
            form_factors: Mapping[int, FormFactorProtocol],
            angle_bins: int = 100,
            brute_force_trials: int = 1_000_000,
            num_top: int = 40000,
            resample_cnt: int = 100,
            weighted: bool = True,
            sigma: float = 0.05,
            plot_diagnostics: bool = False
    ):
        two_thetas, intensities, top, counts = (
            self.calculate_diffraction_pattern(
                atoms,
                form_factors,
                target_accepted_trials=brute_force_trials,
                trials_per_batch=1000,
                angle_bins=angle_bins,
                num_top=num_top,
                weighted=weighted))

        if plot_diagnostics:
            self._plot_diagnostics(
                two_thetas,
                counts,
                intensities,
                top
            )

        intensities_neigh, counts_neigh = self.neighborhood_intensity(
            top[:, 0:3],
            two_thetas,
            atoms,
            form_factors,
            sigma=sigma,
            cnt_per_point=resample_cnt
        )

        if plot_diagnostics:
            self._plot_diagnostics(
                two_thetas,
                counts_neigh,
                intensities_neigh,
                None
            )

        return two_thetas, intensities_neigh

    def neighborhood_intensity(
            self,
            points: np.ndarray,
            two_thetas: np.ndarray,
            atoms: list[Atom],
            form_factors: Mapping[int, FormFactorProtocol],
            sigma: float=0.05,
            cnt_per_point: int=100
    ):
        """
        Calculates the diffraction spectrum by randomly sampling near the supplied
        points (scattering vectors), which are assumed to those with the largest
        contributions to the diffraction spectrum.

        Parameters
        ----------
        points : np.ndarray
            List of scattering vectors. Assumed to be those with the largest
            contributions to the diffraction spectrum.
        two_thetas : np.ndarray
            Left edges of angle bins.
        atoms : list[Atoms]
            List of all atoms in crystal.
        form_factors : Mapping[int, FormFactorProtocol]
            Dictionary mapping atomic number to associated NeutronFormFactor or
            XRayFormFactor.
        sigma : float
            Standard deviation of the 3D Gaussian for sampling around supplied points.
        cnt_per_point : int
            How many vectors to sample around each supplied point.

        Returns
        -------
        intensities : (angle_bins,) ndarray
            intensity calculated for each bin (not normalized)
        counts : (angle_bins,) ndarray
            Number of resampled vectors in each bin. Mostly for diagnostics.
        """
        all_atom_pos = np.array([np.array(atom.position) for atom in atoms])
        all_atoms = np.array([atom.atomic_number for atom in atoms])
        intensities = np.zeros_like(two_thetas, dtype=float)
        counts = np.zeros_like(two_thetas, dtype=int)
        covariance = [[sigma ** 2, 0, 0],
                      [0, sigma ** 2, 0],
                      [0, 0, sigma ** 2]]
        start_time = time.time()
        for i, point in enumerate(points):
            cnt = cnt_per_point
            if i % 1000 == 0:
                per_trial = (time.time() - start_time) * 1e6 / (np.sum(counts) + 0.01)
                print(f"Resampled {i}/{len(points)} points, cnt={cnt}, µs per trial="
                      f"{per_trial:.1f}, "
                      f"Time remaining={(per_trial * (len(points) - i) * cnt_per_point / 1e6):.0f}s")
            scattering_vecs = np.random.multivariate_normal(point, covariance, cnt)
            ks = np.linalg.norm(scattering_vecs, axis=1)
            two_thetas_batch = np.degrees(np.arcsin(ks / 2 / self.k()) * 2)
            # Some vectors may be out of range after resampling from Gaussian
            in_angle_range = np.logical_and(two_thetas_batch > self._min_angle_deg,
                                            two_thetas_batch < self._max_angle_deg)
            scattering_vecs = scattering_vecs[in_angle_range]
            two_thetas_batch = two_thetas_batch[in_angle_range]

            intensity_batch = self.compute_intensities(
                scattering_vecs,
                all_atom_pos,
                all_atoms,
                form_factors
            )

            bins = np.searchsorted(two_thetas, two_thetas_batch) - 1
            intensities[bins] += intensity_batch
            counts += np.bincount(bins, minlength=counts.shape[0])

        return intensities, counts

    @staticmethod
    def compute_intensities_ideal_crystal(
            scattering_vecs: np.ndarray,
            unit_cell_pos: np.ndarray,
            atom_pos_in_uc: np.ndarray,
            atoms_in_uc: np.ndarray,
            form_factors: Mapping[int, FormFactorProtocol],
    ):
        """
        Computes the intensities for each scattering vector.
        """
        # Compute intensities
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
        intensities = np.abs(structure_factors) ** 2
        return intensities

    def calculate_diffraction_pattern_ideal_crystal(
            self,
            form_factors: Mapping[int, FormFactorProtocol],
            target_accepted_trials: int = 5000,
            trials_per_batch: int = 1000,
            unit_cell_reps: tuple[int, int, int] = (8, 8, 8),
            angle_bins: int = 100,
            weighted: bool = False,
            num_top: int = 40000):
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
        angle_bins : int
            Number of bins for scattering angles
        weighted : bool
            Whether to draw scattering vectors from a sphere or via inverse transform
            sampling using pdf.
        num_top : int
            The top num_top scattering trials in intensity will be returned in stream.

        Returns
        -------
        two_thetas : (angle_bins,) ndarray
            The left edges of the bins, evenly spaced within angle range specified
        intensities : (angle_bins,) ndarray
            Intensity calculated for each bin (not normalized)
        stream : ndarray
            Top num_top intensity data points
        counts : ndarray
            Number of trials in each angle bin
        """
        two_thetas = np.linspace(self._min_angle_deg, self._max_angle_deg,
                                 angle_bins + 1)[:-1]
        intensities = np.zeros(angle_bins)
        counts = np.zeros(angle_bins)

        unit_cell_pos = self._unit_cell_positions(unit_cell_reps)

        atoms_in_uc, atom_pos_in_uc = self._atoms_and_pos_in_uc()

        stats = DiffractionMonteCarloRunStats()
        stream = TopIntensityStream(num_top)

        while stats.accepted_data_points < target_accepted_trials:

            if time.time() - stats.prev_print_time_ > 5:
                stats.prev_print_time_ = time.time()
                print(stats)

            if weighted:
                scattering_vecs, two_thetas_batch = (
                    self._get_scattering_vecs_and_angles_weighted(trials_per_batch))
            else:
                scattering_vecs, two_thetas_batch = (
                    self._get_scattering_vecs_and_angles(trials_per_batch))

            intensity_batch = DiffractionMonteCarlo.compute_intensities_ideal_crystal(
                scattering_vecs,
                unit_cell_pos,
                atom_pos_in_uc,
                atoms_in_uc,
                form_factors
            )

            bins = np.searchsorted(two_thetas, two_thetas_batch) - 1
            intensities[bins] += intensity_batch
            counts += np.bincount(bins, minlength=counts.shape[0])

            stats.total_trials += two_thetas_batch.shape[0]
            stats.accepted_data_points += two_thetas_batch.shape[0]

            # Add to stream
            for i, inten in enumerate(intensity_batch):
                stream.add(scattering_vecs[i][0], scattering_vecs[i][1],
                           scattering_vecs[i][2], inten)

        if weighted:
            # Re-normalize intensity distribution
            renormalization = np.ones_like(intensities)
            renormalization /= self._pdf(two_thetas)
            renormalization *= WeightingFunction.natural_distribution(two_thetas)
            intensities *= renormalization

        return two_thetas, intensities, np.array(stream.get_top_n()), counts

    def calculate_neighborhood_diffraction_pattern_ideal_crystal(
            self,
            form_factors: Mapping[int, FormFactorProtocol],
            angle_bins: int = 100,
            brute_force_uc_reps: tuple[int, int, int] = (8, 8, 8),
            neighbor_uc_reps: tuple[int, int, int] = (20, 20, 20),
            brute_force_trials: int = 1_000_000,
            num_top: int = 40000,
            resample_cnt: int = 100,
            weighted: bool = False,
            sigma: float = 0.05,
            plot_diagnostics: bool = False
    ):
        # TODO: condense these parameters into a dictionary? BruteParameters &
        # TODO: NeighborParameters
        two_thetas, intensities, top, counts = (
            self.calculate_diffraction_pattern_ideal_crystal(
                form_factors,
                target_accepted_trials=brute_force_trials,
                trials_per_batch=1000,
                unit_cell_reps=brute_force_uc_reps,
                angle_bins=angle_bins,
                weighted=weighted,
                num_top=num_top))

        if plot_diagnostics:
            self._plot_diagnostics(
                two_thetas,
                counts,
                intensities,
                top
            )

        intensities_neigh, counts_neigh = self.neighborhood_intensity_ideal_crystal(
            top[:, 0:3],
            two_thetas,
            form_factors,
            unit_cell_reps=neighbor_uc_reps,
            sigma=sigma,
            cnt_per_point=resample_cnt
        )

        if plot_diagnostics:
            self._plot_diagnostics(
                two_thetas,
                counts_neigh,
                intensities_neigh,
                None
            )

        return two_thetas, intensities_neigh

    def neighborhood_intensity_ideal_crystal(
            self,
            points: np.ndarray,
            two_thetas: np.ndarray,
            form_factors: Mapping[int, FormFactorProtocol],
            unit_cell_reps: tuple[int, int, int] = (8, 8, 8),
            sigma: float=0.05,
            cnt_per_point: int=100
    ):
        """
        Calculates the diffraction spectrum by randomly sampling near the supplied
        points (scattering vectors), which are assumed to those with the largest
        contributions to the diffraction spectrum.

        Parameters
        ----------
        points : np.ndarray
            List of scattering vectors. Assumed to be those with the largest
            contributions to the diffraction spectrum.
        two_thetas : np.ndarray
            Left edges of angle bins.
        form_factors : Mapping[int, FormFactorProtocol]
            Dictionary mapping atomic number to associated NeutronFormFactor or
            XRayFormFactor.
        unit_cell_reps : tuple[int, int, int]
            How many times to repeat the unit cell in x, y, z directions, forming the
            crystal powder for diffraction.
        sigma : float
            Standard deviation of the 3D Gaussian for sampling around supplied points.
        cnt_per_point : int
            How many vectors to sample around each supplied point.

        Returns
        -------
        intensities : (angle_bins,) ndarray
            intensity calculated for each bin (not normalized)
        counts : (angle_bins,) ndarray
            Number of resampled vectors in each bin. Mostly for diagnostics.
        """
        intensities = np.zeros_like(two_thetas, dtype=float)
        counts = np.zeros_like(two_thetas, dtype=int)
        covariance = [[sigma**2, 0, 0],
                      [0, sigma**2, 0],
                      [0, 0, sigma**2]]
        unit_cell_pos = self._unit_cell_positions(unit_cell_reps)
        atoms_in_uc, atom_pos_in_uc = self._atoms_and_pos_in_uc()
        start_time = time.time()
        for i, point in enumerate(points):
            cnt = cnt_per_point
            if i % 1000 == 0:
                per_trial = (time.time() - start_time) * 1e6 / (np.sum(counts) + 0.01)
                print(f"Resampled {i}/{len(points)} points, cnt={cnt}, µs per trial="
                f"{per_trial:.1f}, "
                f"Time remaining={(per_trial * (len(points) - i) * cnt_per_point / 1e6):.0f}s")
            scattering_vecs = np.random.multivariate_normal(point, covariance, cnt)
            ks = np.linalg.norm(scattering_vecs, axis=1)
            two_thetas_batch = np.degrees(np.arcsin(ks / 2 / self.k()) * 2)
            # Some vectors may be out of range after resampling from Gaussian
            in_angle_range = np.logical_and(two_thetas_batch > self._min_angle_deg,
                                            two_thetas_batch < self._max_angle_deg)
            scattering_vecs = scattering_vecs[in_angle_range]
            two_thetas_batch = two_thetas_batch[in_angle_range]

            intensity_batch = self.compute_intensities_ideal_crystal(
                scattering_vecs,
                unit_cell_pos,
                atom_pos_in_uc,
                atoms_in_uc,
                form_factors
            )

            bins = np.searchsorted(two_thetas, two_thetas_batch) - 1
            intensities[bins] += intensity_batch
            counts += np.bincount(bins, minlength=counts.shape[0])

        return intensities, counts

    @staticmethod
    def compute_intensities_random_occupation(
            scattering_vecs: np.ndarray,
            unit_cell_pos: np.ndarray,
            atom_pos_in_uc: np.ndarray,
            atomic_numbers_vars: list[np.ndarray],
            probs: list[float],
            form_factors: Mapping[int, FormFactorProtocol],
            rng: np.random.Generator
    ):
        """
        Computes the intensities for each scattering vector.
        """
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

        intensities = np.abs(structure_factors) ** 2
        return intensities

    def calculate_diffraction_pattern_random_occupation(
            self,
            atom_from: int,
            atom_to: int,
            probability: float,
            form_factors: Mapping[int, FormFactorProtocol],
            target_accepted_trials: int = 5000,
            trials_per_batch: int = 1000,
            unit_cell_reps: tuple[int, int, int] = (8, 8, 8),
            angle_bins: int = 100,
            weighted: bool = True,
            num_top: int = 40000):
        """
        Calculates the neutron diffraction spectrum using a Monte Carlo method for a
        random occupation crystal.

        The crystal is constructed by selecting from the unit cell varieties
        according to their probability distribution to give the desired concentration.

        Parameters
        ----------
        atom_from : int
            Atomic number of the element to be substituted out.
        atom_to : int
            Atomic number of the element to be substituted in.
        probability : float
            Probability for an atom to be substituted.
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
        angle_bins : int
            Number of bins for scattering angles
        weighted : bool
            Whether to draw scattering vectors from a sphere or via inverse transform
            sampling using pdf.
        num_top : int
            The top num_top scattering trials in intensity will be returned in stream.

        Returns
        -------
        two_thetas : (angle_bins,) ndarray
            The left edges of the bins, evenly spaced within angle range specified
        intensities : (angle_bins,) ndarray
            Intensity calculated for each bin (not normalized)
        stream : ndarray
            Top num_top intensity data points
        counts : ndarray
            Number of trials in each angle bin
        """
        two_thetas = np.linspace(self._min_angle_deg, self._max_angle_deg,
                                 angle_bins + 1)[:-1]
        intensities = np.zeros(angle_bins)
        counts = np.zeros(angle_bins)

        unit_cell_pos = self._unit_cell_positions(unit_cell_reps)

        _, atom_pos_in_uc = self._atoms_and_pos_in_uc()

        uc_vars = UnitCellVarieties(self.unit_cell,
                                    ReplacementProbability(atom_from, atom_to,
                                                           probability))
        atomic_numbers_vars, probs \
            = uc_vars.atomic_number_lists()

        stats = DiffractionMonteCarloRunStats()
        stream = TopIntensityStream(num_top)

        rng = np.random.default_rng() # A NumPy Random Generator

        while stats.accepted_data_points < target_accepted_trials:

            if time.time() - stats.prev_print_time_ > 5:
                stats.prev_print_time_ = time.time()
                print(stats)

            if weighted:
                scattering_vecs, two_thetas_batch = (
                    self._get_scattering_vecs_and_angles_weighted(trials_per_batch))
            else:
                scattering_vecs, two_thetas_batch = (
                    self._get_scattering_vecs_and_angles(trials_per_batch))

            intensity_batch = DiffractionMonteCarlo.compute_intensities_random_occupation(
                scattering_vecs,
                unit_cell_pos,
                atom_pos_in_uc,
                atomic_numbers_vars,
                probs,
                form_factors,
                rng
            )

            # Add to stream
            for i, inten in enumerate(intensity_batch):
                stream.add(scattering_vecs[i][0], scattering_vecs[i][1],
                           scattering_vecs[i][2], inten)

            bins = np.searchsorted(two_thetas, two_thetas_batch) - 1
            intensities[bins] += intensity_batch
            counts += np.bincount(bins, minlength=counts.shape[0])

            stats.total_trials += two_thetas_batch.shape[0]
            stats.accepted_data_points += two_thetas_batch.shape[0]

        if weighted:
            # Re-normalize intensity distribution
            renormalization = np.ones_like(intensities)
            renormalization /= self._pdf(two_thetas)
            renormalization *= WeightingFunction.natural_distribution(two_thetas)
            intensities *= renormalization

        return two_thetas, intensities, np.array(stream.get_top_n()), counts

    def calculate_neighborhood_diffraction_pattern_random_occupation(
            self,
            atom_from: int,
            atom_to: int,
            probability: float,
            form_factors: Mapping[int, FormFactorProtocol],
            angle_bins: int = 100,
            brute_force_uc_reps: tuple[int, int, int] = (8, 8, 8),
            neighbor_uc_reps: tuple[int, int, int] = (20, 20, 20),
            brute_force_trials: int = 1_000_000,
            num_top: int = 40000,
            resample_cnt: int = 100,
            weighted: bool = True,
            sigma: float = 0.05,
            plot_diagnostics: bool = False
    ):
        two_thetas, intensities, top, counts = (
            self.calculate_diffraction_pattern_random_occupation(
                atom_from, atom_to, probability,
                form_factors,
                target_accepted_trials=brute_force_trials,
                trials_per_batch=1000,
                unit_cell_reps=brute_force_uc_reps,
                angle_bins=angle_bins,
                num_top=num_top,
                weighted=weighted))

        if plot_diagnostics:
            self._plot_diagnostics(
                two_thetas,
                counts,
                intensities,
                top
            )

        intensities_neigh, counts_neigh = self.neighborhood_intensity_random_occupation(
            atom_from, atom_to, probability,
            top[:, 0:3],
            two_thetas,
            form_factors,
            unit_cell_reps=neighbor_uc_reps,
            sigma=sigma,
            cnt_per_point=resample_cnt
        )

        if plot_diagnostics:
            self._plot_diagnostics(
                two_thetas,
                counts_neigh,
                intensities_neigh,
                None
            )

        return two_thetas, intensities_neigh

    def neighborhood_intensity_random_occupation(
            self,
            atom_from: int,
            atom_to: int,
            probability: float,
            points: np.ndarray,
            two_thetas: np.ndarray,
            form_factors: Mapping[int, FormFactorProtocol],
            unit_cell_reps: tuple[int, int, int] = (8, 8, 8),
            sigma: float=0.05,
            cnt_per_point: int=100
    ):
        """
        Calculates the diffraction spectrum by randomly sampling near the supplied
        points (scattering vectors), which are assumed to those with the largest
        contributions to the diffraction spectrum.

        Parameters
        ----------
        atom_from : int
            Atomic number of the element to be substituted out.
        atom_to : int
            Atomic number of the element to be substituted in.
        probability : float
            Probability for an atom to be substituted.
        points : np.ndarray
            List of scattering vectors. Assumed to be those with the largest
            contributions to the diffraction spectrum.
        two_thetas : np.ndarray
            Left edges of angle bins.
        form_factors : Mapping[int, FormFactorProtocol]
            Dictionary mapping atomic number to associated NeutronFormFactor or
            XRayFormFactor.
        unit_cell_reps : tuple[int, int, int]
            How many times to repeat the unit cell in x, y, z directions, forming the
            crystal powder for diffraction.
        sigma : float
            Standard deviation of the 3D Gaussian for sampling around supplied points.
        cnt_per_point : int
            How many vectors to sample around each supplied point.

        Returns
        -------
        intensities : (angle_bins,) ndarray
            intensity calculated for each bin (not normalized)
        counts : (angle_bins,) ndarray
            Number of resampled vectors in each bin. Mostly for diagnostics.
        """
        intensities = np.zeros_like(two_thetas, dtype=float)
        counts = np.zeros_like(two_thetas, dtype=int)
        covariance = [[sigma**2, 0, 0],
                      [0, sigma**2, 0],
                      [0, 0, sigma**2]]
        unit_cell_pos = self._unit_cell_positions(unit_cell_reps)
        _, atom_pos_in_uc = self._atoms_and_pos_in_uc()
        uc_vars = UnitCellVarieties(self.unit_cell,
                                    ReplacementProbability(atom_from, atom_to,
                                                           probability))
        atomic_numbers_vars, probs \
            = uc_vars.atomic_number_lists()
        rng = np.random.default_rng()
        start_time = time.time()
        for i, point in enumerate(points):
            cnt = cnt_per_point
            if i % 1000 == 0:
                per_trial = (time.time() - start_time) * 1e6 / (np.sum(counts) + 0.01)
                print(f"Resampled {i}/{len(points)} points, cnt={cnt}, µs per trial="
                      f"{per_trial:.1f}, "
                      f"Time remaining={(per_trial * (len(points) - i) * cnt_per_point / 1e6):.0f}s")
            scattering_vecs = np.random.multivariate_normal(point, covariance, cnt)
            ks = np.linalg.norm(scattering_vecs, axis=1)
            two_thetas_batch = np.degrees(np.arcsin(ks / 2 / self.k()) * 2)
            # Some vectors may be out of range after resampling from Gaussian
            in_angle_range = np.logical_and(two_thetas_batch > self._min_angle_deg,
                                            two_thetas_batch < self._max_angle_deg)
            scattering_vecs = scattering_vecs[in_angle_range]
            two_thetas_batch = two_thetas_batch[in_angle_range]

            intensity_batch = DiffractionMonteCarlo.compute_intensities_random_occupation(
                scattering_vecs,
                unit_cell_pos,
                atom_pos_in_uc,
                atomic_numbers_vars,
                probs,
                form_factors,
                rng
            )

            bins = np.searchsorted(two_thetas, two_thetas_batch) - 1
            intensities[bins] += intensity_batch
            counts += np.bincount(bins, minlength=counts.shape[0])

        return intensities, counts

class TopIntensityStream:
    def __init__(self, N):
        self.N = N
        self.heap = []  # Min-heap to store top N elements

    def add(self, x, y, z, f):
        data_point = (x, y, z, f)
        if len(self.heap) < self.N:
            heapq.heappush(self.heap, (f, data_point))
        elif f > self.heap[0][0]:
            heapq.heappushpop(self.heap, (f, data_point))

    def get_top_n(self):
        return [point for _, point in sorted(self.heap, reverse=True)]