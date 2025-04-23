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
import inspect
from itertools import product
import random
import time
from dataclasses import dataclass, asdict, field
from typing import Mapping, Callable
from abc import ABC, abstractmethod
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from B8_project import utils, file_reading
from B8_project.crystal import UnitCell
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
    def uniform_in_q_space(two_theta: [float, np.ndarray]) -> [float, np.ndarray]:
        """
        The distribution of scattering angles when scattering vectors (Q) are sampled
        uniformly from k-space.
        """
        return np.sin(np.radians(two_theta / 2)) ** 2 * np.cos(
            np.radians(two_theta / 2))

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
class IterationSettings:
    """
    Parent class of UniformSettings and NeighborhoodSettings.
    """
    def __str__(self):
        settings_dict = asdict(self)
        settings_lines = [f"  {key}: {value}" for key, value in settings_dict.items()]
        return f"{self.__class__.__name__}:\n" + "\n".join(settings_lines)


@dataclass
class UniformSettings(IterationSettings):
    """
    Dataclass containing the required settings for the brute-force Monte Carlo method
    of calculating a spectrum.
    """
    total_trials: int = 5000
    trials_per_batch: int = 1000
    angle_bins: int = 100
    weighted: bool = True
    threshold: float = 0.005


@dataclass
class NeighborhoodSettings(IterationSettings):
    """
    Dataclass containing the required settings for the neighborhood sampling Monte
    Carlo method of calculating a spectrum.
    """
    sigma: float = 0.05
    cnt_per_point: int = 100
    trials_per_batch: int = 5000
    threshold: float = 0.005


@dataclass
class UniformPrunedSettings(IterationSettings):
    """
    Dataclass containing the required settings for the uniform Monte Carlo method
    with pruning.
    """
    dist: float = 0.02
    num_cells: tuple = (200, 200, 200)
    total_trials: int = 1_000_000
    trials_per_batch: int = 5_000
    threshold: float = 0.005


@dataclass
class RefinementIteration:
    """
    Defines one iteration of spectrum refinement.
    """
    setup: Callable[[], None]
    settings: IterationSettings

    def __str__(self):
        setup_source = inspect.getsource(self.setup)
        settings_str = str(self.settings)
        return (f"Setup:\n{setup_source}"
                f"{settings_str}")

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
    start_time_: float = field(default_factory=time.time)
    prev_print_time_: float = 0.0
    microseconds_per_trial: float = 0.0

    def recalculate_microseconds_per_trial(self):
        """
        Recalculate average `microseconds_per_trial`.
        """
        if self.accepted_data_points == 0:
            return
        self.microseconds_per_trial = ((time.time() - self.start_time_) /
                                       self.accepted_data_points * 1000000)

    def __str__(self):
        """
        Returns a formatted string containing all attributes that don't
        end in `_`.
        """
        self.recalculate_microseconds_per_trial()
        return "".join([f"{key}={val:.1f} | " if key[-1] != '_' else ""
                        for key, val in self.__dict__.items()])


class DiffractionMonteCarlo(ABC):
    """
    A class to calculate neutron diffraction patterns via Monte Carlo methods.

    Attributes
    ----------
    all_xray_form_factors : Mapping[int, FormFactorProtocol]
        Dictionary containing all xray form factors.
    all_nd_form_factors : Mapping[int, FormFactorProtocol]
        Dictionary containing all neutron form factors.
    wavelength : float
        The wavelength of the incident neutrons (in nm)
    _unit_cell : UnitCell
        The unit cell of the crystal
    _unit_cell_pos : np.array
        Positions of the unit cells that form the crystal.
    _min_angle_deg, _max_angle_deg : float
        Defines angle range of interest. Needed to compute inverse CDF for weighting
        function.
    _pdf: Callable[[np.ndarray], ndarray]
        A probability density function on the angle domain. Can be not normalized.
    _inverse_cdf: Callable[[np.ndarray], ndarray]
        The inverse CDF for given pdf. Numerically computed.
    """
    all_xray_form_factors: Mapping[int, FormFactorProtocol]
    all_nd_form_factors: Mapping[int, FormFactorProtocol]
    wavelength: float
    _unit_cell: UnitCell
    _unit_cell_pos: np.ndarray | None = None
    _min_angle_deg: float
    _max_angle_deg: float
    _pdf: Callable[[np.ndarray], np.ndarray] = None
    _inverse_cdf: Callable[[np.ndarray], np.ndarray] = None

    def __init__(self,
                 wavelength: float,
                 pdf: Callable[[np.ndarray], np.ndarray]=None,
                 min_angle_deg: float=0.,
                 max_angle_deg: float=180.,
                 unit_cell: UnitCell=None):
        self.all_xray_form_factors = file_reading.read_xray_form_factors(
            "data/x_ray_form_factors.csv")
        self.all_nd_form_factors = file_reading.read_neutron_scattering_lengths(
            "data/neutron_scattering_lengths.csv")
        self._unit_cell = unit_cell
        self.wavelength = wavelength
        self._min_angle_deg = min_angle_deg
        self._max_angle_deg = max_angle_deg
        if pdf is not None:
            self.set_pdf(pdf)
        else:
            self.set_pdf(WeightingFunction.uniform_in_q_space)

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

    def setup_cuboid_crystal(self, unit_cell_reps: tuple[int, int, int]):
        unit_cell_pos = np.vstack(
            np.mgrid[0:unit_cell_reps[0], 0:unit_cell_reps[1],
            0:unit_cell_reps[2]]).reshape(3, -1).T
        unit_cell_pos = unit_cell_pos.astype(np.float64)
        np.multiply(unit_cell_pos, self._unit_cell.lattice_constants, out=unit_cell_pos)
        self._unit_cell_pos = unit_cell_pos

    def setup_spherical_crystal(self, r_angstrom: float):
        a, b, c = self._unit_cell.lattice_constants
        max_i = int(np.ceil(2 * r_angstrom / a))
        max_j = int(np.ceil(2 * r_angstrom / b))
        max_k = int(np.ceil(2 * r_angstrom / c))
        center = np.array([r_angstrom, r_angstrom, r_angstrom])
        unit_cell_pos = []
        for i in range(max_i):
            for j in range(max_j):
                for k in range(max_k):
                    x, y, z = i * a, j * b, k * c
                    pos = np.array([x, y, z])
                    if np.linalg.norm(pos - center) < r_angstrom:
                        unit_cell_pos.append([x, y, z])
        self._unit_cell_pos = np.array(unit_cell_pos)
        print(f"INFO: Number of unit cells in spherical crystal: {len(self._unit_cell_pos)}")

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

    def _atoms_and_pos_in_uc(self):
        """
        Returns a list of the atomic number and coordinates w.r.t. the unit cell of
        each atom in one unit cell.
        """
        atoms_in_uc = []
        atom_pos_in_uc = []
        for atom in self._unit_cell.atoms:
            atoms_in_uc.append(atom.atomic_number)
            atom_pos_in_uc.append(np.multiply(atom.position,
                                        self._unit_cell.lattice_constants))
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

    def _get_scattering_vecs_and_angles_box(
            self, n: int, bounding_boxes: list[tuple[np.ndarray, np.ndarray]]):
        """
        Generates scattering vectors which are uniform within defined box in k-space.
        """
        scattering_vecs = np.empty((n, 3))
        for i in range(n):
            min_coords, max_coords = bounding_boxes[random.randrange(len(bounding_boxes))]
            scattering_vecs[i] = np.random.uniform(0, 1, size=3) * (
                    max_coords - min_coords) + min_coords
        ks = np.linalg.norm(scattering_vecs, axis=1)
        two_thetas = np.degrees(np.arcsin(ks / 2 / self.k()) * 2)
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
        ax1.set_title("Diffraction pattern")
        ax1.grid(linestyle=":")

        # Intensity of the top trials - tests for enough decay
        if top_trials is not None:
            ax2.plot(top_trials[:, 3])
        ax2.set_xlim(left=0)
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
            bin_edges = two_thetas - (two_thetas[1] - two_thetas[0]) * 0.5
            bins = np.searchsorted(bin_edges, two_thetas_top) - 1
            top_distribution = np.bincount(bins, minlength=len(two_thetas)) / len(
                two_thetas_top)
            # Hexbin plot
            hb = ax3_2.hexbin(two_thetas_top, top_trials[:, 3], gridsize=80, cmap='YlOrRd',
                            norm=LogNorm())
            cb = plt.colorbar(hb, ax=ax3_2, pad=0.15)
            cb.set_label('Count')
            ax3_2.set_ylim(bottom=0)
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

    @abstractmethod
    def compute_intensities(self,
                            scattering_vecs: np.ndarray,
                            form_factors: Mapping[int, FormFactorProtocol]):
        """Compute intensities based on scattering vectors."""
        pass

    def spectrum_uniform(
            self,
            form_factors: Mapping[int, FormFactorProtocol],
            total_trials: int = 5000,
            trials_per_batch: int = 1000,
            angle_bins: int = 100,
            weighted: bool = False,
            threshold: float = 0.005):
        """
        Calculates the diffraction spectrum using a brute-force Monte Carlo method
        (no neighborhood sampling) for corresponding crystal type.

        For each Monte Carlo trial, randomly choose the incident and scattered k-
        vectors. If the scattering angle is within the range specified, compute the
        lattice and basis structure factors and hence intensity of the trial. Add
        intensity to final diffraction pattern.

        Parameters
        ----------
        form_factors : Mapping[int, FormFactorProtocol]
            Dictionary containing form factors (neutron or x-ray). Can access self.
            all_xray_form_factors or all_nd_form_factors.
        angle_bins : int
            Number of bins for scattering angles
        total_trials : int
            Target number of Monte Carlo trials.
        trials_per_batch : int
            Number of trials calculated at once using NumPy methods
        weighted : bool
            Whether to draw scattering vectors from a sphere or via inverse transform
            sampling using pdf.
        threshold : float
            The scattering trials with intensity greater than threshold times the
            maximum intensity encountered will be returned in stream.

        Returns
        -------
        two_thetas : (angle_bins,) ndarray
            The center of the bins, evenly spaced within angle range specified
        intensities : (angle_bins,) ndarray
            Intensity calculated for each bin (not normalized)
        top : ndarray
            Top intensity data points larger than threshold times the maximum intensity.
        counts : ndarray
            Number of trials in each angle bin
        """
        bin_edges = np.linspace(self._min_angle_deg, self._max_angle_deg,
                                angle_bins + 1)
        two_thetas = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        intensities = np.zeros(angle_bins)
        counts = np.zeros(angle_bins)

        stats = DiffractionMonteCarloRunStats()
        stream = TopIntensityStream(threshold)

        while stats.accepted_data_points < total_trials:
            if time.time() - stats.prev_print_time_ > 5:
                stats.prev_print_time_ = time.time()
                print(stats)

            if weighted:
                scattering_vecs, two_thetas_batch = (
                    self._get_scattering_vecs_and_angles_weighted(trials_per_batch))
            else:
                scattering_vecs, two_thetas_batch = (
                    self._get_scattering_vecs_and_angles(trials_per_batch))

            intensity_batch = self.compute_intensities(
                scattering_vecs,
                form_factors
            )

            bins = np.searchsorted(bin_edges, two_thetas_batch) - 1
            np.add.at(intensities, bins, intensity_batch)
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
            renormalization *= WeightingFunction.uniform_in_q_space(two_thetas)
            intensities *= renormalization

        return two_thetas, intensities, np.array(stream.get_filtered()), counts

    def spectrum_neighborhood(
            self,
            points: np.ndarray,
            two_thetas: np.ndarray,
            form_factors: Mapping[int, FormFactorProtocol],
            sigma: float = 0.05,
            cnt_per_point: int = 100,
            trials_per_batch: int = 5000,
            threshold: float = 0.005
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
            x-axis of diffraction spectrum - for binning.
        form_factors : Mapping[int, FormFactorProtocol]
            Dictionary mapping atomic number to associated NeutronFormFactor or
            XRayFormFactor.
        sigma : float
            Standard deviation of the 3D Gaussian for sampling around supplied points.
        cnt_per_point : int
            How many vectors to sample around each supplied point.
        trials_per_batch : int
            Number of trials calculated at once using NumPy methods
        threshold : float
            The scattering trials with intensity greater than threshold times the
            maximum intensity encountered will be returned in stream.

        Returns
        -------
        intensities : (angle_bins,) ndarray
            intensity calculated for each bin (not normalized)
        top : ndarray
            Top intensity data points larger than threshold times the maximum intensity.
        counts : (angle_bins,) ndarray
            Number of resampled vectors in each bin. Mostly for diagnostics.
        """
        bin_edges = two_thetas - (two_thetas[1] - two_thetas[0]) * 0.5
        intensities = np.zeros_like(two_thetas, dtype=float)
        counts = np.zeros_like(two_thetas, dtype=int)
        covariance = sigma ** 2 * np.eye(3)
        stream = TopIntensityStream(threshold)
        start_time = time.time()

        print(f"INFO: Number of points: {len(points)}")

        scattering_vecs = np.empty((0, 3))
        for i, point in enumerate(points):
            if i != 0 and i % max(1, round(len(points) / 1000) * 100) == 0:
                per_trial = (time.time() - start_time) * 1e6 / (np.sum(counts))
                print(
                    f"Resampled {i}/{len(points)} points, µs per trial={per_trial:.1f}, "
                    f"Time remaining={(per_trial * (len(points) - i) * cnt_per_point / 1e6):.0f}s")

            vecs = np.random.multivariate_normal(point, covariance, cnt_per_point)
            scattering_vecs = np.vstack((scattering_vecs, vecs))

            if len(scattering_vecs) >= trials_per_batch or i == len(points) - 1:
                ks = np.linalg.norm(scattering_vecs, axis=1)
                two_thetas_batch = np.degrees(np.arcsin(ks / 2 / self.k()) * 2)
                # Some vectors may be out of range after resampling from Gaussian
                in_angle_range = np.logical_and(two_thetas_batch >= self._min_angle_deg,
                                                two_thetas_batch <= self._max_angle_deg)
                scattering_vecs = scattering_vecs[in_angle_range]
                two_thetas_batch = two_thetas_batch[in_angle_range]

                intensity_batch = self.compute_intensities(scattering_vecs, form_factors)

                bins = np.searchsorted(bin_edges, two_thetas_batch) - 1
                np.add.at(intensities, bins, intensity_batch)
                counts += np.bincount(bins, minlength=counts.shape[0])

                for j, inten in enumerate(intensity_batch):
                    stream.add(scattering_vecs[j][0], scattering_vecs[j][1],
                               scattering_vecs[j][2], inten)

                # Reset scattering vecs
                scattering_vecs = np.empty((0, 3))

        return intensities, np.array(stream.get_filtered()), counts

    def _get_bounding_boxes(self, points: np.ndarray, num_cells: tuple=(100, 100, 100)):
        min_corner = np.min(points, axis=0)
        max_corner = np.max(points, axis=0)
        bounds_size = max_corner - min_corner
        cell_size = bounds_size / num_cells

        relative_coords = (points - min_corner) / cell_size
        indices = np.floor(relative_coords).astype(int)
        indices = np.clip(indices, 0, np.array(num_cells) - 1)  # Avoid out-of-bounds

        count_grid = np.zeros(num_cells, dtype=int)
        neighbor_offsets = list(product([-1, 0, 1], repeat=3))

        for idx in indices:
            for dx, dy, dz in neighbor_offsets:
                ni = idx[0] + dx
                nj = idx[1] + dy
                nk = idx[2] + dz
                if 0 <= ni < num_cells[0] and 0 <= nj < num_cells[1] and 0 <= nk < \
                        num_cells[2]:
                    count_grid[ni, nj, nk] += 1

        bounding_boxes = []
        for ix in range(num_cells[0]):
            for iy in range(num_cells[1]):
                for iz in range(num_cells[2]):
                    if count_grid[ix, iy, iz] > 0:
                        cell_min = min_corner + np.array([ix, iy, iz]) * cell_size
                        cell_max = cell_min + cell_size
                        bounding_boxes.append((cell_min, cell_max))
        return bounding_boxes

    def spectrum_uniform_pruned(
            self,
            points: np.ndarray,
            two_thetas: np.ndarray,
            form_factors: Mapping[int, FormFactorProtocol],
            dist: float = 0.05,
            num_cells: tuple = (200, 200, 200),
            total_trials: int = 40000,
            trials_per_batch: int = 1000,
            threshold: float = 0.005
    ):
        """
        Calculates the diffraction spectrum by uniformly sampling the Q-sphere, but
        discarding the vectors that are not within distance dist to any points, which
        are assumed to be the main contributions to the diffraction spectrum.

        Parameters
        ----------
        points : np.ndarray
            List of scattering vectors. Assumed to be those with the largest
            contributions to the diffraction spectrum.
        two_thetas : np.ndarray
            Centers of angle bins.
        form_factors : Mapping[int, FormFactorProtocol]
            Dictionary mapping atomic number to associated NeutronFormFactor or
            XRayFormFactor.
        dist : float
            Random scattering vectors must be within distance dist to one of the points
            to be accepted.
        num_cells : tuple[int, int, int]
            Number of voxels to divide the Q-domain into - then generate scattering
            vectors only in those voxels containing points.
        total_trials : int
            Target number of accepted trials.
        trials_per_batch : int
            Number of random scattering vectors generated prior to discarding.
        threshold : float
            The scattering trials with intensity greater than threshold times the
            maximum intensity encountered will be returned in stream.

        Returns
        -------
        intensities : (angle_bins,) ndarray
            intensity calculated for each bin (not normalized)
        top : ndarray
            Top intensity data points larger than threshold times the maximum intensity.
        counts : (angle_bins,) ndarray
            Number of resampled vectors in each bin. Mostly for diagnostics.
        """
        bin_edges = two_thetas - (two_thetas[1] - two_thetas[0]) * 0.5
        intensities = np.zeros_like(two_thetas, dtype=float)
        counts = np.zeros_like(two_thetas, dtype=int)
        stream = TopIntensityStream(threshold)

        print(f"INFO: Number of points: {len(points)}")
        bounding_boxes = self._get_bounding_boxes(points, num_cells=num_cells)
        print(f"INFO: Number of voxels with points: {len(bounding_boxes)}")
        if np.any(bounding_boxes[0][1] - bounding_boxes[0][0] < dist):
            print(f"WARNING: Dist should not be larger than side lengths of voxels.")

        kdtree = scipy.spatial.KDTree(points)
        stats = DiffractionMonteCarloRunStats()

        while stats.accepted_data_points < total_trials:
            if time.time() - stats.prev_print_time_ > 5:
                stats.prev_print_time_ = time.time()
                print(stats)

            scattering_vecs, two_thetas_batch = (
                self._get_scattering_vecs_and_angles_box(trials_per_batch, bounding_boxes))

            # Keep only the vecs close to points
            lengths = kdtree.query_ball_point(scattering_vecs, dist, return_length=True)
            mask = (lengths > 0)
            filtered_vecs = scattering_vecs[mask]
            two_thetas_batch = two_thetas_batch[mask]

            stats.total_trials += trials_per_batch
            stats.accepted_data_points += len(filtered_vecs)

            intensity_batch = self.compute_intensities(filtered_vecs, form_factors)

            bins = np.searchsorted(bin_edges, two_thetas_batch) - 1
            np.add.at(intensities, bins, intensity_batch)
            counts += np.bincount(bins, minlength=counts.shape[0])

            for j, inten in enumerate(intensity_batch):
                stream.add(filtered_vecs[j][0], filtered_vecs[j][1],
                           filtered_vecs[j][2], inten)

        return intensities, np.array(stream.get_filtered()), counts

    def spectrum_iterative_refinement(
            self,
            form_factors: Mapping[int, FormFactorProtocol],
            iterations: list[RefinementIteration],
            plot_diagnostics: bool = False
    ):
        # Validate iterations
        if len(iterations) == 0:
            raise ValueError("Iterations cannot be empty.")
        for i, iteration in enumerate(iterations):
            if i == 0 and not isinstance(iteration.settings, UniformSettings):
                raise ValueError("First iteration must be Uniform Sampling.")
            if i != 0 and isinstance(iteration.settings, UniformSettings):
                raise ValueError("Iterations after the first must not be Uniform Sampling.")

        for it in iterations:
            print(it)

        iterations[0].setup()
        uniform: UniformSettings = iterations[0].settings
        print(f"\nUniform sampling (1/{len(iterations)})\n{iterations[0]}")
        two_thetas, intensities, top, counts = (
            self.spectrum_uniform(
                form_factors,
                total_trials=uniform.total_trials,
                trials_per_batch=uniform.trials_per_batch,
                angle_bins=uniform.angle_bins,
                threshold=uniform.threshold,
                weighted=uniform.weighted))
        if plot_diagnostics:
            self._plot_diagnostics(two_thetas, counts, intensities, top)

        for i, it in enumerate(iterations[1:]):
            it.setup()
            if isinstance(it.settings, NeighborhoodSettings):
                neigh: NeighborhoodSettings = it.settings
                print(f"\nNeighborhood sampling ({i + 2}/{len(iterations)})\n{it}")
                intensities, top, counts = self.spectrum_neighborhood(
                    top[:, 0:3],
                    two_thetas,
                    form_factors,
                    sigma=neigh.sigma,
                    cnt_per_point=neigh.cnt_per_point,
                    trials_per_batch=neigh.trials_per_batch,
                    threshold=neigh.threshold
                )
            elif isinstance(it.settings, UniformPrunedSettings):
                pruned: UniformPrunedSettings = it.settings
                print(f"\nUniform pruned sampling ({i + 2}/{len(iterations)})\n{it}")
                intensities, top, counts = self.spectrum_uniform_pruned(
                    top[:, 0:3],
                    two_thetas,
                    form_factors,
                    dist=pruned.dist,
                    num_cells=pruned.num_cells,
                    total_trials=pruned.total_trials,
                    trials_per_batch=pruned.trials_per_batch,
                    threshold=pruned.threshold
                )

            if plot_diagnostics:
                self._plot_diagnostics(two_thetas, counts, intensities, top )

        return two_thetas, intensities


@dataclass
class TopIntensityStream:
    frac: float
    values: list = field(default_factory=list)
    max_so_far: float = float('-inf')

    def add(self, x, y, z, f):
        if f > self.max_so_far:
            self.max_so_far = f
            threshold = self.frac * self.max_so_far
            self.values = [v for v in self.values if v[3] >= threshold]
        if f >= self.frac * self.max_so_far:
            self.values.append((x, y, z, f))

    def get_filtered(self):
        return sorted(self.values, key=lambda v: v[3], reverse=True)
