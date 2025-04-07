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
import inspect
import time
from dataclasses import dataclass, asdict
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
    num_top: int = 40000


@dataclass
class NeighborhoodSettings(IterationSettings):
    """
    Dataclass containing the required settings for the neighborhood sampling Monte
    Carlo method of calculating a spectrum.
    """
    sigma: float = 0.05
    cnt_per_point: int = 100
    num_top: int = 40000


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
        np.multiply(unit_cell_pos, self._unit_cell.lattice_constants, out=unit_cell_pos)
        return unit_cell_pos

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
            num_top: int = 40000):
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
        num_top : int
            The top num_top scattering trials in intensity will be returned in stream.

        Returns
        -------
        two_thetas : (angle_bins,) ndarray
            The left edges of the bins, evenly spaced within angle range specified
        intensities : (angle_bins,) ndarray
            Intensity calculated for each bin (not normalized)
        top : ndarray
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

        while stats.total_trials < total_trials:
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

            bins = np.searchsorted(two_thetas, two_thetas_batch) - 1
            intensities[bins] += intensity_batch
            counts += np.bincount(bins, minlength=counts.shape[0])

            stats.total_trials += two_thetas_batch.shape[0]

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

    def spectrum_neighborhood(
            self,
            points: np.ndarray,
            two_thetas: np.ndarray,
            form_factors: Mapping[int, FormFactorProtocol],
            sigma: float = 0.05,
            cnt_per_point: int = 100,
            num_top: int = 40000
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
        sigma : float
            Standard deviation of the 3D Gaussian for sampling around supplied points.
        cnt_per_point : int
            How many vectors to sample around each supplied point.
        num_top : int
            The top num_top scattering trials in intensity will be returned in stream.

        Returns
        -------
        intensities : (angle_bins,) ndarray
            intensity calculated for each bin (not normalized)
        top : ndarray
            Top num_top intensity data points
        counts : (angle_bins,) ndarray
            Number of resampled vectors in each bin. Mostly for diagnostics.
        """
        intensities = np.zeros_like(two_thetas, dtype=float)
        counts = np.zeros_like(two_thetas, dtype=int)
        covariance = sigma ** 2 * np.eye(3)
        stream = TopIntensityStream(num_top)
        start_time = time.time()

        for i, point in enumerate(points):
            if i != 0 and i % max(1, round(len(points) / 1000) * 100) == 0:
                per_trial = (time.time() - start_time) * 1e6 / (np.sum(counts))
                print(
                    f"Resampled {i}/{len(points)} points, µs per trial={per_trial:.1f}, "
                    f"Time remaining={(per_trial * (len(points) - i) * cnt_per_point / 1e6):.0f}s")
            scattering_vecs = np.random.multivariate_normal(point, covariance,
                                                            cnt_per_point)
            ks = np.linalg.norm(scattering_vecs, axis=1)
            two_thetas_batch = np.degrees(np.arcsin(ks / 2 / self.k()) * 2)
            # Some vectors may be out of range after resampling from Gaussian
            in_angle_range = np.logical_and(two_thetas_batch >= self._min_angle_deg,
                                            two_thetas_batch <= self._max_angle_deg)
            scattering_vecs = scattering_vecs[in_angle_range]
            two_thetas_batch = two_thetas_batch[in_angle_range]

            intensity_batch = self.compute_intensities(scattering_vecs, form_factors)

            bins = np.searchsorted(two_thetas, two_thetas_batch) - 1
            intensities[bins] += intensity_batch
            counts += np.bincount(bins, minlength=counts.shape[0])

            for j, inten in enumerate(intensity_batch):
                stream.add(scattering_vecs[j][0], scattering_vecs[j][1],
                           scattering_vecs[j][2], inten)

        return intensities, np.array(stream.get_top_n()), counts

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
            if i != 0 and not isinstance(iteration.settings, NeighborhoodSettings):
                raise ValueError("Iterations after the first must be Neighborhood Sampling.")

        for it in iterations:
            print(it)

        iterations[0].setup()
        uniform: UniformSettings = iterations[0].settings
        print(f"\nUniform sampling (1/{len(iterations)})")
        print(iterations[0])
        two_thetas, intensities, top, counts = (
            self.spectrum_uniform(
                form_factors,
                total_trials=uniform.total_trials,
                trials_per_batch=uniform.trials_per_batch,
                angle_bins=uniform.angle_bins,
                num_top=uniform.num_top,
                weighted=uniform.weighted))
        if plot_diagnostics:
            self._plot_diagnostics(
                two_thetas,
                counts,
                intensities,
                top
            )

        for i, it in enumerate(iterations[1:]):
            it.setup()
            neigh: NeighborhoodSettings = it.settings
            print(f"\nNeighborhood sampling ({i + 2}/{len(iterations)})")
            print(it)
            intensities, top, counts = self.spectrum_neighborhood(
                top[:, 0:3],
                two_thetas,
                form_factors,
                sigma=neigh.sigma,
                cnt_per_point=neigh.cnt_per_point,
                num_top=neigh.num_top
            )
            if plot_diagnostics:
                self._plot_diagnostics(
                    two_thetas,
                    counts,
                    intensities,
                    top
                )

        return two_thetas, intensities


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