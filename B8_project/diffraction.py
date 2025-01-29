"""
Diffraction
===========

This module contains a range of functions to calculate diffraction patterns.

Functions
---------
    - _reciprocal_lattice_vector_magnitude: Calculates the magnitude of the 
    reciprocal lattice vector(s) associated with a given deflection angle.
    - _deflection_angle: Calculates the deflection angle associated with a 
    reciprocal lattice vector of a given magnitude.
    - _structure_factor: Returns the structure factor of a crystal evaluated at a 
    given reciprocal lattice vector.
    - get_diffraction_peaks: Calculates the relative intensity of the peak associated 
    with each reciprocal lattice vector.
    - plot_diffraction_pattern: Plots the diffraction pattern for a given crystal and 
    saves the plot as a .pdf file.

"""

from datetime import datetime
import time
from typing import Mapping
import numpy as np
import matplotlib.pyplot as plt
import B8_project.utils as utils
from B8_project.file_reading import *
from B8_project.crystal import UnitCell, ReciprocalLatticeVector
from B8_project.form_factor import FormFactorProtocol, NeutronFormFactor, XRayFormFactor


def _reciprocal_lattice_vector_magnitude(deflection_angle: float, wavelength: float):
    """
    Reciprocal lattice vector magnitude
    ===================================

    Calculates the magnitude of the reciprocal lattice vector(s) associated with a
    given deflection angle.
    """
    if deflection_angle < 0 or deflection_angle > 180:
        raise ValueError("Invalid deflection angle.")

    angle = deflection_angle * np.pi / 360
    return 4 * np.pi * np.sin(angle) / wavelength


def _deflection_angle(
    reciprocal_lattice_vector: ReciprocalLatticeVector, wavelength: float
):
    """
    Deflection angle
    ================

    Calculates the deflection angle associated with a reciprocal lattice vector of
    a given magnitude
    """
    sin_angle = (wavelength * reciprocal_lattice_vector.magnitude()) / (4 * np.pi)

    if sin_angle > 1 or sin_angle < 0:
        raise ValueError("Invalid reciprocal lattice vector magnitude")

    return np.arcsin(sin_angle) * 360 / np.pi


def _structure_factor(
    unit_cell: UnitCell,
    form_factors: Mapping[int, FormFactorProtocol],
    reciprocal_lattice_vector: ReciprocalLatticeVector,
) -> complex:
    """
    Structure factor
    ================

    Returns the structure factor of a crystal evaluated at a given reciprocal lattice vector.

    An instance of `UnitCell` represents the crystal. The form factors are stored
    in a `Mapping` which maps atomic number to form factor.
    """
    structure_factor = 0 + 0j

    for atom in unit_cell.atoms:
        exponent = (2 * np.pi * 1j) * utils.dot_product_tuples(
            reciprocal_lattice_vector.miller_indices, atom.position
        )

        try:
            form_factor = form_factors[atom.atomic_number]

            structure_factor += form_factor.evaluate_form_factor(
                reciprocal_lattice_vector
            ) * np.exp(exponent)

        except KeyError as exc:
            raise KeyError(f"Error reading form factor Mapping: {exc}") from exc
    return structure_factor


def get_diffraction_peaks(
    unit_cell: UnitCell,
    form_factors: Mapping[int, FormFactorProtocol],
    wavelength: float,
    min_deflection_angle: float,
    max_deflection_angle: float,
) -> list[tuple[ReciprocalLatticeVector, float]]:
    """
    Get diffraction peaks
    =====================

    Calculates the relative intensity of the peak associated with each reciprocal
    lattice vector. Returns a list of tuples, with each tuple containing a reciprocal
    lattice vector and the relative intensity of the associated peak.

    Parameters
    ----------
        - unit_cell (UnitCell): the unit cell of the chosen crystal.
        - form_factors (Mapping[int, FormFactorProtocol]): a mapping from atomic
        numbers to a class which represents an atomic form factor. Currently, two
        classes implement the form factor protocol, `NeutronFormFactor` and
        `XRayFormFactor`.
        - wavelength (float): the wavelength of incident particles, given in
        nanometers (nm).
        - min_deflection_angle (float), max_deflection_angle (float): these
        parameters specify the range of deflection angles to be plotted.

    Returns
    -------
        - (list[tuple[ReciprocalLatticeVector, float]]): A list of tuples, with each
        tuple containing a reciprocal lattice vector and the relative intensity of the
        associated peak.
    """
    # Validate min_deflection_angle and max_angle are both greater than 0
    if not (min_deflection_angle >= 0 and max_deflection_angle > 0):
        raise ValueError(
            """min_deflection_angle and max_deflection_angle should be greater than
            or equal to 0."""
        )

    # Validate that max_deflection_angle is larger than min_deflection_angle
    if not max_deflection_angle > min_deflection_angle:
        raise ValueError(
            "max_deflection_angle should be larger than min_deflection_angle"
        )

    # Calculate the minimum and maximum RLV magnitudes
    try:
        min_magnitude = _reciprocal_lattice_vector_magnitude(
            min_deflection_angle, wavelength
        )
        max_magnitude = _reciprocal_lattice_vector_magnitude(
            max_deflection_angle, wavelength
        )
    except ValueError as exc:
        raise ValueError(
            f"Error calculating RLV max and min magnitudes: {exc}"
        ) from exc

    # Generates a list of all reciprocal lattice vectors with valid magnitudes.
    try:
        reciprocal_lattice_vectors = (
            ReciprocalLatticeVector.get_reciprocal_lattice_vectors(
                min_magnitude, max_magnitude, unit_cell
            )
        )
    except ValueError as exc:
        raise ValueError(f"Error generating reciprocal lattice vectors: {exc}") from exc

    # Calculate the structure factor for each reciprocal lattice vectors
    structure_factors = []
    for reciprocal_lattice_vector in reciprocal_lattice_vectors:
        try:
            structure_factors.append(
                _structure_factor(unit_cell, form_factors, reciprocal_lattice_vector)
            )

        except Exception as exc:
            raise ValueError(f"Error computing structure factor: {exc}") from exc

    # Calculate the intensity of each peak and normalize the intensities
    intensities = [np.abs(x) ** 2 for x in structure_factors]
    relative_intensities = [x / max(intensities) for x in intensities]

    return list(zip(reciprocal_lattice_vectors, relative_intensities))


def plot_diffraction_pattern(
    unit_cell: UnitCell,
    form_factors: Mapping[int, FormFactorProtocol],
    wavelength: float,
    min_deflection_angle: float,
    max_deflection_angle: float,
    peak_width: float,
) -> str:
    """
    Plot diffraction pattern
    ========================

    Plots the diffraction pattern for a given crystal and saves the plot as a .pdf
    file in the `results` directory.

    Name of .pdf file
    -----------------
        - For neutron diffraction, the .pdf file has the following name:
        "<material>_<NDP>_<date>_<time>.pdf", where "NDP" stands for Neutron Diffraction
        Pattern.
        - For X-ray diffraction, the .pdf file has the following name:
        "<material>_<XRDP>_<date>_<time>.pdf", where "XRDP" stands for X-Ray Diffraction
        Pattern.

    Parameters
    ----------
        - unit_cell (UnitCell): the unit cell of the chosen crystal.
        - form_factors (Mapping[int, FormFactorProtocol]): a mapping from atomic
        numbers to a class which represents an atomic form factor. Currently, two
        classes implement the form factor protocol, `NeutronFormFactor` and
        `XRayFormFactor`.
        - wavelength (float): the wavelength of incident particles, given in
        nanometers (nm).
        - min_deflection_angle (float), max_deflection_angle (float): these
        parameters specify the range of deflection angles to be plotted.
        - peak_width (float): The width of the intensity peaks. This parameter is
        only used for plotting. A value should be chosen so that all diffraction
        peaks can be observed.

    Returns
    -------
        - (str): The path to the plot.
    """
    try:
        diffraction_peaks = get_diffraction_peaks(
            unit_cell,
            form_factors,
            wavelength,
            min_deflection_angle,
            max_deflection_angle,
        )
    except Exception as exc:
        raise ValueError(f"Error finding diffraction peaks: {exc}") from exc

    reciprocal_lattice_vectors, intensities = zip(*diffraction_peaks)

    # For each reciprocal lattice vector, calculate the associated deflection angle.
    deflection_angles = [
        _deflection_angle(x, wavelength) for x in reciprocal_lattice_vectors
    ]

    # Calculate a sensible number of points
    num_points = np.round(
        10 * (max_deflection_angle - min_deflection_angle) / peak_width
    ).astype(int)

    # Get x coordinates of plotted points.
    x_values = np.linspace(min_deflection_angle, max_deflection_angle, num_points)

    # Get y coordinates of plotted points.
    y_values = np.zeros_like(x_values)

    for deflection_angle, intensity in list(zip(deflection_angles, intensities)):
        y_values += utils.gaussian(x_values, deflection_angle, peak_width, intensity)

    # Normalize the intensities.
    max_intensity = np.max(y_values)
    y_values = y_values / max_intensity

    # Get today's date and format as a string.
    today = datetime.today()
    date_string = today.strftime("%d-%m-%Y_%H-%M")

    # Figure out the diffraction type and correct filename from form_factors.
    if isinstance(form_factors, Mapping) and all(
        isinstance(v, NeutronFormFactor) for v in form_factors.values()
    ):
        diffraction_type = "neutron "
        filename = f"{unit_cell.material}_NDP_{date_string}"
    elif isinstance(form_factors, Mapping) and all(
        isinstance(v, XRayFormFactor) for v in form_factors.values()
    ):
        diffraction_type = "X-ray "
        filename = f"{unit_cell.material}_XRDP_{date_string}"
    else:
        diffraction_type = ""
        filename = f"{unit_cell.material}_DP_{date_string}"

    # Create the figure and axis.
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the data.
    ax.plot(x_values, y_values, color="black")

    # Set axis labels.
    ax.set_xlabel("Deflection angle (°)", fontsize=11)
    ax.set_ylabel("Relative intensity", fontsize=11)

    # Set title.
    ax.set_title(
        f"{unit_cell.material} {diffraction_type}diffraction pattern for λ = {wavelength}nm.",
        fontsize=15,
    )

    # Add grid lines.
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Customize the tick marks.
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.tick_params(axis="both", which="minor", length=4, color="gray")

    # Add minor ticks.
    ax.minorticks_on()

    # Adjust layout to prevent clipping.
    fig.tight_layout()

    # Save the figure.
    fig.savefig(f"results/{filename}.pdf", format="pdf")

    # Return the path to the .pdf file.
    return f"results/{filename}.pdf"




class NeutronDiffractionMonteCarloRunStats:
    def __init__(self):
        self.accepted_data_points = 0
        self.avg_intensity_cnt_ = 0
        self.avg_intensity = 0.0
        self.total_trials = 0
        self.start_time_ = time.time()
        self.prev_print_time_ = 0.0
        self.microseconds_per_trial = 0.0

    def update_avg_intensity(self, intensity: float):
        self.avg_intensity = ((self.avg_intensity * self.avg_intensity_cnt_ + intensity) /
                              (self.avg_intensity_cnt_ + 1))
        self.avg_intensity_cnt_ += 1

    def update(self):
        if self.total_trials == 0: return
        self.microseconds_per_trial = (time.time() - self.start_time_) / self.total_trials * 1000000

    def __str__(self):
        self.update()
        return "".join([f"{key}={val:.1f} | " if key[-1] != '_' else "" for key, val in self.__dict__.items()])


class NeutronDiffractionMonteCarlo:
    def __init__(self, unit_cell: UnitCell, wavelength: float):
        self.unit_cell = unit_cell
        self.wavelength = wavelength

    def calculate_diffraction_pattern(self, N_trials: int = 5000):
        k = 2 * np.pi / self.wavelength
        two_thetas = np.zeros(N_trials)
        intensities = np.zeros(N_trials)

        # read relevant neutron scattering lengths
        all_scattering_lengths = read_neutron_scattering_lengths("data/neutron_scattering_lengths.csv")
        scattering_lengths = {}
        for atom in self.unit_cell.atoms:
            scattering_lengths[atom.atomic_number] = all_scattering_lengths[atom.atomic_number].neutron_scattering_length
        print(scattering_lengths)

        expand_N = 9
        expanded_pos = np.vstack(np.mgrid[0:expand_N, 0:expand_N, 0:expand_N].astype(np.float64)).reshape(3, -1).T
        np.multiply(expanded_pos, self.unit_cell.lattice_constants, out=expanded_pos)
        print(expanded_pos)

        stats = NeutronDiffractionMonteCarloRunStats()

        batch_trials = 10000
        while stats.accepted_data_points < N_trials:

            if time.time() - stats.prev_print_time_ > 5:
                stats.prev_print_time_ = time.time()
                print(stats)

            structure_factors = np.zeros(batch_trials, dtype=np.complex128)
            k_vecs = k * utils.random_uniform_unit_vectors(batch_trials, 3)
            k_primes = k * utils.random_uniform_unit_vectors(batch_trials, 3)
            scattering_vecs = k_primes - k_vecs

            for atom in self.unit_cell.atoms:
                r = np.multiply(atom.position, self.unit_cell.lattice_constants) + expanded_pos
                # r.shape = (expand_N^3, 3)
                # scattering_vec.shape = (batch_trials, 3)
                # structure_factor.shape = (batch_trials, )
                structure_factors += scattering_lengths[atom.atomic_number] * np.sum(np.exp(1j * scattering_vecs @ r.T), axis=1)

            dot_products = np.einsum("ij,ij->i", k_vecs, k_primes)
            two_theta_batch = np.arccos(dot_products / k**2)
            intensity_batch = np.abs(structure_factors)**2

            stats.total_trials += batch_trials

            angles_accepted = np.where(np.logical_and(two_theta_batch > np.radians(15), two_theta_batch < np.radians(60)))
            two_theta_batch = two_theta_batch[angles_accepted]
            intensity_batch = intensity_batch[angles_accepted]

            for i in range(0, two_theta_batch.size, two_theta_batch.size // 50):
                stats.update_avg_intensity(intensity_batch[i])

            intensities_accept = np.where(intensity_batch > 50 * stats.avg_intensity)
            for i in intensities_accept[0]:
                if stats.accepted_data_points == N_trials: break
                two_thetas[stats.accepted_data_points] = np.degrees(two_theta_batch[i])
                intensities[stats.accepted_data_points] = intensity_batch[i]
                stats.accepted_data_points += 1

        intensities /= np.max(intensities)

        return two_thetas, intensities