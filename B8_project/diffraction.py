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
    - _get_diffraction_peaks: Calculates the relative intensity of the peak associated 
    with each reciprocal lattice vector.
    - plot_diffraction_pattern: Plots the diffraction pattern for a given crystal and 
    saves the plot as a .pdf file.
"""

from datetime import datetime
from typing import Mapping
import numpy as np
import matplotlib.pyplot as plt
import B8_project.utils as utils
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


def _get_diffraction_peaks(
    unit_cell: UnitCell,
    form_factors: Mapping[int, FormFactorProtocol],
    wavelength: float,
    min_deflection_angle: float = 10,
    max_deflection_angle: float = 170,
) -> list[tuple[ReciprocalLatticeVector, float]]:
    """
    Get diffraction peaks
    =====================

    Calculates the relative intensity of the peak associated with each reciprocal
    lattice vector. Returns a list of tuples, with each tuple containing a reciprocal
    lattice vector and the relative intensity of the associated peak.
    """
    # Validate min_deflection_angle and max_angle are both greater than 0
    if not (min_deflection_angle >= 0 and max_deflection_angle > 0):
        raise ValueError(
            """min_deflection_angle and max_deflection_angle should be greater than
            or equal to 0."""
        )

    # Validate that max_deflection_angle is larger than min_deflection_angle
    if max_deflection_angle <= min_deflection_angle:
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


def calculate_miller_peaks(
    unit_cell: UnitCell,
    diffraction_type: str,
    neutron_form_factors: Mapping[int, NeutronFormFactor],
    x_ray_form_factors: Mapping[int, XRayFormFactor],
    wavelength: float,
    min_deflection_angle: float = 10,
    max_deflection_angle: float = 170,
    intensity_cutoff: float = 0.01,
) -> list[tuple[float, tuple[int, int, int], float]]:
    """
    Calculate Miller peaks
    ======================

    Calculates the deflection angles, Miller indices (h, k, l) and relative intensities
    of diffraction peaks, and returns this information as a list.

    Parameters
    ----------
        - unit_cell (UnitCell): the unit cell of the chosen crystal.
        - diffraction_type (str): the type of diffraction desired. Should be either
        `"ND"` for neutron diffraction, or `"XRD"` for X-ray diffraction.
        - neutron_form_factors (Mapping[int, NeutronFormFactor]): a mapping from atomic
        numbers to a class which represents a neutron form factor.
        - x_ray_form_factors (Mapping[int, XRayFormFactor]): a mapping from atomic
        numbers to a class which represents an X-ray form factor.
        - wavelength (float): the wavelength of incident particles, given in
        angstroms (Å).
        - min_deflection_angle (float), max_deflection_angle (float): these
        parameters specify the range of deflection angles to be plotted.
        - intensity_cutoff (float): The minimum intensity needed for a peak to be
        recorded.

    Returns
    -------
        - (list[tuple[float, tuple[int, int, int], float]]): Each element in the list
        corresponds to a different diffraction peak, and has the format
        (deflection_angle, miller_indices, relative_intensity).
    """
    if diffraction_type == "ND":
        diffraction_peaks = _get_diffraction_peaks(
            unit_cell,
            neutron_form_factors,
            wavelength,
            min_deflection_angle,
            max_deflection_angle,
        )

    elif diffraction_type == "XRD":
        diffraction_peaks = _get_diffraction_peaks(
            unit_cell,
            x_ray_form_factors,
            wavelength,
            min_deflection_angle,
            max_deflection_angle,
        )

    else:
        raise ValueError("Invalid diffraction type")

    miller_peaks = []

    # Iterates over every diffraction peak and filters any insignificant peaks.
    for reciprocal_lattice_vector, intensity in diffraction_peaks:
        deflection_angle = _deflection_angle(reciprocal_lattice_vector, wavelength)
        miller_indices = reciprocal_lattice_vector.miller_indices

        # Only add intensities which are greater than the cutoff
        if intensity > intensity_cutoff:
            miller_peaks.append((deflection_angle, miller_indices, intensity))

    # Sort miller_peaks based on the deflection angle
    miller_peaks.sort(key=lambda peak: peak[0])

    # Merge peaks which have the same intensity
    i = 0
    while i < len(miller_peaks) - 1:
        deflection_angle = miller_peaks[i][0]
        miller_indices = miller_peaks[i][1]

        # Sort miller indices from largest to smallest and take the absolute value
        miller_indices = tuple(sorted(map(abs, miller_indices), reverse=True))

        while i < len(miller_peaks) - 1 and np.isclose(
            deflection_angle, miller_peaks[i + 1][0], rtol=1e-10
        ):
            miller_peaks[i] = (
                deflection_angle,
                miller_indices,
                miller_peaks[i][2] + miller_peaks[i + 1][2],
            )
            del miller_peaks[i + 1]

        i += 1

    # Normalize the intensities
    max_intensity = max(miller_peaks, key=lambda peak: peak[2])[2]
    miller_peaks = [(x[0], x[1], x[2] / max_intensity) for x in miller_peaks]

    return miller_peaks


def plot_diffraction_pattern(
    unit_cell: UnitCell,
    diffraction_type: str,
    neutron_form_factors: Mapping[int, NeutronFormFactor],
    x_ray_form_factors: Mapping[int, XRayFormFactor],
    wavelength: float = 1.0,
    min_deflection_angle: float = 10,
    max_deflection_angle: float = 170,
    peak_width: float = 0.1,
    plot: bool = True,
    line_width: float = 1.0,
    file_path: str | None = "results/",
) -> tuple[list[float], list[float]]:
    """
    Plot diffraction pattern
    ========================

    Plots the diffraction pattern for a given crystal and saves the plot as a .pdf
    file in a specified directory if desired. Returns the x coordinates and y
    coordinates of the plotted points.

    Name of .pdf file
    -----------------
        - For neutron diffraction, the .pdf file has the following name:
        "<material>_<ND>_<date>.pdf", where "ND" stands for Neutron Diffraction.
        - For X-ray diffraction, the .pdf file has the following name:
        "<material>_<XRD>_<date>.pdf", where "XRD" stands for X-Ray Diffraction.

    Parameters
    ----------
        - unit_cell (UnitCell): the unit cell of the chosen crystal.
        - diffraction_type (str): the type of diffraction desired. Should be either
        `"ND"` for neutron diffraction, or `"XRD"` for X-ray diffraction.
        - neutron_form_factors (Mapping[int, NeutronFormFactor]): a mapping from atomic
        numbers to a class which represents a neutron form factor.
        - x_ray_form_factors (Mapping[int, XRayFormFactor]): a mapping from atomic
        numbers to a class which represents an X-ray form factor.
        - wavelength (float): the wavelength of incident particles, given in
        angstroms (Å).
        - min_deflection_angle (float), max_deflection_angle (float): these
        parameters specify the range of deflection angles to be plotted.
        - peak_width (float): The width of the intensity peaks. This parameter is
        only used for plotting. A value should be chosen so that all diffraction
        peaks can be observed. The default value is 0.1°.
        - plot (bool): True -> plot the diffraction pattern, and save as a .pdf file;
        False -> don't plot the diffraction pattern. The default value is True.
        - line_width (float): The linewidth of the plot. Default value is 1.
        - file_path (str): The path to the directory where the plot will be stored.
        Default value is `results/`.

    Returns
    -------
        - (tuple[list[float], list[float]]): A list of x coordinates and a list of y
        coordinates of the plotted points.
    """
    # Find the diffraction peaks.
    if diffraction_type == "ND":
        try:
            diffraction_peaks = _get_diffraction_peaks(
                unit_cell,
                neutron_form_factors,
                wavelength,
                min_deflection_angle,
                max_deflection_angle,
            )
        except Exception as exc:
            raise ValueError(f"Error finding diffraction peaks: {exc}") from exc

    elif diffraction_type == "XRD":
        try:
            diffraction_peaks = _get_diffraction_peaks(
                unit_cell,
                x_ray_form_factors,
                wavelength,
                min_deflection_angle,
                max_deflection_angle,
            )
        except Exception as exc:
            raise ValueError(f"Error finding diffraction peaks: {exc}") from exc

    else:
        raise ValueError("Invalid diffraction type.")

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

    if plot is True:
        # Get today's date and format as a string.
        today = datetime.today()
        date_string = today.strftime("%d-%m-%Y")

        # Filename
        filename = f"{unit_cell.material}_{diffraction_type}_{date_string}"

        # Create the figure and axis.
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the data.
        ax.plot(
            x_values,
            y_values,
            label=f"{unit_cell.material}, {diffraction_type}, "
            f"λ = {round(wavelength, 4)}Å",
            linewidth=line_width,
        )

        # Set axis labels.
        ax.set_xlabel("Deflection angle (°)", fontsize=11)
        ax.set_ylabel("Relative intensity", fontsize=11)

        # Set title.
        ax.set_title(
            f"Diffraction pattern for {unit_cell.material} ({diffraction_type}).",
            fontsize=15,
        )

        # Add legend.
        ax.legend()

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
        if file_path is None:
            plt.show()
        else:
            fig.savefig(f"{file_path}{filename}.pdf", format="pdf")

        # Print the path to the .pdf file.
        print(f"Plot created at {file_path}{filename}.pdf")

    return x_values.tolist(), y_values.tolist()


def plot_superimposed_diffraction_patterns(
    unit_cells_with_diffraction_types: list[tuple[UnitCell, str]],
    neutron_form_factors: Mapping[int, NeutronFormFactor],
    x_ray_form_factors: Mapping[int, XRayFormFactor],
    wavelength: float = 1.0,
    min_deflection_angle: float = 10,
    max_deflection_angle: float = 170,
    peak_width: float = 0.1,
    variable_wavelength: bool = False,
    line_width: float = 1.0,
    opacity: float = 0.5,
    file_path: str = "results/",
) -> None:
    """
    Plot superimposed diffraction patterns
    ======================================

    Plots the diffraction pattern of a list of crystals on the same plot. Each crystal
    is represented as an instance of `UnitCell`, and the diffraction type (neutron or
    X-ray) should be specified for each crystal.

    Name of .pdf file
    -----------------
    The filename consists of the chemical formula of each crystal followed by the
    diffraction type. For instance, suppose that we wanted to plot the ND pattern of
    NaCl and the XRD pattern of CsCl. The filename would then be
    "NaCl_ND_CsCl_XRD_<date>".

    Parameters
    ----------
        - unit_cells_with_diffraction_types (list[tuple[UnitCell, str]]): Each element
        in the list is a tuple (unit_cell, diffraction_type). unit_cell is an instance
        of `UnitCell`, and represents a crystal. diffraction_type is a string -
        diffraction_type should be `"ND"` for neutron diffraction or `"XRD"` for X-ray
        diffraction.
        - neutron_form_factors (Mapping[int, NeutronFormFactor]): a mapping from atomic
        numbers to a class which represents a neutron form factor.
        - x_ray_form_factors (Mapping[int, XRayFormFactor]): a mapping from atomic
        numbers to a class which represents an X-ray form factor.
        - wavelength (float): the wavelength of incident particles, given in
        angstroms (Å). Default value is 1Å.
        - min_deflection_angle (float), max_deflection_angle (float): these
        parameters specify the range of deflection angles to be plotted. Default values
        are 10°, 170° respectively.
        - peak_width (float): The width of the intensity peaks. This parameter is
        only used for plotting. A value should be chosen so that all diffraction
        peaks can be observed. The default value is 0.1°.
        - variable_wavelength (bool): False (default) -> Each plot uses the same
        wavelength. True -> the first plot uses the wavelength specified when the
        function is called, and the other plots use different wavelengths, such that
        the peaks for all of the plots overlap.
        - line_width (float): The linewidth of each curve. Default value is 1.
        - opacity (float): The opacity of each curve. Default value is 0.5.
        - file_path (str): The path to the directory where the plot will be stored.
        Default value is `results/`.
    """
    # Create the figure and axis.
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the diffraction pattern for each crystal.
    for unit_cell, diffraction_type in unit_cells_with_diffraction_types:
        # Calculate the wavelength for the current crystal.
        if variable_wavelength is True:
            current_wavelength = (
                wavelength
                * unit_cell.lattice_constants[0]
                / unit_cells_with_diffraction_types[0][0].lattice_constants[0]
            )
        else:
            current_wavelength = wavelength

        # Get points to plot for ND.
        if diffraction_type == "ND":
            try:
                x_values, y_values = plot_diffraction_pattern(
                    unit_cell,
                    diffraction_type,
                    neutron_form_factors,
                    x_ray_form_factors,
                    current_wavelength,
                    min_deflection_angle,
                    max_deflection_angle,
                    peak_width,
                    plot=False,
                )
            except Exception as exc:
                raise ValueError(f"Error getting points to plot: {exc}") from exc

        # Get points to plot for XRD
        elif diffraction_type == "XRD":
            try:
                x_values, y_values = plot_diffraction_pattern(
                    unit_cell,
                    diffraction_type,
                    neutron_form_factors,
                    x_ray_form_factors,
                    current_wavelength,
                    min_deflection_angle,
                    max_deflection_angle,
                    peak_width,
                    plot=False,
                )
            except Exception as exc:
                raise ValueError(f"Error getting points to plot: {exc}") from exc
        else:
            raise ValueError("Invalid diffraction type.")

        # Plot the points.
        try:
            ax.plot(
                x_values,
                y_values,
                label=f"{unit_cell.material}, {diffraction_type}"
                f"λ = {round(current_wavelength, 4)}Å",
                linewidth=line_width,
                alpha=opacity,
            )
        except Exception as exc:
            raise ValueError(f"Error plotting points: {exc}") from exc

    # Get today's date and format as a string.
    today = datetime.today()
    date_string = today.strftime("%d-%m-%Y")

    # Create a string for the plot title.
    title_string = "Diffraction pattern for "
    for unit_cell, diffraction_type in unit_cells_with_diffraction_types:
        title_string += f"{unit_cell.material} ({diffraction_type}), "

    title_string = title_string[:-2]
    title_string += "."

    # Create a string for the filename.
    filename = ""
    for unit_cell, diffraction_type in unit_cells_with_diffraction_types:
        filename += f"{unit_cell.material}_{diffraction_type}_"

    filename += f"{date_string}"

    # Set title.
    ax.set_title(title_string, fontsize=15)

    # Set axis labels.
    ax.set_xlabel("Deflection angle (°)", fontsize=11)
    ax.set_ylabel("Relative intensity", fontsize=11)

    # Add legend.
    plt.legend()

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
    fig.savefig(f"{file_path}{filename}.pdf", format="pdf")

    # Print the path to the .pdf file.
    print(f"Plot created at {file_path}{filename}.pdf")
