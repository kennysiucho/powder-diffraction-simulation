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
from B8_project.crystal import UnitCell
from B8_project.form_factor import FormFactorProtocol, NeutronFormFactor, XRayFormFactor


class ReciprocalSpace:
    """
    A class to group functions related to reciprocal space, reciprocal lattice vectors
    and scattering vectors.
    """

    @staticmethod
    def get_reciprocal_lattice_vectors(
        min_magnitude: float,
        max_magnitude: float,
        lattice_constants: np.ndarray,
    ):
        """
        Get reciprocal lattice vectors
        ==============================

        Finds all the reciprocal lattice vectors with a magnitude in between a specified
        minimum and maximum magnitude. Returns a structured NumPy array representing the
        valid reciprocal lattice vectors.

        Array format
        ------------
        The structured NumPy array has the following fields:
            - 'miller_indices': An ndarray representing the Miller indices (h, k, l).
            - 'magnitude': A float representing the magnitude of the reciprocal lattice
            vector.
            - 'components': An ndarray representing the components of the reciprocal
            lattice vector.

        TODO: review documentation.
        """
        # Error handling.
        if not (max_magnitude > 0 and min_magnitude >= 0):
            raise ValueError(
                "max_magnitude and min_magnitude should be greater than or equal to 0."
            )
        if not max_magnitude > min_magnitude:
            raise ValueError("max_magnitude must be greater than min_magnitude.")
        if not (
            isinstance(lattice_constants, np.ndarray)
            and lattice_constants.shape == (3,)
        ):
            raise ValueError("lattice_constants must be a numpy array of length 3.")
        if not np.issubdtype(lattice_constants.dtype, np.floating):
            raise ValueError("lattice_constants must contain only floats.")

        # Upper bounds on Miller indices.
        max_hkl = np.ceil((lattice_constants * max_magnitude) / (2 * np.pi)).astype(int)

        # Generate all possible Miller indices within the bounds.
        miller_indices = np.stack(
            np.meshgrid(
                np.arange(-max_hkl[0], max_hkl[0] + 1),
                np.arange(-max_hkl[1], max_hkl[1] + 1),
                np.arange(-max_hkl[2], max_hkl[2] + 1),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)

        # Compute reciprocal lattice vector components and magnitudes.
        components = (2 * np.pi * miller_indices) / lattice_constants
        magnitudes = np.linalg.norm(components, axis=1)

        # Filter the reciprocal lattice vectors based on their magnitude.
        mask = (magnitudes >= min_magnitude) & (magnitudes <= max_magnitude)
        valid_miller_indices = miller_indices[mask]
        valid_magnitudes = magnitudes[mask]

        # Define a custom datatype to represent reciprocal lattice vectors.
        dtype = np.dtype(
            [("miller_indices", "3i4"), ("magnitudes", "f8"), ("components", "3f8")]
        )

        # Create a structured NumPy array to store the valid reciprocal lattice vectors.
        reciprocal_lattice_vectors = np.empty(
            valid_miller_indices.shape[0], dtype=dtype
        )
        reciprocal_lattice_vectors["miller_indices"] = valid_miller_indices
        reciprocal_lattice_vectors["magnitudes"] = valid_magnitudes
        reciprocal_lattice_vectors["components"] = (
            2 * np.pi * valid_miller_indices
        ) / lattice_constants

        return reciprocal_lattice_vectors

    @staticmethod
    def rlv_magnitudes_from_deflection_angles(
        deflection_angles: np.ndarray, wavelength: float
    ):
        """
        Reciprocal lattice vector magnitude from deflection angle
        =========================================================

        Calculates the magnitudes of the reciprocal lattice vectors associated with a
        range of given deflection angles.

        TODO: review documentation.
        """
        if deflection_angles.min() < 0 or deflection_angles.max() > 180:
            raise ValueError("Invalid deflection angle.")

        angles = deflection_angles * np.pi / 360
        return 4 * np.pi * np.sin(angles) / wavelength

    @staticmethod
    def deflection_angles_from_rlv_magnitudes(
        reciprocal_lattice_vector_magnitudes: np.ndarray, wavelength: float
    ):
        """
        Deflection angle from reciprocal lattice vector magnitude
        =========================================================

        Calculates the deflection angle associated with a reciprocal lattice vector of
        a given magnitude.

        TODO: review documentation.
        """
        sin_angles = (wavelength * reciprocal_lattice_vector_magnitudes) / (4 * np.pi)

        if sin_angles.max() > 1 or sin_angles.min() < 0:
            raise ValueError("Invalid reciprocal lattice vector magnitude(s)")

        return np.arcsin(sin_angles) * 360 / np.pi


class DiffractionPattern:
    """
    A class to group functions related to calculating diffraction patterns.
    """

    @staticmethod
    def get_structure_factors(
        unit_cell: UnitCell,
        form_factors: Mapping[int, FormFactorProtocol],
        reciprocal_lattice_vectors: np.ndarray,
    ) -> np.ndarray:
        """
        Get structure factors
        =====================

        Calculates the structure factor of a crystal for a specified range of
        reciprocal lattice vectors, and returns the structure factors as a NumPy array.

        TODO: review documentation.
        """
        # Initialize the structure factors array.
        structure_factors = np.zeros(
            reciprocal_lattice_vectors.shape[0], dtype=np.complex128
        )

        # Iterate through every atom in the unit cell and calculate the structure
        # factors.
        for atom in unit_cell.atoms:
            exponents = (2 * np.pi * 1j) * np.dot(
                reciprocal_lattice_vectors["miller_indices"], atom.position
            )

            try:
                form_factor = form_factors[atom.atomic_number]

                structure_factors += form_factor.evaluate_form_factors(
                    reciprocal_lattice_vectors["magnitudes"]
                ) * np.exp(exponents)

            except KeyError as exc:
                raise KeyError(f"Error reading form factor Mapping: {exc}") from exc
        return structure_factors

    @staticmethod
    def get_diffraction_peaks(
        unit_cell: UnitCell,
        form_factors: Mapping[int, FormFactorProtocol],
        wavelength: float,
        min_deflection_angle: float = 10,
        max_deflection_angle: float = 170,
        intensity_cutoff: float = 1e-6,
    ) -> np.ndarray:
        """
        Get diffraction peaks
        =====================

        TODO: add documentation.
        """
        # Error handling.
        if not (min_deflection_angle >= 0 and max_deflection_angle > 0):
            raise ValueError(
                """min_deflection_angle and max_deflection_angle should be greater than
                or equal to 0."""
            )
        if max_deflection_angle <= min_deflection_angle:
            raise ValueError(
                "max_deflection_angle should be larger than min_deflection_angle"
            )

        # Calculate the minimum and maximum RLV magnitudes
        try:
            min_magnitude = ReciprocalSpace.rlv_magnitudes_from_deflection_angles(
                np.array(min_deflection_angle), wavelength
            )
            max_magnitude = ReciprocalSpace.rlv_magnitudes_from_deflection_angles(
                np.array(max_deflection_angle), wavelength
            )
        except ValueError as exc:
            raise ValueError(
                f"Error calculating RLV max and min magnitudes: {exc}"
            ) from exc

        # Generate an array of all reciprocal lattice vectors with valid magnitudes.
        try:
            reciprocal_lattice_vectors = ReciprocalSpace.get_reciprocal_lattice_vectors(
                float(min_magnitude),
                float(max_magnitude),
                np.array(unit_cell.lattice_constants),
            )
        except ValueError as exc:
            raise ValueError(
                f"Error generating reciprocal lattice vectors: {exc}"
            ) from exc

        # Generate an array of deflection angles.
        deflection_angles = ReciprocalSpace.deflection_angles_from_rlv_magnitudes(
            reciprocal_lattice_vectors["magnitudes"], wavelength
        )

        # Calculate the structure factor for each reciprocal lattice vectors
        structure_factors = DiffractionPattern.get_structure_factors(
            unit_cell, form_factors, reciprocal_lattice_vectors
        )

        # Calculate the intensity of each peak and normalize the intensities
        intensities = np.abs(structure_factors) ** 2
        relative_intensities = intensities / np.max(intensities)

        # Define a custom datatype to represent intensity peaks.
        dtype = np.dtype(
            [
                ("miller_indices", "3i4"),
                ("deflection_angles", "f8"),
                ("intensities", "f8"),
            ]
        )

        # Create a structured NumPy array to store the intensity peaks.
        diffraction_peaks = np.empty(relative_intensities.shape[0], dtype=dtype)
        diffraction_peaks["miller_indices"] = reciprocal_lattice_vectors[
            "miller_indices"
        ]
        diffraction_peaks["deflection_angles"] = deflection_angles
        diffraction_peaks["intensities"] = relative_intensities

        # Remove duplicate angles and sum the intensities of duplicate peaks.
        diffraction_peaks = DiffractionPattern.merge_peaks(
            diffraction_peaks, intensity_cutoff
        )

        return diffraction_peaks

    @staticmethod
    def merge_peaks(
        diffraction_peaks: np.ndarray, intensity_cutoff: float = 1e-6
    ) -> np.ndarray:
        """
        Merge peaks
        ===========

        TODO: add documentation.
        """
        # Sort diffraction_peaks based on the deflection angle
        diffraction_peaks.sort(order="deflection_angles")

        # Iterate over all diffraction peaks
        i = 0
        while i < len(diffraction_peaks) - 1:
            # Sort miller indices from largest to smallest and take the absolute value
            diffraction_peaks[i]["miller_indices"] = np.abs(
                np.sort(diffraction_peaks[i]["miller_indices"])
            )

            # Remove any duplicate angles and merge intensities
            while i < len(diffraction_peaks) - 1 and np.isclose(
                diffraction_peaks[i]["deflection_angles"],
                diffraction_peaks[i + 1]["deflection_angles"],
                rtol=1e-10,
            ):
                diffraction_peaks["intensities"][i] += diffraction_peaks["intensities"][
                    i + 1
                ]
                diffraction_peaks = np.delete(diffraction_peaks, i + 1)

            i += 1

        # Normalize the intensities
        max_intensity = diffraction_peaks["intensities"].max()
        diffraction_peaks["intensities"] /= max_intensity

        # Remove peaks with intensities below the cutoff.
        diffraction_peaks = diffraction_peaks[
            diffraction_peaks["intensities"] >= intensity_cutoff
        ]

        return diffraction_peaks


class Plotting:
    """
    A class to group functions related to plotting diffraction patterns.
    """

    @staticmethod
    def plot_diffraction_pattern(
        unit_cell: UnitCell,
        diffraction_type: str,
        neutron_form_factors: Mapping[int, NeutronFormFactor],
        x_ray_form_factors: Mapping[int, XRayFormFactor],
        wavelength: float = 0.1,
        min_deflection_angle: float = 10,
        max_deflection_angle: float = 170,
        peak_width: float = 0.1,
        plot: bool = True,
        line_width: float = 1.0,
        file_path: str = "results/",
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
            nanometers (nm).
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
                diffraction_peaks = DiffractionPattern.get_diffraction_peaks(
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
                diffraction_peaks = DiffractionPattern.get_diffraction_peaks(
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

        # Calculate a sensible number of points
        num_points = np.round(
            10 * (max_deflection_angle - min_deflection_angle) / peak_width
        ).astype(int)

        # Get x coordinates of plotted points.
        x_values = np.linspace(min_deflection_angle, max_deflection_angle, num_points)

        # Get y coordinates of plotted points.
        y_values = np.zeros_like(x_values)

        gaussians = utils.gaussian(
            x_values[:, np.newaxis],
            diffraction_peaks["deflection_angles"],
            peak_width,
            diffraction_peaks["intensities"],
        )

        y_values += np.sum(gaussians, axis=1)

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
                f"λ = {round(wavelength, 4)}nm",
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
            fig.savefig(f"{file_path}{filename}.pdf", format="pdf")

            # Print the path to the .pdf file.
            print(f"Plot created at {file_path}{filename}.pdf")

        return x_values.tolist(), y_values.tolist()

    @staticmethod
    def plot_superimposed_diffraction_patterns(
        unit_cells_with_diffraction_types: list[tuple[UnitCell, str]],
        neutron_form_factors: Mapping[int, NeutronFormFactor],
        x_ray_form_factors: Mapping[int, XRayFormFactor],
        wavelength: float = 0.1,
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
            nanometers (nm). Default value is 0.1nm.
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
                    x_values, y_values = Plotting.plot_diffraction_pattern(
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
                    x_values, y_values = Plotting.plot_diffraction_pattern(
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
                    f"λ = {round(current_wavelength, 4)}nm",
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
