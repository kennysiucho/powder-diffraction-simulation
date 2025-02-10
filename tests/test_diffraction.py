"""
This module contains unit tests for the diffraction module.

TODO: add unit tests for the functions in the diffraction.py module.
"""

import numpy as np
from B8_project import diffraction, file_reading, crystal


def test_calculate_structure_factors_normal_operation():
    """
    A unit test for the _calculate_structure_factors function. This unit test tests
    normal operation of the function.
    """
    basis = file_reading.read_basis("tests/data/Na_basis.csv")
    lattice = file_reading.read_lattice("tests/data/Na_lattice.csv")
    unit_cell = crystal.UnitCell.new_unit_cell(basis, lattice)

    neutron_form_factors = file_reading.read_neutron_scattering_lengths(
        "tests/data/neutron_scattering_lengths.csv"
    )

    rlv_magnitudes = crystal.ReciprocalSpace.rlv_magnitudes_from_deflection_angles(
        np.array([20, 60]), 0.1
    )

    reciprocal_lattice_vectors = crystal.ReciprocalSpace.get_reciprocal_lattice_vectors(
        rlv_magnitudes[0], rlv_magnitudes[1], np.array([1.0, 1.0, 1.0])
    )

    # pylint: disable=protected-access
    structure_factors = diffraction._calculate_structure_factors(
        unit_cell, neutron_form_factors, reciprocal_lattice_vectors
    )
    # pylint: enable=protected-access

    assert len(structure_factors) == len(reciprocal_lattice_vectors)


def test_calculate_diffraction_peaks_normal_operation():
    """
    A unit test for the _calculate_diffraction_peaks function. This unit test tests
    normal operation of the function.
    """
    basis = file_reading.read_basis("tests/data/Na_basis.csv")
    lattice = file_reading.read_lattice("tests/data/Na_lattice.csv")
    unit_cell = crystal.UnitCell.new_unit_cell(basis, lattice)

    neutron_form_factors = file_reading.read_neutron_scattering_lengths(
        "tests/data/neutron_scattering_lengths.csv"
    )

    # pylint: disable=protected-access
    diffraction_peaks = diffraction._calculate_diffraction_peaks(
        unit_cell, neutron_form_factors, 0.1, 20, 60, 1e-6
    )
    # pylint: enable=protected-access

    assert isinstance(diffraction_peaks, np.ndarray)

    required_fields = {"miller_indices", "deflection_angles", "intensities"}
    assert set(diffraction_peaks.dtype.names) == required_fields


def test_get_miller_peaks_normal_operation():
    """
    A unit test for the get_miller_peaks function. This unit test tests normal
    operation of the function.
    """
    basis = file_reading.read_basis("tests/data/Na_basis.csv")
    lattice = file_reading.read_lattice("tests/data/Na_lattice.csv")
    unit_cell = crystal.UnitCell.new_unit_cell(basis, lattice)

    neutron_form_factors = file_reading.read_neutron_scattering_lengths(
        "tests/data/neutron_scattering_lengths.csv"
    )
    x_ray_form_factors = file_reading.read_xray_form_factors(
        "tests/data/x_ray_form_factors.csv"
    )

    # To view the output from this function, run pytest with the "-s" flag, i.e. run
    # `pytest tests/test_diffraction.py -s`
    miller_peaks = diffraction.get_miller_peaks(
        unit_cell,
        "ND",
        neutron_form_factors,
        x_ray_form_factors,
        wavelength=0.1,
        min_deflection_angle=20,
        max_deflection_angle=60,
        intensity_cutoff=1e-6,
        print_peak_data=True,
        save_to_csv=False,
    )

    assert isinstance(miller_peaks, np.ndarray)

    required_fields = {"miller_indices", "deflection_angles", "intensities"}
    assert set(miller_peaks.dtype.names) == required_fields


def test_get_diffraction_pattern_normal_operation():
    """
    A unit test for the get_diffraction_pattern function. This unit test tests normal
    operation of the function.
    """
    basis = file_reading.read_basis("tests/data/Na_basis.csv")
    lattice = file_reading.read_lattice("tests/data/Na_lattice.csv")
    unit_cell = crystal.UnitCell.new_unit_cell(basis, lattice)

    neutron_form_factors = file_reading.read_neutron_scattering_lengths(
        "tests/data/neutron_scattering_lengths.csv"
    )
    x_ray_form_factors = file_reading.read_xray_form_factors(
        "tests/data/x_ray_form_factors.csv"
    )

    diffraction_pattern = diffraction.get_diffraction_pattern(
        unit_cell,
        "ND",
        neutron_form_factors,
        x_ray_form_factors,
        wavelength=0.1,
        min_deflection_angle=20,
        max_deflection_angle=60,
        peak_width=0.1,
    )

    assert isinstance(diffraction_pattern, np.ndarray)

    required_fields = {"deflection_angles", "intensities"}
    assert set(diffraction_pattern.dtype.names) == required_fields


def test_plot_diffraction_pattern_normal_operation():
    """
    A unit test for the plot_diffraction_pattern function. This unit test tests normal
    operation of the function.
    """
    basis = file_reading.read_basis("tests/data/Na_basis.csv")
    lattice = file_reading.read_lattice("tests/data/Na_lattice.csv")
    unit_cell = crystal.UnitCell.new_unit_cell(basis, lattice)

    neutron_form_factors = file_reading.read_neutron_scattering_lengths(
        "tests/data/neutron_scattering_lengths.csv"
    )
    x_ray_form_factors = file_reading.read_xray_form_factors(
        "tests/data/x_ray_form_factors.csv"
    )

    # To view the plot, navigate to the `tests/results/` directory.
    diffraction.plot_diffraction_pattern(
        unit_cell,
        "ND",
        neutron_form_factors,
        x_ray_form_factors,
        wavelength=0.1,
        min_deflection_angle=20,
        max_deflection_angle=60,
        peak_width=0.1,
        line_width=1.0,
        file_path="tests/results/",
    )


def test_plot_superimposed_diffraction_patterns_normal_operation():
    """
    A unit test for the plot_superimposed_diffraction_patterns function. This unit test
    tests normal operation of the function.
    """
    basis = file_reading.read_basis("tests/data/NaCl_basis.csv")
    lattice = file_reading.read_lattice("tests/data/NaCl_lattice.csv")
    unit_cell_1 = crystal.UnitCell.new_unit_cell(basis, lattice)

    basis = file_reading.read_basis("tests/data/Cu_basis.csv")
    lattice = file_reading.read_lattice("tests/data/Cu_lattice.csv")
    unit_cell_2 = crystal.UnitCell.new_unit_cell(basis, lattice)

    neutron_form_factors = file_reading.read_neutron_scattering_lengths(
        "tests/data/neutron_scattering_lengths.csv"
    )
    x_ray_form_factors = file_reading.read_xray_form_factors(
        "tests/data/x_ray_form_factors.csv"
    )

    # To view the plot, navigate to the `tests/results/` directory.
    diffraction.plot_superimposed_diffraction_patterns(
        [(unit_cell_1, "ND"), (unit_cell_2, "ND")],
        neutron_form_factors,
        x_ray_form_factors,
        wavelength=0.1,
        min_deflection_angle=20,
        max_deflection_angle=60,
        peak_width=0.1,
        variable_wavelength=True,
        line_width=1.0,
        opacity=0.5,
        file_path="tests/results/",
    )
