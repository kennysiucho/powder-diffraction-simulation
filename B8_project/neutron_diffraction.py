import numpy as np
import scipy
from B8_project.utils import *
from B8_project.unit_cell import UnitCell

# TODO: full list of neutron scattering lengths (read from csv)
scattering_lengths = {
    8: 5.803,
    59: 4.58
}

class NeutronDiffraction:
    def __init__(self, unit_cell: UnitCell, wavelength: float):
        self.unit_cell = unit_cell
        self.wavelength = wavelength

    def basis_structure_factor(self, rlv: tuple[float, float, float]) -> np.complex128:
        s = 0 + 0j
        for atom in self.unit_cell.atoms:
            r = np.multiply(self.unit_cell.lattice_constants, atom.position)
            s += scattering_lengths[atom.atomic_number] * np.exp(1j * np.dot(rlv, r))
        return s

    @classmethod
    def peak_function(cls, x: [float, np.array], location: float, height: float, width: float=0.5):
        return height * np.exp(-0.5 * (x - location)**2 / width**2)

    def diffraction_pattern(self, min_angle: float=0.0, max_angle: float=180.0):
        # TODO: effect of rlv_range?
        rlv_range = 10
        peaks = []
        # TODO: vectorize
        for h in range(-rlv_range, rlv_range + 1):
            for k in range(-rlv_range, rlv_range + 1):
                for l in range(-rlv_range, rlv_range + 1):
                    if (h, k, l) == (0, 0, 0): continue

                    rlv = reciprocal_lattice_vector(self.unit_cell.lattice_constants, h, k, l)
                    s = self.basis_structure_factor(rlv)
                    intensity = np.abs(s)**2

                    rlv_length = np.linalg.norm(rlv)
                    sin_theta = self.wavelength * rlv_length / (4 * np.pi)
                    if sin_theta <= 1.0:
                        peaks.append([np.degrees(2 * np.arcsin(sin_theta)), intensity])
                    # TODO: this double counts?
                    # n = 1
                    # while n * sin_theta <= 1:
                    #     # print(h, k, l, n, sin_theta)
                    #     peaks.append([np.degrees(2 * np.arcsin(n * sin_theta)), intensity])
                    #     n += 1

        two_thetas = np.linspace(min_angle, max_angle, 200)
        intensities = np.zeros(two_thetas.shape)

        for peak in peaks:
            intensities += self.peak_function(two_thetas, peak[0], peak[1])

        intensities /= np.max(intensities)

        return two_thetas, intensities