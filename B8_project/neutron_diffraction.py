from B8_project.unit_cell import Atom, UnitCell
from B8_project.utils import *
import numpy as np
import time

scattering_lengths = {
    8: 5.803,
    59: 4.58
}



class NeutronDiffraction:
    def __init__(self, unit_cell: UnitCell, wavelength: float):
        self.unit_cell = unit_cell
        self.wavelength = wavelength

    def calculate_diffraction_pattern(self, N_trials: int = 5000):
        k = 2 * np.pi / self.wavelength
        two_thetas = np.zeros(N_trials)
        intensities = np.zeros(N_trials)

        expand_N = 9
        expanded_pos = np.vstack(np.mgrid[0:expand_N, 0:expand_N, 0:expand_N].astype(np.float64)).reshape(3, -1).T
        np.multiply(expanded_pos, self.unit_cell.lattice_constants, out=expanded_pos)
        print(expanded_pos)


        i = 0
        prev_print = -1
        cnt = 0
        avg = 0.0
        start_time = time.time()

        while i < N_trials:
            if i != prev_print and i % (N_trials // 100) == 0:
                prev_print = i
                print(f"Data points: {i} | Angle in range: {cnt}, Average intensity={avg}")

            structure_factor = 0 + 0j
            k_vec = k * np.array(random_uniform_unit_vector(3))
            k_prime = k * np.array(random_uniform_unit_vector(3))
            scattering_vec = k_prime - k_vec

            for atom in self.unit_cell.atoms:
                r = np.multiply(atom.position, self.unit_cell.lattice_constants) + expanded_pos
                structure_factor += scattering_lengths[atom.atomic_number] * np.sum(np.exp(1j * np.dot(r, scattering_vec)))

            two_theta = np.arccos(np.dot(k_vec, k_prime) / k**2)
            if two_theta < np.radians(15) or two_theta > np.radians(60): continue
            intensity = abs(structure_factor)**2

            avg = (avg * cnt + intensity) / (cnt + 1)
            cnt += 1
            if intensity > 50 * avg:
                two_thetas[i] = two_theta
                intensities[i] = intensity
                i += 1

            # for atom in self.unit_cell.atoms:
            #     for cell in expanded_pos:
            #         r = np.multiply(atom.position, self.unit_cell.lattice_constants) + cell
            #         structure_factor += scattering_lengths[atom.atomic_number] * np.exp(1j * np.dot(scattering_vec, r))
            # two_thetas[i] = np.arccos(np.dot(k_vec, k_prime) / k**2)
            # intensities[i] = abs(structure_factor)**2

        return np.degrees(two_thetas), intensities / np.max(intensities)