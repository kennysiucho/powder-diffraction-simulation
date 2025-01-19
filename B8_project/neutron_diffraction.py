from B8_project.unit_cell import Atom, UnitCell
from B8_project.utils import *
import numpy as np
import time

scattering_lengths = {
    8: 5.803,
    59: 4.58
}

class NeutronDiffractionRunStats:
    def __init__(self):
        self.accepted_data_points = 0
        self.angles_accepted = 0
        self.angles_accepted_avg_intensity = 0.0
        self.total_trials = 0
        self.start_time_ = time.time()
        self.prev_print_time_ = 0
        self.microseconds_per_trial = 0.0

    def angle_accepted_update(self, intensity: float):
        self.angles_accepted_avg_intensity = ((self.angles_accepted_avg_intensity * self.angles_accepted + intensity) /
                                              (self.angles_accepted + 1))
        self.angles_accepted += 1

    def update(self):
        if self.total_trials == 0: return
        self.microseconds_per_trial = (time.time() - self.start_time_) / self.total_trials * 1000000

    def __str__(self):
        self.update()
        return "".join([f"{key}={val:.1f} | " if key[-1] != '_' else "" for key, val in self.__dict__.items()])

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


        stats = NeutronDiffractionRunStats()

        while stats.accepted_data_points < N_trials:
            stats.total_trials += 1
            if time.time() - stats.prev_print_time_ > 5:
                stats.prev_print_time_ = time.time()
                print(stats)

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

            stats.angle_accepted_update(intensity)
            if intensity > 50 * stats.angles_accepted_avg_intensity:
                two_thetas[stats.accepted_data_points] = np.degrees(two_theta)
                intensities[stats.accepted_data_points] = intensity
                stats.accepted_data_points += 1

        intensities /= np.max(intensities)

        return two_thetas, intensities