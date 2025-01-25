from B8_project.crystal import UnitCell
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
        self.avg_intensity_cnt_ = 0
        self.avg_intensity = 0.0
        self.total_trials = 0
        self.start_time_ = time.time()
        self.prev_print_time_ = 0
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

        batch_trials = 10000
        while stats.accepted_data_points < N_trials:

            if time.time() - stats.prev_print_time_ > 5:
                stats.prev_print_time_ = time.time()
                print(stats)

            structure_factors = np.zeros(batch_trials, dtype=np.complex128)
            k_vecs = k * random_uniform_unit_vectors(batch_trials, 3)
            k_primes = k * random_uniform_unit_vectors(batch_trials, 3)
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