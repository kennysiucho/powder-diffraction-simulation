"""
MC Displacement
===========

This module contains the MCDisplacement class, a child class of MCArbitrary and
MCRandomOccupation. Adds capability to displace atoms by user-defined function.
"""
from typing import Mapping, Callable
import time
import numpy as np
import torch
from B8_project.crystal import UnitCell
from B8_project.mc_arbitrary_crystal import MCArbitraryCrystal
from B8_project.mc_random_occupation import MCRandomOccupation
from B8_project.form_factor import FormFactorProtocol


class MCDisplacement(MCArbitraryCrystal, MCRandomOccupation):
    """
    Child class of MCArbitrary and MCRandomOccupation. Adds capability to displace atoms
    by user-defined function: displace_func(positions, atomic_nums) -> modified_positions.
    """

    def __init__(self,
                 wavelength: float,
                 unit_cell: UnitCell,
                 atom_from: int,
                 atom_to: int,
                 probability: float,
                 displace_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
                 pdf: Callable[[np.ndarray], np.ndarray] = None,
                 min_angle_deg: float = 0.,
                 max_angle_deg: float = 180.):
        """
        Takes a function displace_func which takes in the positions and atomic numbers
        of all atoms in the crystal.
        """
        MCRandomOccupation.__init__(self, wavelength, unit_cell,
                                    atom_from, atom_to, probability,
                                    pdf, min_angle_deg, max_angle_deg)
        self._displace_func = displace_func or (lambda pos, nums: self.gaussian_displaced(
            pos, nums, sigma=0.05, atoms_to_displace=[atom_from, atom_to]
        ))
        self.unique_atomic_numbers = []
        for atom in self._unit_cell.atoms:
            if atom.atomic_number in self.unique_atomic_numbers:
                continue
            self.unique_atomic_numbers.append(atom.atomic_number)
        if atom_to not in self.unique_atomic_numbers:
            self.unique_atomic_numbers.append(atom_to)


    @staticmethod
    def gaussian_displaced(positions: np.ndarray,
                           atoms: np.ndarray,
                           sigma: float,
                           atoms_to_displace: list[int]) -> np.ndarray:
        displacement = np.zeros_like(positions)
        for i in range(len(atoms)):
            if atoms[i] in atoms_to_displace:
                displacement[i] = np.random.normal(scale=sigma, size=3)
        return positions + displacement

    def compute_intensities(self,
                            scattering_vecs: np.ndarray,
                            form_factors: Mapping[int, FormFactorProtocol]):
        """
        Computes the intensities for each scattering vector with profiling output.
        """
        start_total = time.perf_counter()

        if self._unit_cell_pos is None:
            raise ValueError(
                "_unit_cell_pos is None: You must call setup_cuboid_crystal"
                " or setup_spherical_crystal to define the shape of the "
                "crystal particle.")

        t0 = time.perf_counter()

        # Build crystal once per batch
        n_unit_cells = self._unit_cell_pos.shape[0]
        n_atoms_per_uc = len(self._unit_cell.atoms)
        n_uc_varieties = len(self._atomic_numbers_vars)
        n_atoms = n_unit_cells * n_atoms_per_uc

        random_indices = self._rng.choice(np.arange(n_uc_varieties),
                                          size=n_unit_cells,
                                          p=self._probs)
        uc_positions = self._unit_cell_pos[:, np.newaxis, :]  # (n_unit_cells, 1, 3)
        rel_atom_positions = self._atom_pos_in_uc[np.newaxis, :,
                             :]  # (1, n_atoms_per_uc, 3)
        atom_pos = (uc_positions + rel_atom_positions).reshape(n_atoms,
                                                               3)  # (n_atoms, 3)

        # Repeat atomic numbers by variant index
        atomic_nums_per_uc = np.array(self._atomic_numbers_vars, dtype=object)[
            random_indices]  # (n_unit_cells,)
        atomic_nums = np.concatenate(atomic_nums_per_uc)  # Flattened into (n_atoms,)

        # Displace and store
        t0_1 = time.perf_counter()
        atom_pos = self._displace_func(atom_pos, atomic_nums)
        t0_2= time.perf_counter()
        self.set_atoms_pos(atom_pos)
        self.set_atomic_nums(atomic_nums)
        # all_scattering_lengths = (n_atoms,)
        # scattering_vec.shape = (batch_trials, 3)
        # structure_factors.shape = (batch_trials, )
        # k•r[i, j] = scattering_vec[i][k] • all_atom_pos[j][k]

        t1 = time.perf_counter()

        device = torch.device("mps")

        scattering_vecs_t = torch.from_numpy(scattering_vecs).to(torch.float32).to(
            device)
        all_atom_pos_t = torch.from_numpy(self._all_atom_pos).to(torch.float32).to(
            device)
        all_atoms = self._all_atoms
        t2 = time.perf_counter()

        # --- Compute dot product k•r ---
        dot_products = torch.matmul(scattering_vecs_t, all_atom_pos_t.T)
        t3 = time.perf_counter()

        # --- Evaluate form factors (real only) ---
        ff_real_parts = {
            atomic_number: form_factors[atomic_number].evaluate_form_factors_torch(
                scattering_vecs_t)
            for atomic_number in self.unique_atomic_numbers
        }
        t3_5 = time.perf_counter()  # ← New timing point

        # Step 1: Evaluate and stack once
        ordered_atomic_numbers = self.unique_atomic_numbers  # e.g. [6, 8, 12]
        ff_tensor = torch.stack(
            [ff_real_parts[Z] for Z in ordered_atomic_numbers], dim=1
        )  # shape: (n_kvecs, n_unique_Z)

        # Step 2: Build atom index mapping
        Z_to_index = {Z: i for i, Z in enumerate(ordered_atomic_numbers)}
        atom_indices = torch.tensor([Z_to_index[Z] for Z in all_atoms],
                                    dtype=torch.long, device=device)

        # Step 3: Use gather to select form factors per atom
        ff_matrix = ff_tensor.gather(1,
                                     atom_indices.unsqueeze(0).expand(ff_tensor.size(0),
                                                                      -1))
        t4 = time.perf_counter()

        # --- Compute exp(i k•r) = cos(k•r) + i sin(k•r) ---
        # Only real part of form factor used → multiply with cos(k•r)
        cos_theta = torch.cos(dot_products)
        exp_terms = ff_matrix * cos_theta

        # Sum over atoms and square magnitude (imaginary part is 0)
        structure_factors = torch.sum(exp_terms, dim=1)
        intensities = structure_factors ** 2
        t5 = time.perf_counter()

        result = intensities.cpu().numpy()

        # print(f"\n--- Timing breakdown ---")
        # print(f"  Atom generation/setup:      {t0_1 - t0:.4f} sec")
        # print(f"  Atom displacement:          {t0_2 - t0_1:.4f} sec")
        # print(f"  Tensor conversion:          {t2 - t1:.4f} sec")
        # print(f"  Dot product (GPU):          {t3 - t2:.4f} sec")
        # print(f"  Form factor eval only:      {t3_5 - t3:.4f} sec")
        # print(f"  Stack form factors:         {t4 - t3_5:.4f} sec")
        # print(f"  Trig, mult, sum (GPU):      {t5 - t4:.4f} sec")
        # print(f"  Total compute time:         {t5 - start_total:.4f} sec\n")

        return result

