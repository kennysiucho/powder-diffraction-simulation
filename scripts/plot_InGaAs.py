import numpy as np
import matplotlib.pyplot as plt

two_thetas_gaas = np.loadtxt("runs/two_thetas GaAs 10M trials.txt")
intensities_gaas = np.loadtxt("runs/intensities GaAs 10M trials.txt")
two_thetas_ingaas = np.loadtxt("runs/two_thetas InGaAs 2.txt")
intensities_ingaas = np.loadtxt("runs/intensities InGaAs 2.txt")

plt.plot(two_thetas_gaas, intensities_gaas, '-', color='r', label=r"$GaAs$")
plt.plot(two_thetas_ingaas, intensities_ingaas, '-', color='b',
         label=r"$In_{0.4}Ga_{0.6}As$")
plt.xlabel("Scattering angle (2θ) (deg)")
plt.ylabel("Normalized intensity")
plt.title(r"$GaAs$ vs. $In_{0.4}Ga_{0.6}As$ X-ray Diffraction Spectrum")
plt.legend()
plt.show()
