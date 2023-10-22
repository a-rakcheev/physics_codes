import numpy as np
from scipy.optimize import curve_fit

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

import matplotlib.pyplot as plt


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


# parameters
gamma = 20.
T = 95.
dt = 1.e-6
fl = np.linspace(0., 1., 21)

data = np.load("lz_scaling_f_gamma=" + str(gamma).replace(".", "-") + "_T=" + str(T).replace(".", "-")
               + "_dt=" + str(dt).replace(".", "-") + ".npz")

fidelity = data["fidelity"]
energy_variance = data["energy_variance"]
popt_fidelity, pconv_fidelity = curve_fit(quadratic, fl, (1 - fidelity) / (1 - fidelity[0]))
popt_en, pconv_en = curve_fit(quadratic, fl, energy_variance / energy_variance[0])

# print(popt_fidelity)
# print(popt_en)

plt.figure(1, (6, 3.5))

plt.subplot(1, 2, 1)
# plt.plot(fl, (1. - fidelity), color="navy", marker="o", ls="", markersize=4)

plt.plot(fl, (1. - fidelity) / (1. - fidelity[0]), color="navy", marker="o", ls="", markersize=4)
plt.plot(fl, (1 - fl) ** 2, lw=1, ls="-", color="black", label=r"$(1 - f)^2$")

plt.legend()
plt.grid()

plt.xlabel(r"$f$", fontsize=12)
plt.ylabel(r"$\tilde{P}_{ex}$", fontsize=12)

plt.subplot(1, 2, 2)
# plt.plot(fl, energy_variance, color="darkred", marker="o", ls="", markersize=4)

plt.plot(fl, energy_variance / energy_variance[0], color="darkred", marker="o", ls="", markersize=4)
plt.plot(fl, (1 - fl) ** 2, lw=1, ls="-", color="black", label=r"$(1 - f)^2$")

plt.legend()
plt.grid()

plt.xlabel(r"$f$", fontsize=12)
plt.ylabel(r"$\Delta^{2} \tilde{E}$", fontsize=12)

plt.subplots_adjust(wspace=0.4, bottom=0.2)
# plt.savefig("lz_partial_drive_T=" + str(T).replace(".", "-") + ".pdf", format="pdf")
plt.show()
