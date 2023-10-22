import numpy as np
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

import matplotlib.pyplot as plt


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


# parameters
gamma = 20.0
Tl = np.arange(10., 102.5, 2.5)
dt = 1.e-6
fl_idx = [0, 5, 10, 15, 20]
fl = np.linspace(0., 1., 21)

fidelities = np.zeros((len(Tl), len(fl_idx)))
en_var = np.zeros((len(Tl), len(fl_idx)))

for i, T in enumerate(Tl):
    data = np.load("lz_scaling_f_gamma=" + str(gamma).replace(".", "-") + "_T=" + str(T).replace(".", "-")
                   + "_dt=" + str(dt).replace(".", "-") + ".npz")

    fidelity = data["fidelity"]
    energy_variance = data["energy_variance"]
    fidelities[i, :] = fidelity[fl_idx]
    en_var[i, :] = energy_variance[fl_idx]

    # print(T)
    # print(1. - fidelity)
    # print(energy_variance)


plt.figure(1, (6, 3.5))

plt.subplot(1, 2, 1)
plt.plot(Tl, 1. - fidelities, marker="o", ls="", markersize=4)

plt.legend()
plt.grid()

plt.xscale("log")
plt.yscale("log")
plt.ylim(1.e-10, 1.e-1)

plt.xlabel(r"$T$", fontsize=12)
plt.ylabel(r"$P_{ex}$", fontsize=12)

plt.subplot(1, 2, 2)
# plt.plot(fl, energy_variance, color="darkred", marker="o", ls="", markersize=4)

plt.plot(Tl, en_var, marker="o", ls="", markersize=4)

plt.legend()
plt.grid()

plt.xscale("log")
plt.yscale("log")
plt.ylim(1.e-10, 1.e-1)

plt.xlabel(r"$T$", fontsize=12)
plt.ylabel(r"$\Delta^{2} E$", fontsize=12)

plt.subplots_adjust(wspace=0.4, bottom=0.2)
# plt.savefig("lz_partial_drive_T=" + str(T).replace(".", "-") + ".pdf", format="pdf")
plt.show()
