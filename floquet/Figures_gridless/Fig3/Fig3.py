# plot energy density dynamics for different system sizes

import numpy as np
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt

# parameters
Ll = [8, 12, 16, 20, 24, 28]                                        # system sizes
state_name = "x_p"                                                  # state x/y/z plus/minus (p/m)
tau = 1.3                                                           # half-period
cycles = [10000, 10000, 10000, 10000, 5000, 5000]                  # cycles
cycles_max = 5000                                                   # plot until cycles_max

plt.figure(1, figsize=(3.375, 2.5), constrained_layout=True)
for i, L in enumerate(Ll):

    # read data
    name = "hhz_pbc_k=0_p=1_energy_dynamics_" + state_name + "_L=" + str(L) + "_tau" \
           + str(tau).replace(".", "-") + "_cycles=" + str(cycles[i]) + ".npz"

    data = np.load(name)
    en_x = data["en_x"]
    en_z = data["en_z"]
    en_avg = 0.5 * (en_x + en_z)

    # plot
    plt.plot(np.absolute(en_avg[:cycles_max]), lw=1, label=r"$L=" + str(L) + r"$")

    # plt.grid(True)
    plt.legend(fontsize=9)

    plt.xlim(0, cycles_max)
    plt.yscale("log")
    plt.ylim(1.e-6, 1.0)
    plt.xticks(fontsize=12)
    plt.yticks([1.e-6, 1.e-4, 1.e-2, 1.], fontsize=12)

    plt.xlabel("cycles", fontsize=12)
    plt.ylabel(r"$|\epsilon_{0}|$", fontsize=12)


plt.savefig("Fig3.pdf")
# plt.show()

