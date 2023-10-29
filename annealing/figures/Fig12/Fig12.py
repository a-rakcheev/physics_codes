import sys
sys.path.append("E:/Dropbox/codebase/")
sys.path.append("/home/artem/Dropbox/codebase/")

import numpy as np
import hamiltonians_32 as ham32
from scipy.optimize import curve_fit
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt

# fit to line
def lin_fit(log_times, log_corrs):

    def func(x, a, b):
        return a + b * x
    
    params, cov = curve_fit(func, log_times, log_corrs)
    return params, cov

# parameters
Nl = [4, 6, 8]   
MCSl = np.array([1, 2])

steps = 100                    # num of measurements
R = 1000000
nrep = 8
beta = 10.0
step_size = 1
time_steps = 100
instances = [1]                     # instance

prefix = "C:/Users/rakche_a-adm/Downloads/data_sk/data/"

for inst in instances:

    plt.figure(1, figsize=(10, 6))
    labelsize = 15

    for n, N in enumerate(Nl):
        dim = 2 ** (N - 1)
        states = min(100, dim)

        # create Hamiltonians and states
        # input for ZZ hamiltonian
        positions_zz = []
        labels_zz = []

        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):

                positions_zz.append([i, j])
                labels_zz.append([3, 3])

        positions_zz = np.array(positions_zz, dtype=np.int64)
        labels_zz = np.array(labels_zz, dtype=np.int8)

        # bonds from instance file
        name_bonds = prefix + "sk_ising_bimodal_N" + str(N) + "_unique_inst" + str(inst) + ".txt"
        bonds = np.loadtxt(name_bonds)

        h_ising = ham32.operator_sum_real_diag(N, 2, bonds, positions_zz, labels_zz).diagonal()

        # transform to spin flip invariant space
        h_ising = h_ising[0:2 ** (N - 1)]

        probs_final = np.zeros((len(MCSl), states))
        idx_final = np.zeros((len(MCSl), states), dtype=np.int64)

        for k, MCS in enumerate(MCSl):

            # filenames
            name = prefix + "sqa_N" + str(N) + "/N" + str(N) + "/sqa_states_short_inst" + str(inst) + "_MCS" + str(MCS) + "_R=" + str(R) + "_n=" + str(nrep) + "_beta=" + str(beta).replace(".", "-")
            data = np.load(name + ".npz")

            probs_max = data["prob"]
            idx_max = data["idx"]

            probs_final[k, :] = probs_max[-1, :, 0]
            idx_final[k, :] = idx_max[-1, :, 0]

        plt.subplot(2, 3, n + 1)

        for t in [0, 1]:
            plt.plot(np.arange(1, states + 1, 1), dim * probs_final[t, :], label=r"$\mathrm{MCS}=" + str(MCSl[t]) + r"$", ls="", marker="o", markersize=2, zorder=2)

        plt.legend(fontsize=12)
        plt.yscale("log")

        plt.xticks([], [])
        plt.xlabel(r"$z$", fontsize=labelsize)
        if n == 0:
            plt.ylabel(r"$\mathcal{D} \rho_{z}$", fontsize=labelsize)

        plt.title(r"$N=" + str(N) + r"$", fontsize=labelsize)


        # plot dist based on energy
        plt.subplot(2, 3, n + 4)
        for t in [0, 1]:
            plt.plot(h_ising[idx_final[t, :]], dim * probs_final[t, :], label=r"$\mathrm{MCS}=" + str(MCSl[t]) + r"$", ls="", marker="o", markersize=3, zorder=2)


        plt.legend(fontsize=12)
        plt.yscale("log")
 
        plt.xlabel(r"$E_{z}$", fontsize=labelsize)
        if n == 0:
            plt.ylabel(r"$\mathcal{D} \rho_{z}$", fontsize=labelsize)


    plt.subplots_adjust(wspace=0.45, top=0.95, bottom=0.08, left=0.1, right=0.99)
    plt.savefig("Fig12.pdf", format="pdf")
    plt.show()
