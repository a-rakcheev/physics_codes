import sys
sys.path.append("/home/artem/Dropbox/codebase/")
sys.path.append("E:/Dropbox/codebase/")
sys.path.append("C:/Users/ARakc/Dropbox/codebase/")
sys.path.append("C:/Users/rakche_a-adm/Dropbox/codebase/")


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
Nl = [10, 15, 20]   
Tl = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
dtl = np.array([0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001])

steps = 100                    # num of measurements
states = 100
step_size = 1
time_steps = 100
instances = [1]                     # instance

prefix = "C:/Users/rakche_a-adm/Downloads/data_sk/data/"

for inst in instances:

    plt.figure(1, figsize=(10, 6))
    labelsize = 15

    for n, N in enumerate(Nl):
        dim = 2 ** (N - 1)

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
  
        corrs_final = np.zeros((len(Tl), N * (N - 1) // 2))
        amps_final = np.zeros((len(Tl), states), dtype=np.complex128)
        idx_final = np.zeros((len(Tl), states), dtype=np.int64)

        for k, T in enumerate(Tl):
            dt = dtl[k]

            # filenames
            name = prefix + "short_qa/sk_ising_bimodal_nofield_corrs_and_states_N" + str(N) + "_inst" + str(inst) + "_T=" + str(T).replace(".", "-") + "_dt=" + str(dt).replace(".", "-") + "_steps=" + str(steps)
            data = np.load(name + ".npz")

            corrs_zz = data["corrs"]
            amps_max = data["amps"]
            idx_max = data["idx"]

            corrs_final[k, :] = corrs_zz[-1, :]
            amps_final[k, :] = amps_max[-1, :]
            idx_final[k, :] = idx_max[-1, :]

        plt.subplot(2, 3, n + 1)

        for t in [4, 9]:
            plt.plot(np.arange(1, states + 1, 1), dim * np.absolute(amps_final[t, :]) ** 2, label=r"$T=" + str(Tl[t]) + r"$", ls="", marker="o", markersize=2, zorder=2)

        plt.legend(fontsize=12)
        plt.yscale("log")

        plt.xticks([], [])
        plt.xlabel(r"$z$", fontsize=labelsize)
        if n == 0:
            plt.ylabel(r"$\mathcal{D} \rho_{z}$", fontsize=labelsize)

        plt.title(r"$N=" + str(N) + r"$", fontsize=labelsize)


        # plot dist based on energy
        plt.subplot(2, 3, n + 4)
        for t in [4, 9]:
            plt.plot(h_ising[idx_final[t, :]], dim * np.absolute(amps_final[t, :]) ** 2, label=r"$T=" + str(Tl[t]) + r"$", ls="", marker="o", markersize=3, zorder=2)
            plt.plot(h_ising[idx_final[t, :]],  (1 - (Tl[t] ** 2 / 3) * h_ising[idx_final[t, :]]), color="black", lw=1, zorder=1)

        plt.legend(fontsize=12)
        plt.yscale("log")

        plt.xlabel(r"$E_{z}$", fontsize=labelsize)
        if n == 0:
            plt.ylabel(r"$\mathcal{D} \rho_{z}$", fontsize=labelsize)


    plt.subplots_adjust(wspace=0.45, top=0.95, bottom=0.08, left=0.1, right=0.99)
    plt.savefig("Fig11.pdf", format="pdf")
    plt.show()
