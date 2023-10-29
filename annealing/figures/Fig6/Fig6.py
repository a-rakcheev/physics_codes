# DQA of SK Ising with +/- J (bimodal) bonds with degeneracy in the ground state
import numpy as np
import h5py
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.colors as colors

N = 20                               # system size
Tl = [1.0, 10.0, 50.0, 150.0]                              # duration
dt = 0.01                           # time step
steps = 100                         # num of measurements
instances = [9]

states = 1000                         # maximum number of states to keep (should be <= 2 ** (N - 1))

lanczos_order = 40
plot_order = 4

plt.figure(1, figsize=(7.5, 5))
sl = np.linspace(0., 1., steps + 1)

prefix = "C:/Users/rakche_a-adm/Downloads/data_sk/data/"

for inst in instances:
    print("inst:", inst)

    for k, T in enumerate(Tl):

        file = h5py.File(prefix + "quantum_annealing_nofield_states_and_lanczos.hdf5", "r")
        series_group = file["/N" + str(N) + "/inst" + str(inst) + "/series_states=" + str(states) + "_dt=" + str(dt).replace(".", "-") + "_lanczos_order=" + str(lanczos_order)]

        ev_inst = series_group["ev_inst"][:, :, int(T) - 1]
        amps_inst = series_group["amps_inst"][:, :, int(T) - 1]
        ev_comp = series_group["idx_comp"][:, :, int(T) - 1]
        amps_comp = series_group["amps_comp"][:, :, int(T) - 1]

        # instantaneous basis
        plt.subplot(2, 2, k + 1)
        for n in range(min(lanczos_order, plot_order)):
            plt.plot(sl[1:], np.absolute(amps_inst[1:, n]) ** 2, lw=1, label=r"$i=" + str(n) +r"$", zorder=2)

        plt.grid(zorder=1, color="grey", lw=1)
        plt.legend(fontsize=10, framealpha=1)

        if k == 2 or k == 3:
            plt.xlabel(r"$s$", fontsize=13)

        if k == 0 or k == 2:
            plt.ylabel(r"$P_{i}$", fontsize=13)

        plt.title(r"$T=" + str(int(T)) + r"$", fontsize=13)


    plt.subplots_adjust(wspace=0.2, hspace=0.35, left=0.075, right=0.975, bottom=0.1, top=0.95)

    plt.savefig("Fig6.pdf", format="pdf")
    plt.show()
