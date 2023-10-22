import numpy as np
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt


L_d = 26
cycles = 1000
g = 4.0

start = 0.0
end = np.pi
res = 4000000
dE = 0.3

plt.figure(1, figsize=(3.375, 2.5), constrained_layout=True)

# heating from real time
name_data = "hhz_freq_k=0_p=1_energy_dynamics_" + "x_p" + "_L=" + str(L_d) \
                   + "_g=" + str(g).replace(".", "-") + ".txt"

data_x_p = np.loadtxt(name_data, delimiter=",")

name_data = "hhz_freq_k=0_p=1_energy_dynamics_" + "z_p" + "_L=" + str(L_d) \
                   + "_g=" + str(g).replace(".", "-") + ".txt"

data_z_p = np.loadtxt(name_data, delimiter=",")

name_data = "hhz_freq_k=0_p=1_energy_dynamics_" + "z_m" + "_L=" + str(L_d) \
                   + "_g=" + str(g).replace(".", "-") + ".txt"

data_z_m = np.loadtxt(name_data, delimiter=",")

gamma_avg_x_p = data_x_p[:, 4]
gamma_avg_z_p = data_z_p[:, 4]
gamma_avg_z_m = data_z_m[:, 4]

plt.scatter(data_x_p[:, 0], 1. / gamma_avg_x_p, s=12, marker="o", color="black",
         label=r"$| x, + \rangle$", zorder=1)
plt.scatter(data_z_p[:, 0], 1. / gamma_avg_z_p, s=12, marker="s", color="gray",
         label=r"$| z, + \rangle$", zorder=1)
plt.scatter(data_z_m[:, 0], 1. / gamma_avg_z_m, s=12, marker="d", color="silver",
         label=r"$| z, - \rangle$", zorder=1)


colors = ["skyblue", "orange", "mediumorchid", "olive"]


boundaries = dict()
boundaries["4"] = 0
boundaries["6"] = 0
boundaries["8"] = 120
boundaries["10"] = 135


for i, L in enumerate([6, 8, 10]):

    bound = boundaries[str(L)]

    name = "hhz_freq_heating_rate_from_gaps_L" + str(L) + "_g=" + str(g).replace(".", "-") \
           + "_dE" + str(dE).replace(".", "-") + ".npz"

    data = np.load(name)
    wl = data["wl"]
    gamma_gauss = data["gamma_gauss"]

    # convert to cycle time Gamma_cyc = Gamma * T
    gamma_gauss_cyc = gamma_gauss * 2. * np.pi / wl

    plt.plot(np.pi / wl[0:bound + 1], 1. / gamma_gauss_cyc[0:bound + 1], color=colors[i], ls=":", lw=1, zorder=2)
    plt.plot(np.pi / wl[bound:], 1. / gamma_gauss_cyc[bound:], color=colors[i], ls="-", lw=1, label=r"Gap, $L=" + str(L) + r"$", zorder=2)

    # plt.grid("True")
    plt.legend(loc="best", framealpha=1, fontsize=9, handlelength=1.0, ncol=2)
    plt.xlabel(r"$\tau$", fontsize=12)
    plt.ylabel(r"heating time / cycles", fontsize=12)

    plt.yscale("log")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(5, 5.e5)
    plt.xlim(1.025, 1.725)


plt.savefig("Fig10.pdf", format="pdf")
# plt.show()