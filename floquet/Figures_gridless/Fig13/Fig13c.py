import numpy as np
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt

name_data = "ye_k=0_p=1_energy_dynamics_" + "x_p" + "_L=" + str(24) + ".txt"
data_x_p = np.loadtxt(name_data, delimiter=",")

name_data = "ye_k=0_p=1_energy_dynamics_" + "x_m" + "_L=" + str(24) + ".txt"
data_x_m = np.loadtxt(name_data, delimiter=",")


plt.figure(1, figsize=(2.3, 2.5), constrained_layout=True)

# data from real-time
plt.scatter(np.pi / data_x_m[:, 0], 1. / data_x_m[:, 1], s=12, marker="d", color="silver", label=r"$|x, - \rangle$", zorder=2)
plt.scatter(np.pi / data_x_p[:, 0], 1. / data_x_p[:, 1], s=12, marker="o", color="black", label=r"$|x, + \rangle$", zorder=2)

colors = ["skyblue", "orange", "mediumorchid", "olive"]

start = 0.0
end = 1.0
res = 10000000
dE = 0.5

for j, L in enumerate([4, 6, 8, 10]):

    name = "ye_pbc_heating_rate_from_gaps_L" + str(L) + "_dE" + str(dE).replace(".", "-")
    data = np.load(name + ".npz")
    wl = data["wl"]
    gamma_gauss = data["gamma_gauss"]

    # convert to cycles
    gamma_gauss = gamma_gauss * 2. * np.pi / wl
    plt.plot(wl, 1. / gamma_gauss, label=r"Gap, $L=" + str(L) + r"$", color=colors[j], lw=1, zorder=3)


# fgr
name = "ye_pbc_fgr_from_ed_L" + str(18) + "_dE" + str(dE).replace(".", "-") + ".npz"
data = np.load(name)
wl_fgr = data["wl"]
gamma_gauss_fgr = data["gamma_gauss"][0, :]

# convert to cycles
gamma_gauss_fgr = gamma_gauss_fgr * 2. * np.pi / wl_fgr
plt.plot(wl_fgr, 1. / gamma_gauss_fgr, label=r"FGR, $L=" + str(18) + r"$", ls="--", color="lightcoral", lw=1, zorder=1)

# plt.grid(True)
plt.legend(loc="lower right", framealpha=1, fontsize=5, handlelength=1.0, ncol=2)
plt.xlabel(r"$\omega$", fontsize=12)
plt.ylabel(r"heating time / cycles", fontsize=12)

plt.yscale("log")
plt.xlim(np.pi, 12)
plt.ylim(10.0, 1.e5)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.gcf().text(0.03, 0.03, r"$(\mathrm{c})$", fontsize=12)
plt.savefig("Fig13c.pdf", format="pdf")
# plt.show()