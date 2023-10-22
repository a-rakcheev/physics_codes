import numpy as np
from matplotlib import rc
rc('text', usetex=True)

import matplotlib.pyplot as plt

name_data = "hhz_k=0_p=1_energy_dynamics_" + "x_p" + "_L=" + str(28) + ".txt"
data_x_p = np.loadtxt(name_data, delimiter=",")
name_data = "hhz_k=0_p=1_energy_dynamics_" + "z_p" + "_L=" + str(28) + ".txt"
data_z_p = np.loadtxt(name_data, delimiter=",")
name_data = "hhz_k=0_p=1_energy_dynamics_" + "z_m" + "_L=" + str(28) + ".txt"
data_z_m = np.loadtxt(name_data, delimiter=",")


plt.figure(1, figsize=(3.375, 2.75), constrained_layout=True)
plt.scatter(data_x_p[:, 0], 1. / data_x_p[:, 3], s=12, marker="o", color="black", label=r"$|x, + \rangle$", zorder=2)
plt.scatter(data_z_p[:, 0], 1. / data_z_p[:, 3], s=12, marker="s", color="gray", label=r"$|z, + \rangle$", zorder=2)
plt.scatter(data_z_m[:, 0], 1. / data_z_m[:, 3], s=12, marker="d", color="silver", label=r"$|z, - \rangle$", zorder=2)


Ll = [6, 8, 10, 12]
g = 1.0
dE = 0.1

colors = ["skyblue", "orange", "mediumorchid", "olive"]

ls = [(0, ()), (0, ()), (0, ()), (0, ())]
ls2 = [":", ":",  ":",  ":"]

idx_left = [210, 215, 225, 240]
idx_right = [160, 155, 142, 137]

idx_res = [499, 499, 425, 400]
for i, L in enumerate(Ll):

    name = "hhz_fgr_from_gaps_with_energy_transfer_" + "0" + "_L" + str(L) + "_dE" \
           + str(dE).replace(".", "-") + "_g" + str(g).replace(".", "-")

    data = np.load(name + ".npz")
    wl = data["wl"]
    gamma_gauss = data["gamma_gauss"]

    taul = np.pi / wl
    fgr = 1. / (2 * gamma_gauss * taul)
    plt.plot(taul[idx_left[i]:idx_res[i]], fgr[idx_left[i]:idx_res[i]], label=r"Gap, $L=" + str(L) + r"$", lw=1, color=colors[i], zorder=3, ls=ls[i])
    plt.plot(taul[210:idx_left[i]], fgr[210:idx_left[i]], lw=1, color=colors[i], zorder=3, ls=":")

    # resolution
    plt.plot(taul[idx_res[i]:], fgr[idx_res[i]:], lw=1, color=colors[i], zorder=3, ls=":")

    name = "hhz_fgr_from_gaps_with_energy_transfer_" + "pi" + "_L" + str(L) + "_dE" \
           + str(dE).replace(".", "-") + "_g" + str(g).replace(".", "-")

    data = np.load(name + ".npz")
    wl = data["wl"]
    gamma_gauss = data["gamma_gauss"]

    taul = np.pi / wl
    fgr = 1. / (2 * gamma_gauss * taul)
    plt.plot(taul[0:idx_right[i]], fgr[0:idx_right[i]], lw=1, color=colors[i], ls=ls[i])
    plt.plot(taul[idx_right[i]:160], fgr[idx_right[i]:160], lw=1, color=colors[i], ls=ls2[i])


# fgr
L = 18
dE = 0.1
name = "hhz_fgr_from_ed_L" + str(L) + "_dE" + str(dE).replace(".", "-")
data = np.load(name + ".npz")
wl = data["wl"]
gamma_gauss = data["gamma_gauss"]
taul = np.pi / wl
fgr = 1. / (2 * gamma_gauss[0, :] * taul)

plt.plot(taul, fgr, color="lightcoral", label=r"FGR, $L=18$", lw=1, ls="--", zorder=1)

annsize = 6

# plt.grid()
lgnd = plt.legend(loc="upper center", framealpha=1, fontsize=6.5, handlelength=0.75, ncol=2, columnspacing=0.75)

plt.xlabel(r"$\tau$", fontsize=12)
plt.ylabel(r"heating time / cycles", fontsize=12)

plt.yscale("log")
plt.xlim(0.5, 3.0)
plt.ylim(0.1, 1.e7)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig("Fig7.pdf", format="pdf")
plt.show()