import numpy as np
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# parameters
L = 8
start = 2 * np.pi
end = 0.
res = 1000
taul = np.linspace(start, end, res)


data = np.load("data_colored.npz")
angles_full = data["angles_full"]
energy_avg = data["energy_avg"]

plt.figure(1, figsize=(4, 5), constrained_layout=True)
cmap = "seismic"
for i, tau in enumerate(taul):
    plt.scatter(angles_full[i, :], np.full_like(angles_full[i, :], tau), c=energy_avg[i, :],
                s=0.5, marker="o", cmap=cmap, vmin=-0.25, vmax=0.25, zorder=2)

clb = plt.colorbar()
clb.ax.set_title(r"$\epsilon_{0}$", fontsize=12)
# plt.grid()

plt.xlabel(r"$\theta$", fontsize=12)
plt.xticks([-np.pi, -0.5 * np.pi, 0, 0.5 * np.pi, np.pi],
           [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"], fontsize=12)

plt.ylabel(r"$\tau$", fontsize=12)
plt.yticks([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi],
           [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"], fontsize=12)

plt.xlim(-np.pi, np.pi)
plt.ylim(0., 2. * np.pi)


# add circle
coords = [[-2.65, 0.525]]
for coord in coords:

    plt.scatter(coord[0], coord[1], s=75, marker="o", edgecolor="black", color="white", zorder=1)

plt.savefig("Fig1.pdf", format="pdf")
# plt.show()