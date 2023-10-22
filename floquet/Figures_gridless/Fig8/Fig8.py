import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# parameters
L = 8
g = 4.0
start = 0.0
end = 2 * np.pi
res = 1000
taul = np.linspace(start, end, res)

plt.figure(1, figsize=(3.375, 3.75), constrained_layout=True)

data = np.load("data_freq_colored.npz")
angles_full = data["angles_full"]
energy_avg = data["energy_avg"]

cmap = "seismic"

for i, tau in enumerate(taul):
    plt.scatter(angles_full[i, :], np.full_like(angles_full[i, :], tau), c=energy_avg[i, :], s=2, marker="o", cmap=cmap, vmin=-0.25, vmax=0.25)

clb = plt.colorbar()
clb.ax.set_title(r"$\epsilon_{0}$", fontsize=12)

# plt.grid()

plt.xlabel(r"$\theta$", fontsize=12)
plt.xticks([-np.pi, -0.5 * np.pi, 0, 0.5 * np.pi, np.pi],
           [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])

plt.ylabel(r"$\tau$", fontsize=12)
plt.yticks([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi],
           [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])

plt.xlim(-np.pi, np.pi)
plt.ylim(0., 2. * np.pi)

plt.savefig("Fig8.pdf", format="pdf")
# plt.show()