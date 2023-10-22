import numpy as np
import scipy.sparse as sp
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# parameters
L = 6
start = 0.0
end = 0.75
res = 750

taul = np.linspace(start, end, res)

data = np.load("data_colored.npz")
angles_full = data["angles_full"]
spin_flip = data["spin_flip"]

plt.figure(1, figsize=(2.3, 2.5), constrained_layout=True)

cmap = "Spectral"
for i, tau in enumerate(taul):
    scat = plt.scatter(angles_full[i, :], np.full_like(angles_full[i, :], tau), c=spin_flip[i, :], s=0.5, marker="o",
                cmap=cmap, vmin=-1, vmax=1)

ax = plt.gca()
cb = plt.colorbar(scat, ax=ax, pad=0.075, orientation="vertical", aspect=30, ticks=[-1.0, -0.5, 0., 0.5, 1.0])
cb.ax.tick_params(labelsize=10)
cb.ax.set_title(r"$\langle F \rangle $", fontsize=12)

plt.grid(False)
plt.xlabel(r"$\theta$", fontsize=12)
plt.ylabel(r"$\tau$", fontsize=12)

plt.xticks([-np.pi, 0, np.pi],
           [r"$-\pi$", r"$0$", r"$\pi$"], fontsize=12)
plt.yticks([0.0, 0.25, 0.5, 0.75], fontsize=12)

plt.xlim(-np.pi, np.pi)
plt.ylim(0., 0.75)

plt.gcf().text(0.03, 0.03, r"$(\mathrm{a})$", fontsize=12)

plt.savefig("Fig13a.pdf", format="pdf")
# plt.show()
