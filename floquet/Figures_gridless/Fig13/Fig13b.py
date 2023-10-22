import numpy as np
import scipy.sparse as sp
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt

# parameters
L = 10
size = 78
start = 0.0
end = 1.0
res = 10000000
tau_cut = 0.75                   # final tau in plot

data = np.load("data_transfer.npz")
widths_filtered = data["widths"]
tau_min_filtered = data["times"]
widths_F = data["widths_F"]

plt.figure(1, figsize=(2.3, 2.5), constrained_layout=True)

cmap = "jet"
scat = plt.scatter(widths_filtered, tau_min_filtered, marker="o", s=5, c=widths_F, cmap=cmap, vmin=0., vmax=2.)

ax = plt.gca()
cb = plt.colorbar(scat, ax=ax, pad=0.075, orientation="vertical", aspect=20, ticks=[0., 1., 2.])
cb.ax.tick_params(labelsize=10)
cb.ax.set_title(r"$\Delta F_{c}$", fontsize=12)
plt.axvline(2 * np.pi / size, color="black", ls="--", lw=1)

plt.grid(False)
plt.xlabel(r"$\Delta_{c}$", fontsize=12)
plt.ylabel(r"$\tau$", fontsize=12)

plt.xscale("log")
plt.xlim(1.e-10, 2.0)
plt.ylim(0.01, tau_cut)
plt.xticks([1.e-10, 1.e-6, 1.e-2], fontsize=12)
plt.yticks([0., 0.25, 0.5, 0.75], fontsize=12)

plt.gcf().text(0.03, 0.03, r"$(\mathrm{b})$", fontsize=12)
plt.savefig("Fib13b.pdf", format="pdf")
# plt.show()