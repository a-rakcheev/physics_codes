import numpy as np
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt

# parameters
L = 10
size = 78
g = 4.0
start = 0.0
end = np.pi
res = 4000000

# transfer date
name = "hhz_freq_transfers_L=" + str(L) + "_g=" + str(g).replace(".", "-") \
       + "_start=" + str(start).replace(".", "-") + "_end=" + str(end).replace(".", "-") + "_res=" + str(res) + ".npz"
data = np.load(name)

widths = data["widths"]
tau_min = data["times"]
idx = data["idx"]
en_x = data["en_x"]
en_y = data["en_y"]
en_z = data["en_z"]
en_zz = data["en_zz"]
en_avg = 0.5 * (en_x + en_z + en_zz)

idx_gray = 350

plt.figure(1, figsize=(3.375, 3.75), constrained_layout=True)
cmap = "viridis_r"

scat = plt.scatter(widths[0:idx_gray + 1], tau_min[0:idx_gray + 1], marker="o", s=5, c=np.absolute(en_avg[0:idx_gray + 1]), cmap=cmap, vmin=0., vmax=5., zorder=3)

ax = plt.gca()
clb = plt.colorbar(scat, ax=ax, pad=0.075, orientation="vertical", aspect=20)
clb.ax.set_title(r"$\Delta E_{c}$", fontsize=12)
clb.ax.tick_params(labelsize=9)
plt.axvline(2 * np.pi / size, color="black", ls="--", lw=1, zorder=4)

scat = plt.scatter(widths[idx_gray:], tau_min[idx_gray:], marker="o", s=5, color="gray", zorder=3)
plt.axhline(tau_min[idx_gray], lw=1, color="gray", zorder=2)

plt.annotate(r"$\bar{\Delta}$", (4 * np.pi / size, 0.1), fontsize=12, zorder=5)

# plt.grid(color="black", lw=0.5, zorder=1)
plt.xlabel(r"$\Delta_{c}$", fontsize=12)
plt.ylabel(r"$\tau$", fontsize=12)

plt.xscale("log")
plt.xlim(1.e-8, 2.0)
plt.ylim(0.01, end)

plt.xticks([1.e-6, 1.e-2], fontsize=12)
plt.yticks([0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi], [r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"], fontsize=12)

plt.savefig("Fig9.pdf", format="pdf")
# plt.show()