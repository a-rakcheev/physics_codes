# investigate low-energy spectrum
import sys
import numpy as np
import matplotlib
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors
import tqdm

# parameters
N = 20                                                 # system size
inst = 9                                             # instance

prefix = "C:/Users/rakche_a-adm/Downloads/data_sk/data/"
fig = plt.figure(1, figsize=(7.5, 7))

data = np.load(prefix + "spectrum/N" + str(N) + "/sk_ising_bimodal_second_sector_spectrum_and_matel_N=" + str(N) + "_inst=" + str(inst) + ".npz")
spec = data["spec"]
sl = data["sl"]
mags = data["mags"]
# matel_x = data["matel_x"]
# matel_ising = data["matel_ising"]
ent = data["ent"]
amps = data["amps"]
boundary = data["boundary"]

# total number of degenerate states (including the ground state)
states = len(spec[0, :])
colormap = plt.cm.nipy_spectral
line_colors = [colormap(i) for i in np.linspace(0., 1., (states - 1))]

# get magnetizations
mags_diff = np.zeros((states, states))

for j in range(states):
    for k in range(j + 1, states):

        dist = np.sum(np.absolute(mags[j, :] - mags[k, :]) / 2)
        mags_diff[j, k] = min(dist, N - dist)

mags_diff += mags_diff.T

# plot 
gs = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(gs[0, 0])
plt.pcolormesh(np.arange(states + 1), np.arange(states + 1), mags_diff, cmap="jet", vmax=N / 2)
for bound in boundary:

    plt.axvline(bound, ls="-", lw=2, color="black")
    plt.axhline(bound, ls="-", lw=2, color="black")

plt.colorbar()
plt.xticks([], [])
plt.yticks([], [])
plt.title(r"$\textrm{hamming distance}$", fontsize=13)


ax2 = fig.add_subplot(gs[1, 0])

for i in range(states):
    plt.scatter(sl, spec[:, i] - spec[:, 0], s=6, c=ent[:, i] + 1, cmap="jet", vmin=0., vmax=10, label=r"$n=" + str(i) + r"$", zorder=2)

cb = plt.colorbar()
cb.ax.set_title(r"$S_{2}$", fontsize=11)
plt.grid(zorder=1)

plt.xlabel(r"$s$", fontsize=13)
plt.ylabel(r"$\Delta E$", fontsize=13)

plt.title(r"$\textrm{spectrum}$", fontsize=13)

ax5 = fig.add_subplot(gs[:, 1])
states = 10
probs = np.zeros((len(sl), states ** 2))

for i in range(states):
    probs[:, i * states: (i + 1) * states] = np.absolute(amps[:, i , 0:states]) ** 2

plt.pcolormesh(sl, np.arange(states ** 2), probs.T, cmap="nipy_spectral", norm=colors.LogNorm(vmin=0.01, vmax=1.))
for i in range(states):

    plt.axhline(i * states - 0.5, ls="-", lw=1, color="white")

cb = plt.colorbar()

plt.xlabel(r"$s$", fontsize=13)
plt.yticks([], [])
plt.title(r"$\textrm{state decomposition}$", fontsize=13)

plt.subplots_adjust(hspace=0.15, wspace=0.1, left=0.06, right=0.975, top=0.925, bottom=0.075)

plt.savefig("Fig3.png", format="png", dpi=1600)
plt.show()

