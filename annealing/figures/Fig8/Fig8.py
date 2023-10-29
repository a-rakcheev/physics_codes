import sys
sys.path.append("/data/Dropbox/codebase/")
sys.path.append("D:/Dropbox/codebase/")
sys.path.append("C:/Users/ARakc/Dropbox/codebase/")
sys.path.append("E:/Dropbox/codebase/")


import h5py
import numpy as np
import hamiltonians_32 as ham32

from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmasher as cmr

# parameters
N = 10
inst = 39
prefix = "C:/Users/rakche_a-adm/Downloads/data_sk/data/"


# real-time
dt = 0.01
steps = 100
Tl = [1, 10, 50]
corrs_qa = np.zeros((steps, N * (N - 1) // 2))
fidelity_qa = np.zeros(steps)

sl = np.linspace(0., 1., steps + 1)

# for plotting
steps_max = 100  # only plot up to steps max
plotscale = "logscale"  # plot either with linscale or with logscale
# plotscale = "linscale"
cmap_div = cmr.pride
linthresh = 1.e-3

fig = plt.figure(1, figsize=(7.5, 3.5), constrained_layout=True)

# true gs
# bonds from instance file
name_bonds = prefix + "sk_ising_bimodal_N" + str(N) + "_unique_inst" + str(inst) + ".txt"
bonds = np.loadtxt(name_bonds)

# input for ZZ hamiltonian
positions_zz = []
labels_zz = []

for i in range(1, N + 1):
    for j in range(i + 1, N + 1):
        positions_zz.append([i, j])
        labels_zz.append([3, 3])

positions_zz = np.array(positions_zz, dtype=np.int64)
labels_zz = np.array(labels_zz, dtype=np.int8)

# degeneracy breaking local field at first site
h_zz = ham32.operator_sum_real_diag(N, 2, bonds, positions_zz, labels_zz)
h_ising = h_zz

# find ground state index
gs_idx = np.argmin(h_ising.diagonal())

# extract magnetization
mag_gs = np.zeros(N)
for i in range(N):
    mag_gs[N - 1 - i] = np.sign(float((gs_idx & (1 << i)) >> i) - 0.5)

# obtain correlations
corrs_gs = np.zeros((N * (N - 1) // 2))
count = 0

for i in range(N):
    for j in range(i + 1, N):
        corrs_gs[count] = mag_gs[i] * mag_gs[j]
        count += 1

# real-time
file = h5py.File(prefix + "quantum_annealing_nofield.hdf5", "r")
series_group = file["/N" + str(N) + "/inst" + str(inst) + "/series_dt=0-01_steps=100"]

for n, T in enumerate(Tl):

    plt.subplot(1, 3, n + 1)

    # read data - read in chunks based on T
    G_zz = series_group["G_zz"]
    fids_gs = series_group["fids_gs"]

    corrs_qa = G_zz[1:, :, int(T) - 1] 
    fidelity_qa = fids_gs[:, int(T) - 1]

    if plotscale == "logscale":
        plt.pcolormesh(sl, np.arange(1, N * (N - 1) // 2 + 2), corrs_qa.T, cmap=cmap_div, norm=colors.SymLogNorm(linthresh=linthresh, vmin=-1, vmax=1))

    elif plotscale == "linscale":
        plt.pcolormesh(sl, np.arange(1, N * (N - 1) // 2 + 2), corrs_qa.T, cmap=cmap_div, vmin=-1, vmax=1)

    for j in range(1, N):
        plt.axhline(np.sum(np.arange(N - 1, N - j - 1, -1)) + 1, lw=1, color="black")

    if n == 0:
        plt.ylabel(r"$G^{zz}_{ij}$", fontsize=12)

    cb = plt.colorbar(aspect=30)
    cb.ax.tick_params(labelsize=9)
    plt.xlabel(r"$s$", fontsize=12)
    plt.yticks([], [])
    plt.title(r"$T=" + str(int(T)) + r"$", fontsize=10)


plt.savefig("Fig8.png", format="png", dpi=1200)

plt.show()