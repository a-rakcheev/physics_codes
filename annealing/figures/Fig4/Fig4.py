import sys
sys.path.append("/home/artem/Dropbox/codebase/")
sys.path.append("E:/Dropbox/codebase/")
sys.path.append("/data/Dropbox/codebase/")

import h5py
import numpy as np
import hamiltonians_32 as ham32
from scipy.optimize import curve_fit

from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tqdm 

# fit to line
def lin_fit(en, log_probs):

    def func(x, b):
        return -b * x
    
    params, cov = curve_fit(func, en, log_probs)
    return params, cov

# fit to line
def lin_fit_gen(en, log_probs):

    def func(x, b, a):
        return a - b * x
    
    params, cov = curve_fit(func, en, log_probs)
    return params, cov


# parameters
N = 20
states = 1000
scale = "log"

# # Fig 4a
# inst = 1
# T_max = 50

# Fig 4b
inst = 9
T_max = 1000

Tl = np.arange(1, T_max + 1, 1)
fit_range_sa = 20

prefix = "C:/Users/rakche_a-adm/Downloads/data_sk/data/"

gs_indices = np.load(prefix + "gs_indices_N=" + str(N) + ".npy")
fig = plt.figure(1, figsize=(3.75, 3), constrained_layout=True)


# real-time
dt = 0.01
steps = 100

# real-time
file = h5py.File(prefix + "quantum_annealing_nofield.hdf5", "r")
fidelity_qa = file["/N" + str(N) + "/inst" + str(inst) + "/series_dt=0-01_steps=100"]["fids_gs"][:, 0:len(Tl)]
file.close()

# fit (fit last 10 entries since saturation time needed)
fit_params, cov = lin_fit(Tl[-10:-1], np.log(1 - 2 * fidelity_qa[-1, -10:-1]))

plt.plot(Tl, 2 * fidelity_qa[-1, :], ls="-", lw="1", color="black", zorder=3)
plt.plot(Tl, 1 - np.exp(-fit_params[0] * Tl), ls="--", lw="1", color="red", label=r"$\alpha=" + str(-round(fit_params[0], 4)) + r"$", zorder=2)

plt.grid(True, color="grey", lw=1, zorder=1)
plt.legend(loc="upper left", fontsize=10, framealpha=1)

plt.xlabel(r"$T$", fontsize=13)
plt.ylabel(r"$\mathcal{F}$", fontsize=13)

plt.ylim(1.e-3, 1.)
plt.yscale("log")

# # Fig4a
# axin = plt.gca().inset_axes([0.55, 0.2, 0.4, 0.3])
# axin.plot(Tl[-10:-1], 2 * fidelity_qa[-1, -10:-1], ls="-", lw="0.75", color="black", zorder=3)
# axin.plot(Tl[-10:-1], 1 - np.exp(-fit_params[0] * Tl[-10:-1]), ls="--", lw="0.75", color="red", zorder=2)

# axin.grid(True, color="grey", lw=0.75, zorder=1)
# axin.set_yscale("log")

# Fig4b
axin = plt.gca().inset_axes([0.55, 0.2, 0.4, 0.3])
axin.plot(Tl[0:50], 2 * fidelity_qa[-1, 0:50], ls="-", lw="0.75", color="black", zorder=3)

# zoom
axin.set_ylim(1.e-3, 1.0)

axin.grid(True, color="grey", lw=0.75, zorder=1)
axin.set_yscale("log")


plt.savefig("Fig4b.pdf", format="pdf")
plt.show()