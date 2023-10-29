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

# # Fig 7a
# inst = 1

# Fig 7b
inst = 9

states = 1000
scale = "log"
fit_range_sa = 20

prefix = "C:/Users/rakche_a-adm/Downloads/data_sk/data/"

gs_indices = np.load(prefix + "gs_indices_N=" + str(N) + ".npy")
fig = plt.figure(1, figsize=(3.75, 3), constrained_layout=True)


R = 1000000
file = h5py.File(prefix + "simulated_annealing_nofield.hdf5", "r")

# get maximal MCS
instance_group = file["/N" + str(N) + "/inst" + str(inst)]
MCS_max = 100 * len(instance_group.keys())
MCSl = np.arange(100, MCS_max + 100, 100)
fidelity_sa = np.zeros(len(MCSl))

for i, MCS in enumerate(MCSl):

    data_group = instance_group["MCS" + str(MCS) + "/series_" + str(R)]
    solcount = data_group["solcount"][:, 0] / R
    fidelity_sa[i] = solcount[-1]

# fit (fit last 5 entries since saturation time needed)
fit_params, cov = lin_fit_gen(MCSl[-fit_range_sa:-1], np.log(1 - fidelity_sa[-fit_range_sa:-1]))

plt.plot(MCSl, fidelity_sa, ls="-", lw="1", color="black", zorder=3)
plt.plot(MCSl, 1 - np.exp(fit_params[1] - fit_params[0] * MCSl), ls="--", lw="1", color="red", label=r"$\gamma=" + str(round(fit_params[0], 7)) + r"$", zorder=2)
plt.grid(True, color="grey", lw=1, zorder=1)
plt.legend(loc="lower right", fontsize=10, framealpha=1)

plt.xlabel(r"$\mathrm{MCS}$", fontsize=13)
plt.ylabel(r"$\mathcal{F}$", fontsize=13)

plt.ylim(1.e-1, 1.)
plt.yscale("log")

file.close()


plt.savefig("Fig7b.pdf", format="pdf")
plt.show()