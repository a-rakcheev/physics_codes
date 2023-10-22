# this program takes the optimized agp for the LTFI with given parameters (L, l, res_x/y, order)
# and analyzes the coefficient of each string with given range r at all values for g, h (x, y)
# since the agp hast translation and reflection (parity) (in short TP) invariance for all parameters
# only representatives for combinations with this symmetry are taken, for example 11xz, 1xz1, xz11, z11x all have
# the same coefficient due to T symmetry and 11zx, ... also have the same ones due to P symmetry
# the output is a .npz file with three arrays, first the representative strings operator_strings and then 3d arrays
# op_grid_x/y, where the slice op_grid_x[:, :, k] includes the (res_x, res_y) array of coefficients for string
# operator_strings[k]


import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from commute_stringgroups_v2 import *

m = maths()

L = 4                                   # number of spins
res_x = 50                              # number of grid points on x axis
res_y = 50                              # number of grid points on y axis
order = 15                              # order of commutator ansatz
l = 4                                   # range cutoff for variational strings
r = 1                                   # range cutoff for strings to be analyzed

xl = np.linspace(1.e-6, 1.5, res_x)
yl = np.linspace(1.e-6, 1.5, res_y)

# prefix = "/home/artem/Dokumente/"
prefix = "/media/artem/TOSHIBA EXT/Dropbox/data/agp/"
name = prefix + "optimize_agp_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order) \
       + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".pkl"

with open(name, "rb") as readfile:
    agp_x = pickle.load(readfile)
    agp_y = pickle.load(readfile)

readfile.close()

# create list of operators for which the coefficients should be analyzed
# these need to be specified as strings of 1, x, y, z of length L
# note that the gauge potential has translation and reflection symmetry, therefore
# all translated and reflected strings will have the same coefficients and only one representative should be used

# here we will include all operators of range r
# the operators and their number are loaded from pre-computed files
op_file = "operators_TP_l" + str(r) + ".txt"
operator_strings = []

# fill the strings up to the correct system size
with open(op_file, "r") as readfile:
    for line in readfile:

        operator_strings.append(line[0:r] + '1' * (L - r))

operator_strings = np.array(operator_strings, dtype=np.str_)
size = len(operator_strings)

path = np.array([[n, 0] for n in range(50)])
pathlength = len(path)

ops_x = np.zeros((pathlength, size))
ops_y = np.zeros((pathlength, size))

for i, index_tuple in enumerate(path):

    A_x = agp_x[index_tuple[0] * res_y + index_tuple[1]]
    A_y = agp_y[index_tuple[0] * res_y + index_tuple[1]]

    # print(A_y)
    for k, opstr in enumerate(operator_strings):

        if opstr in A_x.keys():
            ops_x[i, k] = A_x[opstr].real

        if opstr in A_y.keys():
            ops_y[i, k] = A_y[opstr].real

plt.subplot(1, 3, 1)
plt.plot(path[:, 0], path[:, 1], marker="", markerfacecolor="red", markeredgecolor="black", markersize=5, color="black")

plt.subplot(1, 3, 2)
plt.pcolormesh(ops_x.T, cmap="Spectral", norm=colors.SymLogNorm(1.e-4, vmin=-10., vmax=10.0))
plt.yticks(np.arange(size) + 0.5, operator_strings)

plt.subplot(1, 3, 3)
plt.pcolormesh(ops_y.T, cmap="Spectral", norm=colors.SymLogNorm(1.e-4, vmin=-10., vmax=10.0))
plt.yticks(np.arange(size) + 0.5, operator_strings)

plt.colorbar()


plt.show()

