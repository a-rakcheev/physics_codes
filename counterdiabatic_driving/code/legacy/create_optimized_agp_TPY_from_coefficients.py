import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from commute_stringgroups_no_quspin import *
m = maths()

L = 12                                             # number of spins
res_h = 200                                        # number of grid points on x axis
res_g = 200                                        # number of grid points on y axis
l = 6                                             # range cutoff for variational strings

# read in TPY operators up to range l
operator_labels = []
variational_operators = []
for k in np.arange(1, l + 1, 1):

    # fill the strings up to the correct system size
    op_file = "operators_TPY_l" + str(k) + ".txt"
    with open(op_file, "r") as readfile:
        for line in readfile:

            operator_labels.append(line[0:k].upper())
            op_str = line[0:k] + '1' * (L - k)
            op_eq = equation()
            for i in range(L):
                op = "".join(roll(list(op_str), i))
                op_eq += equation({op: 1.0})
                op_rev = "".join(roll(list(op_str[::-1]), i))
                op_eq += equation({op_rev: 1.0})

            variational_operators.append(op_eq)

var_order = len(variational_operators)

# name = "optimize_agp_TPY_ltfi_L" + str(L) + "_l" + str(l) + "_res_h" + str(
#     res_h) + "_res_g" + str(res_g) + ".npz"
name = "optimize_agp_TPY_ltfi_precomputed_L" + str(L) + "_l" + str(l) + "_res_h" + str(
    res_h) + "_res_g" + str(res_g) + ".npz"

data = np.load(name)
hl = data["hl"]
gl = data["gl"]
coeff_h = data["ch"]
coeff_g = data["cg"]

# i_start = 0
# i_end = 25
# plots = plt.plot(coeff_g[:, 0, i_start:i_end + 1])
# plt.grid()
#
# plt.legend(plots, operator_labels[i_start:i_end + 1])

op_idx = 735

thresh_h = 1.e-3
vmax_h = 1.e+1
thresh_g = thresh_h
vmax_g = vmax_h
cmap = "Spectral"

plt.pcolormesh(hl, gl, coeff_g[:, :, op_idx].T, cmap=cmap, norm=colors.SymLogNorm(linthresh=thresh_h, vmin=-vmax_h, vmax=vmax_h))
plt.colorbar()

plt.xlabel(r"$h$", fontsize=12)
plt.ylabel(r"$g$", fontsize=12)
plt.title(r"$\mathrm{" + operator_labels[op_idx] + r"}$", fontsize=14)
plt.show()