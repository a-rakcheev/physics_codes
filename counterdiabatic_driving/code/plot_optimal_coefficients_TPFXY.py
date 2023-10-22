import numpy as np

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def sign(sign_bit):

    if sign_bit == 0:
        return "+"
    else:
        return "-"


# parameters
l = 7                                             # range cutoff for variational strings
r = 4                                             # cutoff for plotting

s_0 = 0                                            # signs of operators
s_1 = 0
s_2 = 0

op_name_0 = "xx"                                # operators in the hamiltonian
op_name_1 = "yy"
op_name_2 = "zz"

res_1 = 100                                        # number of grid points on x axis
res_2 = 100                                       # number of grid points on y axis

start1 = 1.e-6
start2 = 1.e-6
end_1 = 2.
end_2 = 2.

sign_0 = (-1) ** s_0
sign_1 = (-1) ** s_1
sign_2 = (-1) ** s_2


# naming conventions
# p1_name = r"\kappa "
# p2_name = r"g "

p1_name = r"\gamma "
p2_name = r"\Delta "

# prefix = "C:/Users/ARakc/Dropbox/data/agp/"
# prefix = "D:/Dropbox/data/agp/"
prefix = ""

name = "optimal_coefficients_TPFXY_l" + str(l) + "_" + op_name_0.upper() + "_" + op_name_1.upper() \
       + "_" + op_name_2.upper() + "_s0=" + str(s_0) + "_s1=" + str(s_1) + "_s2=" + str(s_2) \
       + "_res_1=" + str(res_1) + "_res_2=" + str(res_2) + "_start1=" + str(start1).replace(".", "-") \
       + "_end1=" + str(end_1).replace(".", "-") + "_start2=" + str(start2).replace(".", "-") + "_end2=" \
       + str(end_2).replace(".", "-") + ".npz"

data = np.load(prefix + name)
params1 = data["p1"]
params2 = data["p2"]
coefficients_1 = data["c1"]
coefficients_2 = data["c2"]

operator_names = []
for k in np.arange(3, r + 1, 1):

    # fill the strings up to the correct system size
    op_file = prefix + "operators/operators_TPFXY_l" + str(k) + ".txt"
    with open(op_file, "r") as readfile:
        for line in readfile:
            op_str = line[0:k]
            operator_names.append(op_str)


plt.figure(1, figsize=(9, 3.25))
cmap = "Spectral_r"


for op_idx in range(len(operator_names)):
    plt.clf()

    op_name = operator_names[op_idx]
    print(op_idx, len(operator_names), op_name)
    thresh_h = 1.e-3
    vmax_h = 1.e+1
    thresh_g = thresh_h
    vmax_g = vmax_h

    # A_1
    plt.subplot(1, 2, 1, aspect="equal")
    p = plt.pcolormesh(params1, params2, coefficients_1[:, :, op_idx].T, cmap=cmap,
                   norm=colors.SymLogNorm(linthresh=thresh_h, vmin=-vmax_h, vmax=vmax_h))

    plt.xlabel(r"$" + p1_name + r"$", fontsize=14)
    plt.ylabel(r"$" + p2_name + r"$", fontsize=14)


    plt.title(r"$\mathcal{A}_{" + p1_name + r"}$", fontsize=15)

    ax = plt.gca()
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="10%", pad=0.25)
    cbar = plt.colorbar(p, cax=cax, orientation='vertical')


    # A_1
    # print(op_grid_g[:, :, op_idx])
    plt.subplot(1, 2, 2, aspect="equal")
    p = plt.pcolormesh(params1, params2, coefficients_2[:, :, op_idx].T, cmap=cmap,
                   norm=colors.SymLogNorm(linthresh=thresh_g, vmin=-vmax_g, vmax=vmax_g))

    plt.xlabel(r"$" + p1_name + r"$", fontsize=14)
    plt.ylabel(r"$" + p2_name + r"$", fontsize=14)

    plt.title(r"$\mathcal{A}_{" + p2_name + r"}$", fontsize=15)

    ax = plt.gca()
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="10%", pad=0.25)
    cbar = plt.colorbar(p, cax=cax, orientation='vertical')



    plt.subplots_adjust(wspace=0.65)
    if s_0 == 0:
        plt.suptitle(r"$O=\mathrm{" + op_name.upper() + r"}$, "
                     + r"$H=" + r"\mathrm{" + op_name_0.upper() + "} " + sign(s_1) + " "
                     + p1_name + r"\mathrm{" + op_name_1.upper() + "} " + sign(s_2) + " "
                     + p2_name + r"\mathrm{" + op_name_2.upper() + r"}$", fontsize=14)

    else:
        plt.suptitle(r"$O=\mathrm{" + op_name.upper() + r"}$, "
                     + r"$H=-" + r"\mathrm{" + op_name_0.upper() + "} " + sign(s_1) + " "
                     + p1_name + r"\mathrm{" + op_name_1.upper() + "} " + sign(s_2) + " "
                     + p2_name + r"\mathrm{" + op_name_2.upper() + r"}$", fontsize=14)


    # plt.savefig(prefix + "plots/agp_ltfi_coefficient_" + op_name + "_L" + str(L) + "_l" + str(l) + "_res" + str(res_h)
    #             + ".pdf", format="pdf")
    plt.show()