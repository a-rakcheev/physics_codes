import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

res_h = 100  # number of grid points on x axis
res_g = 50  # number of grid points on y axis
l = 7  # range cutoff for variational strings
L = 13  # number of spins
r = 3   # range for plotting

# name = "optimize_agp_TPY_ltfi_precomputed_L" + str(L) + "_l" + str(l) + "_res_h" + str(
#     res_h) + "_res_g" + str(res_g) + ".npz"

name = "ltfi_coefficients_L" + str(L) + "_l" + str(l) + "_res_h" + str(
    res_h) + "_res_g" + str(res_g) + ".npz"

# prefix = "C:/Users/ARakc/Dropbox/data/agp/"
# prefix = "D:/Dropbox/data/agp/"
prefix = ""

data = np.load(prefix + name)
hl = data["hl"]
gl = data["gl"]
coefficients_h = data["ch"]
coefficients_g = data["cg"]

operator_names = []
for k in np.arange(1, r + 1, 1):

    # fill the strings up to the correct system size
    op_file = prefix + "operators/operators_TPY_l" + str(k) + ".txt"
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

    # A_h
    plt.subplot(1, 2, 1, aspect="equal")
    p = plt.pcolormesh(hl, 2 * gl, coefficients_h[:, :, op_idx].T, cmap=cmap,
                   norm=colors.SymLogNorm(linthresh=thresh_h, vmin=-vmax_h, vmax=vmax_h))

    plt.xticks([0., 0.5, 1.0, 1.5])
    plt.yticks([0., 0.5, 1.0, 1.5], [0., 0.25, 0.5, 0.75])

    plt.xlabel(r"$h$", fontsize=12)
    plt.ylabel(r"$g$", fontsize=12)

    plt.title(r"$\mathrm{" + op_name.upper() + r"}$")
    plt.annotate(r"$\mathcal{A}_{h}$", (0.065, 0.925), xycoords="figure fraction", fontsize=16)

    ax = plt.gca()
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="10%", pad=0.25)
    cbar = plt.colorbar(p, cax=cax, orientation='vertical')


    # A_g
    # print(op_grid_g[:, :, op_idx])
    plt.subplot(1, 2, 2, aspect="equal")
    p = plt.pcolormesh(hl, 2 * gl, coefficients_g[:, :, op_idx].T, cmap=cmap,
                   norm=colors.SymLogNorm(linthresh=thresh_g, vmin=-vmax_g, vmax=vmax_g))

    plt.xticks([0., 0.5, 1.0, 1.5])
    plt.yticks([0., 0.5, 1.0, 1.5], [0., 0.25, 0.5, 0.75])

    plt.xlabel(r"$h$", fontsize=12)
    plt.ylabel(r"$g$", fontsize=12)
    plt.title(r"$\mathrm{" + op_name.upper() + r"}$")
    plt.annotate(r"$\mathcal{A}_{g}$", (0.55, 0.925), xycoords="figure fraction", fontsize=16)

    ax = plt.gca()
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="10%", pad=0.25)
    cbar = plt.colorbar(p, cax=cax, orientation='vertical')



    plt.subplots_adjust(wspace=0.65)

    # plt.savefig(prefix + "plots/agp_ltfi_coefficient_" + op_name + "_L" + str(L) + "_l" + str(l) + "_res" + str(res_h)
    #             + ".pdf", format="pdf")
    plt.show()


# # plot along cut
# for op_idx in range(len(operator_names)):
#     plt.clf()
#
#     op_name = operator_names[op_idx]
#
#     thresh_h = 1.e-3
#     vmax_h = 1.e+1
#     thresh_g = thresh_h
#     vmax_g = vmax_h
#
#     # A_h
#     plt.subplot(1, 2, 1)
#     plt.plot(hl, coefficients_h[:, 0, op_idx])
#
#     # plt.yscale("symlog", linthreshy=thresh_h, vmin=-vmax_h, vmax=vmax_h)
#     plt.grid()
#     plt.ylim(-15, 15)
#     plt.xlabel(r"$h$", fontsize=12)
#     plt.ylabel(r"$g$", fontsize=12)
#
#     plt.title(r"$\mathrm{" + op_name.upper() + r"}$")
#     plt.annotate(r"$\mathcal{A}_{h}$", (0.065, 0.925), xycoords="figure fraction", fontsize=16)
#
#
#
#     # A_g
#     plt.subplot(1, 2, 2)
#     plt.plot(hl, coefficients_g[:, 0, op_idx])
#
#     # plt.yscale("symlog", linthreshy=thresh_g, vmin=-vmax_g, vmax=vmax_g)
#     plt.grid()
#     plt.ylim(-15, 15)
#     plt.xlabel(r"$h$", fontsize=12)
#     plt.ylabel(r"$g$", fontsize=12)
#
#     plt.title(r"$\mathrm{" + op_name.upper() + r"}$")
#     plt.annotate(r"$\mathcal{A}_{g}$", (0.55, 0.925), xycoords="figure fraction", fontsize=16)
#
#
#
#
#     plt.subplots_adjust(wspace=0.65)

    # plt.savefig("agp_ltfi_coefficient_" + op_name + "_L" + str(L) + "_l" + str(l) + "_res" + str(res_h)
    #             + ".pdf", format="pdf")
    # plt.show()