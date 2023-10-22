import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

L = 12  # number of spins
res_kappa = 100  # number of grid points on x axis
res_g = 100  # number of grid points on y axis
l = 7  # range cutoff for variational strings
r = 5   # range for plotting

name = "optimize_agp_TPFY_annni_precomputed_L" + str(L) + "_l" + str(l) + "_res_kappa" + str(
    res_kappa) + "_res_g" + str(res_g) + ".npz"

prefix = "C:/Users/ARakc/Dropbox/data/agp/"
# prefix = "D:/Dropbox/data/agp/"

data = np.load(prefix + name)
kappal = data["kappal"]
gl = data["gl"]
coefficients_kappa = data["ckappa"]
coefficients_g = data["cg"]

operator_names = []
for k in np.arange(2, r + 1, 1):

    # fill the strings up to the correct system size
    op_file = prefix + "operators_TPFY_l" + str(k) + ".txt"
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
    p = plt.pcolormesh(kappal, gl, coefficients_kappa[:, :, op_idx].T, cmap=cmap,
                   norm=colors.SymLogNorm(linthresh=thresh_h, vmin=-vmax_h, vmax=vmax_h))

    plt.xticks([0., 0.5, 1.0, 1.5])
    plt.yticks([0., 0.5, 1.0, 1.5])

    plt.xlabel(r"$\kappa$", fontsize=12)
    plt.ylabel(r"$g$", fontsize=12)

    plt.title(r"$\mathrm{" + op_name.upper() + r"}$")
    plt.annotate(r"$\mathcal{A}_{\kappa}$", (0.065, 0.925), xycoords="figure fraction", fontsize=16)

    ax = plt.gca()
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="10%", pad=0.25)
    cbar = plt.colorbar(p, cax=cax, orientation='vertical')


    # A_g
    # print(op_grid_g[:, :, op_idx])
    plt.subplot(1, 2, 2, aspect="equal")
    p = plt.pcolormesh(kappal, gl, coefficients_g[:, :, op_idx].T, cmap=cmap,
                   norm=colors.SymLogNorm(linthresh=thresh_g, vmin=-vmax_g, vmax=vmax_g))

    plt.xticks([0., 0.5, 1.0, 1.5])
    plt.yticks([0., 0.5, 1.0, 1.5])

    plt.xlabel(r"$\kappa$", fontsize=12)
    plt.ylabel(r"$g$", fontsize=12)
    plt.title(r"$\mathrm{" + op_name.upper() + r"}$")
    plt.annotate(r"$\mathcal{A}_{g}$", (0.55, 0.925), xycoords="figure fraction", fontsize=16)

    ax = plt.gca()
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="10%", pad=0.25)
    cbar = plt.colorbar(p, cax=cax, orientation='vertical')



    plt.subplots_adjust(wspace=0.65)

    plt.savefig(prefix + "plots/agp_annni_coefficient_" + op_name + "_L" + str(L) + "_l" + str(l) + "_res" + str(res_kappa)
                + ".pdf", format="pdf")

    # plt.show()