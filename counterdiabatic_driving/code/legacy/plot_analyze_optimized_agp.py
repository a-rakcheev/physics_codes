import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

L = 6  # number of spins
res_x = 50  # number of grid points on x axis
res_y = 50  # number of grid points on y axis
order = 15  # order of commutator ansatz
l = 5  # range cutoff for variational strings
# r = 1  # range cutoff for strings to be analyzed

hl = np.linspace(1.e-6, 1.5, res_x)
gl = np.linspace(1.e-6, 1.5, res_y)

for r in np.arange(1, l + 1, 1):
# for r in [1]:

    print(r)
    prefix = "C:/Users/ARakc/Dropbox/data/agp/"
    data = np.load(prefix + "analyze_agp_ltfi_L" + str(L) + "_r" + str(r) + "_l" + str(l) + "_order" + str(order) \
                        + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz")

    operator_strings = data["op_str"].astype(str)
    op_grid_h = data["op_x"]
    op_grid_g = data["op_y"]

    plt.figure(1, figsize=(9, 3.25))
    cmap = "Spectral"
    for op_idx in range(len(operator_strings)):

        if np.linalg.norm(op_grid_h[:, :, op_idx]) >= 1.e-2 or np.linalg.norm(op_grid_g[:, :, op_idx]) >= 1.e-2:

            plt.clf()
            # op_idx = 10

            thresh_h = 1.e-3
            vmax_h = 1.e+1
            thresh_g = thresh_h
            vmax_g = vmax_h

            np.set_printoptions(1)
            # A_h
            # print(op_grid_h[:, :, op_idx])
            plt.subplot(1, 2, 1, aspect="equal")
            p = plt.pcolormesh(hl, gl, op_grid_h[:, :, op_idx].T, cmap=cmap,
                           norm=colors.SymLogNorm(linthresh=thresh_h, vmin=-vmax_h, vmax=vmax_h))

            plt.xlabel(r"$h$", fontsize=12)
            plt.ylabel(r"$g$", fontsize=12)
            print(operator_strings[op_idx][0:r])
            plt.title(r"$\mathrm{" + operator_strings[op_idx][0:r].upper() + r"}$")

            plt.xticks([0., 0.5, 1.0, 1.5])
            plt.yticks([0., 0.5, 1.0, 1.5])

            ax = plt.gca()
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="10%", pad=0.25)
            cbar = plt.colorbar(p, cax=cax, orientation='vertical')

            plt.annotate(r"$\mathcal{A}_{h}$", (0.065, 0.925), xycoords="figure fraction", fontsize=16)

            # A_g
            # print(op_grid_g[:, :, op_idx])
            plt.subplot(1, 2, 2, aspect="equal")
            p = plt.pcolormesh(hl, gl, op_grid_g[:, :, op_idx].T, cmap=cmap,
                           norm=colors.SymLogNorm(linthresh=thresh_g, vmin=-vmax_g, vmax=vmax_g))

            plt.xlabel(r"$h$", fontsize=12)
            plt.ylabel(r"$g$", fontsize=12)
            plt.title(r"$\mathrm{" + operator_strings[op_idx][0:r].upper() + r"}$")
            plt.annotate(r"$\mathcal{A}_{g}$", (0.55, 0.925), xycoords="figure fraction", fontsize=16)

            plt.xticks([0., 0.5, 1.0, 1.5])
            plt.yticks([0., 0.5, 1.0, 1.5])

            ax = plt.gca()
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="10%", pad=0.25)
            cbar = plt.colorbar(p, cax=cax, orientation='vertical')



            plt.subplots_adjust(wspace=0.65)

            # plt.savefig("agp_ltfi_coefficient_" + operator_strings[op_idx][0:r] + "_L" + str(L) + "_l" + str(l)
            #             + "_order" + str(order) + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".pdf", format="pdf")
            plt.show()
