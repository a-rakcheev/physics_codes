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
import sys
import multiprocessing as mp
from commute_stringgroups_v2 import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

m = maths()
def analyze_agp(index_tuple):
    print("i, j:", index_tuple[0], index_tuple[1])

    A_x = agp_x[index_tuple[0] * res_y + index_tuple[1]]
    A_y = agp_y[index_tuple[0] * res_y + index_tuple[1]]

    # print(A_y)
    for k, opstr in enumerate(operator_strings):

        if opstr in A_x.keys():
            op_grid_x[(index_tuple[0] * res_y + index_tuple[1]) * size + k] = A_x[opstr].real

        if opstr in A_y.keys():

            if opstr == "yz11":
                print(xl[index_tuple[0]], yl[index_tuple[1]], A_y[opstr].real)
            op_grid_y[(index_tuple[0] * res_y + index_tuple[1]) * size + k] = A_y[opstr].real


if __name__ == '__main__':

    # L = int(sys.argv[1])                    # number of spins
    # res_x = int(sys.argv[2])                # number of grid points on x axis
    # res_y = int(sys.argv[3])                # number of grid points on y axis
    # order = int(sys.argv[4])                # order of commutator ansatz
    # number_of_processes = int(sys.argv[5])  # number of parallel processes (should be equal to number of (logical) cores
    # l = int(sys.argv[6])                    # range cutoff for variational strings
    # r = int(sys.argv[7])

    L = 4                                   # number of spins
    res_x = 20                              # number of grid points on x axis
    res_y = 20                              # number of grid points on y axis
    order = 10                              # order of commutator ansatz
    number_of_processes = 1                 # number of parallel processes (should be equal to number of (logical) cores
    l = 4                                   # range cutoff for variational strings
    r = 2                                   # range cutoff for strings to be analyzed

    xl = np.linspace(1.e-6, 1.5, res_x)
    yl = np.linspace(1.e-6, 1.5, res_y)

    prefix = "/home/artem/Dokumente/"
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

    op_grid_x = mp.Array('d', res_x * res_y * size)
    op_grid_y = mp.Array('d', res_x * res_y * size)

    pool = mp.Pool(processes=number_of_processes)
    computation = pool.map_async(analyze_agp, [(i, j) for i in range(res_x) for j in range(res_y)])

    computation.wait()
    op_grid_x = np.array(op_grid_x).reshape((res_x, res_y, size))
    op_grid_y = np.array(op_grid_y).reshape((res_x, res_y, size))
    # plt.figure(1, figsize=(9, 3.25))

    for op_idx in [4]:
        if np.linalg.norm(op_grid_x[:, :, op_idx]) >= 1.e-2 or np.linalg.norm(op_grid_y[:, :, op_idx]) >= 1.e-2:

            plt.clf()
            thresh_h = 1.e-3
            vmax_h = 1.e+1
            thresh_g = thresh_h
            vmax_g = vmax_h

            np.set_printoptions(1)
            # A_h
            # print(op_grid_h[:, :, op_idx])
            plt.subplot(1, 2, 1, aspect="equal")
            p = plt.pcolormesh(xl, yl, op_grid_x[:, :, op_idx].T, cmap="seismic",
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
            p = plt.pcolormesh(xl, yl, op_grid_y[:, :, op_idx].T, cmap="seismic",
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
            plt.show()
