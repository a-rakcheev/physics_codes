import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

L = 12                                             # number of spins
res_h = 200                                        # number of grid points on x axis
res_g = 200                                        # number of grid points on y axis
lengths = [4, 5, 6, 7]                                             # range cutoff for variational strings

# op_indices = [12, 14, 35, 40]
op_indices = [0, 2, 6, 35]

h_indices = [25, 50, 100]
g_indices = [25, 50, 100]

coefficients_g = np.zeros((4, 4, len(lengths)))
coefficients_h = np.zeros((4, 4, len(lengths)))
g_vals = np.zeros(4)
h_vals = np.zeros(4)

# prefix = "C:/Users/ARakc/Dropbox/data/agp/"
prefix = "D:/Dropbox/data/agp/"

plt.figure(1, figsize=(12, 12))

for s, l in enumerate(lengths):

    print("l", l)
    # read in TPY operators up to range l
    operator_names = []
    for k in np.arange(1, l + 1, 1):

        # fill the strings up to the correct system size
        op_file = prefix + "operators_TPY_l" + str(k) + ".txt"
        with open(op_file, "r") as readfile:
            for line in readfile:
                op_str = line[0:k]
                operator_names.append(op_str)


    name = prefix + "optimize_agp_TPY_ltfi_precomputed_L" + str(L) + "_l" + str(l) + "_res_h" + str(
        res_h) + "_res_g" + str(res_g) + ".npz"

    data = np.load(name)
    hl = data["hl"]
    gl = data["gl"]
    coeff_h = data["ch"]
    coeff_g = data["cg"]

    for i in range(3):

        h_idx = h_indices[i]
        g_idx = g_indices[i]
        h_vals[i] = hl[h_idx]
        g_vals[i] = gl[g_idx]

        for j, op_idx in enumerate(op_indices):
            coefficients_g[i, j, s] = coeff_g[h_idx, g_idx, op_idx]
            coefficients_h[i, j, s] = coeff_h[h_idx, g_idx, op_idx]


for i in range(3):
    for j in range(4):

        plt.subplot(3, 4, j + 1 + 4 * i)
        plt.plot(lengths, coefficients_h[i, j, :], ls="-", marker="o", label=r"$A_{h}$")
        plt.plot(lengths, coefficients_g[i, j, :], ls="--", marker="s", label=r"$A_{g}$")

        plt.legend()
        plt.grid()
        plt.xticks(lengths)
        plt.xlabel(r"$\ell$", fontsize=10)
        plt.title(r"$\mathrm{" + operator_names[op_indices[j]].upper() + r"}$", fontsize=10)

        if j == 0:
            ax = plt.gca()
            plt.text(-0.35, 0.5, r"$h=" + str(round(h_vals[i], 2)) + r", \; g=" + str(round(g_vals[i], 2)) + r"$",
                     rotation=90, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

plt.subplots_adjust(wspace=0.5, hspace=0.75)
# plt.suptitle("Convergence of Coefficients with Different Truncation Range")
# plt.savefig("convergence_demo_long.pdf", format="pdf")
# plt.savefig("convergence_demo_short.pdf", format="pdf")

plt.savefig("convergence_demo.pdf", format="pdf")
# plt.show()