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
l = 6                                             # range cutoff for variational strings
r = l                                             # cutoff for plotting

s_0 = 0                                            # signs of operators
s_1 = 1
s_2 = 1

res_1 = 100                                        # number of grid points on x axis
res_2 = 100                                       # number of grid points on y axis

start1 = 1.e-6
start2 = 1.e-6
end_1 = 2.
end_2 = 2.

sign_0 = (-1) ** s_0
sign_1 = (-1) ** s_1
sign_2 = (-1) ** s_2

p1_name = r"g "
p2_name = r"h "

# prefix = "C:/Users/ARakc/Dropbox/data/agp/"
# prefix = "D:/Dropbox/data/agp/"
prefix = "/home/artem/Dropbox/data/agp/"

name = "optimal_coefficients_ltxy_TPY_l" + str(l) + "_s0=" + str(s_0) + "_s1=" + str(s_1) + "_s2=" + str(s_2) \
       + "_res_1=" + str(res_1) + "_res_2=" + str(res_2) + "_start1=" + str(start1).replace(".", "-") \
       + "_end1=" + str(end_1).replace(".", "-") + "_start2=" + str(start2).replace(".", "-") + "_end2=" \
       + str(end_2).replace(".", "-") + ".npz"


data = np.load(prefix + name)
params1 = data["p1"]
params2 = data["p2"]
coefficients_1 = data["c1"]
coefficients_2 = data["c2"]

operator_names = []
parity_factors = []
for k in np.arange(1, l + 1, 1):

    # fill the strings up to the correct system size
    op_file = prefix + "operators_TPY_l" + str(k) + ".txt"
    with open(op_file, "r") as readfile:
        for line in readfile:
            op_str = line[0:k]
            operator_names.append(op_str)
            if op_str[::-1] == op_str:
                parity_factors.append(2)
            else:
                parity_factors.append(1)

parity_factors = np.array(parity_factors)


metric = np.zeros((res_1, res_2, 2, 2))
plt.figure(1, figsize=(9, 4.5))
cmap = "jet"

for i in range(res_1):
    for j in range(res_2):
        c_1 = coefficients_1[i, j, :]
        c_2 = coefficients_2[i, j, :]

        metric[i, j, 0, 0] = np.sum(parity_factors * c_1 ** 2)
        metric[i, j, 1, 1] = np.sum(parity_factors * c_2 ** 2)
        metric[i, j, 0, 1] = np.sum(parity_factors * c_1 * c_2)
        metric[i, j, 1, 0] = metric[i, j, 0, 1]


xl = params1
yl = params2
X, Y = np.meshgrid(xl, yl)

major_u = np.zeros_like(X)
major_v = np.zeros_like(X)
minor_u = np.zeros_like(X)
minor_v = np.zeros_like(X)

major_norm = np.zeros_like(X)
minor_norm = np.zeros_like(X)
norm = np.zeros_like(X)

for i, x in enumerate(xl):
    for j, y in enumerate(yl):

        g = metric[i, j, :, :]

        ev, evec = np.linalg.eigh(g)
        idx_sort = np.argsort(np.absolute(ev))

        major_u[j, i] = evec[0, idx_sort[1]]
        major_v[j, i] = evec[1, idx_sort[1]]
        major_norm[j, i] = ev[idx_sort[1]]

        minor_u[j, i] = evec[0, idx_sort[0]]
        minor_v[j, i] = evec[1, idx_sort[0]]
        minor_norm[j, i] = ev[idx_sort[0]]

        norm[j, i] = np.sqrt(np.absolute(ev[0] * ev[1]))

weight_major = major_norm / major_norm.max()
weight_minor = minor_norm / major_norm.max()
v_min = weight_minor.min()

# plt.streamplot(X, Y, major_u, major_v, color="black", arrowstyle="-")
# plt.streamplot(X, Y, minor_u, minor_v, color=norm / norm.max(), cmap="inferno_r", norm=colors.Normalize(vmin=0., vmax=1.0), arrowstyle="-")

# plt.grid()
# plt.quiver(X, Y, minor_u, minor_v, norm, norm=colors.LogNorm(vmin=10 ** -2, vmax=10 ** -1), cmap=cmap, pivot="mid")
plt.quiver(X, Y, minor_u, minor_v, norm, cmap=cmap, pivot="mid")

plt.colorbar()

plt.xlim(xl[0], xl[-1])
plt.ylim(yl[0], yl[-1])
plt.xlabel(r"$" + p1_name + r"$", fontsize=14)
plt.ylabel(r"$" + p2_name + r"$", fontsize=14)

plt.subplots_adjust(wspace=0.65)
# if s_0 == 0:
#     plt.suptitle(r"Metric for $T=\infty, l=" + str(l) + r"$: "
#                  + r"$H=" + r"\mathrm{" + op_name_0.upper() + "} " + sign(s_1) + " "
#                  + p1_name + r"\mathrm{" + op_name_1.upper() + "} " + sign(s_2) + " "
#                  + p2_name + r"\mathrm{" + op_name_2.upper() + r"}$", fontsize=14)
#
# else:
#     plt.suptitle(r"Metric for $T=\infty, l=" + str(l) + r"$: "
#                  + r"$H=-" + r"\mathrm{" + op_name_0.upper() + "} " + sign(s_1) + " "
#                  + p1_name + r"\mathrm{" + op_name_1.upper() + "} " + sign(s_2) + " "
#                  + p2_name + r"\mathrm{" + op_name_2.upper() + r"}$", fontsize=14)

# plt.savefig("ltfi_infinite_temperature.pdf", format="pdf")
plt.show()