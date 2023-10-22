# create fidelity metric from the ground state (or average over states)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.sparse as sp
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

# parameters
L = 12
l = 7                                             # range cutoff for variational strings
r = l                                             # cutoff for plotting

s_0 = 0                                            # signs of operators
s_1 = 1
s_2 = 1

res_1 = 50                                        # number of grid points on x axis
res_2 = 25                                       # number of grid points on y axis

start1 = 1.e-6
start2 = 1.e-6
end_1 = 1.5
end_2 = 0.75

sign_0 = (-1) ** s_0
sign_1 = (-1) ** s_1
sign_2 = (-1) ** s_2

dp_1 = (end_1 - start1) / (res_1 - 1)
dp_2 = (end_2 - start2) / (res_2 - 1)

# operators
k_idx = 0                           # momentum index (full momentum is (2 pi / L) * index), should be 0, 1, ..., L - 1
k_name = "0"                        # currently 0 or pi (k_idx = L // 2) available
par = 1                             # parity sector


# prefix = "C:/Users/ARakc/Dropbox/data/agp/"
# prefix = "D:/Dropbox/data/agp/"
# prefix = ""
prefix = "/home/artem/Dropbox/data/agp/"

# name = "groundstate_ltxy_TPY_L" + str(L) + "_s0=" + str(s_0) + "_s1=" + str(s_1) + "_s2=" + str(s_2) \
#        + "_res_1=" + str(res_1) + "_res_2=" + str(res_2) + "_start1=" + str(start1).replace(".", "-") \
#        + "_end1=" + str(end_1).replace(".", "-") + "_start2=" + str(start2).replace(".", "-") + "_end2=" \
#        + str(end_2).replace(".", "-") + ".npz"

name = "ltfi_gs_ed_k=0_p=1_L=" + str(L) + "_h_res=" + str(res_1) + "_g_res=" + str(res_2) + ".npz"

data = np.load(prefix + name)
evec = data["evec"]

# create derivatives
gs_deriv_1 = np.gradient(evec, dp_1, axis=0)
gs_deriv_2 = np.gradient(evec, dp_2, axis=1)

metric = np.zeros((res_1, res_2, 2, 2))
for i in range(res_1):
    for j in range(res_2):

        metric[i, j, 0, 0] = (gs_deriv_1[i, j, :].T.conj() @ gs_deriv_1[i, j, :] -
                              (gs_deriv_1[i, j, :].T.conj() @ evec[i, j, :]) *
                              (evec[i, j, :].T.conj() @ gs_deriv_1[i, j, :])).real

        metric[i, j, 1, 1] = (gs_deriv_2[i, j, :].T.conj() @ gs_deriv_2[i, j, :] -
                              (gs_deriv_2[i, j, :].T.conj() @ evec[i, j, :]) *
                              (evec[i, j, :].T.conj() @ gs_deriv_2[i, j, :])).real

        metric[i, j, 0, 1] = (gs_deriv_1[i, j, :].T.conj() @ gs_deriv_2[i, j, :] -
                              (gs_deriv_1[i, j, :].T.conj() @ evec[i, j, :]) *
                              (evec[i, j, :].T.conj() @ gs_deriv_2[i, j, :])).real

        metric[i, j, 1, 0] = metric[i, j, 0, 1]


# plotting
plt.figure(1, figsize=(6, 6))
cmap = "jet"

xl = np.linspace(start1, end_1, res_1)
yl = np.linspace(start2, end_2, res_2)
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

        print(i, j)
        g = metric[i, j, :, :]

        print(g / L)
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
plt.grid()
plt.quiver(X, Y, minor_u, minor_v, norm / L, norm=colors.LogNorm(vmin=10 ** -3, vmax=10 ** 1), cmap=cmap, pivot="mid")
plt.colorbar()

plt.xlim(xl[0], xl[-1])
plt.ylim(yl[0], yl[-1])
plt.xlabel(r"$g$")
plt.ylabel(r"$h$")

# plt.savefig("ltxy_groundstate.pdf", format="pdf")
plt.show()


