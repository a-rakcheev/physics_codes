import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

l = 4  # range cutoff for variational strings
res_h = 17  # number of grid points on x axis
res_g = 17  # number of grid points on y axis

name = "optimal_coefficients/optimal_coefficients_rydberg_chain_truncated_TPY_l" + str(l) + "_res_g=" + str(res_g) + "_res_h=" + str(res_h) + ".npz"
data = np.load(name)
gl = data["gl"]
hl = data["hl"]
coefficients_g = data["c_g"]
coefficients_h = data["c_h"]

prefix = "/home/artem/Dropbox/bu_research/operators/"
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

metric = np.zeros((len(gl), len(hl), 2, 2))
plt.figure(1, figsize=(9, 3.25))
cmap = "jet"

for i in range(len(gl)):
    for j in range(len(hl)):

        c_g = coefficients_g[i, j, :]
        c_h = coefficients_h[i, j, :]

        metric[i, j, 0, 0] = np.sum(parity_factors * c_g ** 2)
        metric[i, j, 1, 1] = np.sum(parity_factors * c_h ** 2)
        metric[i, j, 0, 1] = np.sum(parity_factors * c_h * c_g)
        metric[i, j, 1, 0] = metric[i, j, 0, 1]


xl = gl
yl = hl
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

# plt.quiver(X, Y, minor_u, minor_v, norm, norm=colors.LogNorm(vmin=10 ** -6, vmax=10 ** 0), cmap=cmap, pivot="mid")

plt.pcolormesh(X, Y, norm, norm=colors.LogNorm(vmin=1.e-6, vmax=1.e2), cmap=cmap)

plt.colorbar()
# plt.grid()

plt.xlim(xl[0], xl[-1])
plt.ylim(yl[0], yl[-1])

plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$g$")
plt.ylabel(r"$h$")

# plt.savefig("ltfi_infinite_temperature.pdf", format="pdf")
plt.show()
