import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

L = 12  # number of spins
res_kappa = 100  # number of grid points on x axis
res_g = 100  # number of grid points on y axis
l = 7  # range cutoff for variational strings
r = 7   # range for plotting

name = "optimize_agp_TPFY_annni_precomputed_L" + str(L) + "_l" + str(l) + "_res_kappa" + str(
    res_kappa) + "_res_g" + str(res_g) + ".npz"

# prefix = "C:/Users/ARakc/Dropbox/data/agp/"
prefix = "D:/Dropbox/data/agp/"

data = np.load(prefix + name)
kappal = data["kappal"]
gl = data["gl"]
coefficients_kappa = data["ckappa"]
coefficients_g = data["cg"]

operator_names = []
parity_factors = []
for k in np.arange(2, r + 1, 1):

    # fill the strings up to the correct system size
    op_file = prefix + "operators_TPFY_l" + str(k) + ".txt"
    with open(op_file, "r") as readfile:
        for line in readfile:
            op_str = line[0:k]
            operator_names.append(op_str)
            if op_str[::-1] == op_str:
                parity_factors.append(2)
            else:
                parity_factors.append(1)

parity_factors = np.array(parity_factors)

metric = np.zeros((res_kappa, res_kappa, 2, 2))
plt.figure(1, figsize=(9, 3.25))
cmap = "Spectral_r"

for i in range(res_kappa):
    for j in range(res_kappa):
        c_kappa = coefficients_kappa[i, j, :]
        c_g = coefficients_g[i, j, :]

        metric[i, j, 0, 0] = np.sum(parity_factors * c_kappa ** 2)
        metric[i, j, 1, 1] = np.sum(parity_factors * c_g ** 2)
        metric[i, j, 0, 1] = np.sum(parity_factors * c_kappa * c_g)
        metric[i, j, 1, 0] = metric[i, j, 0, 1]


xl = kappal
yl = gl
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

        norm[j, i] =np.sqrt(np.absolute(ev[0] * ev[1]))

weight_major = major_norm / major_norm.max()
weight_minor = minor_norm / major_norm.max()
v_min = weight_minor.min()

plt.grid()
plt.quiver(X, Y, minor_u, minor_v, norm, norm=colors.LogNorm(), cmap="jet", pivot="mid")
plt.colorbar()

plt.xlim(xl[0], xl[-1])
plt.ylim(yl[0], yl[-1])
plt.xlabel(r"$\kappa$")
plt.ylabel(r"$g$")

plt.savefig("annni_infinite_temperature.pdf", format="pdf")
plt.show()

