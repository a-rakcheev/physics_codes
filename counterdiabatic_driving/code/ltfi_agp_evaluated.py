# evaluate the expectation values of the agp in order to evaluate metrics
# the evaluation is done in the K=0, P=1 sector

import zipfile
import io

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.sparse as sp
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

# parameters
l = 8                               # range cutoff for variational strings
L_coeff = 2 * l - 1                 # number of spins of the coefficient computation
L = 10                              # number of spins
res_h = 50                         # number of grid points on x axis
res_g = 25                          # number of grid points on y axis
r = 7                               # evaluate up to range (needs to be <= l)

step1 = 1                            # sampling along x axis (every n-th point)
step2 = 1                            # sampling along y axis (every n-th point)

# coefficients
# prefix = "C:/Users/ARakc/Dropbox/data/agp/"
# prefix = "D:/Dropbox/data/agp/"
prefix = ""

name = "ltfi_coefficients_L" + str(L_coeff) + "_l" + str(l) + "_res_h" + str(
    res_h) + "_res_g" + str(res_g) + ".npz"

data = np.load(prefix + name)
hl = data["hl"]
gl = data["gl"]
coefficients_h = data["ch"]
coefficients_g = data["cg"]


# operators
k_idx = 0                           # momentum index (full momentum is (2 pi / L) * index), should be 0, 1, ..., L - 1
k_name = "0"                        # currently 0 or pi (k_idx = L // 2) available
par = 1                             # parity sector

# create operators up to range r and save them in memory
name_zip = "operators/1d_chain_indices_and_periods.zip"
with zipfile.ZipFile(name_zip) as zipper:

    name = "1d_chain_TP_indices_and_periods_L=" + str(L) + "_k=" + k_name + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:

        data = np.load(f)
        periods = data["period"]
        parities = data["parity"]
        size = len(periods)

operators = []
op_names = []

for k in np.arange(1, r + 1, 1):
    # fill the strings up to the correct system size
    op_file = "operators/operators_TPY_l" + str(k) + ".txt"
    with open(op_file, "r") as readfile:
        for line in readfile:

            op_str = line[0:k]
            op_names.append(op_str)

print("op_names read", len(op_names))

name_zip_op = "operators/operators_TP_L=" + str(L) + ".zip"
with zipfile.ZipFile(name_zip_op) as zipper_op:

    for op_str in op_names:
        mat_name = op_str + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"

        with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
            data = np.load(f_op)
            indptr = data["indptr"]
            indices = data["idx"]
            val = data["val"]

        mat = sp.csr_matrix((val, indices, indptr), shape=(size, size))
        operators.append(mat)

print("operators created")

# prefix = "C:/Users/ARakc/Dropbox/data/agp/"
# prefix = "D:/Dropbox/data/agp/"
prefix = ""

name = prefix + "ltfi_gs_ed_k=0_p=1_L=" + str(L) + "_h_res=" + str(res_h) + "_g_res=" + str(res_g) + ".npz"
data = np.load(name)
evec = data["evec"]


# for each parameter point compute the
# compute metric through expectation values

metric = np.zeros((res_h // step1, res_g // step2, 2, 2))

start_time = time.time()
for i in range(res_h // step1):
    print(i)
    for j in range(res_g // step2):

        c_h = coefficients_h[i * step1, j * step2, :]
        c_g = coefficients_g[i * step1, j * step2, :]

        A_h = sp.csr_matrix((size, size), dtype=np.complex128)
        A_g = sp.csr_matrix((size, size), dtype=np.complex128)

        for k in range(len(operators)):

            mat = operators[k]

            A_h = A_h + c_h[k] * mat
            A_g = A_g + c_g[k] * mat

        ground_state = evec[i * step1, j * step2, :]

        state_h = A_h @ ground_state
        state_g = A_g @ ground_state

        metric[i, j, 0, 0] = (state_h.T @ state_h - (ground_state.T @ state_h) ** 2).real
        metric[i, j, 1, 1] = (state_g.T @ state_g - (ground_state.T @ state_g) ** 2).real
        metric[i, j, 0, 1] = (state_g.T @ state_h - (ground_state.T @ state_h) * (ground_state.T @ state_g)).real
        metric[i, j, 1, 0] = metric[i, j, 0, 1]

end_time = time.time()
print("Time:", end_time - start_time)

# plotting
plt.figure(1, figsize=(6, 3.25), constrained_layout=True)
cmap = "jet"

xl = hl[::step1]
yl = gl[::step2]
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

plt.grid()
plt.quiver(X, Y, minor_u, minor_v, norm / L, norm=colors.LogNorm(vmin=10 ** -2, vmax=10 ** 1), cmap=cmap, pivot="tail")
plt.quiver(X, Y, -minor_u, -minor_v, norm / L, norm=colors.LogNorm(vmin=10 ** -2, vmax=10 ** 1), cmap=cmap, pivot="tail")

plt.colorbar()

plt.xlim(xl[0], xl[-1])
plt.ylim(yl[0], yl[-1])
plt.xlabel(r"$h$")
plt.ylabel(r"$g$")

plt.savefig("ltfi_gs_metric_agp_L=" + str(L) + "_l=" + str(l) + "_r=" + str(r) + ".pdf", format="pdf")
plt.show()

