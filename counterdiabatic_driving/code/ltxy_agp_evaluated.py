# evaluate the expectation values of the agp in order to evaluate metrics
# the evaluation is done in the K=0, P=1 sector

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.sparse as sp
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

# parameters
L = 10
l = 5                                             # range cutoff for variational strings
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

# operators
k_idx = 0                           # momentum index (full momentum is (2 pi / L) * index), should be 0, 1, ..., L - 1
k_name = "0"                        # currently 0 or pi (k_idx = L // 2) available
par = 1                             # parity sector

prefix_ops = "operators_TPY/"
name = prefix_ops + "1d_chain_TP_indices_and_periods_L=" + str(L) + "_k=" + k_name + ".npz"
data = np.load(name)
periods = data["period"]
parities = data["parity"]
size = len(periods)


# prefix = "C:/Users/ARakc/Dropbox/data/agp/"
# prefix = "D:/Dropbox/data/agp/"
prefix = ""

name = "optimal_coefficients_ltxy_TPY_l" + str(l) + "_s0=" + str(s_0) + "_s1=" + str(s_1) + "_s2=" + str(s_2) \
       + "_res_1=" + str(res_1) + "_res_2=" + str(res_2) + "_start1=" + str(start1).replace(".", "-") \
       + "_end1=" + str(end_1).replace(".", "-") + "_start2=" + str(start2).replace(".", "-") + "_end2=" \
       + str(end_2).replace(".", "-") + ".npz"


data = np.load(prefix + name)
params1 = data["p1"]
params2 = data["p2"]
coefficients_1 = data["c1"]
coefficients_2 = data["c2"]

# create operators up to range r and save them in memory
operators = []
for k in np.arange(1, r + 1, 1):

    # fill the strings up to the correct system size
    op_file = prefix + "operators/operators_TPY_l" + str(k) + ".txt"
    with open(op_file, "r") as readfile:
        for line in readfile:
            op_str = line[0:k]

            mat_name = prefix_ops + op_str + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
            data = np.load(mat_name)
            indptr = data["indptr"]
            indices = data["idx"]
            val = data["val"]
            mat = sp.csr_matrix((val, indices, indptr), shape=(size, size))

            operators.append(mat)


prefix = "C:/Users/ARakc/Dropbox/data/agp/"
# prefix = "D:/Dropbox/data/agp/"
# prefix = ""

name = "groundstate_ltxy_TPY_L" + str(L) + "_s0=" + str(s_0) + "_s1=" + str(s_1) + "_s2=" + str(s_2) \
       + "_res_1=" + str(res_1) + "_res_2=" + str(res_2) + "_start1=" + str(start1).replace(".", "-") \
       + "_end1=" + str(end_1).replace(".", "-") + "_start2=" + str(start2).replace(".", "-") + "_end2=" \
       + str(end_2).replace(".", "-") + ".npz"

data = np.load(prefix + name)
evec = data["evec"]


# compute metric through expectation values
# for each parameter point compute the
metric = np.zeros((res_1, res_2, 2, 2))

start_time = time.time()
for i in range(res_1):
    print(i)

    for j in range(res_2):

        c_1 = coefficients_1[i, j, :]
        c_2 = coefficients_2[i, j, :]

        A_1 = sp.csr_matrix((size, size), dtype=np.complex128)
        A_2 = sp.csr_matrix((size, size), dtype=np.complex128)

        for k in range(len(operators)):

            mat = operators[k]

            A_1 = A_1 + c_1[k] * mat
            A_2 = A_2 + c_2[k] * mat

        ground_state = evec[i, j, :]

        state_h = A_1 @ ground_state
        state_g = A_2 @ ground_state

        metric[i, j, 0, 0] = (state_h.T @ state_h - (ground_state.T @ state_h) ** 2).real
        metric[i, j, 1, 1] = (state_g.T @ state_g - (ground_state.T @ state_g) ** 2).real
        metric[i, j, 0, 1] = (state_g.T @ state_h - (ground_state.T @ state_h) * (ground_state.T @ state_g)).real
        metric[i, j, 1, 0] = metric[i, j, 0, 1]

end_time = time.time()
print("Time:", end_time - start_time)

# plotting
plt.figure(1, figsize=(9, 3.25))
cmap = "inferno_r"

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
plt.grid()
plt.quiver(X, Y, minor_u, minor_v, norm / L, norm=colors.LogNorm(vmin=10 ** -3, vmax=10 ** 0), cmap=cmap, pivot="mid")
plt.colorbar()

plt.xlim(xl[0], xl[-1])
plt.ylim(yl[0], yl[-1])
plt.xlabel(r"$g$")
plt.ylabel(r"$h$")

# plt.savefig("ltxy_groundstate.pdf", format="pdf")
plt.show()