import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

L = 10
k_idx = 0  # momentum index (full momentum is (2 pi / L) * index), should be 0, 1, ..., L - 1
k_name = "0"  # currently 0 or pi (k_idx = L // 2) available
par = 1

s_0 = 0                                            # signs of operators
s_1 = 1
s_2 = 1

sign_0 = (-1) ** s_0
sign_1 = (-1) ** s_1
sign_2 = (-1) ** s_2

op_name_0 = "zz"                                # operators in the hamiltonian
op_name_1 = "z"
op_name_2 = "x"

res_1 = 100                                        # number of grid points on x axis
res_2 = 100                                       # number of grid points on y axis
step1 = 2
step2 = 4

start1 = 1.e-6
start2 = 1.e-6
end_1 = 3.
end_2 = 1.5

params1 = np.linspace(start1, end_1, res_1)
params2 = np.linspace(start2, end_2, res_2)
dp_1 = params1[1] - params1[0]
dp_2 = params2[1] - params2[0]

param_label_1 = r"$h$"
param_label_2 = r"$g$"

prefix_save = "C:/Users/ARakc/Dropbox/data/agp/"
# prefix_save = "D:/Dropbox/data/agp/"
# prefix_save = ""

name = "groundstate_TPY_L" + str(L) + "_op0=" + op_name_0 + "_op1=" + op_name_1 + "_op2=" + op_name_2\
       + "_s0=" + str(s_0) + "_s1=" + str(s_1) + "_s2=" + str(s_2) \
       + "_res_1=" + str(res_1) + "_res_2=" + str(res_2) + "_start1=" + str(start1).replace(".", "-") \
       + "_end1=" + str(end_1).replace(".", "-") + "_start2=" + str(start2).replace(".", "-") + "_end2=" \
       + str(end_2).replace(".", "-") + ".npz"

data = np.load(prefix_save + name)
ev = data["ev"]
ground_state = data["evec"]

# fidelity (squared)
F_1 = np.zeros((res_1 - 1, res_2 - 1))
F_2 = np.zeros((res_1 - 1, res_2 - 1))
F_12 = np.zeros((res_1 - 1, res_2 - 1))

for i in range(res_1):
    for j in range(res_2):

        state1 = ground_state[i, j, :]

        if i != (res_1 - 1) and j != (res_2 - 1):
            state2 = ground_state[i + 1, j, :]
            F_1[i, j] = np.absolute(state1.T.conj() @ state2) ** 2

            state2 = ground_state[i, j + 1, :]
            F_2[i, j] = np.absolute(state1.T.conj() @ state2) ** 2

            state2 = ground_state[i + 1, j + 1, :]
            F_12[i, j] = np.absolute(state1.T.conj() @ state2) ** 2


# metric components
g_11 = (np.ones_like(F_1) - F_1) / (dp_1 ** 2)
g_22 = (np.ones_like(F_2) - F_2) / (dp_2 ** 2)
g_12 = 0.5 * (F_1 + F_2 - F_12 - np.ones_like(F_12)) / (dp_1 * dp_2)

metric = np.zeros((res_1 - 1, res_2 - 1, 2, 2))
metric[:, :, 0, 0] = g_11
metric[:, :, 1, 1] = g_22
metric[:, :, 0, 1] = g_12
metric[:, :, 1, 0] = g_12

# metric_norm = np.zeros((res_1 - 1, res_2 - 1))
# for i in range(res_1 - 1):
#     for j in range(res_2 - 1):
#
#         metric_norm[i, j] = np.linalg.norm(metric[i, j, :, :], "nuc")
#
#
# plt.pcolormesh(params1, params2, metric_norm.T / L, cmap="jet", norm=colors.LogNorm(vmin=10 ** -4, vmax=10 ** 2))
# plt.colorbar()

# plotting
plt.figure(1, figsize=(6, 3.25), constrained_layout=True)
cmap = "jet"

xl = params1
yl = params2
X, Y = np.meshgrid(xl[::step1], yl[::step2])

major_u = np.zeros_like(X)
major_v = np.zeros_like(X)
minor_u = np.zeros_like(X)
minor_v = np.zeros_like(X)

major_norm = np.zeros_like(X)
minor_norm = np.zeros_like(X)
norm = np.zeros_like(X)

for i, x in enumerate(xl[::step1]):
    for j, y in enumerate(yl[::step2]):

        g = metric[i * step1, j * step2, :, :]

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
plt.quiver(X, Y, minor_u, minor_v, norm / L, norm=colors.LogNorm(vmin=10 ** -3, vmax=5 * 10 ** 0), cmap=cmap, pivot="tail")
plt.quiver(X, Y, -minor_u, -minor_v, norm / L, norm=colors.LogNorm(vmin=10 ** -3, vmax=5 * 10 ** 0), cmap=cmap, pivot="tail")

plt.colorbar()

plt.xlim(xl[0], xl[-1])
plt.ylim(yl[0], yl[-1])
plt.xlabel(param_label_1)
plt.ylabel(param_label_2)

name = "gs_metric_from_state_TPY_L" + str(L) + "_op0=" + op_name_0 + "_op1=" + op_name_1 + "_op2=" + op_name_2\
       + "_s0=" + str(s_0) + "_s1=" + str(s_1) + "_s2=" + str(s_2) \
       + "_res_1=" + str(res_1) + "_res_2=" + str(res_2) + "_start1=" + str(start1).replace(".", "-") \
       + "_end1=" + str(end_1).replace(".", "-") + "_start2=" + str(start2).replace(".", "-") + "_end2=" \
       + str(end_2).replace(".", "-") + ".pdf"

plt.savefig(name, format="pdf")
plt.show()



