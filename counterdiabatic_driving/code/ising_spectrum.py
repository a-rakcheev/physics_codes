import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

L = 8
k_idx = 0  # momentum index (full momentum is (2 pi / L) * index), should be 0, 1, ..., L - 1
k_name = "0"  # currently 0 or pi (k_idx = L // 2) available
par = 1

J = 1.0
g_res = 1001
h_res = 1001

prefix = "operators_TPY/"

# state
name = prefix + "1d_chain_TP_indices_and_periods_L=" + str(L) + "_k=" + k_name + ".npz"
data = np.load(name)
periods = data["period"]
parities = data["parity"]

size = len(periods)
state = np.zeros(size)
state2 = np.zeros(size)

data = None
periods = None
parities = None

# hamiltonians
op_name = "x"
mat_name = prefix + op_name + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
h_x = 0.5 * 0.5 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).todense()

# ham_z is diagonal
op_name = "zz"
mat_name = prefix + op_name + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
h_zz = 0.5 * 0.25 * sp.csr_matrix((val, indices, indptr), shape=(size, size))

op_name = "z"
mat_name = prefix + op_name + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
h_z = 0.5 * 0.5 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).todense()

# gl = np.linspace(0., 2., g_res)
# spectrum = np.zeros((g_res, 2 ** L))
#
# for i, g in enumerate(gl):
#
#     print(i)
#     h_tfi = J * h_zz - g * h_x
#     spectrum[i, :] = np.linalg.eigvalsh(h_tfi)
#
# plt.subplot(1, 2, 1)
# plt.plot(gl, spectrum, color="black")
# plt.grid()
# plt.xlabel(r'$g$', fontsize=12)
# plt.ylabel(r'$E$', fontsize=12)
#
#
# hl = np.linspace(0., 2., h_res)
# spectrum = np.zeros((h_res, 2 ** L))
# for i, h in enumerate(hl):
#
#     print(i)
#     h_lfi = J * h_zz - h * h_z
#     spectrum[i, :] = np.linalg.eigvalsh(h_lfi)
#
# plt.subplot(1, 2, 2)
#
# plt.plot(hl, spectrum, color="black")
# plt.grid()
# plt.xlabel(r'$h$', fontsize=12)
# plt.ylabel(r'$E$', fontsize=12)

gl = np.linspace(0., 1.5, g_res)
spectrum = np.zeros((g_res, size))

for i, g in enumerate(gl):

    print(i)
    h_mix = J * h_zz - g * h_z
    spectrum[i, :] = np.linalg.eigvalsh(h_mix)

plt.plot(gl, spectrum, color="black")
plt.grid()

plt.axvline(1.0, ls="--", color="red")
plt.axvline(0.5, ls="--", color="blue")

plt.xlabel(r'$h$', fontsize=21)
plt.ylabel(r'$E$', fontsize=21)

plt.show()
