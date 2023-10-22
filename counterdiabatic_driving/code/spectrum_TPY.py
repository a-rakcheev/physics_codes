import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import sys

def parity(op_name):
    p = 0
    if op_name == op_name[::-1]:
        p = 1

    return p

L = 10
k_idx = 0  # momentum index (full momentum is (2 pi / L) * index), should be 0, 1, ..., L - 1
k_name = "0"  # currently 0 or pi (k_idx = L // 2) available
par = 1

s_0 = 1                                            # signs of operators
s_1 = 0
s_2 = 1

op_name_0 = "zz"                                # operators in the hamiltonian
op_name_1 = "z1z"
op_name_2 = "x"

res_1 = 100                                        # number of grid points on x axis
res_2 = 2                                       # number of grid points on y axis

start1 = 1.e-6
start2 = 1.e-6
end_1 = 2.
end_2 = 2.

params1 = np.linspace(start1, end_1, res_1)
params2 = np.linspace(start2, end_2, res_2)

sign_0 = (-1) ** s_0
sign_1 = (-1) ** s_1
sign_2 = (-1) ** s_2

factor_0 = 0.5 * (2. - parity(op_name_0))
factor_1 = 0.5 * (2. - parity(op_name_1))
factor_2 = 0.5 * (2. - parity(op_name_2))

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
mat_name = prefix + op_name_0 + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
h_0 = factor_0 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).real

mat_name = prefix + op_name_1 + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
h_1 = factor_1 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).real

mat_name = prefix + op_name_2 + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
h_2 = factor_2 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).real

evals = np.zeros((res_1, res_2, size))

print("Number of States:", size)
for i, p_1 in enumerate(params1):
    for j, p_2 in enumerate(params2):

        print("i, j:", i, j)
        h_tot = sign_0 * h_0 + p_1 * sign_1 * h_1 + p_2 * sign_2 * h_2
        ev = np.linalg.eigvalsh(h_tot.todense())

        evals[i, j, :] = ev


plt.plot(params1, evals[:, 0, :], color="black")
plt.grid()

plt.show()
# prefix_save = "C:/Users/ARakc/Dropbox/data/agp/"
# # prefix_save = "D:/Dropbox/data/agp/"

# name = prefix_save + "ltfi_gs_ed_k=0_p=1_L=" + str(L) + "_h_res=" + str(res_h) + "_g_res=" + str(res_g) + ".npz"
# np.savez_compressed(name, ev=evals, evec=evecs, hl=hl, gl=gl)
