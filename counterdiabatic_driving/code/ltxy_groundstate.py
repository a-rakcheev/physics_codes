import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import sys

# L = int(sys.argv[1])
L = 16
k_idx = 0  # momentum index (full momentum is (2 pi / L) * index), should be 0, 1, ..., L - 1
k_name = "0"  # currently 0 or pi (k_idx = L // 2) available
par = 1

res_1 = 400                                        # number of grid points on x axis
res_2 = 400                                        # number of grid points on y axis

start1 = 1.e-6
start2 = 1.e-6
end_1 = 4.
end_2 = 4.

s_0 = 0
s_1 = 1
s_2 = 1

sign_0 = (-1) ** s_0
sign_1 = (-1) ** s_1
sign_2 = (-1) ** s_2

params1 = np.linspace(start1, end_1, res_1)
params2 = np.linspace(start2, end_2, res_2)

prefix = "D:/Dropbox/dqs/operators_TPY/"

# state
name = prefix + "1d_chain_TP_indices_and_periods_L=" + str(L) + "_k=" + k_name + ".npz"
data = np.load(name)
periods = data["period"]
parities = data["parity"]

size = len(periods)
state = np.zeros(size)
state2 = np.zeros(size)

# hamiltonians
op_name = "x"
mat_name = prefix + op_name + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
h_x = 0.5 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).real

op_name = "z"
mat_name = prefix + op_name + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
h_z = 0.5 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).real

op_name = "xx"
mat_name = prefix + op_name + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
h_xx = 0.5 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).real

op_name = "yy"
mat_name = prefix + op_name + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
h_yy = 0.5 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).real

evals = np.zeros((res_1, res_2))
evecs = np.zeros((res_1, res_2, size), dtype=np.float64)

print("Number of States:", size)
for i, p_1 in enumerate(params1):
    print("i:", i)

    for j, p_2 in enumerate(params2):

        h_tot = sign_0 * (h_xx + h_yy) + sign_1 * p_1 * h_x + sign_2 * p_2 * h_z
        ev, evec = linalg.eigsh(h_tot, k=1, which="SA")

        evals[i, j] = ev[0]
        evecs[i, j, :] = evec[:, 0]

# prefix_save = "C:/Users/ARakc/Dropbox/data/agp/"
prefix_save = "D:/Dropbox/data/agp/"
# prefix_save = ""

name = "groundstate_ltxy_TPY_L" + str(L) + "_s0=" + str(s_0) + "_s1=" + str(s_1) + "_s2=" + str(s_2) \
       + "_res_1=" + str(res_1) + "_res_2=" + str(res_2) + "_start1=" + str(start1).replace(".", "-") \
       + "_end1=" + str(end_1).replace(".", "-") + "_start2=" + str(start2).replace(".", "-") + "_end2=" \
       + str(end_2).replace(".", "-") + ".npz"

np.savez_compressed(prefix_save + name, ev=evals, evec=evecs, p1=params1, p2=params2)
