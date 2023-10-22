import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import sys

# L = int(sys.argv[1])
L = 18
k_idx = 0  # momentum index (full momentum is (2 pi / L) * index), should be 0, 1, ..., L - 1
k_name = "0"  # currently 0 or pi (k_idx = L // 2) available
par = 1

res_h = 50                                        # number of grid points on x axis
res_g = 50                                        # number of grid points on y axis

hl = np.linspace(1.e-6, 3.0, res_h)
gl = np.linspace(1.e-6, 1.5, res_g)

# prefix = "operators_TPY/"
# prefix = "D:/Dropbox/dqs/operators_TPY/"
prefix = "C:/Users/ARakc/Dropbox/dqs/operators_TPY/"

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
h_x = 0.5 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).real

# ham_z is diagonal
op_name = "zz"
mat_name = prefix + op_name + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
h_zz = 0.5 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).real

op_name = "z"
mat_name = prefix + op_name + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
h_z = 0.5 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).real

evals = np.zeros((res_h, res_g))
evecs = np.zeros((res_h, res_g, size), dtype=np.float64)

print("Number of States:", size)
for i, h in enumerate(hl):
    print("i:", i)
    for j, g in enumerate(gl):

        h_tot = h_zz - g * h_x - h * h_z
        ev, evec = linalg.eigsh(h_tot, k=1, which="SA")

        evals[i, j] = ev[0]
        evecs[i, j, :] = evec[:, 0]

prefix_save = "C:/Users/ARakc/Dropbox/data/agp/"
# prefix_save = "D:/Dropbox/data/agp/"
# prefix_save = ""
name = prefix_save + "ltfi_gs_ed_k=0_p=1_L=" + str(L) + "_h_res=" + str(res_h) + "_g_res=" + str(res_g) + ".npz"
np.savez_compressed(name, ev=evals, evec=evecs, hl=hl, gl=gl)
