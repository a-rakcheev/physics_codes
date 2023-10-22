import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg

from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tqdm

# L = int(sys.argv[1])
L = 6
k_idx = 0  # momentum index (full momentum is (2 pi / L) * index), should be 0, 1, ..., L - 1
k_name = "0"  # currently 0 or pi (k_idx = L // 2) available
par = 1

res_g = 129                                        # number of grid points on x axis
res_h = 129                                        # number of grid points on y axis

gl = np.concatenate((np.linspace(1, 9, res_g) * 10 ** -5, np.linspace(1, 9, res_g) * 10 ** -4, np.linspace(1, 9, res_g) * 10 ** -3, np.linspace(1, 9, res_g) * 10 ** -2, np.linspace(1, 9, res_g) * 10 ** -1, np.linspace(1, 5, (res_g + 1) // 2) * 10 ** 0))
hl = np.concatenate((np.linspace(1, 9, res_h) * 10 ** -5, np.linspace(1, 9, res_h) * 10 ** -4, np.linspace(1, 9, res_h) * 10 ** -3, np.linspace(1, 9, res_h) * 10 ** -2, np.linspace(1, 9, res_h) * 10 ** -1, np.linspace(1, 5, (res_h + 1) // 2) * 10 ** 0))

# print(hl)
# print(gl)

prefix = "/home/artem/Dropbox/dqs/operators_TPY/"
# prefix = "D:/Dropbox/dqs/operators_TPY/"
# prefix = "C:/Users/ARakc/Dropbox/dqs/operators_TPY/"

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

op_name = "z"
mat_name = prefix + op_name + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
h_z = 0.5 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).real

op_name = "zz"
mat_name = prefix + op_name + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
op_zz = 0.5 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).real

op_name = "z1z"
mat_name = prefix + op_name + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
op_z1z = 0.5 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).real

op_name = "z11z"
mat_name = prefix + op_name + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
op_z11z = 0.5 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).real

op_name = "z111z"
mat_name = prefix + op_name + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
op_z111z = 0.5 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).real

op_id = L * sp.identity(size, dtype=np.complex128, format="csr")

h_n = 0.5 * (op_id + h_z)
h_nn = 0.25 * (op_id + 2 * h_z + op_zz)
h_n1n = 0.25 * (op_id + 2 * h_z + op_z1z)
h_n11n = 0.25 * (op_id + 2 * h_z + op_z11z)
h_n111n = 0.25 * (op_id + 2 * h_z + op_z111z)

# ham_zdz
h_ndn = h_nn + (1 / 2 ** 6) * h_n1n + (1 / 3 ** 6) * h_n11n 

# arrays for measurements
exp_x = np.zeros((len(gl), len(hl)))
exp_n = np.zeros((len(gl), len(hl)))
exp_nn = np.zeros((len(gl), len(hl)))
exp_n1n = np.zeros((len(gl), len(hl)))
exp_n11n = np.zeros((len(gl), len(hl)))
exp_n111n = np.zeros((len(gl), len(hl)))

print("Number of States:", size)
for i, g in enumerate(tqdm.tqdm(gl)):
    for j, h in enumerate(hl):

        # h_tot = h_ndn - g * h_x - h * h_z
        # change h_z to h_n ? adds constant ? 
        # spectrum
        ev, evec = linalg.eigsh(h_tot, k=1, which="SA")

        # ground state expectation values
        exp_x[i, j] = (evec[:, 0].T.conj() @ h_x @ evec[:, 0]).real / L
        exp_n[i, j] = (evec[:, 0].T.conj() @ h_n @ evec[:, 0]).real / L
        exp_nn[i, j] = (evec[:, 0].T.conj() @ h_nn @ evec[:, 0]).real / L
        exp_n1n[i, j] = (evec[:, 0].T.conj() @ h_n1n @ evec[:, 0]).real / L
        exp_n11n[i, j] = (evec[:, 0].T.conj() @ h_n11n @ evec[:, 0]).real / L
        exp_n111n[i, j] = (evec[:, 0].T.conj() @ h_n111n @ evec[:, 0]).real / L

np.savez_compressed("gs_ed_L=" + str(L) + "_res_g=" + str(res_g) + "_res_h=" + str(res_h) + ".npz", x=exp_x, n=exp_n, nn=exp_nn, n1n=exp_n1n, n11n=exp_n11n, n111n=exp_n111n, hl=hl, gl=gl)

plt.figure(1, figsize=(18, 10))
cmap = "jet"

plt.subplot(2, 3, 1)
plt.pcolormesh(gl, hl, exp_x.T, norm=colors.LogNorm(vmin=1.e-3, vmax=1.), cmap=cmap)
plt.colorbar()

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$g$", fontsize=12)
plt.ylabel(r"$h$", fontsize=12)
plt.title(r"$\langle X \rangle / L$", fontsize=14)


plt.subplot(2, 3, 2)
plt.pcolormesh(gl, hl, exp_n.T, norm=colors.LogNorm(vmin=1.e-3, vmax=1.), cmap=cmap)
plt.colorbar()

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$g$", fontsize=12)
plt.ylabel(r"$h$", fontsize=12)
plt.title(r"$\langle N \rangle / L$", fontsize=14)


plt.subplot(2, 3, 3)
plt.pcolormesh(gl, hl, exp_nn.T, norm=colors.LogNorm(vmin=1.e-3, vmax=1.), cmap=cmap)
plt.colorbar()

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$g$", fontsize=12)
plt.ylabel(r"$h$", fontsize=12)
plt.title(r"$\langle NN \rangle / L$", fontsize=14)


plt.subplot(2, 3, 4)
plt.pcolormesh(gl, hl, exp_n1n.T, norm=colors.LogNorm(vmin=1.e-3, vmax=1.), cmap=cmap)
plt.colorbar()

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$g$", fontsize=12)
plt.ylabel(r"$h$", fontsize=12)
plt.title(r"$\langle N1N \rangle / L$", fontsize=14)


plt.subplot(2, 3, 5)
plt.pcolormesh(gl, hl, exp_n11n.T, norm=colors.LogNorm(vmin=1.e-3, vmax=1.), cmap=cmap)
plt.colorbar()

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$g$", fontsize=12)
plt.ylabel(r"$h$", fontsize=12)
plt.title(r"$\langle N11N \rangle / L$", fontsize=14)


plt.subplot(2, 3, 6)
plt.pcolormesh(gl, hl, exp_n111n.T, norm=colors.LogNorm(vmin=1.e-3, vmax=1.), cmap=cmap)
plt.colorbar()

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$g$", fontsize=12)
plt.ylabel(r"$h$", fontsize=12)
plt.title(r"$\langle N111N \rangle / L$", fontsize=14)

plt.subplots_adjust(wspace=0.25, hspace=0.25, top=0.925, bottom=0.1, left=0.1, right=0.95)
plt.show()
