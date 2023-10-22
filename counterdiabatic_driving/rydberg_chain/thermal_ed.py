import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg

from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tqdm

# L = int(sys.argv[1])
L = 16
k_idx = 0  # momentum index (full momentum is (2 pi / L) * index), should be 0, 1, ..., L - 1
k_name = "0"  # currently 0 or pi (k_idx = L // 2) available
par = 1

# g = 1.e-4
# h = 1.e-0

steps = 17
betal = np.concatenate((np.linspace(1, 9, steps) * 10 ** -3, np.linspace(1, 9, steps) * 10 ** -2, np.linspace(1, 9, steps) * 10 ** -1, np.linspace(1, 9, steps) * 10 ** 0, np.linspace(1, 9, steps) * 10 ** 1))

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

h_n = 0.5 * (op_id + h_z).todense()
h_nn = 0.25 * (op_id + 2 * h_z + op_zz).todense()
h_n1n = 0.25 * (op_id + 2 * h_z + op_z1z).todense()
h_n11n = 0.25 * (op_id + 2 * h_z + op_z11z).todense()
h_n111n = 0.25 * (op_id + 2 * h_z + op_z111z).todense()

# ham_zdz
h_ndn = h_nn + (1 / 2 ** 6) * h_n1n + (1 / 3 ** 6) * h_n11n 

for h in [1.0, 0.1, 0.01, 0.0005]:
    for g in [1.0, 0.1, 0.01, 0.0005]:

        print("h, g:", h, g)

        # spectrum
        h_tot = h_ndn - g * h_x - h * h_n
        ev, evec = np.linalg.eigh(h_tot)
        evec = evec.A

        exp_x = np.zeros(len(betal))
        exp_n = np.zeros(len(betal))
        exp_nn = np.zeros(len(betal))
        exp_n1n = np.zeros(len(betal))
        exp_n11n = np.zeros(len(betal))
        exp_n111n = np.zeros(len(betal))
        partition_function = np.zeros(len(betal))

        for i, beta in enumerate(tqdm.tqdm(betal)):

            Z = np.sum(np.exp(-beta * ev))
            rho = evec @ np.diag(np.exp(-beta * ev)) @ evec.T.conj() / Z

            # expectation values
            partition_function[i] = Z
            exp_x[i] = np.trace(rho @ h_x).real / L
            exp_n[i] = np.trace(rho @ h_n).real / L
            exp_nn[i] = np.trace(rho @ h_nn).real / L
            exp_n1n[i] = np.trace(rho @ h_n1n).real / L
            exp_n11n[i] = np.trace(rho @ h_n11n).real / L
            exp_n111n[i] = np.trace(rho @ h_n111n).real / L

        np.savez_compressed("thermal_ed_logscale_L=" + str(L) + "_steps=" + str(steps) + "_g=" + str(g).replace(".", "-") + "_h=" + str(h).replace(".", "-") + ".npz", betal=betal, Z=partition_function, x=exp_x, n=exp_n, nn=exp_nn, n1n=exp_n1n, n11n=exp_n11n, n111n=exp_n111n)


