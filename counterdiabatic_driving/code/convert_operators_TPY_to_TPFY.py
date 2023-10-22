# convert operators from K=0, P=1 sector to K=0, P=1, F=1 sector

import zipfile
import io

import numpy as np
import scipy.sparse as sp
import time

# parameters
L = 10

# spin_flip_operator
# operators
k_idx = 0                           # momentum index (full momentum is (2 pi / L) * index), should be 0, 1, ..., L - 1
k_name = "0"                        # currently 0 or pi (k_idx = L // 2) available
par = 1                             # parity sector

np.set_printoptions(2, linewidth=200)

# get size
name_zip = "operators/1d_chain_indices_and_periods.zip"
with zipfile.ZipFile(name_zip) as zipper:

    name = "1d_chain_TP_indices_and_periods_L=" + str(L) + "_k=" + k_name + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:

        data = np.load(f)
        periods = data["period"]
        parities = data["parity"]
        size = len(periods)

name_zip_op = "operators/operators_TP_L=" + str(L) + ".zip"
with zipfile.ZipFile(name_zip_op) as zipper_op:

    # hamiltonian operators
    mat_name = L * "x" + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]

    F = sp.csr_matrix((val, indices, indptr), shape=(size, size)).todense() / (2 * L)


ev_F, evec_F = np.linalg.eigh(F)
print("F diagonalized")
print(ev_F)
size_minus = np.count_nonzero(ev_F + 1 == 0.)
size_plus = size - size_minus
print(size_minus, size_plus)

# size_file = "sizes_TPF.txt"
# with open(size_file, "a") as writefile:
#     writefile.write(str(L) + "," + str(size_plus) + " \n")

# # project into basis
# F_proj = evec_F.T.conj() @ F @ evec_F
# F_plus = F_proj[size_minus:, size_minus:]


name_zip_op = "operators/operators_TP_L=" + str(L) + ".zip"
with zipfile.ZipFile(name_zip_op) as zipper_op:

    # all operators with range up to 8
    for l in np.arange(2, 3):

        start_time = time.time()
        # op_names = np.loadtxt("operators/operators_TPFY_l" + str(l) + ".txt", dtype=str)
        # op_names = ["yz"]
        op_names = ["z", "x", "zz", "z1z", "xx", "yy"]

        print(l, len(op_names))

        for op_name in op_names:

            # hamiltonian operators
            mat_name = op_name + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
            with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
                data = np.load(f_op)
                indptr = data["indptr"]
                indices = data["idx"]
                val = data["val"]

            op = sp.csr_matrix((val, indices, indptr), shape=(size, size)).todense()
            op_plus = F_proj = (evec_F.T.conj() @ op @ evec_F)[size_minus:, size_minus:]

            op_plus = sp.csr_matrix(op_plus, shape=(size_plus, size_plus))
            print(op_plus)

            mat_name = "operators/" + op_name + "_TPF_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
            size_name = "operators/" + op_name + "_TPF_L_k=" + k_name + "_p=" + str(par) + ".txt"

            # with open(size_name, "a") as writefile:
            #     writefile.write(str(L) + " " + str(len(op_plus.data)) + "\n")
            # np.savez_compressed(mat_name, indptr=op_plus.indptr, idx=op_plus.indices, val=op_plus.data)

        end_time = time.time()
        print("Conversion Time:", end_time - start_time)