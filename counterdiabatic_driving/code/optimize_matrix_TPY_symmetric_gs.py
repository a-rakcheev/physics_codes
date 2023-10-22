import pauli_string_functions as pauli_func
import numpy as np
from scipy.sparse import save_npz, csr_matrix
from scipy.sparse.linalg import eigsh

import sys
from mpi4py import MPI


def parity(op_name):

    p = 0
    if op_name == op_name[::-1]:
        p = 1

    return p


l = 2
L = 2 * l
param1 = 1.e-6
param2 = 1.e-6

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

# adjust factors due to parity
# the parity of the operator for example zz leads to double counting
# # when creating 1zz1 + zz11 + 11zz + z11z + 1zz1 + 11zz + zz11 + z11z, in contrast to
# # 1xy1 + xy11 + 11xy + y11x + 1yx1 + 11yx + yx11 + x11y
# # the parity is 0, 1 in these cases

factor_0 = 0.5 * (2. - parity(op_name_0))
factor_1 = 0.5 * (2. - parity(op_name_1))
factor_2 = 0.5 * (2. - parity(op_name_2))

prefix = "D:/Dropbox/dqs/operators_TPY/"
# prefix = "C:/Users/ARakc/Dropbox/dqs/operators_TPY/"

# size
name = prefix + "1d_chain_TP_indices_and_periods_L=" + str(L) + "_k=" + k_name + ".npz"
data = np.load(name)
periods = data["period"]
parities = data["parity"]
size = len(periods)


# hamiltonians (if real)
mat_name = prefix + op_name_0 + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
h_0 = factor_0 * csr_matrix((val, indices, indptr), shape=(size, size)).real

mat_name = prefix + op_name_1 + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
h_1 = factor_1 * csr_matrix((val, indices, indptr), shape=(size, size)).real

mat_name = prefix + op_name_2 + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
data = np.load(mat_name)
indptr = data["indptr"]
indices = data["idx"]
val = data["val"]
h_2 = factor_2 * csr_matrix((val, indices, indptr), shape=(size, size)).real

ham = sign_0 * h_0 + sign_1 * param1 * h_1 + sign_2 * param2 * h_2
ev, evec = eigsh(ham, k=1, which="SA")
state = evec[:, 0]

ham = None
h_0 = None
h_1 = None
h_2 = None

names = ["x"]

for op_name_1 in names:

    # initialize
    comm = MPI.COMM_WORLD

    # get rank
    rank = comm.Get_rank()

    # size
    number_of_processes = comm.Get_size()

    # time each process
    start_time = MPI.Wtime()

    # read in basis operators
    num_op = np.loadtxt("operators/operators_TPY_size_full.txt").astype(int)
    size = num_op[l - 1]
    TPY_labels, TPY_coeffs = pauli_func.create_operators_TPY_compact(l, L, size)

    # create commutators
    # create TPY operators from file
    lab_1, c_1 = pauli_func.TP_operator_from_string_compact(op_name_1, L)

    # commutators
    C_1_labels, C_1_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_1, c_1, L)

    step = size // number_of_processes
    rest = size - number_of_processes * step
    if rank == 0:

        print(size, step, number_of_processes * step, rest)


    # create lists with indices
    triplets = []
    for i in range(rank, size, number_of_processes):
        comm_label_i = C_1_labels[i, :, :]
        comm_coeff_i = C_1_coeffs[i, :]

        for j in range(i, size):
            comm_label_j = C_1_labels[j, :, :]
            comm_coeff_j = C_1_coeffs[j, :]

            labels, coeff = pauli_func.multiply_operators_TP(comm_label_i, comm_label_j, comm_coeff_i, comm_coeff_j)

            print(labels)
            print(coeff)

            labels, coeff = pauli_func.operator_cleanup(labels, coeff)
            print(labels)
            print(coeff)

            # create operator
            prod_op = pauli_func.trace_operator_TP(labels, coeff)



            # expectation value
            tr = state.T.conj() @ prod_op @ state

            if abs(tr.real) > 1.e-14:
                triplets.append([i, j, -tr.real / L])
                if i != j:
                    triplets.append([j, i, -tr.real / L])


    end_time = MPI.Wtime()
    # print("Time:", end_time - start_time, rank)

    # gather list
    process_list = comm.gather(triplets, root=0)

    # create sparse matrix
    if rank == 0:

        idx_row = []
        idx_col = []
        val = []

        for triplet_list in process_list:
            for triplet in triplet_list:

                idx_row.append(triplet[0])
                idx_col.append(triplet[1])
                val.append(triplet[2])

        mat = csr_matrix((val, (idx_row, idx_col)), shape=[size, size])
        # print(mat)

        # name = "optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_1.upper() + "_TPY_l=" + str(l) + ".npz"
        # save_npz(name, mat)