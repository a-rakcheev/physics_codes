import pauli_string_functions as pauli_func
import numpy as np
from scipy.sparse import save_npz, csr_matrix
import sys
from mpi4py import MPI


def parity(op_name):

    p = 0
    if op_name == op_name[::-1]:
        p = 1

    return p


l = int(sys.argv[1])
# L = int(sys.argv[2])
# op_name_1 = str(sys.argv[3])

# l = 3
L = 2 * l + 1
# op_name_1 = "x"
# op_name_2 = "z"


names = ["x", "z", "xx", "xz", "yy", "zz", "x1x", "x1z", "xxx", "xxz", "xyy", "xzx", "xzz",
         "y1y", "yxy", "yyz", "yzy", "z1z", "zxz", "zzz"]

for op_name_1 in names:

    tr_I = 2 ** L

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

    # # set spin operators
    # # first factor due to range of operator, second due to parity of the operator for example zz leads to double counting
    # # when creating 1zz1 + zz11 + 11zz + z11z + 1zz1 + 11zz + zz11 + z11z, in contrast to
    # # 1xy1 + xy11 + 11xy + y11x + 1yx1 + 11yx + yx11 + x11y
    # # the parity is 0, 1 in these cases
    #
    # c_1 *= (0.5 ** len(op_name_1)) * 0.5 * (2. - parity(op_name_1))

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
            tr = pauli_func.trace_operator_TP(labels, coeff)

            if abs(tr.real) > 1.e-14:
                triplets.append([i, j, -tr.real / (L * tr_I)])
                if i != j:
                    triplets.append([j, i, -tr.real / (L * tr_I)])


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

        name = "optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_1.upper() + "_TPY_l=" + str(l) + ".npz"
        save_npz(name, mat)