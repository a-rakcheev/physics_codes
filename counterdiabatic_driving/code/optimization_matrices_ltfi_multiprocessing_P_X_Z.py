import pauli_string_functions as pauli_func
import numpy as np
import scipy.sparse as sp
import multiprocessing as mp
import sys

# P matrix
# P_kl = -Tr(C_k C_l), which is real-valued
def create_P_matrix_TP_multiprocessing(index):

    i = index
    print("i", i, flush=True)

    for j in range(size):

        comm_label_i = C_1_labels[i, :, :]
        comm_coeff_i = C_1_coeffs[i, :]

        comm_label_j = C_2_labels[j, :, :]
        comm_coeff_j = C_2_coeffs[j, :]

        labels, coeff = pauli_func.multiply_operators_TP(comm_label_i, comm_label_j, comm_coeff_i, comm_coeff_j)

        tr = pauli_func.trace_operator_TP(labels, coeff)
        P[j + i * size] = -tr.real


if __name__ == "__main__":

    # l = int(sys.argv[1])
    # L = int(sys.argv[2])
    # number_of_processes = int(sys.argv[3])

    l = 4
    L = 4
    number_of_processes = 1
    tr_I = 2 ** L

    # create TPY operators from file
    lab_1, c_1 = pauli_func.TP_operator_from_string_compact("x", L)
    lab_2, c_2 = pauli_func.TP_operator_from_string_compact("z", L)

    # set spin operators
    c_1 *= 0.5 * 0.5
    c_2 *= 0.5 * 0.5

    # read in basis operators
    num_op = np.loadtxt("operators/operators_TPY_size_full.txt").astype(int)
    size = num_op[l - 1]
    TPY_labels, TPY_coeffs = pauli_func.create_operators_TPY_compact(l, L, size)

    # commutators
    C_1_labels, C_1_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_1, c_1, L)
    C_2_labels, C_2_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_2, c_2, L)
    print("Commutators computed", flush=True)

    # create shared array (need to change type of size to int, since mp does not accept np.int64)
    P = mp.Array('d', int(size) ** 2)
    pool = mp.Pool(processes=number_of_processes)
    comp = pool.map_async(create_P_matrix_TP_multiprocessing, [i for i in range(size)])
    comp.wait()
    pool.terminate()

    P = np.array(P).reshape(size, size) / (tr_I * L)
    name = "ltfi_optimization_matrices_P_X_Z_mp_L=" + str(L) + "_l=" + str(l) + ".npz"
    np.savez_compressed(name, P=P)