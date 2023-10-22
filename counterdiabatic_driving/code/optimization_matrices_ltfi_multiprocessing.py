import pauli_string_functions as pauli_func
import numpy as np
import time
import multiprocessing as mp

# P matrix
# P_kl = -Tr(C_k C_l), which is real-valued
def create_P_matrix_TP_multiprocessing(index_tuple):

    i = index_tuple[0]
    j = index_tuple[1]

    print("i, j:", i, j)
    comm_label_i = C_x_labels[i, :, :]
    comm_coeff_i = C_x_coeffs[i, :]

    comm_label_j = C_zz_labels[j, :, :]
    comm_coeff_j = C_zz_coeffs[j, :]

    labels, coeff = pauli_func.multiply_operators_TP(comm_label_i, comm_label_j, comm_coeff_i, comm_coeff_j)

    tr = pauli_func.trace_operator_TP(labels, coeff)
    P[j + i * size] = -tr.real


if __name__ == "__main__":

    # create TPY operators from file
    l = 3
    L = 3
    number_of_processes = 2
    tr_I = 2 ** L

    lab_x, c_x = pauli_func.TP_operator_from_string_compact("x", L)
    lab_z, c_z = pauli_func.TP_operator_from_string_compact("z", L)
    lab_zz, c_zz = pauli_func.TP_operator_from_string_compact("zz", L)

    # set spin operators
    c_x *= 0.5 * 0.5
    c_z *= 0.5 * 0.5
    c_zz *= 0.25 * 0.5

    # read in basis operators
    num_op = np.loadtxt("operators/operators_TPY_size_full.txt").astype(int)
    size = num_op[l - 1]
    TPY_labels, TPY_coeffs = pauli_func.create_operators_TPY_compact(l, L, size)
    print("size:", size)

    start = time.time()
    # commutators
    C_x_labels, C_x_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_x, c_x, L)
    C_z_labels, C_z_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_z, c_z, L)
    C_zz_labels, C_zz_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_zz, c_zz, L)
    print("Commutators computed", flush=True)
    end = time.time()
    print("Time:", end - start)

    start = time.time()
    # create shared array (need to change type of size to int, since mp does not accept np.int64)
    P = mp.Array('d', int(size) ** 2)
    pool = mp.Pool(processes=number_of_processes)
    comp = pool.map_async(create_P_matrix_TP_multiprocessing, [(i, j) for i in range(size) for j in range(size)])
    comp.wait()
    pool.terminate()
    end = time.time()
    print("Time:", end - start)

    P_X_ZZ = np.array(P).reshape(size, size) / (tr_I * L)
    print(P_X_ZZ)

