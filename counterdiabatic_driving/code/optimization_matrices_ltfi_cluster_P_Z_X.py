import pauli_string_functions as pauli_func
import numpy as np
import numba as nb
import sys

l = int(sys.argv[1])
L = int(sys.argv[2])
num_threads = int(sys.argv[3])

nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(num_threads)

print("Threads:", nb.get_num_threads())
print("Threading Layer: %s" % nb.threading_layer())

# create TPY operators from file
tr_I = 2 ** L

lab_x, c_x = pauli_func.TP_operator_from_string_compact("x", L)
lab_z, c_z = pauli_func.TP_operator_from_string_compact("z", L)

# set spin operators
c_x *= 0.5 * 0.5
c_z *= 0.5 * 0.5

num_op = np.loadtxt("operators/operators_TPY_size_full.txt").astype(int)
TPY_labels, TPY_coeffs = pauli_func.create_operators_TPY_compact(l, L, num_op[l - 1])


# commutators
C_x_labels, C_x_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_x, c_x, L)
C_z_labels, C_z_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_z, c_z, L)

P_Z_X = pauli_func.create_P_matrix_TP(C_z_labels, C_x_labels, C_z_coeffs, C_x_coeffs) / tr_I

name = "ltfi_optimization_matrices_P_Z_X_L=" + str(L) + "_l=" + str(l) + ".npz"
np.savez_compressed(name, P_Z_X=P_Z_X / L)