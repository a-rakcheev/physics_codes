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
lab_zz, c_zz = pauli_func.TP_operator_from_string_compact("zz", L)

# set spin operators
c_zz *= 0.25 * 0.5

num_op = np.loadtxt("operators/operators_TPY_size_full.txt").astype(int)
TPY_labels, TPY_coeffs = pauli_func.create_operators_TPY_compact(l, L, num_op[l - 1])

# commutators
C_zz_labels, C_zz_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_zz, c_zz, L)


# P matrices
P_ZZ_ZZ = pauli_func.create_P_matrix_symmetric_TP(C_zz_labels, C_zz_coeffs) / tr_I

name = "ltfi_optimization_matrices_P_ZZ_ZZ_L=" + str(L) + "_l=" + str(l) + ".npz"
np.savez_compressed(name, P_ZZ_ZZ=P_ZZ_ZZ / L)