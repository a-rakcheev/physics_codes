import pauli_string_functions as pauli_func
import numpy as np
import time
import numba as nb
import sys


# l = int(sys.argv[1])
# L = int(sys.argv[2])
# num_threads = int(sys.argv[3])

l = 4
L = 9
num_threads = 1

nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(num_threads)

print("Threads:", nb.get_num_threads())
print("Threading Layer: %s" % nb.threading_layer())


# create TPY operators from file
start = time.time()
tr_I = 2 ** L

lab_x, c_x = pauli_func.TP_operator_from_string_compact("x", L)
lab_z1z, c_z1z = pauli_func.TP_operator_from_string_compact("z1z", L)
lab_zz, c_zz = pauli_func.TP_operator_from_string_compact("zz", L)

# set spin operators
c_x *= 0.5 * 0.5
c_z1z *= 0.25 * 0.5
c_zz *= 0.25 * 0.5

num_op = np.loadtxt("operators/operators_TPY_size_full.txt").astype(int)
TPY_labels, TPY_coeffs = pauli_func.create_operators_TPY_compact(l, L, num_op[l - 1])
print("size:", num_op[l - 1])


# commutators
C_x_labels, C_x_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_x, c_x, L)
C_z1z_labels, C_z1z_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_z1z, c_z1z, L)
C_zz_labels, C_zz_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_zz, c_zz, L)
print("Commutators computed", flush=True)

# R matrices
R_X_X = pauli_func.create_R_matrix_TP(C_x_labels, C_x_coeffs, lab_x, c_x) / tr_I
R_X_Z1Z = pauli_func.create_R_matrix_TP(C_z1z_labels, C_z1z_coeffs, lab_x, c_x) / tr_I
R_X_ZZ = pauli_func.create_R_matrix_TP(C_zz_labels, C_zz_coeffs, lab_x, c_x) / tr_I

R_Z1Z_X = pauli_func.create_R_matrix_TP(C_x_labels, C_x_coeffs, lab_z1z, c_z1z) / tr_I
R_Z1Z_ZZ = pauli_func.create_R_matrix_TP(C_zz_labels, C_zz_coeffs, lab_z1z, c_z1z) / tr_I
R_Z1Z_Z1Z = pauli_func.create_R_matrix_TP(C_z1z_labels, C_z1z_coeffs, lab_z1z, c_z1z) / tr_I
print("R matrices computed")
end = time.time()
print("Time:", end - start, flush=True)


name = "annni_optimization_matrices_R_L=" + str(L) + "_l=" + str(l) + ".npz"
np.savez_compressed(name, R_X_ZZ=R_X_ZZ / L, R_X_Z1Z=R_X_Z1Z / L, R_X_X=R_X_X / L,
                    R_Z1Z_X=R_Z1Z_X / L, R_Z1Z_ZZ=R_Z1Z_ZZ / L, R_Z1Z_Z1Z=R_Z1Z_Z1Z / L)