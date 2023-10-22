import pauli_string_functions as pauli_func
import numpy as np
import time
import numba as nb
import os
os.environ["OMP_NUM_THREADS"] = "1"
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(1)

print("Threading Layer: %s" % nb.threading_layer())
print("Threads:", nb.get_num_threads())

# os.environ["NUMBA_PARALLEL_DIAGNOSTICS"] = "4"

# create TPY operators from file
l = 5
L = 9
tr_I = 2 ** L

lab_x, c_x = pauli_func.TP_operator_from_string_compact("x", L)
lab_z, c_z = pauli_func.TP_operator_from_string_compact("z", L)
lab_zz, c_zz = pauli_func.TP_operator_from_string_compact("zz", L)

# set spin operators
c_x *= 0.5 * 0.5
c_z *= 0.5 * 0.5
c_zz *= 0.25 * 0.5

num_op = np.loadtxt("operators/operators_TPY_size_full.txt").astype(int)
size = num_op[l - 1]
TPY_labels, TPY_coeffs = pauli_func.create_operators_TPY_compact(l, L, size)
print("size:", size)


# commutators
start = time.time()
C_x_labels, C_x_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_x, c_x, L)
C_z_labels, C_z_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_z, c_z, L)
C_zz_labels, C_zz_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_zz, c_zz, L)
print("Commutators computed", flush=True)
#
# # R matrices
# R_X_X = pauli_func.create_R_matrix_TP(C_x_labels, C_x_coeffs, lab_x, c_x) / tr_I
# R_X_Z = pauli_func.create_R_matrix_TP(C_z_labels, C_z_coeffs, lab_x, c_x) / tr_I
# R_X_ZZ = pauli_func.create_R_matrix_TP(C_zz_labels, C_zz_coeffs, lab_x, c_x) / tr_I
#
# R_Z_X = pauli_func.create_R_matrix_TP(C_x_labels, C_x_coeffs, lab_z, c_z) / tr_I
# R_Z_Z = pauli_func.create_R_matrix_TP(C_z_labels, C_z_coeffs, lab_z, c_z) / tr_I
# R_Z_ZZ = pauli_func.create_R_matrix_TP(C_zz_labels, C_zz_coeffs, lab_z, c_z) / tr_I
# print("R matrices computed", flush=True)
end = time.time()
print("Time:", end - start)

# # P matrices
# start = time.time()
# P_X_X = pauli_func.create_P_matrix_symmetric_TP(C_x_labels, C_x_coeffs) / tr_I
# print("P_x_x computed", flush=True)
# end = time.time()
# print("Time:", end - start)
#
# start = time.time()
# P_Z_Z = pauli_func.create_P_matrix_symmetric_TP(C_z_labels, C_z_coeffs) / tr_I
# print("P_z_z computed", flush=True)
# end = time.time()
# print("Time:", end - start)
#
start = time.time()
P_ZZ_ZZ = pauli_func.create_P_matrix_symmetric_TP(C_zz_labels, C_zz_coeffs) / tr_I
print("P_zz_zz computed", flush=True)
end = time.time()
print("Time:", end - start)
print(P_ZZ_ZZ / L)

# P_X_ZZ = pauli_func.create_P_matrix_TP(C_x_labels, C_zz_labels, C_x_coeffs, C_zz_coeffs)
# print("P_x_zz computed", flush=True)

# start = time.time()
# P_X_ZZ = pauli_func.create_P_matrix_TP(C_x_labels, C_zz_labels, C_x_coeffs, C_zz_coeffs)
# end = time.time()
# print("Time:", end - start)
#
# print(P_X_ZZ / (tr_I * L))

# start = time.time()
# P_Z_X = pauli_func.create_P_matrix_TP(C_z_labels, C_x_labels, C_z_coeffs, C_x_coeffs) / tr_I
# print("P_z_x computed", flush=True)
# end = time.time()
# print("Time:", end - start)
#
# start = time.time()
# P_Z_ZZ = pauli_func.create_P_matrix_TP(C_z_labels, C_zz_labels, C_z_coeffs, C_zz_coeffs) / tr_I
# print("P_z_zz computed", flush=True)
# end = time.time()
# print("Time:", end - start)


# np.set_printoptions(3)
# print("R Matrices:")
# print(R_X_X / L)
# print(R_X_Z / L)
# print(R_X_ZZ / L)
# print(R_Z_X / L)
# print(R_Z_Z / L)
# print(R_Z_ZZ / L)
#
# print("P Matrices:")
# print(P_X_X / L)
# print(P_Z_Z / L)
# print(P_ZZ_ZZ / L)
# print(P_X_ZZ / L)
# print(P_Z_X / L)
# print(P_Z_ZZ / L)

# name = "ltfi_optimization_matrices_L=" + str(L) + "_l=" + str(l) + ".npz"
# np.savez_compressed(name,  R_X_ZZ=R_X_ZZ / L, R_X_Z=R_X_Z / L, R_X_X=R_X_X / L, R_Z_X=R_Z_X / L,
#                     R_Z_Z=R_Z_Z / L, R_Z_ZZ=R_Z_ZZ / L, P_X_X=P_X_X / L, P_X_ZZ=P_X_ZZ / L,
#                     P_Z_ZZ=P_Z_ZZ / L, P_Z_X=P_Z_X / L, P_Z_Z=P_Z_Z / L, P_ZZ_ZZ=P_ZZ_ZZ / L)
