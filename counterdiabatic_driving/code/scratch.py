import sys
sys.path.append("/home/artem/Dropbox/codebase/")
sys.path.append("C:/Users/rakche_a-adm/Dropbox/codebase/")
sys.path.append("/mnt/d64a440d-7b6c-45ed-aa6c-c9d991212d84/Dropbox/codebase/")

import pauli_string_functions as pauli_func
import numpy as np
import scipy.sparse as sp
import sys

l = 2
L = 6
op_name1 = "x"
op_name2 = "x"

# create TPY operators from file
tr_I = 2 ** L

lab_1, c_1 = pauli_func.TP_operator_from_string_compact(op_name1, L)
lab_2, c_2 = pauli_func.TP_operator_from_string_compact(op_name2, L)


pauli_func.print_operator(lab_1, c_1)
pauli_func.print_operator(lab_2, c_2)

lab, c = pauli_func.multiply_operators_TP(lab_1, lab_2, c_1, c_2)

pauli_func.print_operator_balanced(lab, c)
print(pauli_func.trace_operator_TP(lab, c) / (tr_I * L))

num_op = np.loadtxt("operators/operators_TPY_size_full.txt").astype(int)
TPY_labels, TPY_coeffs = pauli_func.create_operators_TPY_compact(l, L, num_op[l - 1])

# commutators
C_1_labels, C_1_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_1, c_1, L)
C_2_labels, C_2_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_2, c_2, L)

# pauli_func.print_operator_balanced(C_1_labels, C_1_coeffs)

for i, lab in enumerate(C_1_labels):
    coeff = C_1_coeffs[i]
    pauli_func.print_operator_balanced(lab, coeff)

for i, lab in enumerate(C_2_labels):
    coeff = C_2_coeffs[i]
    pauli_func.print_operator_balanced(lab, coeff)

# P matrix
if op_name1 == op_name2:
    P = pauli_func.create_P_matrix_symmetric_TP(C_1_labels, C_1_coeffs) / (tr_I * L)

else:
    P = pauli_func.create_P_matrix_TP(C_1_labels, C_2_labels, C_1_coeffs, C_2_coeffs) / (tr_I * L)
    P += P.T


print(P)

# name = "ltfi_optimization_matrices_P_X_ZZ_L=" + str(L) + "_l=" + str(l) + ".npz"
# np.savez_compressed(name,  P_X_ZZ=P_X_ZZ)