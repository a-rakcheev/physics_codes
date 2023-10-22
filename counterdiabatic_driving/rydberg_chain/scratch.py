import sys
sys.path.append("/home/artem/Dropbox/codebase/")
sys.path.append("C:/Users/rakche_a-adm/Dropbox/codebase/")
sys.path.append("/mnt/d64a440d-7b6c-45ed-aa6c-c9d991212d84/Dropbox/codebase/")

import pauli_string_functions as pauli_func
import numpy as np
import time
import sys


# l = int(sys.argv[1])
# L = int(sys.argv[2])
# op_name1 = str(sys.argv[3])
# op_name2 = str(sys.argv[4])


l = 2
L = 8
names = ["x", "z", "zz", "z1z", "z11z"]

for op_name in names:

    print("operator:", op_name)
    
    # create TPY operators from file
    tr_I = 2 ** L

    lab_op, c_op = pauli_func.TP_operator_from_string_compact(op_name, L)

    pauli_func.print_operator_balanced(lab_op, c_op)

    num_op = np.loadtxt("/home/artem/Dropbox/bu_research/operators/operators_TPY_size_full.txt").astype(int)
    TPY_labels, TPY_coeffs = pauli_func.create_operators_TPY_compact(l, L, num_op[l - 1])

    # commutators
    C_labels, C_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_op, c_op, L)

    for i, lab in enumerate(C_labels):
        coeff = C_coeffs[i]
        pauli_func.print_operator_balanced(lab, coeff)

    # # R matrices
    # R = pauli_func.create_R_matrix_TP(C_labels, C_coeffs, lab_1, c_1) / (tr_I * L)


