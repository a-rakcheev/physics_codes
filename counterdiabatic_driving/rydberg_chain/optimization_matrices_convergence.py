import sys
sys.path.append("/home/artem/Dropbox/codebase/")
sys.path.append("C:/Users/rakche_a-adm/Dropbox/codebase/")
sys.path.append("/mnt/d64a440d-7b6c-45ed-aa6c-c9d991212d84/Dropbox/codebase/")

import pauli_string_functions as pauli_func
import numpy as np
import time

l = 3
Ll = [3, 4, 5, 6, 7, 8]
names = ["x"]
names2 = ["y", "zz", "z1z"]

for L in Ll:
    for op_name1 in names:
        for op_name2 in names2:

            print(op_name1, op_name2)

            # create TPY operators from file
            start = time.time()
            tr_I = 2 ** L

            lab_1, c_1 = pauli_func.TP_operator_from_string_compact(op_name1, L)
            lab_2, c_2 = pauli_func.TP_operator_from_string_compact(op_name2, L)

            num_op = np.loadtxt("/home/artem/Dropbox/bu_research/operators/operators_TPY_size_full.txt").astype(int)
            TPY_labels, TPY_coeffs = pauli_func.create_operators_TPY_compact(l, L, num_op[l - 1])

            # commutators
            C_labels, C_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_2, c_2, L)

            # for i, lab in enumerate(C_labels):
            #     coeff = C_coeffs[i]
            #     pauli_func.print_operator_balanced(lab, coeff)

            # R matrices
            R = pauli_func.create_R_matrix_TP(C_labels, C_coeffs, lab_1, c_1) / (tr_I * L)

            end = time.time()
            print("Time:", end - start, flush=True)


            print(np.nonzero(R), R[np.nonzero(R)])