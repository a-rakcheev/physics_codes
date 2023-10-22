import sys
sys.path.append("/home/artem/Dropbox/codebase/")
sys.path.append("C:/Users/rakche_a-adm/Dropbox/codebase/")
sys.path.append("/mnt/d64a440d-7b6c-45ed-aa6c-c9d991212d84/Dropbox/codebase/")

import pauli_string_functions as pauli_func
import numpy as np
import time

# l = int(sys.argv[1])
# L = int(sys.argv[2])
# op_name1 = str(sys.argv[3])
# op_name2 = str(sys.argv[4])

L = 4
l = 4
names = ["x", "z"]
names2 = ["x", "z", "zz", "z1z", "z11z"]

for op_name1 in names:
    for op_name2 in names2:

        print(op_name1, op_name2)

        # create TPY operators from file
        start = time.time()
        tr_I = 2 ** L

        lab_1, c_1 = pauli_func.TP_operator_from_string_compact(op_name1, L)
        lab_2, c_2 = pauli_func.TP_operator_from_string_compact(op_name2, L)

        # pauli_func.print_operator_balanced(lab_1, c_1)
        # pauli_func.print_operator_balanced(lab_2, c_2)


        num_op = np.loadtxt("/home/artem/Dropbox/bu_research/operators/operators_TPY_exact_reduced_size_full.txt").astype(int)
        TPY_labels, TPY_coeffs = pauli_func.create_operators_TPY_compact_exact(l, L, num_op[l - 1])

        # commutators
        C_labels, C_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_2, c_2, L)

        # for i, lab in enumerate(C_labels):
        #     coeff = C_coeffs[i]
        #     pauli_func.print_operator_balanced(lab, coeff)

        # R matrices
        R = pauli_func.create_R_matrix_TP(C_labels, C_coeffs, lab_1, c_1) / (tr_I * L)

        end = time.time()
        print("Time:", end - start, flush=True)

        name = "optimization_matrices_TPY/optimization_matrices_R_exact_" + op_name1.upper() + "_" + op_name2.upper() + "_TPY_l=" + str(l) + ".npz"
        np.savez(name, R=R)

        # print(np.nonzero(R), R[np.nonzero(R)])