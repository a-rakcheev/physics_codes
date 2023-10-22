import pauli_string_functions as pauli_func
import numpy as np
import time
import sys


# l = int(sys.argv[1])
# L = int(sys.argv[2])
# op_name1 = str(sys.argv[3])
# op_name2 = str(sys.argv[4])


l = 3
L = l
# names = ["x", "z", "xx", "xz", "yy", "zz", "x1x", "x1z", "xxx", "xxz",
#          "xyy", "xzx", "xzz", "y1y", "yxy", "yyz", "yzy", "z1z", "zxz", "zzz"]
names = ["x", "z", "zz"]

# op_name1 = "x"
# op_name2 = "z"

for op_name1 in names:
    for op_name2 in names:

        print(op_name1, op_name2)
        # create TPY operators from file
        start = time.time()
        tr_I = 2 ** L

        lab_1, c_1 = pauli_func.TP_operator_from_string_compact(op_name1, L)
        lab_2, c_2 = pauli_func.TP_operator_from_string_compact(op_name2, L)

        num_op = np.loadtxt("operators/operators_TPY_exact_reduced_size_full.txt").astype(int)
        TPY_labels, TPY_coeffs = pauli_func.create_operators_TPY_compact_exact(l, L, num_op[l - 1])

        # commutators
        C_labels, C_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_2, c_2, L)

        # R matrices
        R = pauli_func.create_R_matrix_TP(C_labels, C_coeffs, lab_1, c_1) / (tr_I * L)

        print(R)

        end = time.time()
        print("Time:", end - start, flush=True)

        name = "optimization_matrices_R_exact_" + op_name1.upper() + "_" + op_name2.upper() + "_TPY_l=" + str(l) + ".npz"
        np.savez(name, R=R)