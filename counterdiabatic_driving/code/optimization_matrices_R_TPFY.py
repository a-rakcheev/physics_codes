import pauli_string_functions as pauli_func
import numpy as np
import time

l = 8
L = 2 * l + 1
names = ["x", "z", "xx", "xz", "yy", "zz", "x1x", "x1z", "xxx", "xxz",
         "xyy", "xzx", "xzz", "y1y", "yxy", "yyz", "yzy", "z1z", "zxz", "zzz"]

for op_name1 in names:
    print(op_name1)
    start = time.time()
    for op_name2 in names:

        # create TPY operators from file
        tr_I = 2 ** L

        lab_1, c_1 = pauli_func.TP_operator_from_string_compact(op_name1, L)
        lab_2, c_2 = pauli_func.TP_operator_from_string_compact(op_name2, L)


        num_op = np.loadtxt("operators/operators_TPFY_size_full.txt").astype(int)
        TPY_labels, TPY_coeffs = pauli_func.create_operators_TPFY_compact(l, L, num_op[l - 1])

        # commutators
        C_labels, C_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_2, c_2, L)

        # R matrices
        R = pauli_func.create_R_matrix_TP(C_labels, C_coeffs, lab_1, c_1) / (tr_I * L)


        name = "optimization_matrices_R_" + op_name1.upper() + "_" + op_name2.upper() + "_TPFY_l=" + str(l) + ".npz"
        np.savez_compressed(name, R=R)

    end = time.time()
    print("Time:", end - start, flush=True)