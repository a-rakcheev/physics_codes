import sys
sys.path.append("/home/artem/Dropbox/codebase/")
sys.path.append("C:/Users/rakche_a-adm/Dropbox/codebase/")
sys.path.append("/mnt/d64a440d-7b6c-45ed-aa6c-c9d991212d84/Dropbox/codebase/")

import pauli_string_functions as pauli_func
import numpy as np
import scipy.sparse as sp
import sys
import tqdm

l = 4
L = 14
tr_I = 2 ** L

names_list = ["x", "z", "zz", "z1z", "z11z"]
names_list2 = [["x", "z", "zz", "z1z", "z11z"], ["z", "zz", "z1z", "z11z"], ["zz", "z1z", "z11z"], ["z1z", "z11z"], ["z11z"]]


for i, op_name1 in enumerate(names_list):
    for op_name2 in tqdm.tqdm(names_list2[i]):

        # create TPY operators from file
        lab_1, c_1 = pauli_func.TP_operator_from_string_compact(op_name1, L)
        lab_2, c_2 = pauli_func.TP_operator_from_string_compact(op_name2, L)

        # pauli_func.print_operator(lab_1, c_1)
        # pauli_func.print_operator(lab_2, c_2)

        num_op = np.loadtxt("/home/artem/Dropbox/bu_research/operators/operators_TPY_size_full.txt").astype(int)
        TPY_labels, TPY_coeffs = pauli_func.create_operators_TPY_compact(l, L, num_op[l - 1])

        # commutators
        C_1_labels, C_1_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_1, c_1, L)
        C_2_labels, C_2_coeffs = pauli_func.create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_2, c_2, L)

        # for i, lab in enumerate(C_1_labels):
        #     coeff = C_1_coeffs[i]
        #     pauli_func.print_operator(lab, coeff)
        #     print(pauli_func.trace_operator_TP(lab, coeff) / (tr_I * L))

        # for i, lab in enumerate(C_2_labels):
        #     coeff = C_2_coeffs[i]
        #     pauli_func.print_operator(lab, coeff)
        #     print(pauli_func.trace_operator_TP(lab, coeff) / (tr_I * L))

        
        # P matrix
        if op_name1 == op_name2:
            P, P_diag = pauli_func.create_P_matrix_symmetric_TP(C_1_labels, C_1_coeffs)

            name = "/home/artem/Dropbox/bu_research/rydberg_chain/optimization_matrices_TPY/optimization_matrices_P_" + op_name1.upper() + "_" + op_name2.upper() + "_TPY_l=" + str(l) + ".npz"
            name_diag = "/home/artem/Dropbox/bu_research/rydberg_chain/optimization_matrices_TPY/optimization_matrices_P_diag_" + op_name1.upper() + "_" + op_name2.upper() + "_TPY_l=" + str(l) + ".npz"
            
            # convert to sparse matrix
            P = sp.csr_matrix(P  / (tr_I * L))
            sp.save_npz(name, P)

            P_diag = sp.csr_matrix(np.diag(P_diag / (tr_I * L)))
            sp.save_npz(name_diag, P_diag)

        else:
            P = pauli_func.create_P_matrix_TP(C_1_labels, C_2_labels, C_1_coeffs, C_2_coeffs) / (tr_I * L)
            
            name = "/home/artem/Dropbox/bu_research/rydberg_chain/optimization_matrices_TPY/optimization_matrices_P_" + op_name1.upper() + "_" + op_name2.upper() + "_TPY_l=" + str(l) + ".npz"
            
            # convert to sparse matrix
            P = sp.csr_matrix(P)
            sp.save_npz(name, P)
