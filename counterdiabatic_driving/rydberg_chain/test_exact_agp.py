# test commutator with exact gauge potential
# [H, dH/d\lambda - i[A_\lambda, H]] = 0

import sys
sys.path.append("/home/artem/Dropbox/codebase/")
sys.path.append("C:/Users/rakche_a-adm/Dropbox/codebase/")
sys.path.append("/mnt/d64a440d-7b6c-45ed-aa6c-c9d991212d84/Dropbox/codebase/")

import pauli_string_functions as pauli_func
import numpy as np

# parameters
L = 4
l = L                                            # range cutoff for variational strings
res_g = 9                                        # logarithmic resolution
res_h = 9                                        # logarithmic resolution

# hamiltonian operators
lab_I, c_I = pauli_func.TP_operator_from_string_compact("1", L)
lab_x, c_x = pauli_func.TP_operator_from_string_compact("x", L)
lab_z, c_z = pauli_func.TP_operator_from_string_compact("z", L)
lab_zz, c_zz = pauli_func.TP_operator_from_string_compact("zz", L)
lab_z1z, c_z1z = pauli_func.TP_operator_from_string_compact("z1z", L)
lab_z11z, c_z11z = pauli_func.TP_operator_from_string_compact("z11z", L)

lab_zdz, c_zdz = pauli_func.add_operators(lab_I, lab_zz, 0.25 * (1 + (1 / 2 ** 6) + (1 / 3 ** 6)) * L * c_I, c_zz)
lab_zdz, c_zdz = pauli_func.add_operators(lab_zdz, lab_z1z, c_zdz, 0.25 * (1 / 2 ** 6) * c_z1z)
lab_zdz, c_zdz = pauli_func.add_operators(lab_zdz, lab_z11z, c_zdz, 0.25 * (1 / 3 ** 6) * c_z11z)
lab_zdz, c_zdz = pauli_func.add_operators(lab_zdz, lab_z, c_zdz, 0.5 * (1 + (1 / 2 ** 6) + (1 / 3 ** 6)) * c_z)

pauli_func.print_operator_balanced(lab_zdz, c_zdz, digits=6)

name = "optimal_coefficients/optimal_coefficients_exact_rydberg_chain_truncated_TPY_L=" + str(L) + "_res_g=" + str(res_g) + "_res_h=" + str(res_h) + ".npz"
data = np.load(name)
c_g = data["c_g"]
c_h = data["c_h"]
gl = data["gl"]
hl = data["hl"]

# create gauge potential
num_op = np.loadtxt("/home/artem/Dropbox/bu_research/operators/operators_TPY_exact_reduced_size_full.txt").astype(int)
A_labels, A_coeffs = pauli_func.create_operators_TPY_compact_exact(l, L, num_op[l - 1])
A_labels = A_labels[:, 0, :]
A_coeffs = A_coeffs[:, 0]

print(num_op)
print(len(A_coeffs))

# # check if [H, \partial_{lambda}H - i[A, H]] = 0
# for i, g in enumerate(gl):
#     for j, h in enumerate(hl):

#         # set up operators
#         lab_ham, c_ham = pauli_func.add_operators(lab_zdz, lab_x, c_zdz, -g * c_x)
#         lab_ham, c_ham = pauli_func.add_operators(lab_ham, lab_z, c_ham, -h * c_z)

#         A_g_coeffs = c_g[i, j, :]
#         A_h_coeffs = c_h[i, j, :]

#         # test hamiltonian derivative in second direction
#         # commutator [A, H]
#         lab_inner, c_inner = pauli_func.commute_operators_TP(A_labels, lab_ham, A_coeffs * A_g_coeffs, c_ham)

#         # add derivative
#         lab_inner, c_inner = pauli_func.add_operators(lab_x, lab_inner, -c_x, -1.j * c_inner)
#         # pauli_func.print_operator_balanced(lab_inner, c_inner, digits=4)

#         # outer commutator
#         lab_outer, c_outer = pauli_func.commute_operators_TP(lab_ham, lab_inner, c_ham, c_inner)
#         print("Error A_X:", np.linalg.norm(c_outer))

#         # test hamiltonian derivative in first direction
#         # commutator [A, H]
#         lab_inner, c_inner = pauli_func.commute_operators_TP(A_labels, lab_ham, A_coeffs * A_h_coeffs, c_ham)

#         # add derivative
#         lab_inner, c_inner = pauli_func.add_operators(lab_z, lab_inner, -c_z, -1.j * c_inner)
#         # pauli_func.print_operator_balanced(lab_inner, c_inner, digits=4)

#         # outer commutator
#         lab_outer, c_outer = pauli_func.commute_operators_TP(lab_ham, lab_inner, c_ham, c_inner)
#         print("Error A_Z:", np.linalg.norm(c_outer))
