# test commutator with exact gauge potential
# [H, dH/dp_i - i[A_i, H]] = 0

# H = s_0 * H_0 + p_1 * s_1 * H_1 + p_2 * s_2 * H_2
# with parameters p_1, p_2 and signs s_i which are passed as 0/1 => (-1)^0 , (-1)^1

import sys
import zipfile
import io

import numpy as np
import pauli_string_functions as pauli_func


def parity(op_name):
    p = 0
    if op_name == op_name[::-1]:
        p = 1

    return p


# parameters
l = 4  # range cutoff for variational strings
L = l
res_1 = 3  # number of grid points on x axis
res_2 = 3  # number of grid points on y axis

s_0 = 0  # signs of operators
s_1 = 0
s_2 = 0

op_name_0 = "xx"  # operators in the hamiltonian
op_name_1 = "yy"
op_name_2 = "zz"

start1 = 0.
start2 = 0.
end_1 = 2.0
end_2 = 2.0

params1 = np.linspace(start1, end_1, res_1)
params2 = np.linspace(start2, end_2, res_2)

sign_0 = (-1) ** s_0
sign_1 = (-1) ** s_1
sign_2 = (-1) ** s_2

# adjust factors due to parity
# the parity of the operator for example zz leads to double counting
# # when creating 1zz1 + zz11 + 11zz + z11z + 1zz1 + 11zz + zz11 + z11z, in contrast to
# # 1xy1 + xy11 + 11xy + y11x + 1yx1 + 11yx + yx11 + x11y
# # the parity is 0, 1 in these cases

factor_0 = 0.5 * (2. - parity(op_name_0))
factor_1 = 0.5 * (2. - parity(op_name_1))
factor_2 = 0.5 * (2. - parity(op_name_2))


# create operator representation of hamiltonian
lab_0, c_0 = pauli_func.TP_operator_from_string_compact(op_name_0, L)
lab_1, c_1 = pauli_func.TP_operator_from_string_compact(op_name_1, L)
lab_2, c_2 = pauli_func.TP_operator_from_string_compact(op_name_2, L)

c_0 *= factor_0
c_1 *= factor_1
c_2 *= factor_2

# name_zip = "optimization_matrices/optimization_matrices_exact_l=" + str(l) + ".zip"
# with zipfile.ZipFile(name_zip) as zipper:

name = "optimal_coefficients_exact_TPFXY_l" + str(l) + "_" + op_name_0.upper() + "_" + op_name_1.upper() \
       + "_" + op_name_2.upper() + "_s0=" + str(s_0) + "_s1=" + str(s_1) + "_s2=" + str(s_2) \
       + "_res_1=" + str(res_1) + "_res_2=" + str(res_2) + "_start1=" + str(start1).replace(".", "-") \
       + "_end1=" + str(end_1).replace(".", "-") + "_start2=" + str(start2).replace(".", "-") + "_end2=" \
       + str(end_2).replace(".", "-") + ".npz"

data = np.load(name)
coefficients_1 = data["c1"]
coefficients_2 = data["c2"]

# create gauge potential template
num_op = np.loadtxt("operators/operators_TPFXY_size_full.txt").astype(int)
A_labels, A_coeffs = pauli_func.create_operators_TPFXY_compact(l, L, num_op[l - 1])
A_labels = A_labels[:, 0, :]
A_coeffs = A_coeffs[:, 0]

# print(A_labels)
# print(A_coeffs)
#
# print(lab_0, c_0)
# print(lab_1, c_1)
# print(lab_2, c_2)

lab_comm, c_comm = pauli_func.commute_operators_TP(lab_0, lab_1, c_0, c_1)
# print(A_labels.shape, A_coeffs.shape)
# print(lab_0.shape, c_0.shape)
# print(lab_comm.shape, c_comm.shape)

for i in range(res_1):
    p_1 = params1[i]

    for j, p_2 in enumerate(params2):

        print(i, j)
        # set up operators
        lab_h, c_h = pauli_func.add_operators(lab_0, lab_1, sign_0 * c_0, sign_1 * p_1 * c_1)
        lab_h, c_h = pauli_func.add_operators(lab_h, lab_2, c_h, sign_2 * p_2 * c_2)

        pauli_func.print_operator_full(lab_h, c_h)
        A_1_coeffs = coefficients_1[i, j, :]
        A_2_coeffs = coefficients_2[i, j, :]
        print(A_1_coeffs)
        # test hamiltonian derivative in first direction

        # commutator [A, H]
        lab_inner, c_inner = pauli_func.commute_operators_TP(A_labels, lab_h, A_1_coeffs, c_h)

        # add derivative
        lab_inner, c_inner = pauli_func.add_operators(lab_1, lab_inner, sign_1 * c_1, -1.j * c_inner)

        # outer commutator
        lab_outer, c_outer = pauli_func.commute_operators_TP(lab_h, lab_inner, c_h, c_inner)
        print(lab_outer)
        print(c_outer)

        lab_tot, c_tot = pauli_func.operator_cleanup(lab_outer, c_outer)
        print(lab_tot)
        print(c_tot)
