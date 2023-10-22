import hamiltonians_32 as ham32
import numpy as np
import scipy.sparse as sp
from commute_stringgroups_no_quspin import *
m = maths()

L = 3
l = 3

def reflection_operator(number_of_spins):

    mat = np.zeros((2 ** number_of_spins, 2 ** number_of_spins))

    for i in range(2 ** number_of_spins):

        j = int(np.binary_repr(i, number_of_spins)[::-1], 2)
        mat[j, i] = 1.0

    return mat


# input the expression for the translationally invariant operator (like "xyz") and the number of sites
# output are the positions, labels and couplings to be used with hamiltonians32.py to create the appropriate operator
# on which the AGP optimization is based
#
# note the optimization creates all translated and reflected string (with double counting),
# which is also taken into account here in the couplings


def parse_operator_label(op_label, number_of_sites):

    label = []
    position = []

    for j, char in enumerate(op_label):

        if char == "x":
            label.append(1)
            position.append(j)

        elif char == "y":
            label.append(2)
            position.append(j)

        elif char == "z":
            label.append(3)
            position.append(j)

    label = np.array(label, dtype=np.int8)
    position = np.array(position, dtype=np.int64)

    positions = np.zeros((number_of_sites, len(label)), dtype=np.int64)
    labels = np.zeros((number_of_sites, len(label)), dtype=np.int8)

    for i in range(number_of_sites):

        positions[i, :] = (position + i) % number_of_sites + 1
        labels[i, :] = label

    return positions, labels

R = reflection_operator(L)
R_sp = sp.csr_matrix(R)
J = np.full(L, 1.0)


# positions_xyz, labels_x1yz = parse_operator_label("xyx", L)
# positions_xyz_reverse, labels_x1yz_reverse = parse_operator_label("xyx", L)
# h_xyx = ham32.operator_sum_complex(L, 3, J, positions_xyz, labels_x1yz).todense().imag
# h_xyx_reverse = ham32.operator_sum_complex(L, 3, J, positions_xyz_reverse, labels_x1yz_reverse).todense().imag
#
# print(h_xyx - h_xyx_reverse)
# print(h_xyx_reverse - np.dot(np.dot(R.T, h_xyx), R))
#
#
#
#
# positions_x1yz, labels_x1yz = parse_operator_label("x1yz", L)
# positions_x1yz_reverse, labels_x1yz_reverse = parse_operator_label("zy1x", L)
#
# R = reflection_operator(L)
# J = np.full(L, 1.0)
# h_x1yz = ham32.operator_sum_complex(L, 3, J, positions_x1yz, labels_x1yz).todense().imag
# h_x1yz_reverse = ham32.operator_sum_complex(L, 3, J, positions_x1yz_reverse, labels_x1yz_reverse).todense().imag
#
# print(h_x1yz - h_x1yz_reverse)
# print(h_x1yz_reverse - np.dot(np.dot(R.T, h_x1yz), R))

# # read in TPY operators up to range l
# variational_operators = []
# for k in np.arange(1, l + 1, 1):
#
#     # fill the strings up to the correct system size
#     op_file = "operators_TPY_l" + str(k) + ".txt"
#     with open(op_file, "r") as readfile:
#         for line in readfile:
#
#             op_str = line[0:k] + '1' * (L - k)
#             op_eq = equation()
#             for i in range(L):
#                 op = "".join(roll(list(op_str), i))
#                 op_eq += equation({op: 1.0})
#                 op_rev = "".join(roll(list(op_str[::-1]), i))
#                 op_eq += equation({op_rev: 1.0})
#
#             variational_operators.append(op_eq)


operator_labels = []
operator_lengths = []
for k in np.arange(1, l + 1, 1):

    op_file = "operators_TPY_l" + str(k) + ".txt"

    with open(op_file, "r") as readfile:
        for line in readfile:

            label = line[0:k]
            operator_labels.append(label)
            operator_lengths.append(len(label) - label.count("1"))

var_order = len(operator_labels)
print(operator_labels)

h_tot = sp.csr_matrix((2 ** L, 2 ** L), dtype=np.complex128)
for i, op_str in enumerate(operator_labels):

    positions, labels = parse_operator_label(op_str, L)
    op = ham32.operator_sum_complex(L, operator_lengths[i], J, positions, labels)

    if op_str[::-1] == op_str:

        h_tot += 2. * op

    else:

        h_tot += op
        h_tot += R_sp.transpose() * op * R_sp



np.set_printoptions(linewidth=150)
print(h_tot.todense())
print(h_tot.nnz, 4 ** L)
print(np.linalg.eigvalsh(R))