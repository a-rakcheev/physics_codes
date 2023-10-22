# functions that act on pauli strings and operators (sums of strings with coefficients)
# these can be used to efficiently determine products / commutators of operators defined
# in terms of pauli strings

import numpy as np
import numba as nb
from numba.core import types
from numba.typed import Dict
from scipy.sparse import save_npz, csr_matrix
import zipfile
import io

# lookup arrays for multiplication of two pauli operators and operator names for printing
mult_factor = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.j, -1.j], [1.0, -1.j, 1.0, 1.j], [1.0, 1.j, -1.j, 1.0]],
                       dtype=np.complex128)
pauli_name = np.array(["1", "X", "Y", "Z"])

# basic operations on two pauli strings
@nb.jit(nopython=True, cache=True)
def multiply_pauli_strings(label1, label2, coeff1, coeff2):

    label3 = np.bitwise_xor(label1, label2)      # change since this is commutator
    coeff3 = coeff1 * coeff2

    for i in range(len(label1)):

        coeff3 *= mult_factor[label1[i], label2[i]]

    return label3, coeff3


@nb.jit(nopython=True, cache=True)
def commute_pauli_strings(label1, label2, coeff1, coeff2):

    label3 = np.bitwise_xor(label1, label2)      # change since this is commutator
    coeff3_1 = coeff1 * coeff2
    coeff3_2 = coeff1 * coeff2

    for i in range(len(label1)):

        coeff3_1 *= mult_factor[label1[i], label2[i]]
        coeff3_2 *= mult_factor[label2[i], label1[i]]

    return label3, (coeff3_1 - coeff3_2)


# basic operations on operators (weighted sums of pauli strings)
@nb.jit(nopython=True, cache=True)
def add_operators(labels1, labels2, coeff1, coeff2):

    lenght1 = len(coeff1)
    length2 = len(coeff2)

    labels3 = np.zeros((lenght1 + length2, len(labels1[0, :])), dtype=np.int8)
    coeff3 = np.zeros(lenght1 + length2, dtype=np.complex128)

    labels3[0:lenght1, :] = labels1
    labels3[lenght1:, :] = labels2

    coeff3[0:lenght1] = coeff1
    coeff3[lenght1:] = coeff2

    return labels3, coeff3


@nb.jit(nopython=True, cache=True)
def multiply_operators(labels1, labels2, coeff1, coeff2):

    lenght1 = len(coeff1)
    length2 = len(coeff2)

    labels3 = np.zeros((lenght1 * length2, len(labels1[0, :])), dtype=np.int8)
    coeff3 = np.zeros(lenght1 * length2, dtype=np.complex128)

    for i, c1 in enumerate(coeff1):
        for j, c2 in enumerate(coeff2):
            l3, c3 = multiply_pauli_strings(labels1[i], labels2[j], c1, c2)
            labels3[i * length2 + j, :] = l3
            coeff3[i * length2 + j] = c3

    return labels3, coeff3


@nb.jit(nopython=True, cache=True)
def commute_operators(labels1, labels2, coeff1, coeff2):
    lenght1 = len(coeff1)
    length2 = len(coeff2)

    labels3 = np.zeros((lenght1 * length2, len(labels1[0, :])), dtype=np.int8)
    coeff3 = np.zeros(lenght1 * length2, dtype=np.complex128)

    for i, c1 in enumerate(coeff1):
        for j, c2 in enumerate(coeff2):
            l3, c3 = commute_pauli_strings(labels1[i], labels2[j], c1, c2)
            labels3[i * length2 + j, :] = l3
            coeff3[i * length2 + j] = c3

    return labels3, coeff3


@nb.jit(nopython=True, cache=True)
def trace_operator(labels, coeff):

    tr = 0.
    for i, l in enumerate(labels):

        if np.sum(l) == 0:
            tr += coeff[i]

    return tr * (2 ** len(labels[0]))


# the following are the same multiplication, commutation and trace operators
# intended to be used for operators which are translation and parity invariant
# for a single operator only one pauli string should be used reducing memory and computation time
# by a factor of 2L
# the output should also be interpreted accordingly

# the product of two such operators results in 2L new operators, for example
# x11 * xy1
# (x11 + 1x1 + 11x + x11 + 1x1 + 11x) * (xy1 + y1x + 1xy + 1yx + x1y + yx1)             (full expand)
# = 1y1 + i * z1x + xxy + xyx + 11y + i * zx1
# + i * xz1 + yxx + 11y + i * 1zx + xxy + y11
# + xyx + yxx + i * 1xz + 1y1 + i * x1z + yxx
# + 1y1 + i * z1x + xxy + xyx + 11y + i * xx1
# + i * xz1 + yxx + 11y + i * 1zx + xxy + y11
# + xyx + y11 + i * 1xz + 1y1 + i * x1z + yxx
# = (1y1 + i * z1x + xxy + 1y1 + i * z1x + xxy)                                        (compact - note double occurence)

@nb.jit(nopython=True, cache=True)
def multiply_operators_TP(labels1, labels2, coeff1, coeff2):

    size = len(labels1[0])
    lenght1 = len(coeff1)
    length2 = len(coeff2)

    labels3 = np.zeros((lenght1 * length2 * 2 * size, len(labels1[0, :])), dtype=np.int8)
    coeff3 = np.zeros(lenght1 * length2 * 2 * size, dtype=np.complex128)

    for i, c1 in enumerate(coeff1):
        for j, c2 in enumerate(coeff2):

            lab2 = labels2[j]
            k = 0
            for t in range(size):

                lab = np.roll(lab2, t)
                l3, c3 = multiply_pauli_strings(labels1[i], lab, c1, c2)

                labels3[i * length2 * (2 * size) + j * (2 * size) + k, :] = l3
                coeff3[i * length2 * (2 * size) + j * (2 * size) + k] = c3

                k += 1

                lab = np.roll(lab2[::-1], t)
                l3, c3 = multiply_pauli_strings(labels1[i], lab, c1, c2)

                labels3[i * length2 * (2 * size) + j * (2 * size) + k, :] = l3
                coeff3[i * length2 * (2 * size) + j * (2 * size) + k] = c3

                k += 1

    return labels3, coeff3


# same as multiply only changing the operation to commutation
@nb.jit(nopython=True, cache=True)
def commute_operators_TP(labels1, labels2, coeff1, coeff2):
    size = len(labels1[0])
    lenght1 = len(coeff1)
    length2 = len(coeff2)

    labels3 = np.zeros((lenght1 * length2 * 2 * size, len(labels1[0, :])), dtype=np.int8)
    coeff3 = np.zeros(lenght1 * length2 * 2 * size, dtype=np.complex128)

    for i, c1 in enumerate(coeff1):
        for j, c2 in enumerate(coeff2):

            lab2 = labels2[j]
            k = 0
            for t in range(size):
                lab = np.roll(lab2, t)
                l3, c3 = commute_pauli_strings(labels1[i], lab, c1, c2)

                labels3[i * length2 * (2 * size) + j * (2 * size) + k, :] = l3
                coeff3[i * length2 * (2 * size) + j * (2 * size) + k] = c3

                k += 1

                lab = np.roll(lab2[::-1], t)
                l3, c3 = commute_pauli_strings(labels1[i], lab, c1, c2)

                labels3[i * length2 * (2 * size) + j * (2 * size) + k, :] = l3
                coeff3[i * length2 * (2 * size) + j * (2 * size) + k] = c3

                k += 1

    return labels3, coeff3


# if the identity is included it will be enhanced by 2L, since in TP representation 11...11 stands for 1..1 + 1..1 + ... 1..1 (L times due to translation)
#  + 1..1 + 1..1 + ... 1..1 (L times due to reflection)
@nb.jit(nopython=True, cache=True)
def trace_operator_TP(labels, coeff):

    tr = 0.
    for i, l in enumerate(labels):

        if np.sum(l) == 0:
            tr += coeff[i]

    size = len(labels[0])
    return 2 * size * tr * (2 ** size)




@nb.jit(nopython=True, cache=True)
def pauli_string_from_string(name_string):

    pauli_dict = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.int8,
    )

    pauli_dict['1'] = 0
    pauli_dict['x'] = 1
    pauli_dict['y'] = 2
    pauli_dict['z'] = 3

    size = len(name_string)
    label = np.zeros(size, dtype=np.int8)

    for i, char in enumerate(name_string):
        label[i] = pauli_dict[char]

    return label


@nb.jit(nopython=True, cache=True)
def TP_operator_from_string(name_string, size):

    labels = np.zeros((2 * size, size), dtype=np.int8)
    coeffs = np.ones(2 * size, dtype=np.complex128)

    base_pauli_string = np.concatenate((pauli_string_from_string(name_string),
                                        np.zeros(size - len(name_string), dtype=np.int8)))

    for i in range(size):

        labels[i, :] = np.roll(base_pauli_string, i)
        labels[size + i, :] = np.roll(base_pauli_string[::-1], i)

    return labels, coeffs


# represents operator with translation and reflection symmetry by the first string
# for instance xy1 + 1xy + y1x + 1yx + yx1 + x1y is represented by xy
# note that this leads to double counting for x11 + 1x1 + 11x + 11x + 1x1 + x11
# which is compensated for by the factor 1/2 in the coefficient

@nb.jit(nopython=True, cache=True)
def TP_operator_from_string_compact(name_string, size):

    labels = np.zeros((1, size), dtype=np.int8)

    # account for reflection symmetry
    if name_string == name_string[::-1]:
        coeffs = 0.5 * np.ones(1, dtype=np.complex128)
    else:
        coeffs = np.ones(1, dtype=np.complex128)

    base_pauli_string = np.concatenate((pauli_string_from_string(name_string),
                                        np.zeros(size - len(name_string), dtype=np.int8)))

    labels[0, :] = base_pauli_string

    return labels, coeffs


# functions for optimization
# for them we have an array of operators (array of arrays) and coefficients
# and the hamiltonian / derivative as operators

# create list of commutators [H, O_i] from list of operators O_i
@nb.jit(nopython=True, parallel=False, cache=True)
def create_commutators_for_optimization(operators_labels, operators_coeffs, ham_labels, ham_coeffs, system_size):

    number_of_operators = len(operators_coeffs[:, 0])
    commutator_labels = np.zeros((number_of_operators, 4 * system_size ** 2, system_size), dtype=np.int8)
    commutator_coeff = np.zeros((number_of_operators, 4 * system_size ** 2), dtype=np.complex128)

    for i in nb.prange(number_of_operators):

        op_labels = operators_labels[i, :, :]
        op_coeffs = operators_coeffs[i, :]

        labels, coeff = commute_operators(ham_labels, op_labels, ham_coeffs, op_coeffs)

        commutator_labels[i, :, :] = labels
        commutator_coeff[i, :] = coeff

    return commutator_labels, commutator_coeff


# R matrix
# R_k = 2iTr(h_deriv * C_k), which is real-valued
@nb.jit(nopython=True, parallel=False, cache=True)
def create_R_matrix(commutator_labels, commutator_coeffs, ham_deriv_labels, ham_deriv_coeff):

    number_of_operators = len(commutator_coeffs[:, 0])
    mat = np.zeros(number_of_operators, dtype=np.float64)

    for i in nb.prange(number_of_operators):

        # print(i)
        comm_label = commutator_labels[i, :, :]
        comm_coeff = commutator_coeffs[i, :]

        # print_operator(ham_deriv_labels, ham_deriv_coeff)
        # print_operator(comm_label, comm_coeff)

        labels, coeff = multiply_operators(ham_deriv_labels, comm_label, ham_deriv_coeff, comm_coeff)

        # print_operator(labels, coeff)
        tr = trace_operator(labels, coeff)
        # print(tr)
        mat[i] = (2.j * tr).real

    return mat


# P matrix
# P_kl = -Tr(C_k C_l), which is real-valued
@nb.jit(nopython=True, parallel=False)
def create_P_matrix(commutator_labels1, commutator_labels2, commutator_coeffs1, commutator_coeffs2):
    number_of_operators = len(commutator_coeffs1[:, 0])
    mat = np.zeros((number_of_operators, number_of_operators), dtype=np.float64)

    for i in nb.prange(number_of_operators):

        comm_label_i = commutator_labels1[i, :, :]
        comm_coeff_i = commutator_coeffs1[i, :]

        for j in range(number_of_operators):

            comm_label_j = commutator_labels2[j, :, :]
            comm_coeff_j = commutator_coeffs2[j, :]

            labels, coeff = multiply_operators(comm_label_i, comm_label_j, comm_coeff_i, comm_coeff_j)
            tr = trace_operator(labels, coeff)
            mat[i, j] = -tr.real

    return mat

# if both commutator sets are the same P_kl = P_lk
# store only upper triangular part to save space
@nb.jit(nopython=True, parallel=False)
def create_P_matrix_symmetric(commutator_labels1, commutator_coeffs1):
    number_of_operators = len(commutator_coeffs1[:, 0])
    mat = np.zeros((number_of_operators * (number_of_operators + 1)) // 2, dtype=np.float64)

    for i in nb.prange(number_of_operators):
        # print(i)
        # start = 0.
        # with nb.objmode(start='float64'):  # annotate return type
        #     # this region is executed by object-mode.
        #     start = time.time()
        comm_label_i = commutator_labels1[i, :, :]
        comm_coeff_i = commutator_coeffs1[i, :]

        for j in range(i, number_of_operators):

            comm_label_j = commutator_labels1[j, :, :]
            comm_coeff_j = commutator_coeffs1[j, :]

            labels, coeff = multiply_operators(comm_label_i, comm_label_j, comm_coeff_i, comm_coeff_j)
            tr = trace_operator(labels, coeff)
            mat[(j - i) + number_of_operators * i - (i * (i - 1)) // 2] = -tr.real

        # end = 0.
        # with nb.objmode(end='float64'):  # annotate return type
        #     # this region is executed by object-mode.
        #     end = time.time()
        # print(i, end - start)
    return mat


# @nb.jit(nopython=True, cache=True)
def operator_cleanup(operator_labels, operator_coeffs):

    labels = []
    coeffs = []

    for i, lab in enumerate(operator_labels):
        coeff = operator_coeffs[i]

        if np.absolute(coeff) < 1.e-14:

            continue

        else:

            inc = 0
            for j, lab_inc in enumerate(labels):

                if np.array_equal(lab, lab_inc):
                    coeffs[j] += coeff
                    inc = 1

            if inc == 0:

                labels.append(lab)
                coeffs.append(coeff)

    # labels = np.array(labels)
    # coeffs = np.array(coeffs)

    return labels, coeffs

# create an operator list for the previous functions from a file containing the string names of operators
# needs the number of operators and system size in advance
def create_operators_from_file(filename, size, number_of_operators):

    op_array = np.zeros((number_of_operators, 2 * size, size), dtype=np.int8)
    c_array = np.zeros((number_of_operators, 2 * size), dtype=np.complex128)

    names = np.genfromtxt(filename, dtype='str')

    i = 0
    for op_name in names:

        op_lab, op_c = TP_operator_from_string(op_name, size)
        op_array[i, :, :] = op_lab
        c_array[i, :] = op_c

        i += 1

    return op_array, c_array


# create TPY operators up to (including) given range
# needs the number of operators and system size in advance
def create_operators_TPY(op_range, size, number_of_operators):

    op_array = np.zeros((number_of_operators, 2 * size, size), dtype=np.int8)
    c_array = np.zeros((number_of_operators, 2 * size), dtype=np.complex128)

    count = 0

    # extra case for l=1 since file not read in as a list but as single string
    names = ["y"]
    i = 0
    for op_name in names:
        op_lab, op_c = TP_operator_from_string(op_name, size)
        op_array[count + i, :, :] = op_lab
        c_array[count + i, :] = op_c
        i += 1
    count = 1

    # larger ranges
    for r in range(2, op_range + 1, 1):

        filename = "operators/operators_TPY_l" + str(r) + ".txt"
        names = np.genfromtxt(filename, dtype='str')

        i = 0
        for op_name in names:

            op_lab, op_c = TP_operator_from_string(op_name, size)
            op_array[count + i, :, :] = op_lab
            c_array[count + i, :] = op_c
            i += 1

        count += i

    return op_array, c_array


# create TPY operators up to (including) given range
# needs the number of operators and system size in advance
def create_operators_TPY_compact(op_range, size, number_of_operators):

    op_array = np.zeros((number_of_operators, 1, size), dtype=np.int8)
    c_array = np.zeros((number_of_operators, 1), dtype=np.complex128)

    count = 0

    # extra case for l=1 since file not read in as a list but as single string
    names = ["y"]
    i = 0
    for op_name in names:
        op_lab, op_c = TP_operator_from_string_compact(op_name, size)
        op_array[count + i, :, :] = op_lab
        c_array[count + i, :] = op_c
        i += 1
    count = 1

    # larger ranges
    for r in range(2, op_range + 1, 1):

        filename = "/home/artem/Dropbox/bu_research/operators/operators_TPY_l" + str(r) + ".txt"
        names = np.genfromtxt(filename, dtype='str')

        i = 0
        for op_name in names:

            op_lab, op_c = TP_operator_from_string_compact(op_name, size)
            op_array[count + i, :, :] = op_lab
            c_array[count + i, :] = op_c
            i += 1

        count += i

    return op_array, c_array



# create TPY operators up to (including) given range
# needs the number of operators and system size in advance
def create_operators_TPY_compact_exact(op_range, size, number_of_operators):

    op_array = np.zeros((number_of_operators, 1, size), dtype=np.int8)
    c_array = np.zeros((number_of_operators, 1), dtype=np.complex128)

    count = 0

    # extra case for l=1 since file not read in as a list but as single string
    names = ["y"]
    i = 0
    for op_name in names:
        op_lab, op_c = TP_operator_from_string_compact(op_name, size)
        op_array[count + i, :, :] = op_lab
        c_array[count + i, :] = op_c
        i += 1
    count = 1

    # larger ranges
    for r in range(2, op_range, 1):

        filename = "/home/artem/Dropbox/bu_research/operators/operators_TPY_l" + str(r) + ".txt"
        names = np.genfromtxt(filename, dtype='str')

        i = 0
        for op_name in names:

            op_lab, op_c = TP_operator_from_string_compact(op_name, size)
            op_array[count + i, :, :] = op_lab
            c_array[count + i, :] = op_c
            i += 1

        count += i

    # operators for r=l need to be taken from the exact list
    filename = "/home/artem/Dropbox/bu_research/operators/operators_TPY_exact_reduced_l" + str(op_range) + ".txt"
    names = np.genfromtxt(filename, dtype='str')

    i = 0
    for op_name in names:
        op_lab, op_c = TP_operator_from_string_compact(op_name, size)
        op_array[count + i, :, :] = op_lab
        c_array[count + i, :] = op_c
        i += 1

    count += i

    return op_array, c_array



# create TPY operators up to (including) given range
# needs the number of operators and system size in advance
def create_operators_TPFY_compact(op_range, size, number_of_operators):

    op_array = np.zeros((number_of_operators, 1, size), dtype=np.int8)
    c_array = np.zeros((number_of_operators, 1), dtype=np.complex128)

    count = 0

    # extra case for l=1 and l=2 since l=1 empty and for l=2 file not read in as a list but as single string
    names = ["yz"]
    i = 0
    for op_name in names:
        op_lab, op_c = TP_operator_from_string_compact(op_name, size)
        op_array[count + i, :, :] = op_lab
        c_array[count + i, :] = op_c
        i += 1
    count = 1

    # larger ranges
    for r in range(3, op_range + 1, 1):

        filename = "operators/operators_TPFY_l" + str(r) + ".txt"
        names = np.genfromtxt(filename, dtype='str')

        i = 0
        for op_name in names:

            op_lab, op_c = TP_operator_from_string_compact(op_name, size)
            op_array[count + i, :, :] = op_lab
            c_array[count + i, :] = op_c
            i += 1

        count += i

    return op_array, c_array

# create TPFXY operators up to (including) given range
# needs the number of operators and system size in advance
def create_operators_TPFXY_compact(op_range, size, number_of_operators):

    op_array = np.zeros((number_of_operators, 1, size), dtype=np.int8)
    c_array = np.zeros((number_of_operators, 1), dtype=np.complex128)

    count = 0

    # no operators for l=1 and l=2

    # larger ranges
    for r in range(3, op_range + 1, 1):

        filename = "operators/operators_TPFXY_l" + str(r) + ".txt"
        names = np.genfromtxt(filename, dtype='str')

        i = 0
        for op_name in names:

            op_lab, op_c = TP_operator_from_string_compact(op_name, size)
            op_array[count + i, :, :] = op_lab
            c_array[count + i, :] = op_c
            i += 1

        count += i

    return op_array, c_array


# functions for optimization TP operators
# create list of commutators [H, O_i] from list of operators O_i
@nb.jit(nopython=True, parallel=False, cache=True)
def create_commutators_for_optimization_TP(operators_labels, operators_coeffs, ham_labels, ham_coeffs, system_size):

    number_of_operators = len(operators_coeffs[:, 0])
    commutator_labels = np.zeros((number_of_operators, 2 * system_size, system_size), dtype=np.int8)
    commutator_coeff = np.zeros((number_of_operators, 2 * system_size), dtype=np.complex128)

    for i in nb.prange(number_of_operators):

        op_labels = operators_labels[i, :, :]
        op_coeffs = operators_coeffs[i, :]

        labels, coeff = commute_operators_TP(ham_labels, op_labels, ham_coeffs, op_coeffs)

        commutator_labels[i, :, :] = labels
        commutator_coeff[i, :] = coeff

    return commutator_labels, commutator_coeff


# R matrix
# R_k = 2iTr(h_deriv * C_k), which is real-valued

@nb.jit(nopython=True, parallel=False, cache=True)
def create_R_matrix_TP(commutator_labels, commutator_coeffs, ham_deriv_labels, ham_deriv_coeff):

    number_of_operators = len(commutator_coeffs[:, 0])
    mat = np.zeros(number_of_operators, dtype=np.float64)

    for i in nb.prange(number_of_operators):

        # print(i)
        comm_label = commutator_labels[i, :, :]
        comm_coeff = commutator_coeffs[i, :]

        # # parity
        # for n in range(1, len(comm_label)):
        #     print(comm_label, np.roll(comm_label, n))
        #     if np.array_equal(comm_label, np.roll(comm_label, n)):
        #         comm_coeff *= 1 / 2


        # print_operator(ham_deriv_labels, ham_deriv_coeff)
        # print_operator(comm_label, comm_coeff)

        labels, coeff = multiply_operators_TP(ham_deriv_labels, comm_label, ham_deriv_coeff, comm_coeff)

        # print_operator_balanced(labels, coeff)
        tr = trace_operator_TP(labels, coeff)
        # print(tr)
        mat[i] = (2.j * tr).real

    return mat


# P matrix
# P_kl = -Tr(C_k C_l), which is real-valued
@nb.jit(nopython=True, parallel=False)
def create_P_matrix_TP(commutator_labels1, commutator_labels2, commutator_coeffs1, commutator_coeffs2):
    number_of_operators = len(commutator_coeffs1[:, 0])
    mat = np.zeros((number_of_operators, number_of_operators), dtype=np.float64)

    for i in nb.prange(number_of_operators):

        comm_label_i = commutator_labels1[i, :, :]
        comm_coeff_i = commutator_coeffs1[i, :]

        for j in range(number_of_operators):

            comm_label_j = commutator_labels2[j, :, :]
            comm_coeff_j = commutator_coeffs2[j, :]

            labels, coeff = multiply_operators_TP(comm_label_i, comm_label_j, comm_coeff_i, comm_coeff_j)
            tr = trace_operator_TP(labels, coeff)
            mat[i, j] = -tr.real

    return mat

# if both commutator sets are the same P_kl = P_lk
# compute and return only the upper triangular part
@nb.jit(nopython=True, parallel=False)
def create_P_matrix_symmetric_TP(commutator_labels1, commutator_coeffs1):
    number_of_operators = len(commutator_coeffs1[:, 0])
    mat = np.zeros((number_of_operators, number_of_operators), dtype=np.float64)
    diag = np.zeros(number_of_operators, dtype=np.float64)


    for i in nb.prange(number_of_operators):
        # print(i)
        # start = 0.
        # with nb.objmode(start='float64'):  # annotate return type
        #     # this region is executed by object-mode.
        #     start = time.time()
        comm_label_i = commutator_labels1[i, :, :]
        comm_coeff_i = commutator_coeffs1[i, :]

        for j in range(i, number_of_operators):

            comm_label_j = commutator_labels1[j, :, :]
            comm_coeff_j = commutator_coeffs1[j, :]

            labels, coeff = multiply_operators_TP(comm_label_i, comm_label_j, comm_coeff_i, comm_coeff_j)
            tr = trace_operator_TP(labels, coeff)

            if i == j:
                diag[i] = -tr.real
            
            else:
                mat[i, j] = -tr.real

        # end = 0.
        # with nb.objmode(end='float64'):  # annotate return type
        #     # this region is executed by object-mode.
        #     end = time.time()
        # print(i, end - start)
    return mat, diag


# printing functions
@nb.jit(nopython=True, cache=True)
def label_string(label):

    string = ""
    for c in label:
        string += pauli_name[c]
    return string


# @nb.jit(nopython=True, cache=True)
def label_string_op_name_style(label):

    string = ""
    for c in label:
        name = str(pauli_name[c])
        if name != "1":
            string += name.lower()
    return string


def print_operator_full(labels, coeff, digits=2):

    for i, string in enumerate(labels):
        print(np.around(coeff[i], digits), label_string(string))
    print("")


def print_operator(labels, coeff, digits=2):

    for i, string in enumerate(labels):

        if np.absolute(coeff[i]) > 1.e-14:
            print(np.around(coeff[i], digits), label_string(string))
    print("")


def print_operator_balanced(labels, coeffs, digits=2):

    labels = list(labels)
    coeffs = list(coeffs)

    new_labels = []
    new_coeff = []

    while len(labels) > 0:

        # print(labels)
        # print(coeffs)
        # print(new_labels)
        # print(new_coeff)

        string = labels.pop(0)
        coeff = coeffs.pop(0)

        if np.absolute(coeff) > 1.e-12:

            jl = []
            for j, string2 in enumerate(labels):
                if (string2 == string).all():
                    jl.append(j)
                    coeff += coeffs[j]

            for j in sorted(jl, reverse=True):
                del labels[j]
                del coeffs[j]

        new_labels.append(string)
        new_coeff.append(coeff)

    for i, string in enumerate(new_labels):

        coeff = new_coeff[i]
        if np.absolute(coeff) > 1.e-12:

            print(np.around(coeff, digits), label_string(string))

    print("")

# creates the operator matrix (as csr sparse matrix) in the K=0, P=1 symmetry sector
def create_TP_operator_matrix(labels, coeff, system_size):

    k_name = "0"
    par = 1

    # get Hilbert space dimension
    name_zip = "operators/1d_chain_indices_and_periods.zip"
    with zipfile.ZipFile(name_zip) as zipper:

        name = "1d_chain_TP_indices_and_periods_L=" + str(system_size) + "_k=" + k_name + ".npz"
        with io.BufferedReader(zipper.open(name, mode='r')) as f:
            data = np.load(f)
            periods = data["period"]
            parities = data["parity"]
            size = len(periods)

    # empty matrix
    mat = csr_matrix((size, size), dtype=np.complex128)

    # add all non-zero contributions
    for i, string in enumerate(labels):
        if np.absolute(coeff[i]) > 1.e-12:

            # get operator name from label
            # need to remove 1s and convert to lower case
            op_name = label_string_op_name_style(string)

            name_zip_op = "operators/operators_TP_L=" + str(system_size) + ".zip"
            with zipfile.ZipFile(name_zip_op) as zipper_op:
                mat_name = op_name + "_TP_L=" + str(system_size) + "_k=" + k_name + "_p=" + str(par) + ".npz"
                with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
                    data = np.load(f_op)
                    indptr = data["indptr"]
                    indices = data["idx"]
                    val = data["val"]
                op_mat = csr_matrix((val, indices, indptr), shape=(size, size))

            mat += coeff[i] * op_mat

    return mat


#######################################################################################################################
# Testing:

# # test string multiply / commute
# for L in [1, 2, 3, 4, 5]:
#
#     id_string = np.zeros(L, dtype=np.int8)
#     x_string = np.ones(L, dtype=np.int8)
#     y_string = np.full(L, 2, dtype=np.int8)
#     z_string = np.full(L, 3, dtype=np.int8)
#
#     # print(multiply_pauli_strings(id_string, id_string, 1.0, 1.0))
#     # print(multiply_pauli_strings(id_string, x_string, 1.0, 1.0))
#     # print(multiply_pauli_strings(id_string, y_string, 1.0, 1.0))
#     # print(multiply_pauli_strings(id_string, z_string, 1.0, 1.0))
#     #
#     # print(multiply_pauli_strings(x_string, x_string, 1.0, 1.0))
#     # print(multiply_pauli_strings(y_string, y_string, 1.0, 1.0))
#     # print(multiply_pauli_strings(z_string, z_string, 1.0, 1.0))
#     #
#     # print(multiply_pauli_strings(x_string, y_string, 1.0, 1.0))
#     # print(multiply_pauli_strings(y_string, x_string, 1.0, 1.0))
#     #
#     # print(multiply_pauli_strings(x_string, z_string, 1.0, 1.0))
#     # print(multiply_pauli_strings(z_string, x_string, 1.0, 1.0))
#     #
#     # print(multiply_pauli_strings(y_string, z_string, 1.0, 1.0))
#     # print(multiply_pauli_strings(z_string, y_string, 1.0, 1.0))
#
#     print(multiply_pauli_strings(x_string, y_string, 1.0, 1.0))
#     print(commute_pauli_strings(x_string, y_string, 1.0, 1.0))
#
#     start = time.time()
#     for i in range(1000000):
#         # multiply_pauli_strings(x_string, y_string, 1.0, 1.0)
#         commute_pauli_strings(x_string, y_string, 1.0, 1.0)
#     end = time.time()
#     print("L =", L, end - start)


# # test commute operators
# for L in [20]:
#
#     id_string = np.zeros(L, dtype=np.int8)
#     x_string = np.ones(L, dtype=np.int8)
#     y_string = np.full(L, 2, dtype=np.int8)
#     z_string = np.full(L, 3, dtype=np.int8)
#
#     op1_labels = List([id_string, x_string, y_string, z_string, id_string, x_string, y_string, z_string])
#     op1_coeffs = List([1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j])
#
#     op2_labels = List([id_string, x_string, y_string, z_string, id_string, x_string, y_string, z_string])
#     op2_coeffs = List([1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j])
#
#     prod_labels, prod_coeffs = multiply_operators(op1_labels, op2_labels, op1_coeffs, op2_coeffs)
#     comm_labels, comm_coeffs = commute_operators(op1_labels, op2_labels, op1_coeffs, op2_coeffs)
#
#     print("Multiply:")
#     print_operator(prod_labels, prod_coeffs)
#
#     start = time.time()
#     for i in range(1000000):
#         prod_labels, prod_coeffs = multiply_operators(op1_labels, op2_labels, op1_coeffs, op2_coeffs)
#     end = time.time()
#     print("Time:", end - start)
#
#
#     print("Commute:")
#     print_operator(comm_labels, comm_coeffs)
#     start = time.time()
#     for i in range(1000000):
#         comm_labels, comm_coeffs = commute_operators(op1_labels, op2_labels, op1_coeffs, op2_coeffs)
#     end = time.time()
#     print("Time:", end - start)

# test with array

# for L in [20]:
#
#     id_string = np.zeros(L, dtype=np.int8)
#     x_string = np.ones(L, dtype=np.int8)
#     y_string = np.full(L, 2, dtype=np.int8)
#     z_string = np.full(L, 3, dtype=np.int8)
#
#     op1_labels = np.array([id_string, x_string, y_string, z_string, id_string, x_string, y_string, z_string,
#                            id_string, x_string, y_string, z_string, id_string, x_string, y_string, z_string,
#                            id_string, x_string, y_string, z_string, id_string, x_string, y_string, z_string,
#                            id_string, x_string, y_string, z_string, id_string, x_string, y_string, z_string])
#
#     op1_coeffs = np.array([1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j,
#                            1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j,
#                            1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j,
#                            1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j])
#
#     op2_labels = np.array(
#         [id_string, x_string, y_string, z_string, id_string, x_string, y_string, z_string, id_string, x_string,
#          y_string, z_string, id_string, x_string, y_string, z_string, id_string, x_string, y_string, z_string,
#          id_string, x_string, y_string, z_string, id_string, x_string, y_string, z_string, id_string, x_string,
#          y_string, z_string])
#     op2_coeffs = np.array(
#         [1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j,
#          1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j,
#          1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j, 1.0 + 0.j,
#          1.0 + 0.j, 1.0 + 0.j])
#
#     prod_labels, prod_coeffs = multiply_operators_array(op1_labels, op2_labels, op1_coeffs, op2_coeffs)
#     comm_labels, comm_coeffs = commute_operators_array(op1_labels, op2_labels, op1_coeffs, op2_coeffs)
#
#     print("Multiply:")
#     print_operator(prod_labels, prod_coeffs)
#
#     start = time.time()
#     for i in range(10000):
#         prod_labels, prod_coeffs = multiply_operators_array(op1_labels, op2_labels, op1_coeffs, op2_coeffs)
#     end = time.time()
#     print("Time:", end - start)
#
#
#     print("Commute:")
#     print_operator(comm_labels, comm_coeffs)
#     start = time.time()
#     for i in range(10000):
#         comm_labels, comm_coeffs = commute_operators_array(op1_labels, op2_labels, op1_coeffs, op2_coeffs)
#     end = time.time()
#     print("Time:", end - start)


# L = 10
# lab_xz, c_xz = TP_operator_from_string("xz", L)
# lab_xy, c_xy = TP_operator_from_string("xy", L)
#
# # print_operator(lab_xz, c_xz)
# # print_operator(lab_xy, c_xy)
#
# lab_comm, c_comm = commute_operators_array(lab_xz, lab_xy, c_xz, c_xy)
# print_operator(lab_comm, c_comm)
#
# lab_comm_clean, c_comm_clean = operator_cleanup(lab_comm, c_comm)
# print_operator(lab_comm_clean, c_comm_clean)
#
#
# print("Commute:")
# start = time.time()
# for i in range(10000):
#     lab_comm, c_comm = commute_operators_array(lab_xz, lab_xy, c_xz, c_xy)
# end = time.time()
# print("Time:", end - start)
#
# lab_mult, c_mult = multiply_operators_array(lab_xz, lab_xz, c_xz, c_xz)
# lab_mult_clean, c_mult_clean = operator_cleanup(lab_mult, c_mult)
# print_operator(lab_mult_clean, c_mult_clean)
# tr = trace_operator(lab_mult, c_mult)
# print(tr)
#
# print("Trace:")
# start = time.time()
# for i in range(10000):
#     tr = trace_operator(lab_mult, c_mult)
# end = time.time()
# print("Time:", end - start)


# L = 4
# lab_x, c_x = TP_operator_from_string("x", L)
# lab_xz, c_xz = TP_operator_from_string("xz", L)
# lab_xy, c_xy = TP_operator_from_string("xy", L)
#
# print(lab_x)
# op_array = np.array([lab_xz, lab_xy])
# c_array = np.array([c_xz, c_xy])
#
# print(op_array)
# print(op_array[0, :])
# print(c_array)
#
# lab_comm, c_comm = create_commutators_for_optimization(op_array, c_array, lab_x, c_x, L)
# print(lab_comm, c_comm)


# # create TPY operators from file
# L = 12
# l = 3
# tr_I = 2 ** L
#
# lab_x, c_x = TP_operator_from_string("x", L)
# lab_z, c_z = TP_operator_from_string("z", L)
# lab_zz, c_zz = TP_operator_from_string("zz", L)
#
# # set spin operators
# # note factor of 0.5 due to double counting in permutations
# c_x *= 0.5 * 0.5
# c_z *= 0.5 * 0.5
# c_zz *= 0.25 * 0.5
#
# # print("Hamiltonian:")
# # print_operator(lab_x, c_x)
# # print_operator(lab_z, c_z)
# # print_operator(lab_zz, c_zz)
#
# # read in basis operators
# num_op = np.loadtxt("operators_TPY_size_full.txt").astype(int)
# TPY_labels, TPY_coeffs = create_operators_TPY(l, L, num_op[l - 1])
# print("size:", num_op[l - 1])
#
# # print("Operator Basis:")
# # for j in range(num_op[l - 1]):
# #     lab = TPY_labels[j]
# #     c = TPY_coeffs[j]
# #     print_operator(lab, c)
#
# # compile for fair timing comparison
# C_x_labels, C_x_coeffs = create_commutators_for_optimization(TPY_labels, TPY_coeffs, lab_x, c_x, L)
# R_x_x = create_R_matrix(C_x_labels, C_x_coeffs, lab_x, c_x)
# # P_x_x = create_P_matrix_symmetric(C_x_labels, C_x_coeffs)
# # P_x_zz = create_P_matrix(C_x_labels, C_x_labels, C_x_coeffs, C_x_coeffs)
#
#
# start = time.time()
# # commutators
# C_x_labels, C_x_coeffs = create_commutators_for_optimization(TPY_labels, TPY_coeffs, lab_x, c_x, L)
# C_z_labels, C_z_coeffs = create_commutators_for_optimization(TPY_labels, TPY_coeffs, lab_z, c_z, L)
# C_zz_labels, C_zz_coeffs = create_commutators_for_optimization(TPY_labels, TPY_coeffs, lab_zz, c_zz, L)
# end = time.time()
# print("Time - Commutators:", end - start)
#
# # print("Commutators:")
# # for i, lab in enumerate(C_x_labels):
# #     c = C_x_coeffs[i, :]
# #     lab, c = operator_cleanup(lab, c)
# #
# #     if len(lab) > 0:
# #         print_operator(lab, c)
# #     else:
# #         print("empty list")
#
# start = time.time()
# # R matrices
# R_x_x = create_R_matrix(C_x_labels, C_x_coeffs, lab_x, c_x)
# R_x_z = create_R_matrix(C_z_labels, C_z_coeffs, lab_x, c_x)
# R_x_zz = create_R_matrix(C_zz_labels, C_zz_coeffs, lab_x, c_x)
#
# R_z_x = create_R_matrix(C_x_labels, C_x_coeffs, lab_z, c_z)
# R_z_z = create_R_matrix(C_z_labels, C_z_coeffs, lab_z, c_z)
# R_z_zz = create_R_matrix(C_zz_labels, C_zz_coeffs, lab_z, c_z)
# end = time.time()
# print("Time - R:", end - start)
#
# # print("R Matrices:")
# # print(R_x_x)
# # print(R_x_z)
# # print(R_x_zz)
# # print(R_z_x)
# # print(R_z_z)
# # print(R_z_zz)
#
# # P matrices
#
# start = time.time()
#
# P_x_x = create_P_matrix_symmetric(C_x_labels, C_x_coeffs)
# P_z_z = create_P_matrix_symmetric(C_z_labels, C_z_coeffs)
# P_zz_zz = create_P_matrix_symmetric(C_zz_labels, C_zz_coeffs)
#
# end = time.time()
# print("Time - P symmetric:", end - start)
# print("Threading layer chosen: %s" % nb.threading_layer())

# start = time.time()
# P_x_zz = create_P_matrix(C_x_labels, C_zz_labels, C_x_coeffs, C_zz_coeffs)
# P_z_x = create_P_matrix(C_z_labels, C_x_labels, C_z_coeffs, C_x_coeffs)
# P_z_zz = create_P_matrix(C_z_labels, C_zz_labels, C_z_coeffs, C_zz_coeffs)
#
# end = time.time()
# print("Time - P full:", end - start)


# print("P Matrices:")
# print(P_x_x / tr_I)
# print(P_x_zz)
# print(P_z_x)
# print(P_z_z)
# print(P_z_zz)
# print(P_zz_zz)


# # test TP operations
# L = 4
# x_TP_labels = np.array([[1, 0, 0, 0]], dtype=np.int8)
# x_TP_coeffs = np.array([1.0], dtype=np.complex128)
#
# xy_TP_labels = np.array([[1, 2, 0, 0]], dtype=np.int8)
# xy_TP_coeffs = np.array([1.0], dtype=np.complex128)
#
# mult_lab, mult_c = multiply_operators_TP(x_TP_labels, xy_TP_labels, x_TP_coeffs, xy_TP_coeffs)
# print_operator(mult_lab, mult_c)
#
# comm_lab, comm_c = commute_operators_TP(x_TP_labels, xy_TP_labels, x_TP_coeffs, xy_TP_coeffs)
# print_operator(comm_lab, comm_c)
#
# mult_lab, mult_c = multiply_operators_TP(x_TP_labels, x_TP_labels, x_TP_coeffs, x_TP_coeffs)
# print_operator(mult_lab, mult_c)
# print(trace_operator_TP(mult_lab, mult_c))
#
# comm_lab, comm_c = commute_operators_TP(x_TP_labels, x_TP_labels, x_TP_coeffs, x_TP_coeffs)
# print_operator(comm_lab, comm_c)
# print(trace_operator_TP(comm_lab, comm_c))



# ####################################################################################################
# # compile for fair timing comparison
# # create TPY operators from file
# tr_I = 4
#
# lab_x, c_x = TP_operator_from_string_compact("x", 2)
# lab_z, c_z = TP_operator_from_string_compact("z", 2)
# lab_zz, c_zz = TP_operator_from_string_compact("zz", 2)
#
# # set spin operators
# c_x *= 0.5 * 0.5
# c_z *= 0.5 * 0.5
# c_zz *= 0.25 * 0.5
#
# num_op = np.loadtxt("operators_TPY_size_full.txt").astype(int)
# TPY_labels, TPY_coeffs = create_operators_TPY_compact(1, 2, num_op[0])
# C_x_labels, C_x_coeffs = create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_x, c_x, 2)
# R_x_x = create_R_matrix_TP(C_x_labels, C_x_coeffs, lab_x, c_x)
# P_x_x = create_P_matrix_symmetric_TP(C_x_labels, C_x_coeffs)
# P_x_zz = create_P_matrix_TP(C_x_labels, C_x_labels, C_x_coeffs, C_x_coeffs)
# ####################################################################################################
#
#
# # create TPY operators from file
# L = 12
# l = 4
# tr_I = 2 ** L
#
# lab_x, c_x = TP_operator_from_string_compact("x", L)
# lab_z, c_z = TP_operator_from_string_compact("z", L)
# lab_zz, c_zz = TP_operator_from_string_compact("zz", L)
#
# # set spin operators
# c_x *= 0.5 * 0.5
# c_z *= 0.5 * 0.5
# c_zz *= 0.25 * 0.5
#
# # print("Hamiltonian:")
# # print_operator(lab_x, c_x)
# # print_operator(lab_z, c_z)
# # print_operator(lab_zz, c_zz)
# # read in basis operators
# num_op = np.loadtxt("operators_TPY_size_full.txt").astype(int)
# TPY_labels, TPY_coeffs = create_operators_TPY_compact(l, L, num_op[l - 1])
# print("size:", num_op[l - 1])
#
# # print("Operator Basis:")
# # for j in range(num_op[l - 1]):
# #     lab = TPY_labels[j]
# #     c = TPY_coeffs[j]
# #     print_operator(lab, c)
#
#
#
#
# start = time.time()
# # commutators
# C_x_labels, C_x_coeffs = create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_x, c_x, L)
# C_z_labels, C_z_coeffs = create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_z, c_z, L)
# C_zz_labels, C_zz_coeffs = create_commutators_for_optimization_TP(TPY_labels, TPY_coeffs, lab_zz, c_zz, L)
# end = time.time()
# print("Time - Commutators:", end - start)
#
# # print("Commutators:")
# # for i, lab in enumerate(C_x_labels):
# #     c = C_x_coeffs[i, :]
# #     lab, c = operator_cleanup(lab, c)
# #
# #     if len(lab) > 0:
# #         print_operator(lab, c)
# #     else:
# #         print("empty list")
#
# start = time.time()
# # R matrices
# R_x_x = create_R_matrix_TP(C_x_labels, C_x_coeffs, lab_x, c_x)
# R_x_z = create_R_matrix_TP(C_z_labels, C_z_coeffs, lab_x, c_x)
# R_x_zz = create_R_matrix_TP(C_zz_labels, C_zz_coeffs, lab_x, c_x)
#
# R_z_x = create_R_matrix_TP(C_x_labels, C_x_coeffs, lab_z, c_z)
# R_z_z = create_R_matrix_TP(C_z_labels, C_z_coeffs, lab_z, c_z)
# R_z_zz = create_R_matrix_TP(C_zz_labels, C_zz_coeffs, lab_z, c_z)
# end = time.time()
# print("Time - R:", end - start)
#
# print("R Matrices:")
# print(R_x_x / tr_I)
# print(R_x_z / tr_I)
# print(R_x_zz / tr_I)
# print(R_z_x / tr_I)
# print(R_z_z / tr_I)
# print(R_z_zz / tr_I)
#
# # P matrices
#
# start = time.time()
# P_x_x = create_P_matrix_symmetric_TP(C_x_labels, C_x_coeffs)
# P_z_z = create_P_matrix_symmetric_TP(C_z_labels, C_z_coeffs)
# P_zz_zz = create_P_matrix_symmetric_TP(C_zz_labels, C_zz_coeffs)
# end = time.time()
# print("Time - P symmetric:", end - start)
#
# start = time.time()
# P_x_zz = create_P_matrix_TP(C_x_labels, C_zz_labels, C_x_coeffs, C_zz_coeffs)
# P_z_x = create_P_matrix_TP(C_z_labels, C_x_labels, C_z_coeffs, C_x_coeffs)
# P_z_zz = create_P_matrix_TP(C_z_labels, C_zz_labels, C_z_coeffs, C_zz_coeffs)
# end = time.time()
# print("Time - P full:", end - start)
#
# print("P Matrices:")
# print(P_x_x / tr_I)
# print(P_z_z / tr_I)
# print(P_zz_zz / tr_I)
#
# print(P_x_zz / tr_I)
# print(P_z_x / tr_I)
# print(P_z_zz / tr_I)
#
# print("Threading layer chosen: %s" % nb.threading_layer())
