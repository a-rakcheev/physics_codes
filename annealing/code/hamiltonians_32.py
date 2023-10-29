import numpy as np
import numba as nb
import scipy.sparse as sp
# import qa_functions as qa
import scipy.sparse.linalg as spla


# # hamiltonians for testing
# # --------------------------------------------------------------
# def hamiltonian_xz(number_of_spins, boundary_condition):
#
#     size = 2 ** number_of_spins
#     mat = np.zeros((size, size))
#
#     for i in range(size):
#         for j in range(number_of_spins - 1):
#
#             idx_flip = number_of_spins - 1 - j
#             idx_measure = idx_flip - 1
#
#             # measure bit
#             bit = (i >> idx_measure) % 2
#
#             # flip bit
#             k = np.bitwise_xor(i, 2 ** idx_flip)
#
#             # matrix element
#             mat[k, i] = 0.5 * (bit - 0.5)
#
#         if boundary_condition == "pbc":
#
#             idx_flip = 0
#             idx_measure = number_of_spins - 1
#
#             # measure bit
#             bit = (i >> idx_measure) % 2
#
#             # flip bit
#             k = np.bitwise_xor(i, 2 ** idx_flip)
#
#             # matrix element
#             mat[k, i] = 0.5 * (bit - 0.5)
#
#     return mat
#
#
# def hamiltonian_zx(number_of_spins, boundary_condition):
#     size = 2 ** number_of_spins
#     mat = np.zeros((size, size))
#
#     for i in range(size):
#         for j in range(number_of_spins - 1):
#             idx_measure = number_of_spins - 1 - j
#             idx_flip = idx_measure - 1
#
#             # measure bit
#             bit = (i >> idx_measure) % 2
#
#             # flip bit
#             k = np.bitwise_xor(i, 2 ** idx_flip)
#
#             # matrix element
#             mat[k, i] = 0.5 * (bit - 0.5)
#
#         if boundary_condition == "pbc":
#             idx_flip = number_of_spins - 1
#             idx_measure = 0
#
#             # measure bit
#             bit = (i >> idx_measure) % 2
#
#             # flip bit
#             k = np.bitwise_xor(i, 2 ** idx_flip)
#
#             # matrix element
#             mat[k, i] = 0.5 * (bit - 0.5)
#
#     return mat
#
#
# def hamiltonian_y_sparse(number_of_spins, field):
#
#     mat = qa.hamiltonian_x_sparse_from_diag(number_of_spins, field)
#     mat -= sp.tril(mat) * 2.
#     return 1.j * mat
#
#
# def hamiltonian_xx_sparse(number_of_spins, boundary_condition, field):
#
#     spin_first = qa.spin_x_matrix_sparse(1, number_of_spins, 1.)
#     spin_second = qa.spin_x_matrix_sparse(2, number_of_spins, 1.)
#
#     mat = field * spin_first * spin_second
#
#     for i in np.arange(3, number_of_spins + 1, 1):
#
#         spin_first = spin_second
#         spin_second = qa.spin_x_matrix_sparse(i, number_of_spins, 1.)
#         mat += field * spin_first * spin_second
#
#     if boundary_condition == "pbc":
#
#         spin_first = spin_second
#         spin_second = qa.spin_x_matrix_sparse(1, number_of_spins, 1.)
#         mat += field * spin_first * spin_second
#
#     return mat
#
#
# def hamiltonian_xx_broken_pbc(number_of_spins, field, field_boundary):
#
#     spin_first = qa.spin_x_matrix_sparse(1, number_of_spins, 1.)
#     spin_second = qa.spin_x_matrix_sparse(2, number_of_spins, 1.)
#
#     mat = field * spin_first * spin_second
#
#     for i in np.arange(3, number_of_spins + 1, 1):
#         spin_first = spin_second
#         spin_second = qa.spin_x_matrix_sparse(i, number_of_spins, 1.)
#         mat += field * spin_first * spin_second
#
#     spin_first = spin_second
#     spin_second = qa.spin_x_matrix_sparse(1, number_of_spins, 1.)
#     mat += field_boundary * spin_first * spin_second
#
#     return mat
#
#
# def hamiltonian_xx_2_broken_pbc(number_of_spins, field, field_boundary):
#
#     spin_first = qa.spin_x_matrix_sparse(1, number_of_spins, 1.)
#     spin_second = qa.spin_x_matrix_sparse(3, number_of_spins, 1.)
#
#     mat = field * spin_first * spin_second
#
#     for i in np.arange(2, number_of_spins - 1, 1):
#         spin_first = qa.spin_x_matrix_sparse(i, number_of_spins, 1.)
#         spin_second = qa.spin_x_matrix_sparse(i + 2, number_of_spins, 1.)
#         mat += field * spin_first * spin_second
#
#     # N - 1, 1 term, breaks translational invariance due to special coupling
#     spin_first = qa.spin_x_matrix_sparse(number_of_spins - 1, number_of_spins, 1.)
#     spin_second = qa.spin_x_matrix_sparse(1, number_of_spins, 1.)
#     mat += field_boundary * spin_first * spin_second
#
#     # N, 2 term, normal coupling
#     spin_first = qa.spin_x_matrix_sparse(number_of_spins, number_of_spins, 1.)
#     spin_second = qa.spin_x_matrix_sparse(2, number_of_spins, 1.)
#     mat += field * spin_first * spin_second
#
#     return mat

# ---------------------------------------


# apply pauli matrix on spin state defined by index
@nb.jit("Tuple((i4, f8))(i4, i8, i8)", nopython=True, cache=True)
def apply_pauli_z(index, site, number_of_spins):

    bit_index = number_of_spins - site
    val = 2. * ((index >> bit_index) % 2) - 1.
    index_out = index
    return index_out, val


@nb.jit("Tuple((i4, f8))(i4, i8, i8)", nopython=True, cache=True)
def apply_pauli_x(index, site, number_of_spins):
    bit_index = number_of_spins - site
    val = 1.
    index_out = index ^ 2 ** bit_index
    return index_out, val


@nb.jit("Tuple((i4, c16))(i4, i8, i8)", nopython=True, cache=True)
def apply_pauli_y(index, site, number_of_spins):
    bit_index = number_of_spins - site
    val = 1.j * (2. * ((index >> bit_index) % 2) - 1.)
    index_out = index ^ 2 ** bit_index
    return index_out, val


# create n-body operator J * tprod_i^n sigma^(alpha_i)_(j_i)
# note that all sites need to be different (so for example sigma^z_1 tprod sigma^x_1 does not work
# site labeling is 1 ... N
# naming is 1, 2, 3 = x, y, z
# these need to be integers
# the output type valtype need to be specified currently
# depending on whether the matrix elements should be complex (1) or real (0)

# # there are checks on this and on the correct length of the site list and the names list
# # also on the correct site indexing and names

@nb.njit("Tuple((i4[:], f8[:]))(i8, i8, f8, i8[:], i1[:])", parallel=False, cache=True)
def single_operator_real(number_of_spins, order, strength, sites, names):

    # checks
    assert(len(sites) and len(names) == order), "Given site or name list is not consistent with the order"
    assert(len(np.unique(sites)) == order), "Some sites appear twice. This is not supported."

    # indices for sparse matrix
    size = 2 ** number_of_spins
    idx_row = np.zeros(size, dtype=np.int32)
    values = np.zeros(size, dtype=np.float64)

    # create indices and matrix elements
    for i in range(size):
        c = strength
        i2 = i
        for j in range(order):

            if names[j] == 1:
                i2, d = apply_pauli_x(i2, sites[j], number_of_spins)

            elif names[j] == 2:
                raise TypeError("Pauli-Y operator used in real operator function.")

            else:
                i2, d = apply_pauli_z(i2, sites[j], number_of_spins)

            c *= d

        idx_row[i] = i2
        values[i] = c

    return idx_row, values


# @nb.njit("Tuple((i4[:], c16[:]))(i8, i8, f8, i8[:], i1[:])", parallel=False, cache=True)
@nb.njit(parallel=False, cache=True)
def single_operator_complex(number_of_spins, order, strength, sites, names):

    # checks
    assert(len(sites) and len(names) == order), "Given site or name list is not consistent with the order"
    assert(len(np.unique(sites)) == order), "Some sites appear twice. This is not supported."

    # indices for sparse matrix
    size = 2 ** number_of_spins
    idx_row = np.zeros(size, dtype=np.int32)
    values = np.zeros(size, dtype=np.complex128)

    # create indices and matrix elements
    for i in range(size):
        c = strength
        i2 = i
        for j in range(order):

            if names[j] == 1:
                i2, d = apply_pauli_x(i2, sites[j], number_of_spins)

            elif names[j] == 2:
                i2, d = apply_pauli_y(i2, sites[j], number_of_spins)

            else:
                i2, d = apply_pauli_z(i2, sites[j], number_of_spins)

            c *= d

        idx_row[i] = i2
        values[i] = c

    return idx_row, values


@nb.jit(forceobj=True)
def operator_sum_real(number_of_spins, order, strengths, sites, names):

    size = 2 ** number_of_spins
    number_of_terms = len(names)
    idx_row = np.zeros(number_of_terms * size, dtype=np.int32)
    idx_col = np.zeros(number_of_terms * size, dtype=np.int32)
    values = np.zeros(number_of_terms * size, dtype=np.float64)

    for i in range(number_of_terms):

        idx, vals = single_operator_real(number_of_spins, order, strengths[i], sites[i, :], names[i, :])
        idx_row[i * size: (i + 1) * size] = idx
        idx_col[i * size: (i + 1) * size] = np.arange(0, 2 ** number_of_spins, 1)
        values[i * size: (i + 1) * size] = vals

    return sp.csr_matrix((values, (idx_row, idx_col)), shape=(size, size))


@nb.jit(forceobj=True)
def operator_sum_real_diag(number_of_spins, order, strengths, sites, names):

    number_of_terms = len(names)
    size = 2 ** number_of_spins
    values = np.zeros(size)

    for i in range(number_of_terms):

        idx, vals = single_operator_real(number_of_spins, order, strengths[i], sites[i, :], names[i, :])
        values[idx] += vals

    diags = sp.diags(values, shape=(size, size), format="csr")
    return diags


@nb.jit(forceobj=True)
def operator_sum_complex(number_of_spins, order, strengths, sites, names):

    size = 2 ** number_of_spins
    number_of_terms = len(names)
    idx_row = np.zeros(number_of_terms * size, dtype=np.int32)
    idx_col = np.zeros(number_of_terms * size, dtype=np.int32)
    values = np.zeros(number_of_terms * size, dtype=np.complex128)

    for i in range(number_of_terms):

        idx, vals = single_operator_complex(number_of_spins, order, strengths[i], sites[i, :], names[i, :])
        idx_row[i * size: (i + 1) * size] = idx
        idx_col[i * size: (i + 1) * size] = np.arange(0, 2 ** number_of_spins, 1)
        values[i * size: (i + 1) * size] = vals

    return sp.csr_matrix((values, (idx_row, idx_col)), shape=(size, size))



# apply operators matrix-free onto a single vector
@nb.njit(parallel=False, cache=True)
def single_operator_real_mf(vec, number_of_spins, order, strength, sites, names):

    # checks
    assert(len(sites) and len(names) == order), "Given site or name list is not consistent with the order"
    assert(len(np.unique(sites)) == order), "Some sites appear twice. This is not supported."

    vec2 = np.zeros_like(vec)
    size = 2 ** number_of_spins

    # create indices and matrix elements
    for i in range(size):
        c = strength
        i2 = i
        for j in range(order):

            if names[j] == 1:
                i2, d = apply_pauli_x(i2, sites[j], number_of_spins)

            elif names[j] == 2:
                raise TypeError("Pauli-Y operator used in real operator function.")

            else:
                i2, d = apply_pauli_z(i2, sites[j], number_of_spins)

            c *= d

        vec2[i] += c * vec[i2]
    return vec2


@nb.njit(parallel=False, cache=True)
def single_operator_complex_mf(vec, number_of_spins, order, strength, sites, names):

    # checks
    assert(len(sites) and len(names) == order), "Given site or name list is not consistent with the order"
    assert(len(np.unique(sites)) == order), "Some sites appear twice. This is not supported."

    vec2 = np.zeros_like(vec)
    size = 2 ** number_of_spins

    # create indices and matrix elements
    for i in range(size):
        c = strength
        i2 = i
        for j in range(order):

            if names[j] == 1:
                i2, d = apply_pauli_x(i2, sites[j], number_of_spins)

            elif names[j] == 2:
                i2, d = apply_pauli_y(i2, sites[j], number_of_spins)

            else:
                i2, d = apply_pauli_z(i2, sites[j], number_of_spins)

            c *= d
        print(i, i2, c)
        vec2[i] += c * vec[i2]
    return vec2



@nb.njit("Tuple((i8[:, :], i1[:, :]))(i8)", parallel=False, cache=True)
def input_h_x(number_of_spins):

    site_array = np.arange(1, number_of_spins + 1, 1, np.int64).reshape(number_of_spins, 1)
    name_array = np.full(number_of_spins, 1, np.int8).reshape(number_of_spins, 1)

    return site_array, name_array


@nb.njit("Tuple((i8[:, :], i1[:, :]))(i8)", parallel=False, cache=True)
def input_h_y(number_of_spins):
    site_array = np.arange(1, number_of_spins + 1, 1, np.int64).reshape(number_of_spins, 1)
    name_array = np.full(number_of_spins, 2, np.int8).reshape(number_of_spins, 1)

    return site_array, name_array


@nb.njit("Tuple((i8[:, :], i1[:, :]))(i8)", parallel=False, cache=True)
def input_h_z(number_of_spins):
    site_array = np.arange(1, number_of_spins + 1, 1, np.int64).reshape(number_of_spins, 1)
    name_array = np.full(number_of_spins, 3, np.int8).reshape(number_of_spins, 1)

    return site_array, name_array


@nb.njit("Tuple((i8[:, :], i1[:, :]))(i8, i1)", parallel=False, cache=True)
def input_h_xx_1d(number_of_spins, boundary_condition):

    if boundary_condition == 0:

        site_array = np.zeros((number_of_spins - 1, 2), np.int64)
        name_array = np.zeros((number_of_spins - 1, 2), np.int8)

        for i in range(number_of_spins - 1):

            site_array[i, :] = np.array([i + 1, i + 2], np.int64)
            name_array[i, :] = np.array([1, 1], np.int8)

    elif boundary_condition == 1:

        site_array = np.zeros((number_of_spins, 2), np.int64)
        name_array = np.zeros((number_of_spins, 2), np.int8)

        for i in range(number_of_spins - 1):

            site_array[i, :] = np.array([i + 1, i + 2], np.int64)
            name_array[i, :] = np.array([1, 1], np.int8)

        site_array[number_of_spins - 1, :] = np.array([number_of_spins, 1], np.int64)
        name_array[number_of_spins - 1, :] = np.array([1, 1], np.int8)

    else:

        raise ValueError("Unknown Boundary Condition. Only obc (0) and pbc (1) supported.")

    return site_array, name_array


@nb.njit("Tuple((i8[:, :], i1[:, :]))(i8, i1)", parallel=False, cache=True)
def input_h_yy_1d(number_of_spins, boundary_condition):
    if boundary_condition == 0:

        site_array = np.zeros((number_of_spins - 1, 2), np.int64)
        name_array = np.zeros((number_of_spins - 1, 2), np.int8)

        for i in range(number_of_spins - 1):
            site_array[i, :] = np.array([i + 1, i + 2], np.int64)
            name_array[i, :] = np.array([2, 2], np.int8)

    elif boundary_condition == 1:

        site_array = np.zeros((number_of_spins, 2), np.int64)
        name_array = np.zeros((number_of_spins, 2), np.int8)

        for i in range(number_of_spins - 1):
            site_array[i, :] = np.array([i + 1, i + 2], np.int64)
            name_array[i, :] = np.array([3, 3], np.int8)

        site_array[number_of_spins - 1, :] = np.array([number_of_spins, 1], np.int64)
        name_array[number_of_spins - 1, :] = np.array([3, 3], np.int8)

    else:

        raise ValueError("Unknown Boundary Condition. Only obc (0) and pbc (1) supported.")

    return site_array, name_array


@nb.njit("Tuple((i8[:, :], i1[:, :]))(i8, i1)", parallel=False, cache=True)
def input_h_zz_1d(number_of_spins, boundary_condition):
    if boundary_condition == 0:

        site_array = np.zeros((number_of_spins - 1, 2), np.int64)
        name_array = np.zeros((number_of_spins - 1, 2), np.int8)

        for i in range(number_of_spins - 1):
            site_array[i, :] = np.array([i + 1, i + 2], np.int64)
            name_array[i, :] = np.array([3, 3], np.int8)

    elif boundary_condition == 1:

        site_array = np.zeros((number_of_spins, 2), np.int64)
        name_array = np.zeros((number_of_spins, 2), np.int8)

        for i in range(number_of_spins - 1):
            site_array[i, :] = np.array([i + 1, i + 2], np.int64)
            name_array[i, :] = np.array([3, 3], np.int8)

        site_array[number_of_spins - 1, :] = np.array([number_of_spins, 1], np.int64)
        name_array[number_of_spins - 1, :] = np.array([3, 3], np.int8)

    else:

        raise ValueError("Unknown Boundary Condition. Only obc (0) and pbc (1) supported.")

    return site_array, name_array


@nb.njit("Tuple((i8[:, :], i1[:, :]))(i8, i1)", parallel=False, cache=True)
def input_h_xy_1d(number_of_spins, boundary_condition):
    if boundary_condition == 0:

        site_array = np.zeros((number_of_spins - 1, 2), np.int64)
        name_array = np.zeros((number_of_spins - 1, 2), np.int8)

        for i in range(number_of_spins - 1):
            site_array[i, :] = np.array([i + 1, i + 2], np.int64)
            name_array[i, :] = np.array([1, 2], np.int8)

    elif boundary_condition == 1:

        site_array = np.zeros((number_of_spins, 2), np.int64)
        name_array = np.zeros((number_of_spins, 2), np.int8)

        for i in range(number_of_spins - 1):
            site_array[i, :] = np.array([i + 1, i + 2], np.int64)
            name_array[i, :] = np.array([1, 2], np.int8)

        site_array[number_of_spins - 1, :] = np.array([number_of_spins, 1], np.int64)
        name_array[number_of_spins - 1, :] = np.array([1, 2], np.int8)

    else:

        raise ValueError("Unknown Boundary Condition. Only obc (0) and pbc (1) supported.")

    return site_array, name_array


@nb.njit("Tuple((i8[:, :], i1[:, :]))(i8, i1)", parallel=False, cache=True)
def input_h_xz_1d(number_of_spins, boundary_condition):
    if boundary_condition == 0:

        site_array = np.zeros((number_of_spins - 1, 2), np.int64)
        name_array = np.zeros((number_of_spins - 1, 2), np.int8)

        for i in range(number_of_spins - 1):
            site_array[i, :] = np.array([i + 1, i + 2], np.int64)
            name_array[i, :] = np.array([1, 3], np.int8)

    elif boundary_condition == 1:

        site_array = np.zeros((number_of_spins, 2), np.int64)
        name_array = np.zeros((number_of_spins, 2), np.int8)

        for i in range(number_of_spins - 1):
            site_array[i, :] = np.array([i + 1, i + 2], np.int64)
            name_array[i, :] = np.array([1, 3], np.int8)

        site_array[number_of_spins - 1, :] = np.array([number_of_spins, 1], np.int64)
        name_array[number_of_spins - 1, :] = np.array([1, 3], np.int8)

    else:

        raise ValueError("Unknown Boundary Condition. Only obc (0) and pbc (1) supported.")

    return site_array, name_array


@nb.njit("Tuple((i8[:, :], i1[:, :]))(i8, i1)", parallel=False, cache=True)
def input_h_yx_1d(number_of_spins, boundary_condition):
    if boundary_condition == 0:

        site_array = np.zeros((number_of_spins - 1, 2), np.int64)
        name_array = np.zeros((number_of_spins - 1, 2), np.int8)

        for i in range(number_of_spins - 1):
            site_array[i, :] = np.array([i + 1, i + 2], np.int64)
            name_array[i, :] = np.array([2, 1], np.int8)

    elif boundary_condition == 1:

        site_array = np.zeros((number_of_spins, 2), np.int64)
        name_array = np.zeros((number_of_spins, 2), np.int8)

        for i in range(number_of_spins - 1):
            site_array[i, :] = np.array([i + 1, i + 2], np.int64)
            name_array[i, :] = np.array([2, 1], np.int8)

        site_array[number_of_spins - 1, :] = np.array([number_of_spins, 1], np.int64)
        name_array[number_of_spins - 1, :] = np.array([2, 1], np.int8)

    else:

        raise ValueError("Unknown Boundary Condition. Only obc (0) and pbc (1) supported.")

    return site_array, name_array


@nb.njit("Tuple((i8[:, :], i1[:, :]))(i8, i1)", parallel=False, cache=True)
def input_h_yz_1d(number_of_spins, boundary_condition):
    if boundary_condition == 0:

        site_array = np.zeros((number_of_spins - 1, 2), np.int64)
        name_array = np.zeros((number_of_spins - 1, 2), np.int8)

        for i in range(number_of_spins - 1):
            site_array[i, :] = np.array([i + 1, i + 2], np.int64)
            name_array[i, :] = np.array([2, 3], np.int8)

    elif boundary_condition == 1:

        site_array = np.zeros((number_of_spins, 2), np.int64)
        name_array = np.zeros((number_of_spins, 2), np.int8)

        for i in range(number_of_spins - 1):
            site_array[i, :] = np.array([i + 1, i + 2], np.int64)
            name_array[i, :] = np.array([2, 3], np.int8)

        site_array[number_of_spins - 1, :] = np.array([number_of_spins, 1], np.int64)
        name_array[number_of_spins - 1, :] = np.array([2, 3], np.int8)

    else:

        raise ValueError("Unknown Boundary Condition. Only obc (0) and pbc (1) supported.")

    return site_array, name_array


@nb.njit("Tuple((i8[:, :], i1[:, :]))(i8, i1)", parallel=False, cache=True)
def input_h_zx_1d(number_of_spins, boundary_condition):
    if boundary_condition == 0:

        site_array = np.zeros((number_of_spins - 1, 2), np.int64)
        name_array = np.zeros((number_of_spins - 1, 2), np.int8)

        for i in range(number_of_spins - 1):
            site_array[i, :] = np.array([i + 1, i + 2], np.int64)
            name_array[i, :] = np.array([3, 1], np.int8)

    elif boundary_condition == 1:

        site_array = np.zeros((number_of_spins, 2), np.int64)
        name_array = np.zeros((number_of_spins, 2), np.int8)

        for i in range(number_of_spins - 1):
            site_array[i, :] = np.array([i + 1, i + 2], np.int64)
            name_array[i, :] = np.array([3, 1], np.int8)

        site_array[number_of_spins - 1, :] = np.array([number_of_spins, 1], np.int64)
        name_array[number_of_spins - 1, :] = np.array([3, 1], np.int8)

    else:

        raise ValueError("Unknown Boundary Condition. Only obc (0) and pbc (1) supported.")

    return site_array, name_array


@nb.njit("Tuple((i8[:, :], i1[:, :]))(i8, i1)", parallel=False, cache=True)
def input_h_zy_1d(number_of_spins, boundary_condition):
    if boundary_condition == 0:

        site_array = np.zeros((number_of_spins - 1, 2), np.int64)
        name_array = np.zeros((number_of_spins - 1, 2), np.int8)

        for i in range(number_of_spins - 1):
            site_array[i, :] = np.array([i + 1, i + 2], np.int64)
            name_array[i, :] = np.array([3, 2], np.int8)

    elif boundary_condition == 1:

        site_array = np.zeros((number_of_spins, 2), np.int64)
        name_array = np.zeros((number_of_spins, 2), np.int8)

        for i in range(number_of_spins - 1):
            site_array[i, :] = np.array([i + 1, i + 2], np.int64)
            name_array[i, :] = np.array([3, 2], np.int8)

        site_array[number_of_spins - 1, :] = np.array([number_of_spins, 1], np.int64)
        name_array[number_of_spins - 1, :] = np.array([3, 2], np.int8)

    else:

        raise ValueError("Unknown Boundary Condition. Only obc (0) and pbc (1) supported.")

    return site_array, name_array


@nb.njit("Tuple((i8[:, :], i1[:, :]))(i8, i1, i8)", parallel=False, cache=True)
def input_h_xx_distance_1d(number_of_spins, boundary_condition, distance):
    if boundary_condition == 0:

        site_array = np.zeros((number_of_spins - distance, 2), np.int64)
        name_array = np.zeros((number_of_spins - distance, 2), np.int8)

        for i in range(number_of_spins - distance):
            site_array[i, :] = np.array([i + 1, i + 1 + distance], np.int64)
            name_array[i, :] = np.array([1, 1], np.int8)

    elif boundary_condition == 1:

        site_array = np.zeros((number_of_spins, 2), np.int64)
        name_array = np.zeros((number_of_spins, 2), np.int8)

        for i in range(number_of_spins):
            site_array[i, :] = np.array([i + 1, (i + distance) % number_of_spins + 1], np.int64)
            name_array[i, :] = np.array([1, 1], np.int8)

    else:

        raise ValueError("Unknown Boundary Condition. Only obc (0) and pbc (1) supported.")

    return site_array, name_array


@nb.njit("Tuple((i8[:, :], i1[:, :]))(i8, i1, i8)", parallel=False, cache=True)
def input_h_yy_distance_1d(number_of_spins, boundary_condition, distance):
    if boundary_condition == 0:

        site_array = np.zeros((number_of_spins - distance, 2), np.int64)
        name_array = np.zeros((number_of_spins - distance, 2), np.int8)

        for i in range(number_of_spins - distance):
            site_array[i, :] = np.array([i + 1, i + 1 + distance], np.int64)
            name_array[i, :] = np.array([2, 2], np.int8)

    elif boundary_condition == 1:

        site_array = np.zeros((number_of_spins, 2), np.int64)
        name_array = np.zeros((number_of_spins, 2), np.int8)

        for i in range(number_of_spins):
            site_array[i, :] = np.array([i + 1, (i + distance) % number_of_spins + 1], np.int64)
            name_array[i, :] = np.array([2, 2], np.int8)

    else:

        raise ValueError("Unknown Boundary Condition. Only obc (0) and pbc (1) supported.")

    return site_array, name_array


@nb.njit("Tuple((i8[:, :], i1[:, :]))(i8, i1, i8)", parallel=False, cache=True)
def input_h_zz_distance_1d(number_of_spins, boundary_condition, distance):
    if boundary_condition == 0:

        site_array = np.zeros((number_of_spins - distance, 2), np.int64)
        name_array = np.zeros((number_of_spins - distance, 2), np.int8)

        for i in range(number_of_spins - distance):
            site_array[i, :] = np.array([i + 1, i + 1 + distance], np.int64)
            name_array[i, :] = np.array([3, 3], np.int8)

    elif boundary_condition == 1:

        site_array = np.zeros((number_of_spins, 2), np.int64)
        name_array = np.zeros((number_of_spins, 2), np.int8)

        for i in range(number_of_spins):
            site_array[i, :] = np.array([i + 1, (i + distance) % number_of_spins + 1], np.int64)
            name_array[i, :] = np.array([3, 3], np.int8)

    else:

        raise ValueError("Unknown Boundary Condition. Only obc (0) and pbc (1) supported.")

    return site_array, name_array


# half-chain inputs
@nb.njit("Tuple((i8[:, :], i1[:, :]))(i8)", parallel=False, cache=True)
def input_h_z_half_1d(number_of_spins):

    if number_of_spins % 2 == 1:

        raise ValueError("Number of spins must be even")

    else:

        site_array = np.arange(number_of_spins // 2 + 1, number_of_spins + 1, 1, np.int64).reshape(number_of_spins // 2, 1)
        name_array = np.full(number_of_spins // 2, 3, np.int8).reshape(number_of_spins // 2, 1)

    return site_array, name_array


# half-chain inputs
@nb.njit("Tuple((i8[:, :], i1[:, :]))(i8)", parallel=False, cache=True)
def input_h_y_half_1d(number_of_spins):
    if number_of_spins % 2 == 1:

        raise ValueError("Number of spins must be even")

    else:

        site_array = np.arange(number_of_spins // 2 + 1, number_of_spins + 1, 1, np.int64).reshape(number_of_spins // 2, 1)
        name_array = np.full(number_of_spins // 2, 2, np.int8).reshape(number_of_spins // 2, 1)

    return site_array, name_array


# matrix-free matvec for quantum annealing with H(s) = s * H_Ising + (1 - s) * H_x
# with H_Ising diagonal in the computational basis and H_x = -\sum\limits_{i} \sigma_{i}^{x}
# H_x only creates bit flips so we do not need to compute the action every time but rather generate all flips
# H_Ising is also computed once and stored - if needed it can be computed matrix-free as well

# sequential version
@nb.njit(parallel=False, cache=True)
def matvec_qa(h_ising, vec, s, system_size):

    # new vector - use two vectors to prevent writing to the same vector by multiple threads
    new_vec = np.zeros_like(vec)

    # compute in parallel over rows
    for i in range(2 ** system_size):

        # diagonal
        new_vec[i] = s * h_ising[i] * vec[i]

        # off-diagonal
        for n in range(system_size):

            # find bit flipped index
            j = i ^ (2 ** n)

            # add matrix element
            new_vec[i] += (s - 1) * vec[j]

    return new_vec


# parallel version
# COMPILES CORRECTLY BUT SEEMS TO NOT BE IMPROVING PERFORMANCE
@nb.njit(parallel=True, cache=True)
def matvec_qa_par(h_ising, vec, s, system_size):

    # new vector - use two vectors to prevent writing to the same vector by multiple threads
    new_vec = np.zeros_like(vec)

    # compute in parallel over rows
    for i in nb.prange(2 ** system_size):

        # diagonal
        new_vec[i] = s * h_ising[i] * vec[i]

        # off-diagonal
        for n in range(system_size):

            # find bit flipped index
            j = i ^ (2 ** n)

            # add matrix element
            new_vec[i] += (s - 1) * vec[j]

    return new_vec


# # test
# N = 3
# bc = 0
# J = np.full(N, 1.0)
# s = 0.42
#
# positions_z, labels_z = input_h_z(N)
# positions_x, labels_x = input_h_x(N)
# positions_zz, labels_zz = input_h_zz_1d(N, bc)
#
# mat_z = operator_sum_real_diag(N, 1, J, positions_z, labels_z)
# mat_x = operator_sum_real(N, 1, J, positions_x, labels_x)
# mat_zz = operator_sum_real_diag(N, 2, J, positions_zz, labels_zz)
# ham_ising = mat_z.diagonal() + mat_zz.diagonal()
#
# # np.random.seed(1)
# test_vec = np.random.rand(2 ** N) + 1.j * np.random.rand(2 ** N)
# test_vec /= np.linalg.norm(test_vec)
#
# # true multiply
# mult_vec = s * mat_z @ test_vec + s * mat_zz @ test_vec - (1 - s) * mat_x @ test_vec
#
# # matrix-free multiply
# mult_mf_vec = matvec_qa(ham_ising, test_vec, s, N)
#
# print(mult_vec)
# print(mult_mf_vec)
# print(np.linalg.norm(mult_vec - mult_mf_vec))


# # test against qa functions
# N = 8
# bc = 1
# bc_string = ["obc", "pbc"]
# J = np.full(N, 1.0)
# j = 1.0
# j_spc = 0.45
# J_spc = np.full(N, 1.0)
# J_spc[N - 1] = j_spc
# J_spc_2 = np.full(N, 1.0)
# J_spc_2[N - 2] = j_spc
# # test
#
# spin_matrix = qa.spin_matrix(N)
# h_z = qa.hamiltonian_z_sparse(N, spin_matrix, j)
# h_x = qa.hamiltonian_x_sparse_from_diag(N, j)
# h_zz = qa.hamiltonian_j_sparse(N, spin_matrix, bc_string[bc])
# h_y = hamiltonian_y_sparse(N, j)
# h_xx = hamiltonian_xx_sparse(N, bc_string[bc], j)
# h_xz = sp.csr_matrix(hamiltonian_xz(N, bc_string[bc]))
# h_zx = sp.csr_matrix(hamiltonian_zx(N, bc_string[bc]))
# h_zz_2 = qa.hamiltonian_j_sparse_distance(2, N, spin_matrix, bc_string[bc], j)
# h_xx_spc = hamiltonian_xx_broken_pbc(N, j, j_spc)
# h_xx_2_spc = hamiltonian_xx_2_broken_pbc(N, j, j_spc)
#
# print("original created")
#
# positions_z, labels_z = input_h_z(N)
# positions_x, labels_x = input_h_x(N)
# positions_y, labels_y = input_h_y(N)
# positions_zz, labels_zz = input_h_zz_1d(N, bc)
# positions_xx, labels_xx = input_h_xx_1d(N, bc)
# positions_xz, labels_xz = input_h_xz_1d(N, bc)
# positions_zx, labels_zx = input_h_zx_1d(N, bc)
# positions_zz_2, labels_zz_2 = input_h_zz_distance_1d(N, bc, 2)
# positions_xx_2, labels_xx_2 = input_h_xx_distance_1d(N, bc, 2)
#
#
# print("inputs obtained")
#
# mat_z = 0.5 * operator_sum_real(N, 1, J, positions_z, labels_z)
# mat_x = 0.5 * operator_sum_real(N, 1, J, positions_x, labels_x)
# mat_y = 0.5 * operator_sum_complex(N, 1, J, positions_y, labels_y)
# mat_xx = 0.25 * operator_sum_real(N, 2, J, positions_xx, labels_xx)
# mat_zz = 0.25 * operator_sum_real(N, 2, J, positions_zz, labels_zz)
# mat_xz = 0.25 * operator_sum_real(N, 2, J, positions_xz, labels_xz)
# mat_zx = 0.25 * operator_sum_real(N, 2, J, positions_zx, labels_zx)
# mat_zz_2 = 0.25 * operator_sum_real(N, 2, J, positions_zz_2, labels_zz_2)
# mat_xx_spc = 0.25 * operator_sum_real(N, 2, J_spc, positions_xx, labels_xx)
# mat_xx_2_spc = 0.25 * operator_sum_real(N, 2, J_spc_2, positions_xx_2, labels_xx_2)
#
# print("operators created")
#
# print("h_z:", mat_z - h_z)
# print(spla.norm(mat_z))
# print(spla.norm(h_z))
# print("h_x:", mat_x - h_x)
# print(spla.norm(mat_x))
# print(spla.norm(h_x))
# print("h_y:", mat_y - h_y)
# print(spla.norm(mat_y))
# print(spla.norm(h_y))
#
#
# print("h_zz:", mat_zz - h_zz)
# print(spla.norm(mat_zz))
# print(spla.norm(h_zz))
# print("h_xx:", mat_xx - h_xx)
# print(spla.norm(mat_xx))
# print(spla.norm(h_xx))
# print("h_xz:", mat_xz - h_xz)
# print(spla.norm(mat_xz))
# print(spla.norm(h_xz))
# print("h_zx:", mat_zx - h_zx)
# print(spla.norm(mat_zx))
# print(spla.norm(h_zx))
# print("h_zzz_2:", mat_zz_2 - h_zz_2)
# print(spla.norm(mat_zz_2))
# print(spla.norm(h_zz_2))
#
#
# print("h_xx_bpbc:", mat_xx_spc - h_xx_spc)
# print(spla.norm(mat_xx_spc))
# print(spla.norm(h_xx_spc))
# print("h_zzz_2_bpbc:", mat_xx_2_spc - h_xx_2_spc)
# print(spla.norm(mat_xx_2_spc))
# print(spla.norm(h_xx_2_spc))

# need to add:
#
# mpre checks
# pauli plus/minus
# symmetries
# fermionic operator (anti-symmetric wedge)
