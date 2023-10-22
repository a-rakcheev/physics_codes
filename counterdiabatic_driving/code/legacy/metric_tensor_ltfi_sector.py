import numpy as np
import scipy.sparse.linalg as spla
import sys
from commute_stringgroups_v2 import *
import time
import multiprocessing as mp
m = maths()


# create a variational basis from some set of hermitian operators, that allows for a unique solution to
# the quadratic optimization problem. This means that no linear combination of the
# resulting operators has a vanishing commutator with the Hamiltonian

def unique_variational_basis(operator_set, hamiltonian):

    # create set of commutators
    commutator_set = []
    for operator in operator_set:

        commutator_set.append(-1.j * m.c(hamiltonian, operator))

    # orthogonalize the commutators using gram-schmidt, applying the same linear combination to the operators

    operator_basis = []
    commutator_basis = []

    norm = np.sqrt(m.tracedot(commutator_set[0], commutator_set[0]))

    operator_basis.append((1. / norm) * operator_set[0])
    commutator_basis.append((1. / norm) * commutator_set[0])

    for i in np.arange(1, len(operator_set)):

        commutator = commutator_set[i]
        operator = operator_set[i]
        reduced_commutator = commutator

        for j in range(len(commutator_basis)):
            overlap = m.tracedot(commutator_basis[j], commutator)
            reduced_commutator = reduced_commutator - overlap * commutator_basis[j]
            operator = operator - overlap * operator_basis[j]

        norm = np.sqrt(m.tracedot(reduced_commutator, reduced_commutator))

        # check if the commutator is 0, if yes it will not be added to the basis
        if norm <= 1.e-10:
            continue

        else:

            # normalize
            reduced_commutator = (1. / norm) * reduced_commutator
            operator = (1. / norm) * operator

            commutator_basis.append(reduced_commutator)
            operator_basis.append(operator)

    return operator_basis


def fill_metric(index_tuple):

    x = xl[index_tuple[0]]
    y = yl[index_tuple[1]]

    # define hamiltonian

    print("i, j:", index_tuple[0], index_tuple[1])
    ham = h_zz - y * h_x - x * h_z
    ham_deriv_x = (-1) * h_z
    ham_deriv_y = (-1) * h_x

    ###################################################################################################################
    # create gauge potential

    # compute A_g
    # compute the gauge potential hamiltonians
    # note the factor of 1.j in the 1st order
    gauge_pot = []
    for i in range(order):

        if i == 0:

            pot = 1.j * m.c(ham, ham_deriv_x)
            gauge_pot.append(pot)

        else:

            pot = gauge_pot[i - 1]
            pot = m.c(ham, pot)
            pot = m.c(ham, pot)
            gauge_pot.append(pot)

    # print("Commutators Computed")

    gauge_full = gauge_pot[0]
    for i in range(len(gauge_pot) - 1):
        gauge_full += gauge_pot[i + 1]

    # optimize the coefficients based on variational operators
    variational_strings = split_equation_TP_length_cutoff(gauge_full, l, "pbc")
#    print("Variational Basis Computed:", len(variational_strings))
    variational_operators = unique_variational_basis(variational_strings, ham)
#    print("Operators Orthogonalized", len(variational_operators))

    var_order = len(variational_operators)
    R = np.zeros(var_order)

    # create commutators and fill R matrix
    for i in range(var_order):
        comm = m.c(ham, variational_operators[i])
        prod = ham_deriv_x * comm
        R[i] = (2.j * prod.trace()).real

    coeff = 0.5 * R / (2 ** N)

    # add up operators from gauge potential, since the analytical formula is defined for pure pauli strings

    A_x = variational_operators[0] * coeff[0]
    for i in range(var_order - 1):
        A_x += variational_operators[i + 1] * coeff[i + 1]

    print(A_x)
    # commutator test
    comm = m.c(ham, 1.j * ham_deriv_x - m.c(A_x, ham))
    norm_x[index_tuple[0] * res_y + index_tuple[1]] = np.absolute(np.sqrt(m.tracedot(comm, comm)))
    # print("Norm x:", np.absolute(np.sqrt(m.tracedot(comm, comm))))

    # compute A_h
    # compute the gauge potential hamiltonians
    # note the factor of 1.j in the 1st order
    gauge_pot = []
    for i in range(order):

        if i == 0:

            pot = 1.j * m.c(ham, ham_deriv_y)
            gauge_pot.append(pot)

        else:

            pot = gauge_pot[i - 1]
            pot = m.c(ham, pot)
            pot = m.c(ham, pot)
            gauge_pot.append(pot)

    # print("Commutators Computed")
    gauge_full = gauge_pot[0]
    for i in range(len(gauge_pot) - 1):
        gauge_full += gauge_pot[i + 1]

    # optimize the coefficients based on variational operators
    variational_strings = split_equation_TP_length_cutoff(gauge_full, l, "pbc")
#    print("Variational Basis Computed:", len(variational_strings))
    variational_operators = unique_variational_basis(variational_strings, ham)
#    print("Operators Orthogonalized", len(variational_operators))

    var_order = len(variational_operators)
    R = np.zeros(var_order)

    # create commutators and fill R matrix
    for i in range(var_order):
        comm = m.c(ham, variational_operators[i])
        prod = ham_deriv_y * comm
        R[i] = (2.j * prod.trace()).real

    coeff = 0.5 * R / (2 ** N)

    # add up operators from gauge potential, since the analytical formula is defined for pure pauli strings

    A_y = variational_operators[0] * coeff[0]
    for i in range(var_order - 1):
        A_y += variational_operators[i + 1] * coeff[i + 1]

    print(A_y)
    # commutator test
    comm = m.c(ham, 1.j * ham_deriv_y - m.c(A_y, ham))
    norm_y[index_tuple[0] * res_y + index_tuple[1]] = np.absolute(np.sqrt(m.tracedot(comm, comm)))
    # print("Norm y:", np.absolute(np.sqrt(m.tracedot(comm, comm))))
    ###################################################################################################################

#    # create gauge potential as operator
#    Ax_mat = A_x.make_operator().todense()
#    Ay_mat = A_y.make_operator().todense()
#    ham_mat = ham.make_operator().todense()

#    Axx_mat = np.dot(Ax_mat, Ax_mat)
#    Axy_mat = np.dot(Ax_mat, Ay_mat)
#    Ayx_mat = np.dot(Ay_mat, Ax_mat)
#    Ayy_mat = np.dot(Ay_mat, Ay_mat)

#    tr_x = 0.
#    tr_y = 0.
#    tr_xy = 0.
#    tr_xx = 0.
#    tr_yx = 0.
#    tr_yy = 0.

#    ev, evec = np.linalg.eigh(ham_mat)

#    ###################################################################################################################
#    # compute covariance covariance based on subspace

#    for i in range(S):
#        ground_state = evec[:, i]

#        tr_x += np.dot(ground_state.T.conj(), np.dot(Ax_mat, ground_state))[0, 0].real
#        tr_y += np.dot(ground_state.T.conj(), np.dot(Ay_mat, ground_state))[0, 0].real

#        tr_xx += np.dot(ground_state.T.conj(), np.dot(Axx_mat, ground_state))[0, 0].real
#        tr_xy += np.dot(ground_state.T.conj(), np.dot(Axy_mat, ground_state))[0, 0].real
#        tr_yx += np.dot(ground_state.T.conj(), np.dot(Ayx_mat, ground_state))[0, 0].real
#        tr_yy += np.dot(ground_state.T.conj(), np.dot(Ayy_mat, ground_state))[0, 0].real

#    tr_x /= S
#    tr_y /= S
#    tr_xy /= S
#    tr_xx /= S
#    tr_yx /= S
#    tr_yy /= S

#    metric = np.zeros((2, 2))
#    metric[0, 0] = tr_xx - tr_x ** 2
#    metric[0, 1] = 0.5 * (tr_xy + tr_yx) - tr_x * tr_y
#    metric[1, 0] = metric[0, 1]
#    metric[1, 1] = tr_yy - tr_y ** 2

    ###################################################################################################################
    # # covariance based on coherent finite-temperature
    #
    # partition_function = np.sum(np.exp(-beta * ev))
    # for i in range(2 ** N):
    #     ground_state = evec[:, i]
    #
    #     tr_x += np.exp(-beta * ev[i]) * np.dot(ground_state.T.conj(), np.dot(Ax_mat, ground_state))[0, 0].real
    #     tr_y += np.exp(-beta * ev[i]) * np.dot(ground_state.T.conj(), np.dot(Ay_mat, ground_state))[0, 0].real
    #
    #     tr_xx += np.exp(-beta * ev[i]) * np.dot(ground_state.T.conj(), np.dot(Axx_mat, ground_state))[0, 0].real
    #     tr_xy += np.exp(-beta * ev[i]) * np.dot(ground_state.T.conj(), np.dot(Axy_mat, ground_state))[0, 0].real
    #     tr_yx += np.exp(-beta * ev[i]) * np.dot(ground_state.T.conj(), np.dot(Ayx_mat, ground_state))[0, 0].real
    #     tr_yy += np.exp(-beta * ev[i]) * np.dot(ground_state.T.conj(), np.dot(Ayy_mat, ground_state))[0, 0].real
    #
    # tr_x /= partition_function
    # tr_y /= partition_function
    # tr_xy /= partition_function
    # tr_xx /= partition_function
    # tr_yx /= partition_function
    # tr_yy /= partition_function
    #
    # metric = np.zeros((2, 2))
    # metric[0, 0] = tr_xx - tr_x ** 2
    # metric[0, 1] = 0.5 * (tr_xy + tr_yx) - tr_x * tr_y
    # metric[1, 0] = metric[0, 1]
    # metric[1, 1] = tr_yy - tr_y ** 2

    ###################################################################################################################
    # # covariance based on coherent infinite temperature
    #
    # partition_function = 2 ** N
    # tr_x = np.trace(Ax_mat).real
    # tr_y = np.trace(Ay_mat).real
    #
    # tr_xx = np.trace(Axx_mat).real
    # tr_xy = np.trace(Axy_mat).real
    # tr_yx = tr_xy
    # tr_yy = np.trace(Ayy_mat).real
    #
    # tr_x /= partition_function
    # tr_y /= partition_function
    # tr_xy /= partition_function
    # tr_xx /= partition_function
    # tr_yx /= partition_function
    # tr_yy /= partition_function
    #
    # metric = np.zeros((2, 2))
    # metric[0, 0] = tr_xx - tr_x ** 2
    # metric[0, 1] = 0.5 * (tr_xy + tr_yx) - tr_x * tr_y
    # metric[1, 0] = metric[0, 1]
    # metric[1, 1] = tr_yy - tr_y ** 2

    ###################################################################################################################
    # # covariance based on incoherent finite temperature
    #
    # partition_function = np.sum(np.exp(-beta * ev))
    # for i in range(2 ** N):
    #     ground_state = evec[:, i]
    #
    #     tr_x = np.dot(ground_state.T.conj(), np.dot(Ax_mat, ground_state))[0, 0].real
    #     tr_y = np.dot(ground_state.T.conj(), np.dot(Ay_mat, ground_state))[0, 0].real
    #
    #     tr_xx += np.exp(-beta * ev[i]) * (np.dot(ground_state.T.conj(), np.dot(Axx_mat, ground_state))[0, 0].real - tr_x ** 2)
    #     tr_xy += np.exp(-beta * ev[i]) * (np.dot(ground_state.T.conj(), np.dot(Axy_mat, ground_state))[0, 0].real - tr_x * tr_y)
    #     tr_yx += np.exp(-beta * ev[i]) * (np.dot(ground_state.T.conj(), np.dot(Ayx_mat, ground_state))[0, 0].real - tr_x * tr_y)
    #     tr_yy += np.exp(-beta * ev[i]) * (np.dot(ground_state.T.conj(), np.dot(Ayy_mat, ground_state))[0, 0].real - tr_y ** 2)
    #
    # tr_xy /= partition_function
    # tr_xx /= partition_function
    # tr_yx /= partition_function
    # tr_yy /= partition_function
    #
    # metric = np.zeros((2, 2))
    # metric[0, 0] = tr_xx - tr_x ** 2
    # metric[0, 1] = 0.5 * (tr_xy + tr_yx)
    # metric[1, 0] = metric[0, 1]
    # metric[1, 1] = tr_yy - tr_y ** 2

    ###################################################################################################################
    # # covariance based on incoherent infinite temperature
    # partition_function = 2 ** N
    # for i in range(S):
    #     ground_state = evec[:, i]
    #
    #     tr_x = np.dot(ground_state.T.conj(), np.dot(Ax_mat, ground_state))[0, 0].real
    #     tr_y = np.dot(ground_state.T.conj(), np.dot(Ay_mat, ground_state))[0, 0].real
    #
    #     tr_xx += (np.dot(ground_state.T.conj(), np.dot(Axx_mat, ground_state))[0, 0].real - tr_x ** 2)
    #     tr_xy += (np.dot(ground_state.T.conj(), np.dot(Axy_mat, ground_state))[0, 0].real - tr_x * tr_y)
    #     tr_yx += (np.dot(ground_state.T.conj(), np.dot(Ayx_mat, ground_state))[0, 0].real - tr_x * tr_y)
    #     tr_yy += (np.dot(ground_state.T.conj(), np.dot(Ayy_mat, ground_state))[0, 0].real - tr_y ** 2)
    #
    # tr_xy /= partition_function
    # tr_xx /= partition_function
    # tr_yx /= partition_function
    # tr_yy /= partition_function
    #
    # metric = np.zeros((2, 2))
    # metric[0, 0] = tr_xx - tr_x ** 2
    # metric[0, 1] = 0.5 * (tr_xy + tr_yx)
    # metric[1, 0] = metric[0, 1]
    # metric[1, 1] = tr_yy - tr_y ** 2
    ###################################################################################################################
    # set metric

#    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4] = metric[0, 0]
#    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 1] = metric[0, 1]
#    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 2] = metric[1, 0]
#    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 3] = metric[1, 1]

    # print("Metric Obtained")


if __name__ == '__main__':

    # parameters
#    N = int(sys.argv[1])			# number of spins
#    res_x = int(sys.argv[2])			# number of grid points on x axis
#    res_y = int(sys.argv[3])			# number of grid points on y axis
#    order = int(sys.argv[4]) 			# order of commutator ansatz
#    number_of_processes = int(sys.argv[5])     	# number of parallel processes (should be equal to number of (logical) cores
#    l = int(sys.argv[6]) 			# range cutoff for variational strings
#    S = int(sys.argv[7])			# number of states in sector (if covariance based on sector expectation values)
#    beta = float(sys.argv[8])			# inverse temperature (if covariance based on thermal expectation values)


    N = 4                       # number of spins
    res_x = 2                  # number of grid points on x axis
    res_y = 2                  # number of grid points on y axis
    order = 10                   # order of commutator ansatz
    number_of_processes = 1     # number of parallel processes (should be equal to number of (logical) cores
    l = 4                       # range cutoff for variational strings
    S = 2                       # number of states in sector (if covariance based on sector expectation values)
    beta = 10.0                  # inverse temperature (if covariance based on thermal expectation values)

    xl = np.linspace(1.e-6, 1.5, res_x)
    yl = np.linspace(1.e-6, 1.5, res_y)

    metric_grid = mp.Array('d', res_x * res_y * 4)
    norm_x = mp.Array('d', res_x * res_y)
    norm_y = mp.Array('d', res_x * res_y)

    np.set_printoptions(2)

    # hamiltonians
    h_x = equation()
    for i in range(N):
        op = ''.join(roll(list('x' + '1' * (N - 1)), i))
        h_x[op] = 0.5

    h_z = equation()
    for i in range(N):
        op = ''.join(roll(list('z' + '1' * (N - 1)), i))
        h_z[op] = 0.5

    h_zz = equation()
    for i in range(N):
        op = ''.join(roll(list('zz' + '1' * (N - 2)), i))
        h_zz += equation({op: 0.25})

    # start = time.time()

    pool = mp.Pool(processes=number_of_processes)
    computation = pool.map(fill_metric, [(i, j) for i in range(res_x) for j in range(res_y)])

    # end = time.time()
    # print("Time:", end - start)

    metric = np.array(metric_grid).reshape((res_x, res_y, 2, 2))
    norm_x = np.array(norm_x).reshape((res_x, res_y))
    norm_y = np.array(norm_y).reshape((res_x, res_y))

    # check how good the ansatz is by looking at the norm of the commutator [H, iH' - [A, H]]
    # print(norm_x)
    # print(norm_y)
    # save metric to file for plotting

#    name = "metric_subspace_ltfi_N" + str(N) + "_l" + str(l) + "_order" + str(order) + "_S" + str(S) + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
    # name = "metric_coherent_finT_ltfi_N" + str(N) + "_l" + str(l) + "_order" + str(order) + "_beta" + str(beta).replace(".", "-") + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
    # name = "metric_incoherent_finT_ltfi_N" + str(N) + "_l" + str(l) + "_order" + str(order) + "_beta" + str(beta).replace(".", "-") + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
    # name = "metric_coherent_infT_ltfi_N" + str(N) + "_l" + str(l) + "_order" + str(order) + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
    # name = "metric_incoherent_infT_ltfi_N" + str(N) + "_l" + str(l) + "_order" + str(order) + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"

#    np.savez_compressed(name, metric=metric, norm_x=norm_x, norm_y=norm_y)

