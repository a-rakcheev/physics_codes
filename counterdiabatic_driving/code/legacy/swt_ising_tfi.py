import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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


def fill_coefficients(k):

    g = gl[k]
    print("g:", g)

    ham = h_zz - g * h_x
    R = np.zeros(var_order)

    variational_operators = unique_variational_basis(variational_strings, ham)
    print("Basis Orthonormalized", len(variational_operators))

    # create commutators and fill R matrix
    for i in range(var_order):
        comm = m.c(ham, variational_operators[i])
        prod = ham_deriv_x * comm
        R[i] = (2.j * prod.trace()).real

    coeff = 0.5 * R / (2 ** N)
    coeff_grid[k * var_order: (k+1) * var_order] = coeff


if __name__ == '__main__':

    # parameters
    N = 4
    res = 50
    order = 10
    number_of_processes = 4
    l = 4
    gl = np.linspace(0.01, 1.5, res)
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

    # compute everything for every point (inefficient)
    start = time.time()

    ham = h_zz - 0.24 * h_x
    ham_deriv_x = (-1) * h_x

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
    variational_operators = unique_variational_basis(variational_strings, ham)

    print("Operators Split", len(variational_strings))
    var_order = len(variational_operators)

    coeff_grid = mp.Array('d', res * var_order)
    pool = mp.Pool(processes=number_of_processes)
    computation = pool.map(fill_coefficients, range(res))

    end = time.time()
    print("Time:", end - start)

    # # add up operators from gauge potential, since the analytical formula is defined for pure pauli strings
    # A_g = variational_operators[0] * coeff[0]
    # for i in range(var_order - 1):
    #     A_g += variational_operators[i + 1] * coeff[i + 1]
    #
    # print(A_g)
    #
    # # commutator test
    # comm = m.c(ham, 1.j * ham_deriv_x - m.c(A_g, ham))
    # print("Norm:", np.absolute(np.sqrt(m.tracedot(comm, comm))))

    coefficients = np.array(coeff_grid[:]).reshape((res, var_order))
    print(coefficients)
