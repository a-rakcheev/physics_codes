import sys
import pickle
import time
import numpy as np
import multiprocessing as mp
from commute_stringgroups_v2 import *
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


def optimize_agp(index_tuple):

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

    coeff = 0.5 * R / (2 ** L)

    # add up operators from gauge potential, since the analytical formula is defined for pure pauli strings

    A_x = variational_operators[0] * coeff[0]
    for i in range(var_order - 1):
        A_x += variational_operators[i + 1] * coeff[i + 1]

    agp_x[index_tuple[0] * res_y + index_tuple[1]] = A_x

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

    coeff = 0.5 * R / (2 ** L)

    # add up operators from gauge potential, since the analytical formula is defined for pure pauli strings

    A_y = variational_operators[0] * coeff[0]
    for i in range(var_order - 1):
        A_y += variational_operators[i + 1] * coeff[i + 1]

    agp_y[index_tuple[0] * res_y + index_tuple[1]] = A_y

if __name__ == '__main__':
    
    # parameters
    L = int(sys.argv[1])			                    # number of spins
    res_x = int(sys.argv[2])			                # number of grid points on x axis
    res_y = int(sys.argv[3])			                # number of grid points on y axis
    order = int(sys.argv[4]) 			                # order of commutator ansatz
    number_of_processes = int(sys.argv[5])     	        # number of parallel processes (should be equal to number of (logical) cores
    l = int(sys.argv[6]) 			                    # range cutoff for variational strings

    # L = 4                                             # number of spins
    # res_x = 20                                        # number of grid points on x axis
    # res_y = 20                                        # number of grid points on y axis
    # order = 10                                        # order of commutator ansatz
    # number_of_processes = 8                           # number of parallel processes (should be equal to number of (logical) cores
    # l = 4                                             # range cutoff for variational strings

    xl = np.linspace(1.e-6, 1.5, res_x)
    yl = np.linspace(1.e-6, 1.5, res_y)


    # hamiltonians
    h_x = equation()
    for i in range(L):
        op = ''.join(roll(list('x' + '1' * (L - 1)), i))
        h_x[op] = 0.5

    h_z = equation()
    for i in range(L):
        op = ''.join(roll(list('z' + '1' * (L - 1)), i))
        h_z[op] = 0.5

    h_zz = equation()
    for i in range(L):
        op = ''.join(roll(list('zz' + '1' * (L - 2)), i))
        h_zz += equation({op: 0.25})

    start = time.time()

    manager = mp.Manager()
    agp_x = manager.list(range(res_x * res_y))
    agp_y = manager.list(range(res_x * res_y))
    pool = mp.Pool(processes=number_of_processes)
    computation = pool.map_async(optimize_agp, [(i, j) for i in range(res_x) for j in range(res_y)])

    computation.wait()

    agp_x = list(agp_x) 
    agp_y = list(agp_y) 

    name = "optimize_agp_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order) + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".pkl"
    with open(name, "wb") as writefile:
        
        pickle.dump(agp_x, writefile)
        pickle.dump(agp_y, writefile)

    writefile.close()






