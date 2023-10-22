# test first orders gauge potentials for tfi
import numpy as np
from commute_stringgroups_v2 import *
import hamiltonians_32 as ham32

# convert an equation to input for hamiltonian_32 program
# to create matrices


def parse_equation(eq):

    strings = eq.keys()
    vals = eq.values()

    # need to extract the pauli labels and sites of the xyz operators (identity is filled up automatically)
    number_of_spins = len(eq.keys()[0])

    sites = []
    labels = []

    for string in strings:

        site = []
        label = []
        for i, char in enumerate(list(string)):

            if char == "1":

                continue

            elif char == "x":

                site.append(i + 1)
                label.append(1)

            elif char == "y":

                site.append(i + 1)
                label.append(2)

            elif char == "z":

                site.append(i + 1)
                label.append(3)

            else:

                raise ValueError("Unknown Pauli Label - Use x, y, z")

        sites.append(site)
        labels.append(label)

    sites = np.array(sites, dtype=np.int64)
    labels = np.array(labels, dtype=np.int8)
    values = np.array(vals, dtype=np.float64)

    return sites, labels, values


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

    print("Operator Set:", operator_set)
    print("Commutator Set:", commutator_set)

    for i in np.arange(1, len(operator_set)):

        print(i)
        commutator = commutator_set[i]
        operator = operator_set[i]

        reduced_commutator = commutator
        for j in range(i):

            overlap = m.tracedot(commutator_basis[j], commutator)
            print("Overlap:", overlap)

            reduced_commutator = reduced_commutator - overlap * commutator_basis[j]
            print("Reduced Commutator:", reduced_commutator)

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


# parameters
N = 4               # number of spins
order = 10          # order of variational gauge potential
l_max = 2

# hamiltonians
m = maths()
np.set_printoptions(1)

h_zz = equation()
for i in range(N):
    op = ''.join(roll(list('zz' + '1' * (N - 2)), i))
    h_zz[op] = -1.


h_x = equation()
for i in range(N):
    op = ''.join(roll(list('x' + '1' * (N - 1)), i))
    h_x[op] = -1.


h_sum = h_zz + h_x
ham_deriv = h_x - h_zz

J = 1.0
h = 0.5
ham = J * h_zz + h * h_x
print("Hamiltonian:", h_sum)
print("Hamiltonian Derivative:", ham_deriv)

# we need a list of equations for the different orders
# and then another list of coefficients


# compute the gauge potential hamiltonians
# note the factor of 1.j in the 1st order
gauge_pot = []
for i in range(order):

    if i == 0:

        pot = 1.j * m.c(h_sum, ham_deriv)
        gauge_pot.append(pot)

    else:

        pot = gauge_pot[i - 1]
        pot = m.c(h_sum, pot)
        pot = m.c(h_sum, pot)
        gauge_pot.append(pot)


# optimize the coefficients based on variational operators

gauge_full = gauge_pot[0]
for i in range(len(gauge_pot) - 1):

    gauge_full += gauge_pot[i + 1]

# print("Gauge Full:", gauge_full)

# # split operators
variational_strings = split_equation_length_cutoff(gauge_full, l_max, "pbc")
print("Gauge Operators:", variational_strings)

variational_operators = unique_variational_basis(variational_strings, ham)
print("Variational Operators:", variational_operators)


# check orthogonality of variational operators
for i, op in enumerate(variational_operators):
    for j, op2 in enumerate(variational_operators[i:]):

        print(i, j + i, m.tracedot(op, op2), m.tracedot(m.c(ham, op), m.c(ham, op2)))

var_order = len(variational_operators)
C = []
P = np.zeros((var_order, var_order))
R = np.zeros(var_order)

# create commutators and fill R matrix
for i in range(var_order):

    comm = m.c(ham, variational_operators[i])
    C.append(comm)
    prod = ham_deriv * comm
    R[i] = (2.j * prod.trace()).real

# fill P matrix
for i in range(var_order):
    for j in np.arange(i, var_order, 1):

        prod = C[i] * C[j]
        tr = prod.trace().real
        P[i, j] = -tr
        P[j, i] = -tr

print("R:", R)
print("P:", P)
print("Eigenvalues:", np.linalg.eigvalsh(P))

# the optimal coefficient is given by 1/2 * (P)^(-1) R
# if (-P) is positive definite
coeff = 0.5 * np.dot(np.linalg.inv(P), R)


# add up operators from gauge potential, since the analytical formula is defined for pure pauli strings

gauge_operator = variational_operators[0] * coeff[0]
for i in range(var_order - 1):

    gauge_operator += variational_operators[i + 1] * coeff[i + 1]

print("Optimized Gauge Potential:", gauge_operator)


# commutator test
print("Commutator:", m.c(ham, 1.j * ham_deriv - m.c(gauge_operator, ham)))