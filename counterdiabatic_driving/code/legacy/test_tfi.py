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


# parameters
N = 6               # number of spins
order = 10          # order of variational gauge potential
l_max = 6           # cutoff for pauli string range
np.set_printoptions(2)

# hamiltonians
m = maths()
J = 1.
h = 1.


h_xx = equation()
for i in range(N):
    op = ''.join(roll(list('xx' + '1' * (N - 2)), i))
    h_xx[op] = 1.


h_z = equation()
for i in range(N):
    op = ''.join(roll(list('z' + '1' * (N - 1)), i))
    h_z[op] = 1.

h_xy = equation()
for i in range(N):
    op = ''.join(roll(list('xy' + '1' * (N - 2)), i))
    h_xy[op] = 1.

# create variational gauge potential for the TFI (derivative wrt h)
ham = -h * h_z - J * h_xx
ham_deriv = (-1) * h_z

print("Hamiltonian:", ham)
print("Hamiltonian Derivative:", ham_deriv)

# we need a list of equations for the different orders
# and then another list of coefficients


# compute the gauge potential hamiltonians
# note the factor of 1.j in the 1st order
gauge_pot = []
for i in range(order):

    if i == 0:

        pot = 1.j * m.c(ham, ham_deriv)
        gauge_pot.append(pot)

    else:

        pot = gauge_pot[i - 1]
        pot = m.c(ham, pot)
        pot = m.c(ham, pot)
        gauge_pot.append(pot)


gauge_full = gauge_pot[0]
for i in range(len(gauge_pot) - 1):

    gauge_full += gauge_pot[i + 1]

print("Gauge Full:", gauge_full)

# split operators
variational_strings = split_equation_length_cutoff(gauge_full, l_max, "pbc")
print("Gauge Operators:", variational_strings)

variational_operators = unique_variational_basis(variational_strings, ham)
print("Variational Operators:", variational_operators)

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
# print("Coefficient Vector:", coeff)


# optimized gauge potential
gauge_operator = variational_operators[0] * coeff[0]
for i in range(var_order - 1):

    gauge_operator += variational_operators[i + 1] * coeff[i + 1]

print("Optimized Gauge Potential:")
print(gauge_operator)

# commutator test
comm = m.c(ham, 1.j * ham_deriv - m.c(gauge_operator, ham))
print("Commutator:", comm)
print("Norm:", np.absolute(np.sqrt(m.tracedot(comm, comm))))

# def exact_coefficients_full(number_of_spins, field):
#
#     alphas = np.zeros(number_of_spins - 1)
#     kl = (2 * np.arange(number_of_spins) - 1.) * np.pi / number_of_spins
#     # kl = np.arange(number_of_spins) * np.pi / number_of_spins
#     print(kl)
#     for l in range(number_of_spins - 1):
#
#         numerator = np.sin(kl) * np.sin((l + 1) * kl)
#         denominator = np.full_like(numerator, 1 + field ** 2) - 2. * field * np.cos(kl)
#         alphas[l] = -(0.25 / number_of_spins) * np.sum(numerator / denominator)
#
#     return alphas
#
#
# exact_coeff = exact_coefficients_full(N, h)
# print("Exact Coefficients:", exact_coeff)
# exact_gauge_pot = variational_operators[0] * exact_coeff[0]
# for i in range(len(exact_coeff) - 1):
#
#     gauge_operator += variational_operators[i + 1] * exact_coeff[i + 1]
# print("Commutator:", m.c(ham, 1.j * ham_deriv - m.c(exact_gauge_pot, ham)))


#
#
# print(m.c(ham, 1.j * ham_deriv))
# print(m.c(h_xy, h_z))
# print(m.c(h_xy, h_xx))
#
# print("[Sxx, Sz]", m.c(h_xx, h_z))
# print("[Sz, Syy-Sxx]", m.c(h_z, h_yy - h_xx))
# print("[Sxx, Syy-Sxx]", m.c(h_xx, h_yy - h_xx))


# test basis rotation for tfi

# ham_z = h_z.make_operator().todense()
# ham_xx = h_xx.make_operator().todense()
# ham_xy = h_xy.make_operator().todense()

# positions_z, labels_z, values_z = parse_equation(h_z)
# positions_xx, labels_xx, values_xx = parse_equation(h_xx)
# positions_xy, labels_xy, values_xy = parse_equation(h_xy)
#
# ham_z = ham32.operator_sum_complex(N, 1, values_z, positions_z, labels_z).todense()
# ham_xx = ham32.operator_sum_complex(N, 2, values_xx, positions_xx, labels_xx).todense()
# ham_xy = ham32.operator_sum_complex(N, 2, values_xy, positions_xy, labels_xy).todense()
#
# print(positions_z, labels_z, values_z)
# print(positions_xx, labels_xx, values_xx)
# print(positions_xy, labels_xy, values_xy)
#
#
# def phase_integral(field):
#
#     return 0.125 * (np.pi - 2. * np.arctan(2. * field))
#
#
# # diagonalize hamiltonian
# ham_tfi = -ham_xx - h * ham_z
# ev_tfi, evec_tfi = np.linalg.eigh(ham_tfi)
#
#
# # compute propagator
# ev_xy, evec_xy = np.linalg.eigh(ham_xy)
#
# phase = phase_integral(h)
# exp_diag = np.diag(np.exp(-1.j * phase * ev_xy))
# W = np.dot(exp_diag, evec_xy.T.conj())
# W = np.dot(evec_xy, W)
#
# # rotate initial eigenstates of h_z
# ev_z, evec_z = np.linalg.eigh(-h * ham_z)
# evec_rotated = np.dot(W, evec_z)
#
# np.set_printoptions(2)
# print(ham_z)
# print(ham32_z)
# print(ham_xx)
# print(ham32_xx)
# print(ham_xy)
# print(ham32_xy)

# print("Eigenstates TFI:")
# print(evec_tfi)
#
# print("Eigenstates Rotated:")
# print(evec_rotated)
#
# print("Probability Ground State:", np.absolute(np.dot(evec_rotated[:, 0].T.conj(), evec_tfi[:, 0])) ** 2)