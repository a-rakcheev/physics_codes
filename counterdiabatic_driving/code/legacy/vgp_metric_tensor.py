import numpy as np
from commute_stringgroups_v2 import *
import matplotlib.pyplot as plt


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
L = 4                          # number of spins
order = 2                       # order of variational gauge potential
n = 2                           # number of rings for resolution
dp = 0.1                         # resolution in parameter space
m = maths()

# initial position
J_0 = 1.
h_0 = 0.5

# parameter tuples
N = 1 + 3 * n * (n + 1)         # number of tuples
params = np.zeros((N, 2))

count = 0
for k in range(n + 1):
    N_ring = 6 * k

    if k == 0:

        params[count, 0] = J_0
        params[count, 1] = h_0
        count += 1

    else:

        for i in range(N_ring):

            if i == 0:

                params[count, 0] = params[0, 0] + k * dp
                params[count, 1] = params[0, 1]
                count += 1

            else:

                turning_angle = 2. * np. pi / 3. + np.floor_divide(i - 1, k) * np.pi / 3.
                vector = dp * np.array([np.cos(turning_angle), np.sin(turning_angle)])

                params[count, 0] = params[count - 1, 0] + vector[0]
                params[count, 1] = params[count - 1, 1] + vector[1]
                count += 1


# print(params)
# plt.scatter(params[:, 0], params[:, 1], marker="o", color="black")
# plt.grid()
# plt.xlabel("J")
# plt.ylabel("h")
# plt.show()


# hamiltonians
h_x = equation()
for i in range(L):
    op = ''.join(roll(list('x' + '1' * (L - 1)), i))
    h_x[op] = -1.


h_zz = equation()
for i in range(L):
    op = ''.join(roll(list('zz' + '1' * (L - 2)), i))
    h_zz[op] = -1.


h_deriv_J = h_zz
h_deriv_h = h_x
h_sum = h_zz + h_x
print("Hamiltonians Created")


# variational basis
# compute the gauge potential hamiltonians
gauge_pot_J = []
gauge_pot_h = []

for i in range(order):

    if i == 0:

        pot = 1.j * m.c(h_sum, h_deriv_J)
        gauge_pot_J.append(pot)

    else:

        pot = gauge_pot_J[i - 1]
        pot = m.c(h_sum, pot)
        pot = m.c(h_sum, pot)
        gauge_pot_J.append(pot)

print(gauge_pot_J)
gauge_full_J = gauge_pot_J[0]
for i in range(len(gauge_pot_J) - 1):

    gauge_full_J += gauge_pot_J[i + 1]


for i in range(order):

    if i == 0:

        pot = 1.j * m.c(h_sum, h_deriv_h)
        gauge_pot_h.append(pot)

    else:

        pot = gauge_pot_h[i - 1]
        pot = m.c(h_sum, pot)
        pot = m.c(h_sum, pot)
        gauge_pot_h.append(pot)

print(gauge_pot_h)
gauge_full_h = gauge_pot_h[0]
for i in range(len(gauge_pot_h) - 1):

    gauge_full_h += gauge_pot_h[i + 1]

# split operators
variational_strings_J = split_equation_length_cutoff(gauge_full_J, order, "pbc")
variational_strings_h = split_equation_length_cutoff(gauge_full_h, order, "pbc")

variational_operators_J = unique_variational_basis(variational_strings_J, J_0 * h_zz + h_0 * h_x)
variational_operators_h = unique_variational_basis(variational_strings_h, J_0 * h_zz + h_0 * h_x)

print("Variational Basis Computed")
print("VB J:", len(variational_operators_J), variational_operators_J)
print("VB_h:", len(variational_operators_h) ,variational_operators_h)

# optimize the coefficients_for each parameter choice
gauge_correlators = np.zeros((N, 6))


for k, tuples in enumerate(params):
    print(k)

    J = tuples[0]
    h = tuples[1]

    ham = J * h_zz + h * h_x

    # optimize A_J
    var_order = len(variational_operators_J)

    C = []
    R = np.zeros(var_order)
    P = np.zeros((var_order, var_order))

    # create commutators and fill R matrix
    for i in range(var_order):

        comm = m.c(ham, variational_operators_J[i])
        C.append(comm)

        prod = h_deriv_J * comm
        R[i] = (2.j * prod.trace()).real

    # fill P matrix
    for i in range(var_order):
        for j in np.arange(i, var_order, 1):

            prod = C[i] * C[j]
            tr = prod.trace().real
            P[i, j] = -tr
            P[j, i] = -tr

    coeff_J = 0.5 * np.dot(np.linalg.inv(P), R).real
    # gauge potential
    A_J = variational_operators_J[0] * coeff_J[0]
    for i in range(var_order - 1):
        A_J += variational_operators_J[i + 1] * coeff_J[i + 1]

    print("A_J computed")

    # optimize A_h
    var_order = len(variational_operators_h)

    C = []
    R = np.zeros(var_order)
    P = np.zeros((var_order, var_order))

    # create commutators and fill R matrix
    for i in range(var_order):

        comm = m.c(ham, variational_operators_h[i])
        C.append(comm)

        prod = h_deriv_h * comm
        R[i] = (2.j * prod.trace()).real

    # fill P matrix
    for i in range(var_order):
        for j in np.arange(i, var_order, 1):

            prod = C[i] * C[j]
            tr = prod.trace().real
            P[i, j] = -tr
            P[j, i] = -tr

    coeff_h = 0.5 * np.dot(np.linalg.inv(P), R).real

    # gauge potential
    A_h = variational_operators_h[0] * coeff_h[0]
    for i in range(var_order - 1):
        A_h += variational_operators_h[i + 1] * coeff_h[i + 1]
    print("A_h computed")

    # compute gauge correlators
    gauge_correlators[k, 0] = A_J.trace() / (2 ** L)
    gauge_correlators[k, 1] = A_h.trace() / (2 ** L)
    gauge_correlators[k, 2] = m.tracedot(A_J, A_J).real
    gauge_correlators[k, 3] = m.tracedot(A_J, A_h).real
    gauge_correlators[k, 4] = m.tracedot(A_h, A_J).real
    gauge_correlators[k, 5] = m.tracedot(A_h, A_h).real


# compute tensors

g_JJ = gauge_correlators[:, 2] - gauge_correlators[:, 0] ** 2
g_Jh = 0.5 * (gauge_correlators[:, 3] + gauge_correlators[:, 4]) - gauge_correlators[:, 0] * gauge_correlators[:, 1]
g_hh = gauge_correlators[:, 5] - gauge_correlators[:, 1] ** 2
F_Jh = gauge_correlators[:, 3] - gauge_correlators[:, 4]

print(g_JJ)
print(g_Jh)
print(g_hh)
print(F_Jh)


# plot 3d


