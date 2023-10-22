import numpy as np
from commute_stringgroups_v2 import *
import matplotlib.pyplot as plt
import qa_functions as qa


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
N = 8                       # number of spins
order = 20                  # order of variational gauge potential
cutoff = 2                  # string length cutoff
res = 50                   # resolution of the interval (0, 1) in s
T = 100.0                     # protocol duration
s_initial = 0.3
s_final = 0.9

m = maths()
sl = np.linspace(s_initial, s_final, res + 1)
ds = (s_final - s_initial) / res

np.set_printoptions(3)


# hamiltonians
h_0 = equation()
for i in range(N):
    op = ''.join(roll(list('x' + '1' * (N - 1)), i))
    h_0[op] = -0.5


h_1 = equation()
for i in range(N):
    op = ''.join(roll(list('zz' + '1' * (N - 2)), i))
    h_1[op] = -0.25


h_sum = h_0 + h_1
h_deriv = h_1 - h_0

print("Hamiltonians Created")

# variational basis

# compute the gauge potential hamiltonians
# note the factor of 1.j in the 1st order
gauge_pot = []
for i in range(order):

    if i == 0:

        pot = 1.j * m.c(h_sum, h_deriv)
        gauge_pot.append(pot)

    else:

        pot = gauge_pot[i - 1]
        pot = m.c(h_sum, pot)
        pot = m.c(h_sum, pot)
        gauge_pot.append(pot)

gauge_full = gauge_pot[0]
for i in range(len(gauge_pot) - 1):

    gauge_full += gauge_pot[i + 1]

# split operators
variational_strings = split_equation_length_cutoff(gauge_full, cutoff, "pbc")
print("Variational Strings:", variational_strings)


# evolution with actual matrices and vectors
# start in ground state of h_0

ham_0 = h_0.make_operator().todense()
ham_1 = h_1.make_operator().todense()
ham_deriv = h_deriv.make_operator().todense()

ev_0, evec_0 = np.linalg.eigh((1 - s_initial) * ham_0 + s_initial * ham_1)
state = evec_0[:, 0].A

# # evolution with the gauge potential only
# # compute overlaps with the instantaneous ground state
#
# prob = [1.]
#
# for i in range(res):
#
#     # if (i % (res // 100)) == 0:
#     #
#     #     print(i // (res // 100), "%")
#
#     # get effective hamiltonian in first order ME
#     if i == 0:
#
#         h_inst = (1 - s_initial) * h_0 + s_initial * h_1
#
#         variational_operators = unique_variational_basis(variational_strings, h_inst)
#         var_order = len(variational_operators)
#         R = np.zeros(var_order)
#
#         # create commutators and fill R matrix
#         for k in range(var_order):
#             comm = m.c(h_inst, variational_operators[k])
#             prod = h_deriv * comm
#             R[k] = (2.j * prod.trace()).real
#
#         alphas = 0.5 * R / (2 ** N)
#
#         A_current = variational_operators[0] * alphas[0]
#         for j in range(var_order - 1):
#
#             A_current += variational_operators[j + 1] * alphas[j + 1]
#
#         h_next = (1 - s_initial - ds) * h_0 + (s_initial + ds) * h_1
#
#         variational_operators = unique_variational_basis(variational_strings, h_next)
#         var_order = len(variational_operators)
#         R = np.zeros(var_order)
#
#         # create commutators and fill R matrix
#         for k in range(var_order):
#             comm = m.c(h_next, variational_operators[k])
#             prod = h_deriv * comm
#             R[k] = (2.j * prod.trace()).real
#
#         alphas = 0.5 * R / (2 ** N)
#
#         A_next = variational_operators[0] * alphas[0]
#         for j in range(var_order - 1):
#             A_next += variational_operators[j + 1] * alphas[j + 1]
#
#         A_avg = 0.5 * (A_current.make_operator().todense() + A_next.make_operator().todense())
#
#     else:
#
#         A_current = A_next
#         h_next = (1 - s_initial - (i + 1) * ds) * h_0 + (s_initial + (i + 1) * ds) * h_1
#
#         variational_operators = unique_variational_basis(variational_strings, h_next)
#         var_order = len(variational_operators)
#         R = np.zeros(var_order)
#
#         # create commutators and fill R matrix
#         for k in range(var_order):
#             comm = m.c(h_next, variational_operators[k])
#             prod = h_deriv * comm
#             R[k] = (2.j * prod.trace()).real
#
#         alphas = 0.5 * R / (2 ** N)
#
#         A_next = variational_operators[0] * alphas[0]
#         for j in range(var_order - 1):
#             A_next += variational_operators[j + 1] * alphas[j + 1]
#
#         A_avg = 0.5 * (A_current.make_operator().todense() + A_next.make_operator().todense())
#
#     # exponentiate using full diagonalization
#     ev_eff, evec_eff = np.linalg.eigh(A_avg)
#
#     exp_diag = np.diag(np.exp(-1.j * ds * ev_eff))
#     M = np.dot(exp_diag, evec_eff.T.conj())
#     M = np.dot(evec_eff, M)
#     state = np.dot(M, state).A
#
#     # probability to be in the ground state
#     ham_inst = (1 - s_initial - (i + 1) * ds) * ham_0 + (s_initial + (i + 1) * ds) * ham_1
#     ev_inst, evec_inst = np.linalg.eigh(ham_inst)
#     ground_state = evec_inst[:, 0].A
#
#     overlap = np.vdot(ground_state, state)
#     prob.append(np.absolute(overlap) ** 2)
#
#     # print(ham_inst[0:4, 0:4])
#     # print(ev_inst[1] - ev_inst[0])
#     # print(state[0:8].T)
#     # print(ground_state[0:8].T)
#
# # np.savez_compressed("vqa_tfi_gauge_only_N" + str(N) + "_res" + str(res) + "_order" + str(order) + ".npz", prob=prob)

# evolution with the counter-diabatic Hamiltonian
# compute overlaps with the instantaneous ground state and the 2 spin rdm
# from the rdm take the product state probabilities

prob = [1.]
rdm_array = np.zeros((res + 1, 2 ** (N / 2), 2 ** (N / 2)), dtype=np.complex128)
rdm = qa.reduced_density_matrix(state, 2, N / 2)
rdm_array[0, :, :] = rdm

for i in range(res):

    if (i % (res // 100)) == 0:

        print(i // (res // 100), "%")

    # get effective hamiltonian in first order ME
    if i == 0:

        h_inst = (1 - s_initial) * h_0 + s_initial * h_1

        variational_operators = unique_variational_basis(variational_strings, h_inst)
        var_order = len(variational_operators)
        R = np.zeros(var_order)

        # create commutators and fill R matrix
        for k in range(var_order):
            comm = m.c(h_inst, variational_operators[k])
            prod = h_deriv * comm
            R[k] = (2.j * prod.trace()).real

        alphas = 0.5 * R / (2 ** N)

        A_current = variational_operators[0] * alphas[0]
        for j in range(var_order - 1):
            A_current += variational_operators[j + 1] * alphas[j + 1]

        h_next = (1 - s_initial - ds) * h_0 + (s_initial + ds) * h_1

        variational_operators = unique_variational_basis(variational_strings, h_next)
        var_order = len(variational_operators)
        R = np.zeros(var_order)

        # create commutators and fill R matrix
        for k in range(var_order):
            comm = m.c(h_next, variational_operators[k])
            prod = h_deriv * comm
            R[k] = (2.j * prod.trace()).real

        alphas = 0.5 * R / (2 ** N)

        A_next = variational_operators[0] * alphas[0]
        for j in range(var_order - 1):
            A_next += variational_operators[j + 1] * alphas[j + 1]

        A_avg = 0.5 * (A_current.make_operator().todense() + A_next.make_operator().todense())

    else:

        A_current = A_next
        h_next = (1 - s_initial - (i + 1) * ds) * h_0 + (s_initial + (i + 1) * ds) * h_1

        variational_operators = unique_variational_basis(variational_strings, h_next)
        var_order = len(variational_operators)
        R = np.zeros(var_order)

        # create commutators and fill R matrix
        for k in range(var_order):
            comm = m.c(h_next, variational_operators[k])
            prod = h_deriv * comm
            R[k] = (2.j * prod.trace()).real

        alphas = 0.5 * R / (2 ** N)

        A_next = variational_operators[0] * alphas[0]
        for j in range(var_order - 1):
            A_next += variational_operators[j + 1] * alphas[j + 1]

        A_avg = 0.5 * (A_current.make_operator().todense() + A_next.make_operator().todense())

    # exponentiate using full diagonalization
    h_eff = A_avg + T * ((1 - s_initial - i * ds) * ham_0 + (s_initial + i * ds) * ham_1 + 0.5 * ds * ham_deriv)
    ev_eff, evec_eff = np.linalg.eigh(h_eff)

    exp_diag = np.diag(np.exp(-1.j * ds * ev_eff))
    M = np.dot(exp_diag, evec_eff.T.conj())
    M = np.dot(evec_eff, M)
    state = np.dot(M, state).A

    # probability to be in the ground state
    ham_inst = (1 - s_initial - (i + 1) * ds) * ham_0 + (s_initial + (i + 1) * ds) * ham_1
    ev_inst, evec_inst = np.linalg.eigh(ham_inst)
    ground_state = evec_inst[:, 0].A

    overlap = np.vdot(ground_state, state)
    prob.append(np.absolute(overlap) ** 2)

    rdm = qa.reduced_density_matrix(state, 2, N / 2)
    rdm_array[i + 1, :, :] = rdm


name = "vqa_cd_tfi_L" + str(N) + "_l_max" + str(cutoff) + "_T" + str(T).replace(".", "-")\
       + "_s" + str(s_final).replace(".", "-") + "_order" + str(order) + ".npz"
#
# np.savez_compressed(name, prob=prob, rdm=rdm_array)

plt.plot(sl, prob, marker="^", ls="")

plt.grid()
plt.xlabel("s")
plt.ylabel("P")
plt.ylim(0., 1.0)
plt.show()
