import numpy as np
from commute_stringgroups_v2 import *
import matplotlib.pyplot as plt

# parameters
N = 4               # number of spins
order = 3          # order of variational gauge potential
res = 100          # resolution of the interval (0, 1) in s
T = 1.0         # protocol duration

m = maths()
J = 1.0
K = 0.5
g = 1.0

sl = np.linspace(0., 0.99, res + 1)
ds = 0.99 / res

np.set_printoptions(1)

# hamiltonians
h_zz = equation()
for i in range(N):
    op = ''.join(roll(list('zz' + '1' * (N - 2)), i))
    h_zz[op] = 1.

h_zz2 = equation()
for i in range(N):
    op = ''.join(roll(list('z1z' + '1' * (N - 3)), i))
    h_zz2[op] = 1.

h_x = equation()
for i in range(N):
    op = ''.join(roll(list('x' + '1' * (N - 1)), i))
    h_x[op] = 1.

h_0 = -g * h_x
h_1 = -J * h_zz + J * K * h_zz2
h_deriv = h_1 - h_0
h_sum = h_0 + h_1

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
# variational_operators = split_equation_shifts_and_conjugates(gauge_full)

variational_operators = gauge_pot
print("Variational Basis Complete")
print(variational_operators)
var_order = len(variational_operators)
C_0 = []
C_1 = []

R_0 = np.zeros(var_order)
R_1 = np.zeros(var_order)

P_0 = np.zeros((var_order, var_order))
P_01 = np.zeros((var_order, var_order))
P_1 = np.zeros((var_order, var_order))

# create commutators and fill R matrix
for i in range(var_order):

    comm_0 = m.c(h_0, variational_operators[i])
    comm_1 = m.c(h_1, variational_operators[i])

    C_0.append(comm_0)
    C_1.append(comm_1)

    prod_0 = h_deriv * comm_0
    prod_1 = h_deriv * comm_1

    R_0[i] = (2.j * prod_0.trace()).real
    R_1[i] = (2.j * prod_1.trace()).real


# fill P matrix
for i in range(var_order):
    for j in np.arange(i, var_order, 1):

        prod_0 = C_0[i] * C_0[j]
        tr_0 = prod_0.trace().real
        P_0[i, j] = -tr_0
        P_0[j, i] = -tr_0

        prod_01 = C_0[i] * C_1[j] + C_1[i] * C_0[j]
        tr_01 = prod_01.trace().real
        P_01[i, j] = -tr_01
        P_01[j, i] = -tr_01

        prod_1 = C_1[i] * C_1[j]
        tr_1 = prod_1.trace().real
        P_1[i, j] = -tr_1
        P_1[j, i] = -tr_1

print("Variational Matrices Created")

# obtain coefficients for each s
alphas = np.zeros((res + 1, var_order))

for i, s in enumerate(sl):

    R = (1 - s) * R_0 + s * R_1
    P = (1 - s) * (1 - s) * P_0 + s * (1 - s) * P_01 + s * s * P_1

    alphas[i, :] = 0.5 * np.dot(np.linalg.inv(P), R).real

print("Optimal Coefficients Computed")

# evolution with actual matrices and vectors
# start in ground state of h_0

ham_0 = h_0.make_operator().todense()
ham_1 = h_1.make_operator().todense()
ham_deriv = h_deriv.make_operator().todense()

ev_0, evec_0 = np.linalg.eigh(ham_0)
state = evec_0[:, 0].A

# evolution with the gauge potential only
# compute overlaps with the instantaneous ground state
#
# prob = [1.]
#
# for i in range(res):
#
#     if (i % (res // 100)) == 0:
#
#         print(i // (res // 100), "%")
#
#     # get effective hamiltonian in first order ME
#     if i == 0:
#
#         A_current = variational_operators[0] * alphas[i, 0]
#         for j in range(var_order - 1):
#
#             A_current += variational_operators[j + 1] * alphas[i, j + 1]
#
#         A_next = variational_operators[0] * alphas[i + 1, 0]
#         for j in range(var_order - 1):
#             A_next += variational_operators[j + 1] * alphas[i + 1, j + 1]
#
#         A_avg = 0.5 * (A_current.make_operator().todense() + A_next.make_operator().todense())
#
#     else:
#
#         A_current = A_next
#         A_next = variational_operators[0] * alphas[i + 1, 0]
#         for j in range(var_order - 1):
#             A_next += variational_operators[j + 1] * alphas[i + 1, j + 1]
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
#     ham_inst = (1 - (i + 1) * ds) * ham_0 + (i + 1) * ds * ham_1
#     ev_inst, evec_inst = np.linalg.eigh(ham_inst)
#     ground_state = evec_inst[:, 0].A
#
#     overlap = np.vdot(ground_state, state)
#     prob.append(np.absolute(overlap) ** 2)

# np.savez_compressed("vqa_tfi_gauge_only_N" + str(N) + "_res" + str(res) + "_order" + str(order) + ".npz", prob=prob)

# evolution with the counterdiabatic Hamiltonian
# compute overlaps with the instantaneous ground state

prob = [1.]

for i in range(res):

    if (i % (res // 100)) == 0:

        print(i // (res // 100), "%")

    # get effective hamiltonian in first order ME
    if i == 0:

        A_current = variational_operators[0] * alphas[i, 0]
        for j in range(var_order - 1):

            A_current += variational_operators[j + 1] * alphas[i, j + 1]

        A_next = variational_operators[0] * alphas[i + 1, 0]
        for j in range(var_order - 1):
            A_next += variational_operators[j + 1] * alphas[i + 1, j + 1]

        A_avg = 0.5 * (A_current.make_operator().todense() + A_next.make_operator().todense())

    else:

        A_current = A_next
        A_next = variational_operators[0] * alphas[i + 1, 0]
        for j in range(var_order - 1):
            A_next += variational_operators[j + 1] * alphas[i + 1, j + 1]

        A_avg = 0.5 * (A_current.make_operator().todense() + A_next.make_operator().todense())

    # exponentiate using full diagonalization
    h_eff = A_avg + T * ((1 - i * ds) * ham_0 + i * ds * ham_1 + 0.5 * ds * ham_deriv)
    ev_eff, evec_eff = np.linalg.eigh(h_eff)

    exp_diag = np.diag(np.exp(-1.j * ds * ev_eff))
    M = np.dot(exp_diag, evec_eff.T.conj())
    M = np.dot(evec_eff, M)
    state = np.dot(M, state).A

    # probability to be in the ground state
    ham_inst = (1 - (i + 1) * ds) * ham_0 + (i + 1) * ds * ham_1
    ev_inst, evec_inst = np.linalg.eigh(ham_inst)
    ground_state = evec_inst[:, 0].A

    overlap = np.vdot(ground_state, state)
    prob.append(np.absolute(overlap) ** 2)


plt.plot(sl, prob, marker="o", markersize=3, ls="--")

plt.grid()
plt.xlabel(r"$s$", fontsize=12)
plt.ylabel(r"$P_{g}$", fontsize=12)
plt.ylim(0., 1.)

plt.show()