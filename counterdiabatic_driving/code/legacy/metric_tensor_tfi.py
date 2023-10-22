import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from commute_stringgroups_v2 import *
m = maths()


def streamplot_metric_tensor(xl, yl, gl, cmap):

    X, Y = np.meshgrid(xl, yl)

    major_u = np.zeros_like(X)
    major_v = np.zeros_like(X)
    minor_u = np.zeros_like(X)
    minor_v = np.zeros_like(X)

    norm = np.zeros_like(X)

    for i, x in enumerate(xl):
        for j, y in enumerate(yl):

            g = gl[i, j, :, :]

            ev, evec = np.linalg.eigh(g)
            idx_sort = np.argsort(np.absolute(ev))

            major_u[j, i] = evec[0, idx_sort[1]]
            major_v[j, i] = evec[1, idx_sort[1]]

            minor_u[j, i] = evec[0, idx_sort[0]]
            minor_v[j, i] = evec[1, idx_sort[0]]

            norm[j, i] =np.sqrt(np.absolute(ev[0] * ev[1]))

    # plt.streamplot(X, Y, major_u, major_v, color="black", arrowstyle="-")
    plt.streamplot(X, Y, minor_u, minor_v, color=norm / norm.max(), cmap=cmap,
                   norm=colors.Normalize(vmin=0., vmax=1.0), arrowstyle="-")
    plt.colorbar()


def quiver_metric_tensor(xl, yl, gl, cmap):

    X, Y = np.meshgrid(xl, yl)

    major_u = np.zeros_like(X)
    major_v = np.zeros_like(X)
    minor_u = np.zeros_like(X)
    minor_v = np.zeros_like(X)

    norm = np.zeros_like(X)

    for i, x in enumerate(xl):
        for j, y in enumerate(yl):

            g = gl[i, j, :, :]

            ev, evec = np.linalg.eigh(g)
            idx_sort = np.argsort(np.absolute(ev))

            major_u[j, i] = evec[0, idx_sort[1]]
            major_v[j, i] = evec[1, idx_sort[1]]

            minor_u[j, i] = evec[0, idx_sort[0]]
            minor_v[j, i] = evec[1, idx_sort[0]]

            norm[j, i] =np.sqrt(np.absolute(ev[0] * ev[1]))

    plt.grid()
    plt.quiver(X, Y, minor_u, minor_v, norm / norm.max(), cmap=cmap, pivot="mid")
    plt.colorbar()


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
N = 5
res_x = 20
res_y = 20
order = 6

xl = np.linspace(0.1, 1.0, res_x)
yl = np.linspace(0.1, 1.0, res_y)

metric_grid = np.zeros((res_x, res_y, 2, 2))


# hamiltonians
h_zz = equation()
for i in range(N):
    op = ''.join(roll(list('zz' + '1' * (N - 2)), i))
    h_zz[op] = 0.25

h_x = equation()
for i in range(N):
    op = ''.join(roll(list('x' + '1' * (N - 1)), i))
    h_x[op] = 0.5


# compute everything for every point (inefficient)
for l, g in enumerate(xl):
    for n, h in enumerate(yl):

        print("i, j:", l, n)
        ham = -g * h_x - h * h_zz
        ham_deriv_g = (-1) * h_x
        ham_deriv_h = (-1) * h_zz

        # compute A_g
        # compute the gauge potential hamiltonians
        # note the factor of 1.j in the 1st order
        gauge_pot = []
        for i in range(order):

            if i == 0:

                pot = 1.j * m.c(ham, ham_deriv_g)
                gauge_pot.append(pot)

            else:

                pot = gauge_pot[i - 1]
                pot = m.c(ham, pot)
                pot = m.c(ham, pot)
                gauge_pot.append(pot)

        print("Commutators Computed")

        # optimize the coefficients based on variational operators
        variational_operators = unique_variational_basis(gauge_pot, ham)
        print("Operators Orthogonalized")

        var_order = len(variational_operators)
        C = []
        P = np.zeros((var_order, var_order))
        R = np.zeros(var_order)

        # create commutators and fill R matrix
        for i in range(var_order):
            comm = m.c(ham, variational_operators[i])
            C.append(comm)
            prod = ham_deriv_g * comm
            R[i] = (2.j * prod.trace()).real

        coeff = 0.5 * R / (2 ** N)

        # add up operators from gauge potential, since the analytical formula is defined for pure pauli strings

        A_g = variational_operators[0] * coeff[0]
        for i in range(var_order - 1):
            A_g += variational_operators[i + 1] * coeff[i + 1]

        # commutator test
        comm = m.c(ham, 1.j * ham_deriv_g - m.c(A_g, ham))
        print("Norm g:", np.absolute(np.sqrt(m.tracedot(comm, comm))))
        # print("A_g:", A_g)

        # compute A_h
        # compute the gauge potential hamiltonians
        # note the factor of 1.j in the 1st order
        gauge_pot = []
        for i in range(order):

            if i == 0:

                pot = 1.j * m.c(ham, ham_deriv_h)
                gauge_pot.append(pot)

            else:

                pot = gauge_pot[i - 1]
                pot = m.c(ham, pot)
                pot = m.c(ham, pot)
                gauge_pot.append(pot)

        print("Commutators Computed")
        # optimize the coefficients based on variational operators
        variational_operators = unique_variational_basis(gauge_pot, ham)
        print("Operators Orthogonalized")

        var_order = len(variational_operators)
        C = []
        P = np.zeros((var_order, var_order))
        R = np.zeros(var_order)

        # create commutators and fill R matrix
        for i in range(var_order):
            comm = m.c(ham, variational_operators[i])
            C.append(comm)
            prod = ham_deriv_h * comm
            R[i] = (2.j * prod.trace()).real

        coeff = 0.5 * R / (2 ** N)

        # add up operators from gauge potential, since the analytical formula is defined for pure pauli strings

        A_h = variational_operators[0] * coeff[0]
        for i in range(var_order - 1):
            A_h += variational_operators[i + 1] * coeff[i + 1]

        # commutator test
        comm = m.c(ham, 1.j * ham_deriv_h - m.c(A_h, ham))
        print("Norm h:", np.absolute(np.sqrt(m.tracedot(comm, comm))))
        # print("A_h:", A_h)

        # coherent infinite temperature
        # tr_g = A_g.trace() / (2 ** N)
        # tr_h = A_h.trace() / (2 ** N)
        # tr_gg = m.tracedot(A_g, A_g)
        # tr_hh = m.tracedot(A_h, A_h)
        # tr_gh = m.tracedot(A_g, A_h)
        # tr_hg = m.tracedot(A_g, A_h)

        # ground state (find by using selective diagonalization of sparse matrix)
        Ag_mat = A_g.make_operator()
        Ah_mat = A_h.make_operator()
        ham_mat = ham.make_operator()

        ev, evec = spla.eigsh(ham_mat, k=1, which="SA")
        ground_state = evec[:, 0]

        tr_g = np.dot(ground_state.T.conj(), Ag_mat.dot(ground_state))
        tr_h = np.dot(ground_state.T.conj(), Ah_mat.dot(ground_state))

        Agg_mat = Ag_mat * Ag_mat
        Agh_mat = Ag_mat * Ah_mat
        Ahg_mat = Ah_mat * Ag_mat
        Ahh_mat = Ah_mat * Ah_mat

        tr_gg = np.dot(ground_state.T.conj(), Agg_mat.dot(ground_state))
        tr_gh = np.dot(ground_state.T.conj(), Agh_mat.dot(ground_state))
        tr_hg = np.dot(ground_state.T.conj(), Ahg_mat.dot(ground_state))
        tr_hh = np.dot(ground_state.T.conj(), Ahh_mat.dot(ground_state))

        metric = np.zeros((2, 2))
        metric[0, 0] = tr_gg - tr_g ** 2
        metric[0, 1] = 0.5 * (tr_gh + tr_hg) - tr_g * tr_h
        metric[1, 0] = 0.5 * (tr_gh + tr_hg) - tr_g * tr_h
        metric[1, 1] = tr_hh - tr_h ** 2

        metric_grid[l, n, :, :] = metric
        print("Metric Obtained")

plt.subplot(1, 2, 1)
streamplot_metric_tensor(xl, yl, metric_grid, "jet")
plt.xlabel(r"$g$", fontsize=12)
plt.ylabel(r"$h$", fontsize=12)

plt.subplot(1, 2, 2)
quiver_metric_tensor(xl, yl, metric_grid, "jet")
plt.xlabel(r"$g$", fontsize=12)
plt.ylabel(r"$h$", fontsize=12)

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.suptitle("Ground State Metric of the TFI", fontsize=12)
plt.show()