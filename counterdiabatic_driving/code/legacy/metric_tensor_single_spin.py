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

            norm[j, i] = np.sqrt(np.absolute(ev[0] * ev[1]))

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
N = 1
res_x = 5
res_y = 5
order = 6

xl = np.linspace(0.01, np.pi - 0.001, res_x)
yl = np.linspace(0.01, 2. * np.pi - 0.001, res_y)

metric_grid = np.zeros((res_x, res_y, 2, 2))
np.set_printoptions(2)

# hamiltonians
h_x = equation()
for i in range(N):
    op = ''.join(roll(list('x' + '1' * (N - 1)), i))
    h_x[op] = 1.

h_y = equation()
for i in range(N):
    op = ''.join(roll(list('y' + '1' * (N - 1)), i))
    h_y[op] = 1.

h_z = equation()
for i in range(N):
    op = ''.join(roll(list('z' + '1' * (N - 1)), i))
    h_z[op] = 1.

# compute everything for every point (inefficient)
for l, x in enumerate(xl):
    for n, y in enumerate(yl):

        print("theta, phi:", x, y)
        ham = -np.cos(x) * h_z - np.sin(x) * np.cos(y) * h_x - np.sin(x) * np.sin(y) * h_y
        ham_deriv_x = np.sin(x) * h_z - np.cos(x) * np.cos(y) * h_x - np.cos(x) * np.sin(y) * h_y
        ham_deriv_y = np.sin(x) * np.sin(y) * h_x - np.sin(x) * np.cos(y) * h_y

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

        # optimize the coefficients based on variational operators
        variational_operators = unique_variational_basis(gauge_pot, ham)
        # print("Operators Orthogonalized")

        var_order = len(variational_operators)
        C = []
        P = np.zeros((var_order, var_order))
        R = np.zeros(var_order)

        # create commutators and fill R matrix
        for i in range(var_order):
            comm = m.c(ham, variational_operators[i])
            C.append(comm)
            prod = ham_deriv_x * comm
            R[i] = (2.j * prod.trace()).real

        coeff = 0.5 * R / (2 ** N)

        # add up operators from gauge potential, since the analytical formula is defined for pure pauli strings

        A_x = variational_operators[0] * coeff[0]
        for i in range(var_order - 1):
            A_x += variational_operators[i + 1] * coeff[i + 1]

        # commutator test
        comm = m.c(ham, 1.j * ham_deriv_x - m.c(A_x, ham))
        print("Norm x:", np.absolute(np.sqrt(m.tracedot(comm, comm))))
        # print("A_theta:", A_x)

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
        # optimize the coefficients based on variational operators
        variational_operators = unique_variational_basis(gauge_pot, ham)
        # print("Operators Orthogonalized")

        var_order = len(variational_operators)
        C = []
        P = np.zeros((var_order, var_order))
        R = np.zeros(var_order)

        # create commutators and fill R matrix
        for i in range(var_order):
            comm = m.c(ham, variational_operators[i])
            C.append(comm)
            prod = ham_deriv_y * comm
            R[i] = (2.j * prod.trace()).real

        coeff = 0.5 * R / (2 ** N)

        # add up operators from gauge potential, since the analytical formula is defined for pure pauli strings

        A_y = variational_operators[0] * coeff[0]
        for i in range(var_order - 1):
            A_y += variational_operators[i + 1] * coeff[i + 1]

        # commutator test
        comm = m.c(ham, 1.j * ham_deriv_y - m.c(A_y, ham))
        print("Norm y:", np.absolute(np.sqrt(m.tracedot(comm, comm))))
        # print("A_phi:", A_y)

        # coherent infinite temperature
        # tr_g = A_g.trace() / (2 ** N)
        # tr_h = A_h.trace() / (2 ** N)
        # tr_gg = m.tracedot(A_g, A_g)
        # tr_hh = m.tracedot(A_h, A_h)
        # tr_gh = m.tracedot(A_g, A_h)
        # tr_hg = m.tracedot(A_g, A_h)

        # ground state (find by using selective diagonalization of sparse matrix)
        # Ax_mat = A_x.make_operator()
        # Ay_mat = A_y.make_operator()
        # ham_mat = ham.make_operator()

        Ax_mat = A_x.make_operator().todense()
        Ay_mat = A_y.make_operator().todense()
        ham_mat = ham.make_operator().todense()

        # print("A_theta:", Ax_mat)
        # print("A_phi:", Ay_mat)
        # print("ham:", ham_mat)

        # ev, evec = spla.eigsh(ham_mat)
        ev, evec = np.linalg.eigh(ham_mat)

        ground_state = evec[:, 0]

        # tr_x = np.dot(ground_state.T.conj(), Ax_mat.dot(ground_state))
        # tr_y = np.dot(ground_state.T.conj(), Ay_mat.dot(ground_state))

        tr_x = np.dot(ground_state.T.conj(), np.dot(Ax_mat, ground_state))[0, 0].real
        tr_y = np.dot(ground_state.T.conj(), np.dot(Ay_mat, ground_state))[0, 0].real

        # Axx_mat = Ax_mat * Ax_mat
        # Axy_mat = Ax_mat * Ay_mat
        # Ayx_mat = Ay_mat * Ax_mat
        # Ayy_mat = Ay_mat * Ay_mat

        Axx_mat = np.dot(Ax_mat, Ax_mat)
        Axy_mat = np.dot(Ax_mat, Ay_mat)
        Ayx_mat = np.dot(Ay_mat, Ax_mat)
        Ayy_mat = np.dot(Ay_mat, Ay_mat)

        # tr_xx = np.dot(ground_state.T.conj(), Axx_mat.dot(ground_state))
        # tr_xy = np.dot(ground_state.T.conj(), Axy_mat.dot(ground_state))
        # tr_yx = np.dot(ground_state.T.conj(), Ayx_mat.dot(ground_state))
        # tr_yy = np.dot(ground_state.T.conj(), Ayy_mat.dot(ground_state))

        tr_xx = np.dot(ground_state.T.conj(), np.dot(Axx_mat, ground_state))[0, 0].real
        tr_xy = np.dot(ground_state.T.conj(), np.dot(Axy_mat, ground_state))[0, 0].real
        tr_yx = np.dot(ground_state.T.conj(), np.dot(Ayx_mat, ground_state))[0, 0].real
        tr_yy = np.dot(ground_state.T.conj(), np.dot(Ayy_mat, ground_state))[0, 0].real

        # print(tr_x)
        # print(tr_y)
        # print(tr_xx)
        # print(tr_yy)
        # print(tr_xy)
        # print(tr_yx)

        metric = np.zeros((2, 2))
        metric[0, 0] = tr_xx - tr_x ** 2
        metric[0, 1] = 0.5 * (tr_xy + tr_yx) - tr_x * tr_y
        metric[1, 0] = metric[0, 1]
        metric[1, 1] = tr_yy - tr_y ** 2

        metric_ana = np.zeros((2, 2))
        metric_ana[0, 0] = 0.25
        metric_ana[1, 1] = 0.25 * np.sin(x) ** 2

        # print(np.linalg.norm(metric_ana - metric))

        metric_grid[l, n, :, :] = metric
        # print("Metric Obtained")


quiver_metric_tensor(xl, yl, metric_grid, "jet")
plt.xlabel(r"$\theta$", fontsize=12)
plt.ylabel(r"$\phi$", fontsize=12)
plt.title("Ground State Metric of a Spin in a Field", fontsize=12)
plt.show()