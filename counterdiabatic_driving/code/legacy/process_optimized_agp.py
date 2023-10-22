import pickle
import numpy as np
import sys
import multiprocessing as mp
from commute_stringgroups_v2 import *
m = maths()


def fill_metric_subspace_coherent(index_tuple):

    print("i, j:", index_tuple[0], index_tuple[1])

    x = xl[index_tuple[0]]
    y = yl[index_tuple[1]]

    A_x = agp_x[index_tuple[0] * res_y + index_tuple[1]]
    A_y = agp_y[index_tuple[0] * res_y + index_tuple[1]]

    # define hamiltonian
    ham = h_zz - y * h_x - x * h_z

    # create gauge potential as operator
    Ax_mat = A_x.make_operator().todense()
    Ay_mat = A_y.make_operator().todense()
    ham_mat = ham.make_operator().todense()

    Axx_mat = np.dot(Ax_mat, Ax_mat)
    Axy_mat = np.dot(Ax_mat, Ay_mat)
    Ayy_mat = np.dot(Ay_mat, Ay_mat)

    tr_x = 0.
    tr_y = 0.
    tr_xy = 0.
    tr_xx = 0.
    tr_yy = 0.

    ev, evec = np.linalg.eigh(ham_mat)

    # compute covariance covariance based on subspace

    for i in range(S):
        ground_state = evec[:, i]

        tr_x += np.dot(ground_state.T.conj(), np.dot(Ax_mat, ground_state))[0, 0].real
        tr_y += np.dot(ground_state.T.conj(), np.dot(Ay_mat, ground_state))[0, 0].real

        tr_xx += np.dot(ground_state.T.conj(), np.dot(Axx_mat, ground_state))[0, 0].real
        tr_xy += np.dot(ground_state.T.conj(), np.dot(Axy_mat, ground_state))[0, 0].real
        tr_yy += np.dot(ground_state.T.conj(), np.dot(Ayy_mat, ground_state))[0, 0].real

    tr_x /= S
    tr_y /= S
    tr_xy /= S
    tr_xx /= S
    tr_yy /= S

    metric = np.zeros((2, 2))
    metric[0, 0] = tr_xx - tr_x ** 2
    metric[0, 1] = tr_xy - tr_x * tr_y
    metric[1, 1] = tr_yy - tr_y ** 2

    # set metric
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4] = metric[0, 0]
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 1] = metric[0, 1]
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 2] = metric[0, 1]
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 3] = metric[1, 1]


def fill_metric_subspace_incoherent(index_tuple):

    print("i, j:", index_tuple[0], index_tuple[1])

    x = xl[index_tuple[0]]
    y = yl[index_tuple[1]]

    A_x = agp_x[index_tuple[0] * res_y + index_tuple[1]]
    A_y = agp_y[index_tuple[0] * res_y + index_tuple[1]]

    # define hamiltonian
    ham = h_zz - y * h_x - x * h_z

    # create gauge potential as operator
    Ax_mat = A_x.make_operator().todense()
    Ay_mat = A_y.make_operator().todense()
    ham_mat = ham.make_operator().todense()

    Axx_mat = np.dot(Ax_mat, Ax_mat)
    Axy_mat = np.dot(Ax_mat, Ay_mat)
    Ayy_mat = np.dot(Ay_mat, Ay_mat)

    metric = np.zeros((2, 2))

    ev, evec = np.linalg.eigh(ham_mat)

    # compute covariance covariance based on subspace

    for i in range(S):
        ground_state = evec[:, i]

        tr_x = np.dot(ground_state.T.conj(), np.dot(Ax_mat, ground_state))[0, 0].real
        tr_y = np.dot(ground_state.T.conj(), np.dot(Ay_mat, ground_state))[0, 0].real

        tr_xx = np.dot(ground_state.T.conj(), np.dot(Axx_mat, ground_state))[0, 0].real
        tr_xy = np.dot(ground_state.T.conj(), np.dot(Axy_mat, ground_state))[0, 0].real
        tr_yy = np.dot(ground_state.T.conj(), np.dot(Ayy_mat, ground_state))[0, 0].real

        metric[0, 0] += tr_xx - tr_x ** 2
        metric[0, 1] += tr_xy - tr_x * tr_y
        metric[1, 1] += tr_yy - tr_y ** 2

    # set metric
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4] = metric[0, 0] / S
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 1] = metric[0, 1] / S
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 2] = metric[0, 1] / S
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 3] = metric[1, 1] / S


def fill_metric_finite_temperature_coherent(index_tuple):

    print("i, j:", index_tuple[0], index_tuple[1])

    x = xl[index_tuple[0]]
    y = yl[index_tuple[1]]

    A_x = agp_x[index_tuple[0] * res_y + index_tuple[1]]
    A_y = agp_y[index_tuple[0] * res_y + index_tuple[1]]

    # define hamiltonian
    ham = h_zz - y * h_x - x * h_z

    # create gauge potential as operator
    Ax_mat = A_x.make_operator().todense()
    Ay_mat = A_y.make_operator().todense()
    ham_mat = ham.make_operator().todense()

    Axx_mat = np.dot(Ax_mat, Ax_mat)
    Axy_mat = np.dot(Ax_mat, Ay_mat)
    Ayy_mat = np.dot(Ay_mat, Ay_mat)

    tr_x = 0.
    tr_y = 0.
    tr_xy = 0.
    tr_xx = 0.
    tr_yy = 0.

    ev, evec = np.linalg.eigh(ham_mat)

    # covariance based on coherent finite-temperature
    partition_function = np.sum(np.exp(-beta * ev))
    for i in range(2 ** L):
        ground_state = evec[:, i]
    
        tr_x += np.exp(-beta * ev[i]) * np.dot(ground_state.T.conj(), np.dot(Ax_mat, ground_state))[0, 0].real
        tr_y += np.exp(-beta * ev[i]) * np.dot(ground_state.T.conj(), np.dot(Ay_mat, ground_state))[0, 0].real
    
        tr_xx += np.exp(-beta * ev[i]) * np.dot(ground_state.T.conj(), np.dot(Axx_mat, ground_state))[0, 0].real
        tr_xy += np.exp(-beta * ev[i]) * np.dot(ground_state.T.conj(), np.dot(Axy_mat, ground_state))[0, 0].real
        tr_yy += np.exp(-beta * ev[i]) * np.dot(ground_state.T.conj(), np.dot(Ayy_mat, ground_state))[0, 0].real
    
    tr_x /= partition_function
    tr_y /= partition_function
    tr_xy /= partition_function
    tr_xx /= partition_function
    tr_yy /= partition_function
    
    metric = np.zeros((2, 2))
    metric[0, 0] = tr_xx - tr_x ** 2
    metric[0, 1] = tr_xy - tr_x * tr_y
    metric[1, 1] = tr_yy - tr_y ** 2

    # set metric
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4] = metric[0, 0]
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 1] = metric[0, 1]
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 2] = metric[0, 1]
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 3] = metric[1, 1]


def fill_metric_infinite_temperature_coherent(index_tuple):

    print("i, j:", index_tuple[0], index_tuple[1])

    A_x = agp_x[index_tuple[0] * res_y + index_tuple[1]]
    A_y = agp_y[index_tuple[0] * res_y + index_tuple[1]]

    A_xx = A_x * A_x
    A_xy = A_x * A_y
    A_yy = A_y * A_y

    # covariance based on coherent infinite temperature
    partition_function = 2 ** L
    tr_x = A_x.trace().real
    tr_y = A_y.trace().real
    
    tr_xx = A_xx.trace().real
    tr_xy = A_xy.trace().real
    tr_yy = A_yy.trace().real
    
    tr_x /= partition_function
    tr_y /= partition_function
    tr_xy /= partition_function
    tr_xx /= partition_function
    tr_yy /= partition_function
    
    metric = np.zeros((2, 2))
    metric[0, 0] = tr_xx - tr_x ** 2
    metric[0, 1] = tr_xy - tr_x * tr_y
    metric[1, 1] = tr_yy - tr_y ** 2

    # set metric
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4] = metric[0, 0]
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 1] = metric[0, 1]
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 2] = metric[0, 1]
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 3] = metric[1, 1]


def fill_metric_finite_temperature_incoherent(index_tuple):

    print("i, j:", index_tuple[0], index_tuple[1])

    x = xl[index_tuple[0]]
    y = yl[index_tuple[1]]

    A_x = agp_x[index_tuple[0] * res_y + index_tuple[1]]
    A_y = agp_y[index_tuple[0] * res_y + index_tuple[1]]

    # define hamiltonian
    ham = h_zz - y * h_x - x * h_z

    # create gauge potential as operator
    Ax_mat = A_x.make_operator().todense()
    Ay_mat = A_y.make_operator().todense()
    ham_mat = ham.make_operator().todense()

    Axx_mat = np.dot(Ax_mat, Ax_mat)
    Axy_mat = np.dot(Ax_mat, Ay_mat)
    Ayy_mat = np.dot(Ay_mat, Ay_mat)

    metric = np.zeros((2, 2))

    ev, evec = np.linalg.eigh(ham_mat)

    # compute covariance covariance based on subspace
    partition_function = np.sum(np.exp(-beta * ev))
    for i in range(2 ** L):
        ground_state = evec[:, i]

        tr_x = np.exp(-beta * ev[i]) * np.dot(ground_state.T.conj(), np.dot(Ax_mat, ground_state))[0, 0].real
        tr_y = np.exp(-beta * ev[i]) * np.dot(ground_state.T.conj(), np.dot(Ay_mat, ground_state))[0, 0].real

        tr_xx = np.exp(-beta * ev[i]) * np.dot(ground_state.T.conj(), np.dot(Axx_mat, ground_state))[0, 0].real
        tr_xy = np.exp(-beta * ev[i]) * np.dot(ground_state.T.conj(), np.dot(Axy_mat, ground_state))[0, 0].real
        tr_yy = np.exp(-beta * ev[i]) * np.dot(ground_state.T.conj(), np.dot(Ayy_mat, ground_state))[0, 0].real

        metric[0, 0] += tr_xx - tr_x ** 2
        metric[0, 1] += tr_xy - tr_x * tr_y
        metric[1, 1] += tr_yy - tr_y ** 2

    # set metric
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4] = metric[0, 0] / partition_function
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 1] = metric[0, 1] / partition_function
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 2] = metric[0, 1] / partition_function
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 3] = metric[1, 1] / partition_function


def fill_metric_infinite_temperature_incoherent(index_tuple):
    print("i, j:", index_tuple[0], index_tuple[1])

    x = xl[index_tuple[0]]
    y = yl[index_tuple[1]]

    A_x = agp_x[index_tuple[0] * res_y + index_tuple[1]]
    A_y = agp_y[index_tuple[0] * res_y + index_tuple[1]]

    # define hamiltonian
    ham = h_zz - y * h_x - x * h_z

    # create gauge potential as operator
    Ax_mat = A_x.make_operator().todense()
    Ay_mat = A_y.make_operator().todense()
    ham_mat = ham.make_operator().todense()

    metric = np.zeros((2, 2))

    ev, evec = np.linalg.eigh(ham_mat)

    # compute covariance covariance based on subspace
    partition_function = 2 ** L

    A_xx = A_x * A_x
    A_xy = A_x * A_y
    A_yy = A_y * A_y

    tr_xx = A_xx.trace().real
    tr_xy = A_xy.trace().real
    tr_yy = A_yy.trace().real

    metric[0, 0] = tr_xx
    metric[0, 1] = tr_xy
    metric[1, 1] = tr_yy

    for i in range(2 ** L):

        ground_state = evec[:, i]

        tr_x = np.dot(ground_state.T.conj(), np.dot(Ax_mat, ground_state))[0, 0].real
        tr_y = np.dot(ground_state.T.conj(), np.dot(Ay_mat, ground_state))[0, 0].real

        metric[0, 0] -= tr_x ** 2
        metric[0, 1] -= tr_x * tr_y
        metric[1, 1] -= tr_y ** 2


    # set metric
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4] = metric[0, 0] / partition_function
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 1] = metric[0, 1] / partition_function
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 2] = metric[0, 1] / partition_function
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 3] = metric[1, 1] / partition_function


def fill_metric_error_metric(index_tuple):

    print("i, j:", index_tuple[0], index_tuple[1])

    x = xl[index_tuple[0]]
    y = yl[index_tuple[1]]

    A_x = agp_x[index_tuple[0] * res_y + index_tuple[1]]
    A_y = agp_y[index_tuple[0] * res_y + index_tuple[1]]

    # define hamiltonian
    ham = h_zz - y * h_x - x * h_z
    ham_deriv_x = (-1) * h_z
    ham_deriv_y = (-1) * h_x

    chi_x = ham_deriv_x + 1.j * m.c(A_x, ham)
    chi_y = ham_deriv_y + 1.j * m.c(A_y, ham)
    ham_trace = (ham * ham).trace().real

    comm_x = 1.j * m.c(ham, chi_x)
    comm_y = 1.j * m.c(ham, chi_y)

    metric = np.zeros((2, 2))
    metric[0, 0] = (comm_x * comm_y).trace().real / ham_trace
    metric[1, 1] = (comm_y * comm_y).trace().real / ham_trace
    metric[0, 1] = (comm_x * comm_y).trace().real / ham_trace

    # set metric
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4] = metric[0, 0]
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 1] = metric[0, 1]
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 2] = metric[0, 1]
    metric_grid[(index_tuple[0] * res_y + index_tuple[1]) * 4 + 3] = metric[1, 1]


if __name__ == '__main__':

    L = int(sys.argv[1])                                               # number of spins
    res_x = int(sys.argv[2])                                          # number of grid points on x axis
    res_y = int(sys.argv[3])                                          # number of grid points on y axis
    order = int(sys.argv[4])                                          # order of commutator ansatz
    number_of_processes = int(sys.argv[5])                             # number of parallel processes (should be equal to number of (logical) cores
    l = int(sys.argv[6])                                               # range cutoff for variational strings
    S = int(sys.argv[7])                                               # subspace size (for subspace computations)
    beta = float(sys.argv[8])                                          # inverse temperature (for finite temperature computations)
    computation = str(sys.argv[9])

    # L = 4                                               # number of spins
    # res_x = 50                                          # number of grid points on x axis
    # res_y = 50                                          # number of grid points on y axis
    # order = 15                                          # order of commutator ansatz
    # number_of_processes = 4                             # number of parallel processes (should be equal to number of (logical) cores
    # l = 3                                               # range cutoff for variational strings
    # S = 14                                               # subspace size (for subspace computations)
    # beta = 0.01                                          # inverse temperature (for finite temperature computations)

# select which computation should be performed (subspace, (in-)finite_temperature_(in-)coherent, chi_metric (see box between eq. 56 and 57)))
    # computations = ["subspace_coherent", "subspace_incoherent", "finite_temperature_coherent",
    #                 "infinite_temperature_coherent", "finite_temperature_incoherent",
    #                 "infinite_temperature_incoherent", "error_metric"]

    computations = ["error_metric"]

    xl = np.linspace(1.e-6, 1.5, res_x)
    yl = np.linspace(1.e-6, 1.5, res_y)

    metric_grid = mp.Array('d', res_x * res_y * 4)
    # prefix = "/home/artem/Dropbox/bu_research/data/"
    # prefix = "/media/artem/TOSHIBA EXT/Dropbox/data/agp/"
    prefix = ""

    name = prefix + "optimize_agp_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order) + "_res_x" + str(res_x) + "_res_y" + str(
    res_y) + ".pkl"

    with open(name, "rb") as readfile:
        agp_x = pickle.load(readfile)
        agp_y = pickle.load(readfile)

    readfile.close()


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

    for computation in computations[:]:

        print(computation)

        # compute metric based on gauge potential and commutator (chi)
        if computation == "subspace_coherent":

            pool = mp.Pool(processes=number_of_processes)
            comp = pool.map_async(fill_metric_subspace_coherent, [(i, j) for i in range(res_x) for j in range(res_y)])
            comp.wait()
            metric = np.array(metric_grid).reshape((res_x, res_y, 2, 2))
            name = "metrics_subspace_coherent_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order) \
                   + "_S" + str(S) + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
            np.savez_compressed(name, metric=metric)

        elif computation == "subspace_incoherent":

            pool = mp.Pool(processes=number_of_processes)
            comp = pool.map_async(fill_metric_subspace_incoherent, [(i, j) for i in range(res_x) for j in range(res_y)])
            comp.wait()
            metric = np.array(metric_grid).reshape((res_x, res_y, 2, 2))
            name = "metrics_subspace_incoherent_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order) \
                   + "_S" + str(S) + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
            np.savez_compressed(name, metric=metric)

        elif computation == "finite_temperature_coherent":

            pool = mp.Pool(processes=number_of_processes)
            comp = pool.map_async(fill_metric_finite_temperature_coherent, [(i, j) for i in range(res_x) for j in range(res_y)])
            comp.wait()

            metric = np.array(metric_grid).reshape((res_x, res_y, 2, 2))
            name = "metrics_coherent_finT_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order) \
                   + "_beta" + str(beta).replace(".", "-") + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
            np.savez_compressed(name, metric=metric)

        elif computation == "infinite_temperature_coherent":

            pool = mp.Pool(processes=number_of_processes)
            comp = pool.map_async(fill_metric_infinite_temperature_coherent, [(i, j) for i in range(res_x) for j in range(res_y)])
            comp.wait()

            metric = np.array(metric_grid).reshape((res_x, res_y, 2, 2))
            name = "metrics_coherent_infT_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order) \
                   + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
            np.savez_compressed(name, metric=metric)

        elif computation == "finite_temperature_incoherent":

            pool = mp.Pool(processes=number_of_processes)
            comp = pool.map_async(fill_metric_finite_temperature_incoherent, [(i, j) for i in range(res_x) for j in range(res_y)])
            comp.wait()

            metric = np.array(metric_grid).reshape((res_x, res_y, 2, 2))
            name = "metrics_incoherent_finT_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order) \
                   + "_beta" + str(beta).replace(".", "-") + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
            np.savez_compressed(name, metric=metric)

        elif computation == "infinite_temperature_incoherent":

            pool = mp.Pool(processes=number_of_processes)
            comp = pool.map_async(fill_metric_infinite_temperature_incoherent, [(i, j) for i in range(res_x) for j in range(res_y)])
            comp.wait()

            metric = np.array(metric_grid).reshape((res_x, res_y, 2, 2))
            name = "metrics_incoherent_infT_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order) \
                   + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
            np.savez_compressed(name, metric=metric)

        elif computation == "error_metric":

            pool = mp.Pool(processes=number_of_processes)
            comp = pool.map_async(fill_metric_error_metric, [(i, j) for i in range(res_x) for j in range(res_y)])
            comp.wait()

            metric = np.array(metric_grid).reshape((res_x, res_y, 2, 2))
            name = "metrics_error_metric_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order) \
                   + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
            np.savez_compressed(name, metric=metric)