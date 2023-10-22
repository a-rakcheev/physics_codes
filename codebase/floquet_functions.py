# some convenience functions for Floquet Systems
# here the considered protocol (for now) is
# U(T) = exp(-i(H-H')T/2 * exp(-i(H + H')T/2
# H' is the drive and H the driven Hamiltonian and also the average hamiltonian

import numpy as np
from scipy.signal import argrelmin
import numba as nb
import matplotlib.pyplot as plt

# coefficients for Trotter formula
c_5 = 1. / (2. - 2. ** (1 / 3))
coeff = [[1., 1.], [0.5, 1., 0.5], [7 / 24, 2 / 3, 3 / 4, -2 / 3, -1 / 24, 1.],
         [0.5 * c_5, c_5, 0.5 * (1. - c_5), 1. - 2 * c_5, 0.5 * (1. - c_5), c_5, 0.5 * c_5]]


# computes the eigenangles for a sequence of drive periods and returns as array
# the propagators are created based on an exact diagonalization of the hamiltonians
# at half the periods, these should be given as dense matrices
# this function deletes the original hamiltonians during the process
# since only the combinations H + H, H - H' are needed
# the sequence of periods is determined by the start, end and steps in between
# however these should be given for the half-period tau = T/2
@nb.jit(forceobj=True)
def eigenangles_ed(average, drive, start, end, steps):

    # needed arrays
    taul = np.linspace(start, end, steps)
    angle_array = np.zeros((steps, len(average)), dtype=np.float64)

    # hamiltonians
    h_plus = average + drive
    h_minus = average - drive

    # diagonalize
    ev_plus, evec_plus = np.linalg.eigh(h_plus)
    evec_plus = evec_plus.astype(np.complex128)

    ev_minus, evec_minus = np.linalg.eigh(h_minus)
    evec_minus = evec_minus.astype(np.complex128)

    for i, tau in enumerate(taul):

        # propagators
        exp_plus = np.diag(np.exp(-1.j * ev_plus * tau))
        u_plus = exp_plus @ evec_plus.T.conj()
        u_plus = evec_plus @ u_plus
        exp_plus = None

        exp_minus = np.diag(np.exp(-1.j * ev_minus * tau))
        u_minus = exp_minus @ evec_minus.T.conj()
        u_minus = evec_minus @ u_minus
        exp_minus = None

        u = u_minus @ u_plus
        u_plus = None
        u_minus = None

        ev = np.linalg.eigvals(u)
        u = None
        angle_array[i, :] = -np.angle(ev)

    return angle_array


# computes the eigenangles and energies of the average hamiltonian for a sequence of drive periods and returns as array
# the propagators are created based on an exact diagonalization of the hamiltonians
# at half the periods, these should be given as dense matrices
# this function deletes the original hamiltonians during the process
# since only the combinations H + H, H - H' are needed
# the sequence of periods is determined by the start, end and steps in between
# however these should be given for the half-period tau = T/2
@nb.jit(forceobj=True)
def eigenangles_and_energies_ed(h_x, h_z, h_zz, start, end, steps, size):

    # needed arrays
    taul = np.linspace(start, end, steps)
    angle_array = np.zeros((steps, size), dtype=np.float64)
    energy_x = np.zeros((steps, size), dtype=np.float64)
    energy_z = np.zeros((steps, size), dtype=np.float64)
    energy_zz = np.zeros((steps, size), dtype=np.float64)

    # hamiltonians
    h_plus = h_x
    h_minus = h_z + h_zz

    # diagonalize
    ev_plus, evec_plus = np.linalg.eigh(h_plus)
    evec_plus = evec_plus.astype(np.complex128)

    ev_minus, evec_minus = np.linalg.eigh(h_minus)
    evec_minus = evec_minus.astype(np.complex128)

    for i, tau in enumerate(taul):

        # propagators
        exp_plus = np.diag(np.exp(-1.j * ev_plus * tau))
        u_plus = exp_plus @ evec_plus.T.conj()
        u_plus = evec_plus @ u_plus
        exp_plus = None

        exp_minus = np.diag(np.exp(-1.j * ev_minus * tau))
        u_minus = exp_minus @ evec_minus.T.conj()
        u_minus = evec_minus @ u_minus
        exp_minus = None

        u = u_minus @ u_plus
        u_plus = None
        u_minus = None

        ev, evec = np.linalg.eig(u)

        energy = np.diag(evec.T.conj() @ h_x @ evec)
        energy_x[i, :] = energy.real

        energy = np.diag(evec.T.conj() @ h_z @ evec)
        energy_z[i, :] = energy.real

        energy = np.diag(evec.T.conj() @ h_zz @ evec)

        u = None
        angle_array[i, :] = -np.angle(ev)

    return angle_array


# computes the eigenangles for a sequence of drive periods and returns as array
# based on the n-th order Trotter formula (this gives an error of order n+1)
# the inputs must be the matrices H_+, H_-, where exp(-i tau H_- * coeff) is starting the exponential product
# also one can indicate, whether H_- is diagonal, which reduced computational effort,
#  and in this case one needs to pass the diagonal

@nb.jit(forceobj=True)
def eigenangles_ed_trotter(h_plus, h_minus, start, end, steps, order, diagonal=False):

    # get coefficients
    c = coeff[order - 1]

    # needed arrays
    taul = np.linspace(start, end, steps)
    angle_array = np.zeros((steps, len(h_minus)), dtype=np.float64)

    # diagonal case
    if diagonal:

        # diagonalize
        ev_plus, evec_plus = np.linalg.eigh(h_plus)
        evec_plus = evec_plus.astype(np.complex128)

        for i, tau in enumerate(taul):
            u = np.identity(len(h_minus), dtype=np.complex128)

            for j in range(len(c)):

                # apply H_-
                if j % 2 == 0:

                    prop = np.diag(np.exp(-1.j * h_minus * tau * c[j]))

                # apply H_+
                else:

                    # propagators
                    exp_plus = np.diag(np.exp(-1.j * ev_plus * tau * c[j]))
                    prop = exp_plus @ evec_plus.T.conj()
                    prop = evec_plus @ prop
                    exp_plus = None

                u = u @ prop

            ev = np.linalg.eigvals(u)
            angle_array[i, :] = -np.angle(ev)

    else:

        # diagonalize
        ev_plus, evec_plus = np.linalg.eigh(h_plus)
        evec_plus = evec_plus.astype(np.complex128)

        ev_minus, evec_minus = np.linalg.eigh(h_minus)
        evec_minus = evec_minus.astype(np.complex128)

        for i, tau in enumerate(taul):
            u = np.identity(len(h_minus), dtype=np.complex128)

            for j in range(len(c)):

                # apply H_-
                if j % 2 == 0:

                    exp_minus = np.diag(np.exp(-1.j * ev_minus * tau * c[j]))
                    prop = exp_minus @ evec_minus.T.conj()
                    prop = evec_minus @ prop
                    exp_minus = None

                # apply H_+
                else:

                    # propagators
                    exp_plus = np.diag(np.exp(-1.j * ev_plus * tau * c[j]))
                    prop = exp_plus @ evec_plus.T.conj()
                    prop = evec_plus @ prop
                    exp_plus = None

            ev = np.linalg.eigvals(u)
            angle_array[i, :] = -np.angle(ev)

    return angle_array


# same but as a function of frequency instead of period
# go from large to small frequency
@nb.jit(forceobj=True)
def eigenangles_ed_freq(average, drive, start, end, steps):

    # needed arrays
    freql = np.linspace(end, start, steps)
    taul = np.pi / freql
    angle_array = np.zeros((steps, len(average)), dtype=np.float64)

    # hamiltonians
    h_plus = average + drive
    h_minus = average - drive
    average = None
    drive = None

    # diagonalize
    ev_plus, evec_plus = np.linalg.eigh(h_plus)
    h_plus = None
    ev_minus, evec_minus = np.linalg.eigh(h_minus)
    h_minus = None

    for i, tau in enumerate(taul):

        print(i)
        # propagators
        exp_plus = np.diag(np.exp(-1.j * ev_plus * tau))
        u_plus = exp_plus @ evec_plus.T.conj()
        u_plus = evec_plus @ u_plus
        exp_plus = None

        exp_minus = np.diag(np.exp(-1.j * ev_minus * tau))
        u_minus = exp_minus @ evec_minus.T.conj()
        u_minus = evec_minus @ u_minus
        exp_minus = None

        u = u_minus @ u_plus
        u_plus = None
        u_minus = None

        ev = np.linalg.eigvals(u)
        u = None
        angle_array[i, :] = -np.angle(ev)

    return angle_array


# computes the eigenangles for a sequence of drive strengths and returns as array
# the propagators are created based on an exact diagonalization of the hamiltonians
# at half the periods, these should be given as dense matrices
# # this function deletes the original hamiltonians during the process
# since only the combinations H + h * H, H - h * H' are needed
# the sequence of strengths is determined by the start, end and steps in between
@nb.jit(forceobj=True)
def eigenangles_ed_vary_strength(average, drive, start, end, steps, freq):

    tau = np.pi / freq
    # needed arrays
    hl = np.linspace(start, end, steps)
    angle_array = np.zeros((steps, len(average)), dtype=np.float64)
    for i, h in enumerate(hl):

        # hamiltonians
        h_plus = average + h * drive
        h_minus = average - h * drive

        # diagonalize
        ev_plus, evec_plus = np.linalg.eigh(h_plus)
        h_plus = None
        ev_minus, evec_minus = np.linalg.eigh(h_minus)
        h_minus = None

        # propagators
        exp_plus = np.diag(np.exp(-1.j * ev_plus * tau))
        u_plus = exp_plus @ evec_plus.T.conj()
        u_plus = evec_plus @ u_plus
        exp_plus = None

        exp_minus = np.diag(np.exp(-1.j * ev_minus * tau))
        u_minus = exp_minus @ evec_minus.T.conj()
        u_minus = evec_minus @ u_minus
        exp_minus = None

        u = u_minus @ u_plus
        u_plus = None
        u_minus = None

        ev = np.linalg.eigvals(u)
        u = None
        angle_array[i, :] = -np.angle(ev)

    return angle_array


# evolve state based on ed and compute energy wrt the average hamiltonian
# the hamiltonians need to numpy arrays (not numpy.matrix)
# the state needs to be a numpy array of shape

@nb.jit(forceobj=True)
def energy_dynamics_ed(state, average, plus, minus, hperiod, steps):

    en = np.zeros(steps + 1)
    en[0] = (state.T.conj() @ average @ state).real

    # propagator
    ev, evec = np.linalg.eigh(plus)
    exp = np.diag(np.exp(-1.j * ev * hperiod))
    u_plus = exp @ evec.T.conj()
    u_plus = evec @ u_plus
    exp = None

    ev, evec = np.linalg.eigh(minus)
    exp = np.diag(np.exp(-1.j * ev * hperiod))
    u_minus = exp @ evec.T.conj()
    u_minus = evec @ u_minus
    exp = None

    u = u_minus @ u_plus
    u_plus = None
    u_minus = None
    
    for i in range(steps):

        print(i)
        state = u @ state
        en[i + 1] = (state.T.conj() @ average @ state).real

    return en


# plot angles with given color and marker
# this does not return anything, so it can be used in a script with multiple subplots
@nb.jit(forceobj=True)
def plot_angles(angle_array, start, end, steps, color, marker, markersize):

    taul = np.linspace(start, end, steps)
    for i, tau in enumerate(taul):

        plt.scatter(angle_array[i, :], np.full_like(angle_array[i, :], tau), s=markersize, color=color, marker=marker)

        plt.grid()
        plt.xlabel(r"$\theta$", fontsize=12)
        plt.ylabel(r"$\tau$", fontsize=12)
        plt.xticks(
            [-np.pi, -0.5 * np.pi, 0., 0.5 * np.pi, np.pi],
            [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
        # plt.xticks(
        #     [-np.pi, -0.75 * np.pi, -0.5 * np.pi, -0.25 * np.pi, 0., 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi],
        #     [r"$-\pi$", r"$-\frac{3\pi}{4}$", r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", r"$\frac{\pi}{4}$",
        #      r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])

        plt.ylim(start, end)
        plt.xlim(-np.pi, np.pi)


# same as above but with frequency on y axis
# plot angles with given color and marker
# this does not return anything, so it can be used in a script with multiple subplots
@nb.jit(forceobj=True)
def plot_angles_freq(angle_array, start, end, steps, color, marker, markersize):

    freql = np.linspace(end, start, steps)
    for i, w in enumerate(freql):

        plt.scatter(angle_array[i, :], np.full_like(angle_array[i, :], w), s=markersize, color=color, marker=marker)

        plt.grid()
        plt.xlabel(r"$\theta$", fontsize=12)
        plt.ylabel(r"$\omega$", fontsize=12)
        plt.xticks(
            [-np.pi, -0.75 * np.pi, -0.5 * np.pi, -0.25 * np.pi, 0., 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi],
            [r"$-\pi$", r"$-\frac{3\pi}{4}$", r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", r"$\frac{\pi}{4}$",
             r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])

        plt.ylim(end, start)
        plt.xlim(-np.pi, np.pi)


# plot angles with given color and marker
# this does not return anything, so it can be used in a script with multiple subplots
@nb.jit(forceobj=True)
def plot_angles_divided(angle_array, start, end, steps, color, marker, markersize):

    taul = np.linspace(start, end, steps)
    for i, tau in enumerate(taul):

        plt.scatter(angle_array[i, :] / (2. * tau), np.full_like(angle_array[i, :], tau), s=markersize, color=color, marker=marker)

        plt.grid()
        plt.xlabel(r"$\theta$", fontsize=12)
        plt.ylabel(r"$\tau$", fontsize=12)
        plt.xticks(
            [-np.pi, -0.75 * np.pi, -0.5 * np.pi, -0.25 * np.pi, 0., 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi],
            [r"$-\pi$", r"$-\frac{3\pi}{4}$", r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", r"$\frac{\pi}{4}$",
             r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])

        plt.ylim(start, end)
        plt.xlim(-np.pi, np.pi)


# plot angles with given color and marker
# this does not return anything, so it can be used in a script with multiple subplots
@nb.jit(forceobj=True)
def plot_angles_vary_strength(angle_array, start, end, steps, color, marker, markersize):

    hl = np.linspace(start, end, steps)
    for i, h in enumerate(hl):

        plt.scatter(angle_array[i, :], np.full_like(angle_array[i, :], h), s=markersize, color=color, marker=marker)

        plt.grid()
        plt.xlabel(r"$\theta$", fontsize=12)
        plt.ylabel(r"$h$", fontsize=12)
        plt.xticks(
            [-np.pi, -0.75 * np.pi, -0.5 * np.pi, -0.25 * np.pi, 0., 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi],
            [r"$-\pi$", r"$-\frac{3\pi}{4}$", r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", r"$\frac{\pi}{4}$",
             r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])

        plt.ylim(start, end)
        plt.xlim(-np.pi, np.pi)


# find gaps from angles
@nb.jit(forceobj=True)
def crossing_data(angle_array, times, threshold):

    # the array has the angles for fixed time as rows
    # these columns need to be sorted from smallest to larges angles

    sorted_array = np.sort(angle_array, axis=1)
    # we now compute the angle gaps at fixed time

    gap_array = sorted_array - np.roll(sorted_array, 1, axis=1)
    # the first entry can be outside the brillouin zone
    # correct by adding 2 pi to the first column

    gap_array[:, 0] += np.full_like(gap_array[:, 0], 2. * np.pi)

    # wrapping can lead to discontinuities, which we need to take care of
    # before looking at the minima

    difference_array = gap_array[1, :] - gap_array[0, :]
    for i in range(len(times) - 2):

        difference_array_next = gap_array[i + 2, :] - gap_array[i + 1, :]

        # check if the differences are discontinuous
        # need to set a threshold for this
        # if discontinuous wrap around until the difference satisfies criterium
        # start wrapping to the right

        if np.average(np.absolute(difference_array_next)) > threshold * np.average(np.absolute(difference_array))\
                and np.average(np.absolute(difference_array_next)) > 1.e-14:

            count = 1
            difference_shifted = np.roll(gap_array[i + 2, :], count) - gap_array[i + 1, :]

            # if still too large wrap again
            while np.average(np.absolute(difference_shifted)) > threshold * np.average(np.absolute(difference_array)):

                count += 1
                difference_shifted = np.roll(gap_array[i + 2, :], count) - gap_array[i + 1, :]

            # if correct update gap_array and difference
            gap_array[i + 2, :] = np.roll(gap_array[i + 2, :], count)
            difference_array = difference_shifted

        else:

            difference_array = difference_array_next

    # find local minima along each column
    # return times and widths (angles would be desirable but hard to recover due to shifts)

    idx_min = argrelmin(gap_array, axis=0)
    return gap_array, idx_min


# unwrap data depdning on floquet eigenstates, by following the wrap around of each state
# and returning the corresponding indices that should be used to sort arrays
# example after definition
@nb.jit(forceobj=True)
def wrapping_indices(angle_array, times, threshold):

    # the array has the angles for fixed time as rows
    # these columns need to be sorted from smallest to larges angles

    sorted_array = np.sort(angle_array, axis=1)
    # we now compute the angle gaps at fixed time

    gap_array = sorted_array - np.roll(sorted_array, 1, axis=1)
    # the first entry can be outside the brillouin zone
    # correct by adding 2 pi to the first column

    gap_array[:, 0] += np.full_like(gap_array[:, 0], 2. * np.pi)

    # wrapping can lead to discontinuities, which we need to take care of
    # before looking at the minima

    time_indices = [0]
    wrap_counts = [0]

    difference_array = gap_array[1, :] - gap_array[0, :]
    for i in range(len(times) - 2):

        difference_array_next = gap_array[i + 2, :] - gap_array[i + 1, :]

        # check if the differences are discontinuous
        # need to set a threshold for this
        # if discontinuous wrap around until the difference satisfies criterium
        # start wrapping to the right
        count = 0
        if np.average(np.absolute(difference_array_next)) > threshold * np.average(
                np.absolute(difference_array)) and np.average(np.absolute(difference_array_next)) > 1.e-14:

            count += 1
            difference_shifted = np.roll(gap_array[i + 2, :], count) - gap_array[i + 1, :]

            # if still too large wrap again
            while np.average(np.absolute(difference_shifted)) > threshold * np.average(
                    np.absolute(difference_array)):

                count += 1
                difference_shifted = np.roll(gap_array[i + 2, :], count) - gap_array[i + 1, :]

            # if correct update gap_array and difference
            gap_array[i + 2, :] = np.roll(gap_array[i + 2, :], count)
            difference_array = difference_shifted

        else:

            difference_array = difference_array_next

        if count != wrap_counts[-1]:
            time_indices.append(i + 2)
            wrap_counts.append(count)

    return np.array(time_indices), np.array(wrap_counts)

# example

# idx_tau, wraps = wrapping_indices(angle_array_symmetric, taul, 10)
# for j, idx in enumerate(idx_tau):
#
#     # wrap only if wrap count is not 0
#
#     if wraps[j] == 0:
#
#         continue
#
#     else:
#
#         # define tau interval for wrapping
#         # if it extends till the end set the last index
#
#         if j < len(idx_tau) - 1:
#
#             idx_next = idx_tau[j + 1]
#
#         else:
#
#             idx_next = tau_res - 1
#
#         # wrap all time slices in the interval according to the count
#         probabilities_avg[idx:idx_next, :, :] = np.roll(probabilities_avg[idx:idx_next, :, :], wraps[j], axis=2)
#
#         matrix_elements_symmetric[idx:idx_next, :, :] = np.roll(matrix_elements_symmetric[idx:idx_next, :, :],
#                                                                 [wraps[j], wraps[j]], axis=(2, 1))


# the above setup is further combined into wrap functions for probabilities and matrix elements
@nb.jit(forceobj=True)
def wrap_probabilities(probabilities, angle_array, times, threshold):

    tau_res = len(times)
    idx_tau, wraps = wrapping_indices(angle_array, times, threshold)
    for j, idx in enumerate(idx_tau):

        # wrap only if wrap count is not 0
        if wraps[j] == 0:

            continue

        else:

            # define tau interval for wrapping
            # if it extends till the end set the last index

            if j < len(idx_tau) - 1:

                idx_next = idx_tau[j + 1]

            else:

                idx_next = tau_res - 1

            # wrap all time slices in the interval according to the count
            probabilities[idx:idx_next, :, :] = np.roll(probabilities[idx:idx_next, :, :], wraps[j], axis=2)

    return probabilities


@nb.jit(forceobj=True)
def wrap_matrix_elements(matrix_elements, angle_array, times, threshold):

    tau_res = len(times)
    idx_tau, wraps = wrapping_indices(angle_array, times, threshold)
    for j, idx in enumerate(idx_tau):

        # wrap only if wrap count is not 0
        if wraps[j] == 0:

            continue

        else:

            # define tau interval for wrapping
            # if it extends till the end set the last index

            if j < len(idx_tau) - 1:

                idx_next = idx_tau[j + 1]

            else:

                idx_next = tau_res - 1

            # wrap all time slices in the interval according to the count
            matrix_elements[idx:idx_next, :, :] = np.roll(matrix_elements[idx:idx_next, :, :],
                                                          [wraps[j], wraps[j]], axis=(2, 1))

    return matrix_elements


# plot angles with given color and marker
# this does not return anything, so it can be used in a script with multiple subplots
@nb.jit(forceobj=True)
def print_angles_divided(angle_array, start, end, steps):

    taul = np.linspace(start, end, steps)
    for i, tau in enumerate(taul):

        print(tau, np.around(angle_array[i, :] / tau, 1))
