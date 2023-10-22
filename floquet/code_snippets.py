# simple python code snippets to find gap widths, energy differences and compute the heating rate
# they only require numpy and argrelmin from scipy.signal
# however for maximal speed they might be compiled using something like numba, cython or pythran

import numpy as np
from scipy.signal import argrelmin


# find gap widths, times at which the crossings occur and which levels cross
# the indexing for the last one starts at 0, corresponding to the difference between smallest and largest angle and then corresponds to differences of consecutive levels

# the inputs are:
# grid of eigenangles of the Floquet propagator - (M x D) array with M the time discretization and D the Hilbert space dimension
# a threshold for comparing differences between consecutive eigenangles to identify wrappings (see paper) - we used a value of 10

# the output is:
# gap widths
# time and space indices of local minima
def crossing_data(angle_array, treshold):

    # the array has the angles for fixed time as rows
    # these columns need to be sorted from smallest to larges angles
    sorted_array = np.sort(angle_array, axis=1)

    # find number of time steps
    size_time = len(angle_array[:, 0])

    # we now compute the angle gaps at fixed time
    gap_array = sorted_array - np.roll(sorted_array, 1, axis=1)

    # the first entry is the difference between the first and last level and has therefore a redundant factor of -2 pi
    # correct by adding 2 pi to the first column
    gap_array[:, 0] += np.full_like(gap_array[:, 0], 2. * np.pi)

    # wrapping can lead to discontinuities, which we need to take care of before looking at the minima
    difference_array = gap_array[1, :] - gap_array[0, :]
    for i in range(size_time - 2):
        difference_array_next = gap_array[i + 2, :] - gap_array[i + 1, :]

        # check if the differences are discontinuous
        # need to set a threshold for this
        # if discontinuous wrap around until the difference satisfies criterium
        # start wrapping to the right (THIS WILL LOOP IF THE RESOLUTION OR THRESHOLD IS CHOSEN POORLY)

        if np.average(np.absolute(difference_array_next)) > treshold * np.average(np.absolute(difference_array)) and np.average(np.absolute(difference_array_next)) > 1.e-14:

            count = 1
            difference_shifted = np.roll(gap_array[i + 2, :], count) - gap_array[i + 1, :]

            # if still too large wrap again
            while np.average(np.absolute(difference_shifted)) > treshold * np.average(np.absolute(difference_array)):

                count += 1
                difference_shifted = np.roll(gap_array[i + 2, :], count) - gap_array[i + 1, :]

            # if correct update gap_array and difference
            gap_array[i + 2, :] = np.roll(gap_array[i + 2, :], count)
            difference_array = difference_shifted

        else:

            difference_array = difference_array_next


    # find local minima along each column
    # return times and widths
    idx_min = argrelmin(gap_array, axis=0)

    # idx_min[0] gives the time indices of local minima
    # idx_min[1] gives the space indices of local minima

    # obtain gap widths
    widths = gap_array[idx_min[0], idx_min[1]]

    return widths, idx_min


# compute energy differences of a crossing based on backtracking to the origin (average Hamiltonian at infinite freq. or special Hamiltonian at time crystal)

# inputs:
# spectrum of Hamiltonian at the origin sorted in ascending order - note that the transfer wrt any observable can be obtained by changing
# the spectrum with the expectation value of the desired observable wrt the corresponding eigenstate

# indices of avoided crossings as obtained from the crossing_data algorithm (they should be ascending from smallest half-period to largest)

# output:
# energy transfer at every avoided crossing

def energy_difference_backtracking(en_avg_sort_0, indices):
    # find the energy difference of avoided crossings starting from 0
    ev_avg_0 = []
    ev_avg_1 = []

    ev_idx = np.arange(0, size, 1)

    # indices of eigenstates are based on indices of crossings
    # crossing 0 is between first and last eigenstate
    # after each crossings one needs to switch the indices in ev_idx, since the eigenstates
    # are switched after the crossing

    for i, idx in enumerate(indices):
        if idx == 0:

            idx1 = 0
            idx2 = size - 1

        else:

            idx1 = idx - 1
            idx2 = idx

        # energies of crossing states
        ev_avg_0.append(en_avg_sort_0[ev_idx[idx1]])
        ev_avg_1.append(en_avg_sort_0[ev_idx[idx2]])

        # switch indices
        ev_idx[[idx1, idx2]] = ev_idx[[idx2, idx1]]

    ev_avg_0 = np.array(ev_avg_0)
    ev_avg_1 = np.array(ev_avg_1)

    return np.absolute(ev_avg_0 - ev_avg_1)


# compute heating rate (Gamma) based on the infinite temperature expansion with gap widths and energy transfers

# broaden the delta function as a Gaussian
def delta_gauss(x, dE):

    sigma = dE / 2.
    norm = 1. / (sigma * np.sqrt(2 * np.pi))
    func = np.exp(-0.5 * ((x / sigma) ** 2))
    return norm * func

# compute heating rate at specified frequencies
# inputs:
# beta of high temperature ansatz - for small enough beta (we use 0.001) the result is independent of beta
# list of frequencies wl for which the rate is computed
# spectrum at the origin ev_eff (average Hamiltonian or something else depending on the situation - see paper)
# gaussian width dE for broadening the delta function (we use dE around 0.3, but if an accurate estimate is needed one should be careful here)
# gap widths - as obtained earlier
# crossing times of avoided crossings - as obtained earlier
# energy differences at avoided crossings - as obtained earlier

def heating_rate(beta, wl, ev_eff, dE, widths, times, energy_differences):

    # hilbert space dimension
    size = len(ev_eff)

    # factors for high temperature ansatz
    boltzmann_factor = np.exp(-beta * ev_eff)
    Z = np.sum(boltzmann_factor)

    E_infty = np.sum(ev_eff) / size
    E_beta = np.sum(ev_eff * boltzmann_factor) / Z

    # compute energy absorption rate
    ear = np.zeros_like(wl)
    for i, gap in enumerate(widths):

        w_c = np.pi / times[i]                # frequency of crossing
        en_diff_c = energy_differences[i]     # energy transfer

        ear_gap = en_diff_c * (gap * w_c / (2. * np.pi)) ** 2
        P_ij = en_diff_c                        # occupation difference factor (a common factor of beta / Z is introduced in the end) in high temperature ansatz

        ear += (delta_gauss(wl - w_c, dE)) * ear_gap * P_ij

    # heating rate
    gamma = 0.5 * np.pi * (beta / Z) * ear_gauss / (E_infty - E_beta)

    return gamma