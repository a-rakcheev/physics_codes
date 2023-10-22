import numpy as np
from scipy.signal import argrelmin
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmasher as cmr

cmap_div = cmr.pride


def crossing_data(angle_array, times, treshold):

    # the array has the angles for fixed time as rows
    # these columns need to be sorted from smallest to larges angles

    sorted_array = np.sort(angle_array, axis=1)

    # we now compute the angle gaps at fixed time
    gap_array = sorted_array - np.roll(sorted_array, 1, axis=1)

    plt.subplot(1, 6, 2)
    plt.pcolormesh(np.arange(len(angle_array[0, :])), times, gap_array, cmap=cmap_div, vmax=2., vmin=-2.)
    plt.colorbar()
    plt.xlabel(r"$i$", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks([], [], fontsize=12)
    plt.title("gaps raw", fontsize=12)

    # the first entry can be outside the brillouin zone
    # correct by adding 2 pi to the first column

    gap_array[:, 0] += np.full_like(gap_array[:, 0], 2. * np.pi)

    plt.subplot(1, 6, 3)
    plt.pcolormesh(np.arange(len(angle_array[0, :])), times, gap_array, cmap=cmap_div, vmax=2., vmin=-2.)
    plt.colorbar()
    plt.xlabel(r"$i$", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks([], [], fontsize=12)
    plt.title("gaps modified", fontsize=12)

    # wrapping can lead to discontinuities, which we need to take care of
    # before looking at the minima

    difference_array = gap_array[1, :] - gap_array[0, :]
    for i in range(len(times) - 2):
#         print(i)
        difference_array_next = gap_array[i + 2, :] - gap_array[i + 1, :]

        # check if the differences are discontinuous
        # need to set a threshold for this
        # if discontinuous wrap around until the difference satisfies criterion
        # start wrapping to the right

        # print(i, np.average(np.absolute(difference_array)), np.average(np.absolute(difference_array_next)))
        if np.average(np.absolute(difference_array_next)) > treshold * np.average(np.absolute(difference_array)) and np.average(np.absolute(difference_array_next)) > 1.e-14:

            # print(i, "Wrap Around")
            count = 1
            difference_shifted = np.roll(gap_array[i + 2, :], count) - gap_array[i + 1, :]

            # if still too large wrap again
            while np.average(np.absolute(difference_shifted)) > treshold * np.average(np.absolute(difference_array)):

                # print("wrap again")
                count += 1
                difference_shifted = np.roll(gap_array[i + 2, :], count) - gap_array[i + 1, :]

            # if correct update gap_array and difference
            gap_array[i + 2, :] = np.roll(gap_array[i + 2, :], count)
            difference_array = difference_shifted

        else:

            difference_array = difference_array_next

    plt.subplot(1, 6, 4)
    plt.pcolormesh(np.arange(len(angle_array[0, :])), times, gap_array, cmap=cmap_div, vmax=2., vmin=-2.)
    plt.colorbar()
    plt.xlabel(r"$i$", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks([], [], fontsize=12)
    plt.title("gaps re-wrapped", fontsize=12)

    # find local minima along each column
    # return times and widths (angles would be desirable but hard to recover due to shifts)

    idx_min = argrelmin(gap_array, axis=0)
    return gap_array, idx_min


# parameters
N = 6
bc = "pbc"
m = 1.0
h = 1.0
j = 1.0

tau_start = 0.
tau_end = 1.5
tau_res = 100000
taul = np.linspace(tau_start, tau_end, tau_res)
name = "eigenangles_tfi_symmetric_" + bc + "_N" + str(N) + "_j" + str(j).replace(".", "-") + "_m" + str(m).replace(".", "-") \
       + "_h" + str(h).replace(".", "-") + "_tau_start" + str(tau_start).replace(".", "-") + "_tau_end" \
       + str(tau_end).replace(".", "-") + "_tau_res" + str(tau_res) + ".npz"

data = np.load(name)
angle_array = data["angles"]
size = len(angle_array[0, :])

plt.figure(1, figsize=(8, 2))
plot_step = 100                     # plot only every n-th angular slice in the angle plot

gaps, indices = crossing_data(angle_array, taul, 5)


# get times and widths of gaps
tau_min = tau_start + indices[0] * (tau_end - tau_start) / tau_res
widths = gaps[indices[0], indices[1]]

# eigenangles
plt.subplot(1, 6, 1)
for i, tau in enumerate(taul[::plot_step]):

    # print(i)
    plt.scatter(angle_array[i * plot_step, :], tau * np.ones_like(angle_array[i * plot_step, :]), marker="o", s=0.5,
                color="darkred")
plt.xlim(-np.pi, np.pi)
plt.ylim(tau_start, tau_end)
plt.title("eigenangles", fontsize=12)
plt.xlabel(r"$\theta$", fontsize=12)
plt.xticks([-np.pi, 0, np.pi], [r"$-\pi$", r"$0$", r"$\pi$"], fontsize=12)
plt.ylabel(r"$\tau$", fontsize=12)


# positions of minima (note that the first position corresponds to the wrapped around gap)
plt.subplot(1, 6, 5)
plt.scatter(indices[1], tau_min, marker="o", s=6)
plt.ylim(bottom=tau_start, top=tau_end)
plt.title("crossing levels", fontsize=12)
plt.xlabel(r"$i$", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks([], [], fontsize=12)

# gap widths
plt.subplot(1, 6, 6)
plt.scatter(widths, tau_min, marker="^", s=5, color="green")
plt.xlabel(r"$\Delta_{c}$", fontsize=12)
plt.xscale("log")
plt.xlim(left=1.e-5, right=10)
plt.xticks(fontsize=12)
plt.ylim(bottom=tau_start, top=tau_end)
plt.yticks([], [], fontsize=12)
plt.title("gap widths", fontsize=12)

# show averga spacing based on Hilbert space dimesion
plt.axvline(2. * np.pi / size, ls="--", color="black", lw=1)


plt.subplots_adjust(wspace=0.35, left=0.07, right=0.975, top=0.9, bottom=0.15)

# plt.savefig("Fig12.pdf", format="pdf")                # very large due to many plot elements - use png instead
plt.savefig("Fig12.png", format="png", dpi=1200)
# plt.show()
