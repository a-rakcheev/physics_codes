# investigate low-energy spectrum
import numpy as np
import matplotlib
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import tqdm

# parameters
Nl = [15, 18, 20, 22]                                                          # system size
steps = 100                                                     # num of measurements
instance_list = [np.arange(1, 101, 1), np.arange(1, 101, 1), np.arange(1, 201, 1), np.arange(1, 301, 1)]                                # instance
marked_instances = [[], [], [8, 77, 85, 146, 156, 195], [73, 147, 154, 170, 172, 192, 228, 237, 262]]

prefix = "C:/Users/rakche_a-adm/Downloads/data_sk/data/"
fig = plt.figure(1, figsize=(7.5, 6))

for n, N in enumerate(Nl):

    instances = instance_list[n]
    hard_instances = marked_instances[n]
    mags_diff = np.zeros((len(instances), 2))

    for i, inst in enumerate(tqdm.tqdm(instances)):

        filename = prefix + "spectrum/N" + str(N) + "/sk_ising_bimodal_spectrum_analysis_N=" + str(N) + "_inst=" + str(inst) + ".npz"
        data = np.load(filename)

        spec = data["en"]
        
        # find degeneracy of first 6 excited states
        E = spec[0]
        energies = [E]
        boundary = []

        for k, en in enumerate(spec):

            if en > E:
                boundary.append(k)
                E = en
                energies.append(E)        

            if len(boundary) >= 7:
                break

        boundary = np.array(boundary)
        energies = np.array(energies[:-1])

        # total number of degenerate states (including the ground state)
        states = boundary[-1]

        # test
        # print(spec[0:states + 1])
        # print(boundary)
        # print(energies)

        # get magnetizations
        mags_gs = data["mags"][0, :]
        mags_gs_flip = data["mags"][1, :]

        # print(mags_gs)
        # print(mags_gs_flip)

        for j in range(2):
            mags_ex = data["mags"][boundary[j]:boundary[j + 1], :]
            # print(mags_ex)
            # find hamming distance to gs
            deg = boundary[j + 1] - boundary[j]
            # print(deg)
            diff = 0.

            for s in range(deg):
                diff += min(np.sum(np.absolute(mags_gs - mags_ex[s, :]) / 2), np.sum(np.absolute(mags_gs_flip - mags_ex[s, :])) / 2)
            # print(diff)
            mags_diff[i, j] = diff / deg


    # plot barplot
    plt.subplot(2, 2, n + 1)

    # first sector
    plt.bar(instances, mags_diff[:, 0], width=1.5, color="black", zorder=2)

    for inst in hard_instances:
        plt.bar(instances[hard_instances], mags_diff[hard_instances, 0], width=1.5, color="red", zorder=3)

    # # second sector
    # plt.bar(instances, mags_diff[:, 1], width=1.5, color="black", zorder=2)

    # for inst in hard_instances:
    #     plt.bar(instances[hard_instances], mags_diff[hard_instances, 1], width=1.5, color="red", zorder=3)

    plt.grid(zorder=1, color="grey", lw=1)
    plt.xlabel(r"$\mathrm{instance}$", fontsize=13)

    if n == 0 or n == 2:
        plt.ylabel(r"$\textrm{hamming distance}$", fontsize=13)
    plt.xlim(0, instances[-1])
    plt.ylim(0, N // 2)
    # plt.title(r"$n = " + str(n + 1) + r"$", fontsize=14)
    plt.title(r"$N = " + str(N) + r"$", fontsize=13)


# plt.suptitle("'hamming distance' of the n lowest excited sectors to the ground state")
plt.subplots_adjust(hspace=0.4, wspace=0.2, left=0.075, right=0.975, top=0.95, bottom=0.075)

plt.savefig("Fig2.pdf")
plt.show()