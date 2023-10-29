import numpy as np
import matplotlib
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt

Nl = [15, 18, 20, 22]                               # system size

# hard instances for each size - each actual instance number is reduced by 1 to act as index
marked_instances = [[], [], [8, 77, 85, 146, 156, 195], [73, 147, 154, 170, 172, 192, 228, 237, 262]]

prefix = "C:/Users/rakche_a-adm/Downloads/data_sk/data/"
fig = plt.figure(1, figsize=(7.5, 6))

for i, N in enumerate(Nl):
    
    # minimal gaps for each instance
    gaps = np.load(prefix + "minimal_gaps_N=" + str(N) + ".npy")
    instances = np.arange(1, len(gaps) + 1, 1)

    # plot gaps on logscale - plot marked instances on top
    plt.subplot(2, 2, i + 1)
    plt.scatter(instances, gaps, s=7, marker="o", color="black", zorder=2)
    plt.scatter(instances[marked_instances[i]], gaps[marked_instances[i]], s=7, marker="o", color="red", zorder=3)

    plt.grid(zorder=1)
    plt.yscale("log")
    plt.ylim(1.e-3, 1.0)
    plt.xlim(1, instances[-1])

    if i > 1:
        plt.xlabel(r"instance", fontsize=13)

    if i % 2 == 0:
        plt.ylabel(r"$\Delta E_{\mathrm{min}}$", fontsize=13)

    plt.title(r"$N=" + str(N) + r"$", fontsize=13)


plt.subplots_adjust(wspace=0.25, hspace=0.3, top=0.95, bottom=0.1, left=0.09, right=0.975)
plt.savefig("Fig1.pdf", format="pdf")
plt.show()