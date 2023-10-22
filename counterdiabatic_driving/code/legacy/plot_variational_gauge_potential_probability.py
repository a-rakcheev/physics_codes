import numpy as np
import matplotlib.pyplot as plt


# parameters
N = 8               # number of spins
res = 100          # resolution of the interval (0, 1) in s
orders = [1, 2, 3, 4, 5, 6, 7]
sl = np.linspace(0., 1., res + 1)
plt.figure(1, figsize=(8, 5))

for order in orders:

    name = "vqa_tfi_gauge_only_N" + str(N) + "_res" + str(res) + "_order" + str(order) + ".npz"
    data = np.load(name)
    prob = data["prob"]

    plt.plot(sl, prob, marker="o", markersize=3, ls="-", label="Order " + str(order))

plt.grid()
plt.xlabel(r"$s$", fontsize=12)
plt.ylabel(r"$P_{g}$", fontsize=12)
plt.legend()
plt.title("Variational QA of the TFI with the Gauge Potential")
plt.ylim(0., 1.)

# plt.show()
plt.savefig("vqa_tfi_gauge_only_N" + str(N) + ".pdf", format="pdf", dpi=300)
