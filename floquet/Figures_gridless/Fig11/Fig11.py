import numpy as np
import scipy.sparse as sp
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# parameters
L = 8
bc = "obc"
# tau = 0.5249327354        # (a)
# tau = 0.5249327359        # (b)
tau = 0.5249327364          # (c)

probability = np.load("data_tau" + str(tau).replace(".", "-") + ".npy")

plt.figure(1, figsize=(3.3, 2.5), constrained_layout=True)

plt.plot(probability[:, 0], color="navy", ls="-", lw=1, label=r"First State")
plt.plot(probability[:, -1], color="red", ls="--", lw=1, label=r"Second State")
plt.plot(np.full_like(probability[:, 0], 1.) - probability[:, 0] - probability[:, -1], color="palegreen", ls="-.",
         lw=1, label=r"Rest")

plt.legend(loc="center", framealpha=1, fontsize=8.5, handlelength=1.0, ncol=3)
# plt.grid()

plt.xlabel(r"$10^{7}$ cycles", fontsize=12)
plt.ylabel(r"Probability", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig("Fig11c.pdf", format="pdf")
# plt.show()

