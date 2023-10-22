import numpy as np
from scipy.optimize import curve_fit
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt


# fit to line
def linear_fit(x, gamma, a):
       return a - gamma * x


def exp_fit(x, gamma, a):
    return np.exp(a - gamma * x)

L = 28
state_name = "x_p"
taus = [1.1, 1.35, 2.2, 2.3]

plt.figure(1, figsize=(3.375, 2.5), constrained_layout=True)

# cycles for different tau
cycles = dict()
cycles["1.1"] = 5000
cycles["1.15"] = 5000
cycles["1.2"] = 5000
cycles["1.25"] = 5000
cycles["1.3"] = 5000
cycles["1.35"] = 3000
cycles["1.4"] = 3000
cycles["2.0"] = 1000
cycles["2.1"] = 1000
cycles["2.2"] = 3000
cycles["2.3"] = 5000
cycles["2.4"] = 5000
cycles["2.5"] = 5000
cycles["2.6"] = 5000

# fit in defined boundaries
boundaries = dict()
boundaries["1.1"] = [3000, 5000]
boundaries["1.15"] = [3000, 5000]
boundaries["1.2"] = [3000, 5000]
boundaries["1.25"] = [3000, 5000]
boundaries["1.3"] = [1000, 3000]
boundaries["1.35"] = [400, 1000]
boundaries["1.4"] = [200, 400]
boundaries["2.0"] = [50, 100]
boundaries["2.1"] = [50, 200]
boundaries["2.2"] = [100, 500]
boundaries["2.3"] = [2000, 3000]
boundaries["2.4"] = [3000, 5000]
boundaries["2.5"] = [3000, 5000]
boundaries["2.6"] = [3000, 5000]

colors = ["skyblue", "orange", "mediumorchid", "olive"]
zorder = [1, 2, 1, 1]
for i, tau in enumerate(taus):

    fit_start = boundaries[str(tau)][0]
    fit_end = boundaries[str(tau)][1]

    prefix = ""
    name = "hhz_pbc_k=0_p=1_energy_dynamics_" + state_name + "_L=" + str(L) + "_tau" + str(tau).replace(".", "-") \
           + "_cycles=" + str(cycles[str(tau)]) + ".npz"

    data = np.load(prefix + name)

    en_x = data["en_x"]
    en_z = data["en_z"]
    en_avg = 0.5 * (en_x + en_z)

    # fit
    popt_avg, pcov_avg = curve_fit(linear_fit, np.arange(cycles[str(tau)] + 1)[fit_start: fit_end + 1].astype(float),
                                   np.log(np.absolute(en_avg[fit_start:fit_end + 1])))

    plt.plot(np.absolute(en_avg), label=r"$\tau=" + str(tau) + r"$", color=colors[i], zorder=zorder[i], lw=2)
    plt.plot(exp_fit(np.arange(cycles[str(tau)] + 1), *popt_avg), ls="--", color="black", lw=1, zorder=3)


plt.legend(framealpha=1, fontsize=8, handlelength=0.5, handleheight=0.1)
# plt.grid()
plt.xlabel("cycles", fontsize=12)
plt.ylabel(r"$|\epsilon_{0}|$", fontsize=12)

plt.yscale("log")
plt.ylim(1.e-6, 1.0)
plt.xlim(0, 3000)

plt.xticks(fontsize=12)
plt.yticks([1.0, 1.e-2, 1.e-4, 1.e-6], fontsize=12)

plt.savefig("Fig2.pdf", format="pdf")
# plt.show()


