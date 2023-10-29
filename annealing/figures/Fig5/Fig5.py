import numpy as np
import matplotlib
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt

prefix = "C:/Users/rakche_a-adm/Downloads/data_sk/data/"

# Fig 5a
N = 18                               # system size
instances_marked = []

# # Fig 5b
# N = 20                               # system size
# instances_marked = [9, 78, 86, 147, 157, 196]

rates = np.load(prefix + "qa_rates_new_N=" + str(N) + ".npy")

instances = np.arange(1, 201, 1)
dt = 0.01
steps = 100

fig = plt.figure(1, figsize=(3.75, 3), constrained_layout=True)
gaps = np.load(prefix + "minimal_gaps_N=" + str(N) + ".npy")

gaps_marked = []
rates_marked = []

for inst in instances_marked:
    gaps_marked.append(gaps[inst - 1])
    rates_marked.append(rates[inst - 1])

gaps_marked = np.array(gaps_marked)
rates_marked = np.array(rates_marked)

plt.scatter(1. / gaps[0:len(instances)] ** 2, 1. / rates, s=20, c="black", marker="o", zorder=3)
plt.scatter(1. / gaps_marked ** 2, 1. / rates_marked, s=20, c="red", marker="o", zorder=4)

plt.plot(1. / gaps ** 2, 1. / gaps ** 2, ls="-", color="blue", lw=1, zorder=2)

plt.grid(zorder=1, color="grey", lw=1)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$1/(\Delta E_{\mathrm{min}})^2$", fontsize=13)
plt.ylabel(r"$t_{\mathrm{sol}}$", fontsize=13)

plt.savefig("Fig5a.pdf", format="pdf")
plt.show()