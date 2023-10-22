import numpy as np
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# parameters
L = 8
g = 1.0
m = 1.0
h = 1.0
start = 0.0
end = 3.141
res = 1000000

name = "hhz_scaled_gap_widths_and_energies_L" + str(L) \
       + "_tau_start" + str(start).replace(".", "-") + "_tau_end" + str(end).replace(".", "-") \
       + "_tau_res" + str(res) + ".npz"

data = np.load(name)
indices = data["idx"]
widths = data["widths"]
tau_min = start + indices[0] * (end - start) / res

# high res
start_hr = 0.25
end_hr = 1.0
res_hr = 15000000
name = "hhz_scaled_gap_widths_and_energies_from_gaps_L" + str(L) \
           + "_tau_start" + str(start_hr).replace(".", "-") + "_tau_end" + str(end_hr).replace(".", "-") \
           + "_tau_res" + str(res_hr) + ".npz"

data = np.load(name)
tau_min_hr = data["times"]
widths_hr = data["widths"]


idx_hr = 0

# plot results
markersize = 3
fig = plt.figure(1, figsize=(3.375, 2.5))


plt.subplot(1, 2, 1)
plt.scatter(widths, tau_min, s=markersize, color="darkred")


# plt.grid()
plt.xlabel(r"$\Delta_{c}$", fontsize=12)
plt.ylabel(r"$\tau$", fontsize=12)

plt.xscale("log")
plt.xlim(1.e-8, 1.e-2)
plt.xticks([1.e-8, 1.e-6, 1.e-4, 1.e-2], fontsize=12)

plt.ylim(0.0, 1.0)
plt.yticks([0, 0.5, 1.0], fontsize=12)


plt.subplot(1, 2, 2)
plt.scatter(widths_hr, tau_min_hr, s=markersize, color="darkred")

# plt.grid()
plt.xlabel(r"$\Delta_{c}$", fontsize=12)

plt.xscale("log")
plt.xlim(1.e-8, 1.e-2)
plt.xticks([1.e-8, 1.e-6, 1.e-4, 1.e-2], fontsize=12)

plt.ylim(0.0, 1.)
plt.yticks([0, 0.5, 1.0], [], fontsize=12)

plt.subplots_adjust(wspace=0.3, bottom=0.2, top=0.95, left=0.15, right=0.95)
plt.savefig("Fig4.pdf", format="pdf")
# plt.show()