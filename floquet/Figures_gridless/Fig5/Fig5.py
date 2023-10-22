import numpy as np
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# parameters
L = 8
bc = "pbc"
m = 1.0
h = 1.0
j = 1.0
m_max = 5

tau_start = 0.25
tau_end = 1.25
tau_res = 20000000
tau_res_max = 20000000
taul = np.linspace(tau_start, tau_end, tau_res)

# matrix_elements
name = "hhz_matel_drive_ed_L=" + str(L) + "_g=" + str(1.0).replace(".", "-") + ".npz"

data = np.load(name)
ev = data["ev"]
h_drive = data["drive"]

plt.figure(1, figsize=(3.375, 4.))
markersize = 4
color_matel = "skyblue"
color_gap = "darkred"

plt.subplot(2, 1, 1)
g = 0.01

# crossings
name = "angular_gaps_tfi_scaled_symmetric_" + bc + "_N" + str(L) + "_j" + str(j).replace(".", "-") + "_m" + str(
    m).replace(".", "-") \
       + "_h" + str(h).replace(".", "-") + "_tau_start" + str(tau_start).replace(".", "-") + "_tau_end" \
       + str(tau_end).replace(".", "-") + "_tau_res" + str(tau_res) + "_g" + str(g).replace(".", "-") + ".npz"

data = np.load(name)
tau_min = data["times"]
widths = data["widths"]

matel = []
wl = []

for i, ev_i in enumerate(ev):
    for k, ev_k in enumerate(ev):

        wl.append(np.absolute(ev_i - ev_k))
        matel.append(np.absolute(h_drive[i, k]) * 4 / np.pi)

wl = np.array(wl)

plt.plot(np.pi / wl, matel, marker="s", ls="", color=color_matel, label=r"Mat. El.", markersize=markersize)
plt.plot(tau_min, widths / (2 * tau_min * g), marker="x", ls="", color=color_gap, label=r"Gap", markersize=markersize)

plt.legend(loc="upper left", framealpha=1, ncol=2)
# plt.grid()

plt.xlim(0.5, 1.25)
plt.xticks([0.5, 0.75, 1.0, 1.25], [], fontsize=12)

plt.yscale("log")
plt.yticks(fontsize=12)
plt.ylim(1.e-6, 5.e-2)

plt.subplot(2, 1, 2)
g = 1.0

# crossings
name = "angular_gaps_tfi_scaled_symmetric_" + bc + "_N" + str(L) + "_j" + str(j).replace(".", "-") + "_m" + str(
    m).replace(".", "-") \
       + "_h" + str(h).replace(".", "-") + "_tau_start" + str(tau_start).replace(".", "-") + "_tau_end" \
       + str(tau_end).replace(".", "-") + "_tau_res" + str(tau_res) + "_g" + str(g).replace(".", "-") + ".npz"

data = np.load(name)
tau_min = data["times"]
widths = data["widths"]

matel = []
wl = []

for i, ev_i in enumerate(ev):
    for k, ev_k in enumerate(ev):

        wl.append(np.absolute(ev_i - ev_k))
        matel.append(np.absolute(h_drive[i, k]) * 4 / np.pi)

wl = np.array(wl)

plt.plot(np.pi / wl, matel, marker="s", ls="", color=color_matel, label=r"Mat. El.", markersize=markersize)
plt.plot(tau_min, widths / (2 * tau_min * g), marker="x", ls="", color=color_gap, label=r"Gap", markersize=markersize)

plt.legend(loc="upper left", framealpha=1, ncol=2)
# plt.grid()

plt.xlim(0.5, 1.25)
plt.xticks([0.5, 0.75, 1.0, 1.25], fontsize=12)
plt.xlabel(r"$\tau$", fontsize=12)

plt.yscale("log")
plt.yticks(fontsize=12)
plt.ylim(1.e-6, 5.e-2)

plt.subplots_adjust(hspace=0.2, top=0.95, right=0.95)
plt.savefig("Fig5.pdf", format="pdf")
# plt.show()


