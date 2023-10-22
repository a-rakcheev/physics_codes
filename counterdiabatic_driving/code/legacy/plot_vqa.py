import numpy as np
from commute_stringgroups_v2 import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# parameters
N = 6                  # number of spins
order = 10              # order of variational gauge potential
cutoff = 3              # length cutoff
res = 60                # resolution of the interval (0, 1) in s
T = 100.0               # protocol duration
s_initial = 0.3         # initial s
s_final = 0.9           # final s
kappa = 0.5

sl = np.linspace(s_initial, s_final, res + 1)

plt.figure(1, figsize=(14, 8))
plt.subplot(1, 2, 1)
# for cutoff in np.arange(2, N + 1, 1):
#
#     print("l_max:", cutoff)
#     name = "vqa_cd_tfi_L" + str(N) + "_l" + str(cutoff) + "_T" + str(T).replace(".", "-") \
#            + "_res" + str(res) + "_s_i" + str(s_initial).replace(".", "-") + "_s_f" + str(s_final).replace(".", "-") \
#            + "_order" + str(order) + ".npz"
#
#     data = np.load(name)
#     prob = data["prob"]
#     plt.plot(sl, prob, marker="^", markersize=4, ls="", label=r"$l_{max}=" + str(cutoff) + r"$")


# name = "vqa_cd_tfi_L" + str(N) + "_l" + str(cutoff) + "_T" + str(T).replace(".", "-") \
#        + "_res" + str(res) + "_s_i" + str(s_initial).replace(".", "-") + "_s_f" + str(s_final).replace(".", "-") \
#        + "_order" + str(order) + ".npz"

name = "vqa_cd_annni_L" + str(N) + "_l" + str(cutoff) + "_k" + str(kappa).replace(".", "-") \
       + "_T" + str(T).replace(".", "-") + "_res" + str(res) + "_s_i" + str(s_initial).replace(".", "-") + "_s_f" \
       + str(s_final).replace(".", "-") + "_order" + str(order) + ".npz"

data = np.load(name)
prob = data["prob"]
states = data["state"]

np.set_printoptions(1)

plt.plot(sl, prob, marker="^", markersize=4, ls="", label=r"$l_{max}=" + str(cutoff) + r"$")

plt.legend()
plt.grid()
plt.xlabel("s")
plt.ylabel("P")
plt.title("Fidelity")

# plot unconnected correlation function

correlations = np.zeros((res + 1, N // 2))
for d in np.arange(1, N // 2 + 1, 1):
    print("d:", d)
    O = equation()

    for i in range(N):
        op = ''.join(roll(list("z" + (d - 1) * "1" + "z" + "1" * (N - d - 1)), i))
        O[op] = 1. / N

    O_op = O.make_operator()
    for j in range(res + 1):

        print(j)
        state = states[j, :]
        state2 = O_op.dot(state)
        correlations[j, d - 1] = np.vdot(state, state2)


plt.subplot(1, 2, 2)
# plt.pcolormesh(np.arange(N // 2 + 1), sl, np.absolute(correlations), cmap="inferno_r",
#                norm=colors.LogNorm(vmin=1.e-3, vmax=1.0))
plt.pcolormesh(np.arange(N // 2 + 1), sl, correlations, cmap="seismic",
               vmin=-1., vmax=1.0)
plt.colorbar()

plt.xlabel(r"$d$", fontsize=12)
plt.ylabel(r"$s$", fontsize=12)
plt.xticks(np.arange(N // 2) + 0.5, np.arange(1, N // 2 + 1, 1))
plt.title("Spin-Spin Correlations")


plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.suptitle("Counter-Diabatic Driving of the TFI with a String Length Cutoff")
plt.show()