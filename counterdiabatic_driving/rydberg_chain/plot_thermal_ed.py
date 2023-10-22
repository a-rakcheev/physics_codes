        
import numpy as np
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.colors as colors

L = 16
steps = 17
    
for h in [1.0, 0.1, 0.01, 0.0005]:
    for g in [1.0, 0.1, 0.01, 0.0005]:

        print("h, g:", h, g)
        data = np.load("thermal_ed_logscale_L=" + str(L) + "_steps=" + str(steps) + "_g=" + str(g).replace(".", "-") + "_h=" + str(h).replace(".", "-") + ".npz")
        betal = data["betal"]
        partition_function = data["Z"]
        exp_x = data["x"]
        exp_n = data["n"]
        exp_nn = data["nn"]
        exp_n1n = data["n1n"]
        exp_n11n = data["n11n"]
        exp_n111n = data["n111n"]

        print(betal[0])
        print("Z:", partition_function[0])

        print("X:", exp_x[0])
        print("N:", exp_n[0])
        print("NN:", exp_nn[0])
        print("N1N:", exp_n1n[0])
        print("N11N:", exp_n11n[0])
        print("N111N:", exp_n111n[0])

        print(betal[-1])
        print("Z:", partition_function[-1])

        print("X:", exp_x[-1])
        print("N:", exp_n[-1])
        print("NN:", exp_nn[-1])
        print("N1N:", exp_n1n[-1])
        print("N11N:", exp_n11n[-1])
        print("N111N:", exp_n111n[-1])

        # energy density
        u = exp_nn + (1 / 2 ** 6) * exp_n1n + (1 / 3 ** 6) * exp_n11n - g * exp_x - h * exp_n 
        # subtract identity terms
        u = u - u[0]

        plt.figure(1, figsize=(20, 12))

        plt.subplot(2, 2, 1)
        plt.plot(betal, exp_x - exp_x[0], lw=1, label=r"$x$", zorder=2)
        plt.plot(betal, exp_n - exp_n[0], lw=1, label=r"$n$", zorder=2)
        plt.plot(betal, exp_nn - exp_nn[0], lw=1, label=r"$nn$", zorder=2)
        plt.plot(betal, exp_n1n - exp_n1n[0], lw=1, label=r"$n1n$", zorder=2)
        plt.plot(betal, exp_n11n - exp_n11n[0], lw=1, label=r"$n11n$", zorder=2)
        plt.plot(betal, exp_n111n - exp_n111n[0], lw=1, label=r"$n111n$", zorder=2)

        plt.grid(zorder=1)
        plt.legend()
        plt.xscale("log")
        plt.xlim(betal[0], 1.e2)
        plt.xlabel(r"$\beta$", fontsize=12)
        plt.ylabel(r"$\langle o \rangle - \langle o \rangle_{\beta=0}$", fontsize=12)


        plt.subplot(2, 2, 2)
        plt.plot(betal, u, lw=1, color="black", zorder=2)

        plt.grid(zorder=1)
        plt.xscale("log")
        plt.xlim(betal[0], 1.e2)
        plt.xlabel(r"$\beta$", fontsize=12)
        plt.ylabel(r"$\langle u \rangle - \langle u \rangle_{\beta=0}$", fontsize=12)


        plt.subplot(2, 2, 3)
        plt.plot(betal, np.gradient(u, betal), lw=1, color="black", zorder=2)

        plt.grid(zorder=1)
        plt.xscale("log")
        plt.xlim(betal[0], 1.e2)
        plt.xlabel(r"$\beta$", fontsize=12)
        plt.ylabel(r"$\partial_{\beta}u$", fontsize=12)

        plt.subplot(2, 2, 4)
        plt.plot(betal, np.gradient(np.gradient(u, betal), betal), lw=1, color="black", zorder=2)

        plt.grid(zorder=1)
        plt.xscale("log")
        plt.xlim(betal[0], 1.e2)
        plt.xlabel(r"$\beta$", fontsize=12)
        plt.ylabel(r"$\partial^{2}_{\beta}u$", fontsize=12)

        plt.suptitle(r"$L=" + str(L) + r",\; g=" + str(g) + r",\; h=" + str(h) + r"$")
        plt.subplots_adjust(wspace=0.25, hspace=0.25, top=0.925, bottom=0.1, left=0.1, right=0.95)
        plt.show()

        plt.clf()