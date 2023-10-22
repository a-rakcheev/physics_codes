import numpy as np
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.colors as colors

L = 6
res_g = 65
res_h = 65

plt.figure(1, figsize=(18, 10))
cmap = "nipy_spectral"
plotscale = "linscale"

data = np.load("gs_ed_L=" + str(L) + "_res_g=" + str(res_g) + "_res_h=" + str(res_h) + ".npz")

exp_x = data["x"]
exp_n = data["n"]
exp_nn = data["nn"]
exp_n1n = data["n1n"]
exp_n11n = data["n11n"]
exp_n111n = data["n111n"]

hl = data["hl"]
gl = data["gl"]

plt.subplot(2, 3, 1)

if plotscale == "logscale":
    plt.pcolormesh(gl, hl, exp_x.T, norm=colors.LogNorm(vmin=1.e-3, vmax=1.), cmap=cmap)
elif plotscale == "linscale":
    plt.pcolormesh(gl, hl, exp_x.T, cmap=cmap, vmin=1.e-3, vmax=1)

plt.colorbar()

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$g$", fontsize=12)
plt.ylabel(r"$h$", fontsize=12)
plt.title(r"$\langle X \rangle / L$", fontsize=14)


plt.subplot(2, 3, 2)
if plotscale == "logscale":
    plt.pcolormesh(gl, hl, exp_n.T, norm=colors.LogNorm(vmin=1.e-3, vmax=1.), cmap=cmap)
elif plotscale == "linscale":
    plt.pcolormesh(gl, hl, exp_n.T, cmap=cmap, vmin=1.e-3, vmax=1)
    
plt.colorbar()

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$g$", fontsize=12)
plt.ylabel(r"$h$", fontsize=12)
plt.title(r"$\langle N \rangle / L$", fontsize=14)


plt.subplot(2, 3, 3)
if plotscale == "logscale":
    plt.pcolormesh(gl, hl, exp_nn.T, norm=colors.LogNorm(vmin=1.e-3, vmax=1.), cmap=cmap)
elif plotscale == "linscale":
    plt.pcolormesh(gl, hl, exp_nn.T, cmap=cmap, vmin=1.e-3, vmax=1)

plt.colorbar()

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$g$", fontsize=12)
plt.ylabel(r"$h$", fontsize=12)
plt.title(r"$\langle NN \rangle / L$", fontsize=14)


plt.subplot(2, 3, 4)
if plotscale == "logscale":
    plt.pcolormesh(gl, hl, exp_n1n.T, norm=colors.LogNorm(vmin=1.e-3, vmax=1.), cmap=cmap)
elif plotscale == "linscale":
    plt.pcolormesh(gl, hl, exp_n1n.T, cmap=cmap, vmin=1.e-3, vmax=1)
plt.colorbar()

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$g$", fontsize=12)
plt.ylabel(r"$h$", fontsize=12)
plt.title(r"$\langle N1N \rangle / L$", fontsize=14)


plt.subplot(2, 3, 5)
if plotscale == "logscale":
    plt.pcolormesh(gl, hl, exp_n11n.T, norm=colors.LogNorm(vmin=1.e-3, vmax=1.), cmap=cmap)
elif plotscale == "linscale":
    plt.pcolormesh(gl, hl, exp_n11n.T, cmap=cmap, vmin=1.e-3, vmax=1)
plt.colorbar()

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$g$", fontsize=12)
plt.ylabel(r"$h$", fontsize=12)
plt.title(r"$\langle N11N \rangle / L$", fontsize=14)


plt.subplot(2, 3, 6)
if plotscale == "logscale":
    plt.pcolormesh(gl, hl, exp_n111n.T, norm=colors.LogNorm(vmin=1.e-3, vmax=1.), cmap=cmap)
elif plotscale == "linscale":
    plt.pcolormesh(gl, hl, exp_n111n.T, cmap=cmap, vmin=1.e-3, vmax=1)
plt.colorbar()

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$g$", fontsize=12)
plt.ylabel(r"$h$", fontsize=12)
plt.title(r"$\langle N111N \rangle / L$", fontsize=14)

plt.subplots_adjust(wspace=0.25, hspace=0.25, top=0.925, bottom=0.1, left=0.1, right=0.95)
plt.savefig("gs_ed_L=" + str(L) + ".png", format="png", dpi=300)
plt.show()