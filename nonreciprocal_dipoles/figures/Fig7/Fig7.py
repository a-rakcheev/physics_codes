import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'].join([r'\usepackage{amsfonts}', r'\usepackage{amsmath}', r'\usepackage{amssymb}'])

L = 2
theta = 45
eta = 0.0
z_0_set = 0.1  # plate distance
r_max = 20  # maximal radius
r_res = 21  # radial
gc_order = 10000
factor1 = 0.75
factor2 = 0.75

tl = [100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]

labelsize= 15
legendsize = 11

prefix = "E:/Dropbox/data/dipoles/repo/data/"                            # repo data directory

plt.figure(1, figsize=(6, 4), constrained_layout=True)        
for t_final in tl:

    data = np.load(prefix + "nonreciprocal_dynamics_pair_energy_vary_q_z0=" + str(z_0_set).replace(".", "-") + "_theta=" + str(theta) + "_eta=" + str(eta).replace(".", "-")  
    + "_tf=" + str(t_final).replace(".", "-") + "_f1=" + str(round(factor1, 2)) + "_f2=" + str(round(factor2, 2)) + ".npz") 

    ear = data["en"]
    ql = data["ql"]

    plt.plot(ql[1:], 2 * np.sqrt(ear[1:] / t_final), marker="o", lw=1, ls="--", markersize=3, label=r"$\tau_{\mathrm{final}}=" + str(int(t_final)) + r"$")

plt.grid()
plt.legend(fontsize=legendsize)
plt.xlabel(r"$qa$", fontsize=labelsize)
plt.ylabel(r"$\langle \ddot{\varphi}_{-} \rangle $", fontsize=labelsize)

plt.xscale("log")
plt.yscale("log")
plt.tick_params(labelsize=12)
plt.savefig("Fig7" + ".pdf", format="pdf")
plt.show()
