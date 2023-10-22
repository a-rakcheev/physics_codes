import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def quiver_metric_tensor(xl, yl, gl, cmap_major, cmap_minor):

    X, Y = np.meshgrid(xl, yl)

    major_u = np.zeros_like(X)
    major_v = np.zeros_like(X)
    minor_u = np.zeros_like(X)
    minor_v = np.zeros_like(X)

    norm = np.zeros_like(X)

    for i, x in enumerate(xl):
        for j, y in enumerate(yl):

            g = gl[i, j, :, :]

            ev, evec = np.linalg.eigh(g)
            idx_sort = np.argsort(np.absolute(ev))

            major_u[j, i] = evec[0, idx_sort[1]]
            major_v[j, i] = evec[1, idx_sort[1]]

            minor_u[j, i] = evec[0, idx_sort[0]]
            minor_v[j, i] = evec[1, idx_sort[0]]

            norm[j, i] = np.sqrt(np.absolute(ev[0] * ev[1]))

    plt.grid()
    # plt.quiver(X, Y, minor_u, minor_v, norm / norm.max(), cmap=cmap, pivot="mid")

    if cmap_maj == "None":
        # p_min = plt.quiver(X, Y, minor_u, minor_v, norm, norm=colors.LogNorm(vmin=1.e-5, vmax=1.e-2), cmap=cmap_minor, pivot="mid")
        p_min = plt.quiver(X, Y, minor_u, minor_v, norm, cmap=cmap_minor, pivot="mid")

        cbar = plt.colorbar(p_min)
        cbar.ax.tick_params(labelsize=5)

    elif cmap_min == "None":
        p_maj = plt.quiver(X, Y, major_u, major_v, norm, norm=colors.LogNorm(vmin=1.e-5, vmax=1.e-2), cmap=cmap_major, pivot="mid")
        cbar = plt.colorbar(p_maj)
        cbar.ax.tick_params(labelsize=5)

    else:
        p_maj = plt.quiver(X, Y, major_u, major_v, norm, norm=colors.LogNorm(vmin=1.e-5, vmax=1.e-2), cmap=cmap_major, pivot="mid")
        p_min = plt.quiver(X, Y, minor_u, minor_v, norm, norm=colors.LogNorm(vmin=1.e-5, vmax=1.e-2), cmap=cmap_minor, pivot="mid")

        cbar_min = plt.colorbar(p_min)
        cbar_min.ax.tick_params(labelsize=5)

        cbar_maj = plt.colorbar(p_maj)
        cbar_maj.ax.tick_params(labelsize=5)


# parameters
L = 4
res_x = 50
res_y = 50
l = 2
order = 15
S = 2
beta = 0.01

cmap_maj = "None"
cmap_min = "inferno_r"
# computations = ["subspace_coherent", "subspace_incoherent", "finite_temperature_coherent",
#                 "infinite_temperature_coherent", "finite_temperature_incoherent",
#                 "infinite_temperature_incoherent", "error_metric"]
computations = ["error_metric"]

plt.figure(1, figsize=(3.4, 3.))
save = "yes"                                                               # yes if plot should be saved
# prefix = "/home/artem/Dropbox/bu_research/data/"
# prefix = "D:/Dropbox/bu_research/code/"
prefix = ""

for computation in computations[:]:

    print(computation)
    plt.clf()

    # compute metric based on gauge potential and commutator (chi)
    if computation == "subspace_coherent":

        name = prefix + "metrics_subspace_coherent_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order)\
               + "_S" + str(S) + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
        data = np.load(name)
        metric_grid = data["metric"]
        xl = np.linspace(1.e-6, 1.5, res_x)
        yl = np.linspace(1.e-6, 1.5, res_y)

        quiver_metric_tensor(xl, yl, metric_grid / L, cmap_major=cmap_maj, cmap_minor=cmap_min)
        plt.xticks([0., 0.5, 1., 1.5], fontsize=5)
        plt.yticks([0., 0.5, 1., 1.5], fontsize=5)
        plt.minorticks_on()
        plt.xlabel(r"$h$", fontsize=6)
        plt.ylabel(r"$g$", fontsize=6)
        plt.title("Coherent Subspace Metric Density of the LTFI with S=" + str(S) + r", $l=" + str(l) + "$", fontsize=5.5)

        if save == "yes":
            plt.savefig("metric_ltfi_" + computation + "_L" + str(L) + "_l" + str(l) + "_order" + str(order)
                        + "_S" + str(S) + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".pdf", format="pdf")

    elif computation == "subspace_incoherent":

        name = prefix + "metrics_subspace_incoherent_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order) \
               + "_S" + str(S) + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
        data = np.load(name)
        metric_grid = data["metric"]
        xl = np.linspace(1.e-6, 1.5, res_x)
        yl = np.linspace(1.e-6, 1.5, res_y)

        quiver_metric_tensor(xl, yl, metric_grid / L, cmap_major=cmap_maj, cmap_minor=cmap_min)
        plt.xticks([0., 0.5, 1., 1.5], fontsize=5)
        plt.yticks([0., 0.5, 1., 1.5], fontsize=5)
        plt.minorticks_on()
        plt.xlabel(r"$h$", fontsize=6)
        plt.ylabel(r"$g$", fontsize=6)
        plt.title("Incoherent Subspace Metric Density of the LTFI with S=" + str(S) + r", $l=" + str(l) + "$", fontsize=5.5)

        if save == "yes":
            plt.savefig("metric_ltfi_" + computation + "_L" + str(L) + "_l" + str(l) + "_order" + str(order)
                        + "_S" + str(S) + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".pdf", format="pdf")

    elif computation == "finite_temperature_coherent":

        name = prefix + "metrics_coherent_finT_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order)\
               + "_beta" + str(beta).replace(".", "-") + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
        data = np.load(name)
        metric_grid = data["metric"]
        xl = np.linspace(1.e-6, 1.5, res_x)
        yl = np.linspace(1.e-6, 1.5, res_y)

        quiver_metric_tensor(xl, yl, metric_grid / L, cmap_major=cmap_maj, cmap_minor=cmap_min)
        plt.xticks([0., 0.5, 1., 1.5], fontsize=5)
        plt.yticks([0., 0.5, 1., 1.5], fontsize=5)
        plt.minorticks_on()
        plt.xlabel(r"$h$", fontsize=6)
        plt.ylabel(r"$g$", fontsize=6)
        plt.title(r"Coherent Finite-Temperature Metric Density of the LTFI with $\beta=" + str(beta) + r"$, $l=" + str(l) + "$", fontsize=5.5)

        if save == "yes":
            plt.savefig("metric_ltfi_" + computation + "_L" + str(L) + "_l" + str(l) + "_order" + str(order)
                        + "_beta"+ str(beta).replace(".", "-") + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".pdf", format="pdf")

    elif computation == "infinite_temperature_coherent":

        name = prefix + "metrics_coherent_infT_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order) \
               + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
        data = np.load(name)
        metric_grid = data["metric"]
        xl = np.linspace(1.e-6, 1.5, res_x)
        yl = np.linspace(1.e-6, 1.5, res_y)

        quiver_metric_tensor(xl, yl, metric_grid / L, cmap_major=cmap_maj, cmap_minor=cmap_min)
        plt.xticks([0., 0.5, 1., 1.5], fontsize=5)
        plt.yticks([0., 0.5, 1., 1.5], fontsize=5)
        plt.minorticks_on()
        plt.xlabel(r"$h$", fontsize=6)
        plt.ylabel(r"$g$", fontsize=6)
        plt.title(r"Coherent Infinite-Temperature Metric Density of the LTFI with $l=" + str(l) + "$", fontsize=5.5)

        if save == "yes":
            plt.savefig("metric_ltfi_" + computation + "_L" + str(L) + "_l" + str(l) + "_order" + str(order)
                        + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".pdf", format="pdf")

    elif computation == "finite_temperature_incoherent":

        name = prefix + "metrics_incoherent_finT_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order) \
               + "_beta" + str(beta).replace(".", "-") + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
        data = np.load(name)
        metric_grid = data["metric"]
        xl = np.linspace(1.e-6, 1.5, res_x)
        yl = np.linspace(1.e-6, 1.5, res_y)

        quiver_metric_tensor(xl, yl, metric_grid / L, cmap_major=cmap_maj, cmap_minor=cmap_min)
        plt.xticks([0., 0.5, 1., 1.5], fontsize=5)
        plt.yticks([0., 0.5, 1., 1.5], fontsize=5)
        plt.minorticks_on()
        plt.xlabel(r"$h$", fontsize=6)
        plt.ylabel(r"$g$", fontsize=6)
        plt.title(r"Incoherent Finite-Temperature Metric Density of the LTFI with $\beta=" + str(beta) + r"$, $l=" + str(l) + "$", fontsize=5.5)

        if save == "yes":

            plt.savefig("metric_ltfi_" + computation + "_L" + str(L) + "_l" + str(l) + "_order" + str(order) + "_beta"
                        + str(beta).replace(".", "-") + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".pdf", format="pdf")

    elif computation == "infinite_temperature_incoherent":

        name = prefix + "metrics_incoherent_infT_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order) \
               + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
        data = np.load(name)
        metric_grid = data["metric"]
        xl = np.linspace(1.e-6, 1.5, res_x)
        yl = np.linspace(1.e-6, 1.5, res_y)

        quiver_metric_tensor(xl, yl, metric_grid / L, cmap_major=cmap_maj, cmap_minor=cmap_min)
        plt.xticks([0., 0.5, 1., 1.5], fontsize=5)
        plt.yticks([0., 0.5, 1., 1.5], fontsize=5)
        plt.minorticks_on()
        plt.xlabel(r"$h$", fontsize=6)
        plt.ylabel(r"$g$", fontsize=6)
        plt.title(r"Incoherent Infinite-Temperature Metric Density of the LTFI with $l=" + str(l) + "$", fontsize=5.5)

        if save == "yes":
            plt.savefig("metric_ltfi_" + computation + "_L" + str(L) + "_l" + str(l) + "_order" + str(order)
                        + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".pdf", format="pdf")

    elif computation == "chi_metric":

        name = prefix + "metrics_chi_metric_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order) \
               + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
        data = np.load(name)
        metric_grid = data["metric"]
        xl = np.linspace(1.e-6, 1.5, res_x)
        yl = np.linspace(1.e-6, 1.5, res_y)

        quiver_metric_tensor(xl, yl, metric_grid / L, cmap_major=cmap_maj, cmap_minor=cmap_min)
        plt.xticks([0., 0.5, 1., 1.5], fontsize=5)
        plt.yticks([0., 0.5, 1., 1.5], fontsize=5)
        plt.minorticks_on()
        plt.xlabel(r"$h$", fontsize=6)
        plt.ylabel(r"$g$", fontsize=6)
        plt.title(r"$\chi$-Metric Density of the LTFI with $l=" + str(l) + "$", fontsize=5.5)

        if save == "yes":
            plt.savefig("metric_ltfi_" + computation + "_L" + str(L) + "_l" + str(l) + "_order" + str(order)
                        + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".pdf", format="pdf")

    elif computation == "error_metric":

        name = prefix + "metrics_error_metric_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order) \
               + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
        data = np.load(name)
        metric_grid = data["metric"]
        xl = np.linspace(1.e-6, 1.5, res_x)
        yl = np.linspace(1.e-6, 1.5, res_y)

        quiver_metric_tensor(xl, yl, metric_grid / L, cmap_major=cmap_maj, cmap_minor=cmap_min)
        plt.xticks([0., 0.5, 1., 1.5], fontsize=5)
        plt.yticks([0., 0.5, 1., 1.5], fontsize=5)
        plt.minorticks_on()
        plt.xlabel(r"$h$", fontsize=6)
        plt.ylabel(r"$g$", fontsize=6)
        plt.title(r"Error Metric Density of the LTFI with $l=" + str(l) + "$", fontsize=5.5)

        if save == "yes":
            plt.savefig("metric_error_" + computation + "_L" + str(L) + "_l" + str(l) + "_order" + str(order)
                        + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".pdf", format="pdf")
plt.show()
