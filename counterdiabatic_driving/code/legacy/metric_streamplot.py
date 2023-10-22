import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

res = 10
plt.figure(1, figsize=(3, 3))

# # sphere
# xl = np.linspace(0, 2. * np.pi, res)
# yl = np.linspace(0, np.pi, res)
#
# X, Y = np.meshgrid(xl, yl)
#
# major_u = np.zeros_like(X)
# major_v = np.zeros_like(X)
# minor_u = np.zeros_like(X)
# minor_v = np.zeros_like(X)
#
# major_norm = np.zeros_like(X)
# minor_norm = np.zeros_like(X)
# norm = np.zeros_like(X)
#
# for i, x in enumerate(xl):
#     for j, y in enumerate(yl):
#
#         g = np.zeros((2, 2))
#         g[0, 0] = np.sin(y) ** 2
#         g[0, 1] = 0
#         g[1, 0] = g[0, 1]
#         g[1, 1] = 1.
#
#         ev, evec = np.linalg.eigh(g)
#         idx_sort = np.argsort(np.absolute(ev))
#
#         major_u[j, i] = evec[0, idx_sort[1]]
#         major_v[j, i] = evec[1, idx_sort[1]]
#         major_norm[j, i] = ev[idx_sort[1]]
#
#         minor_u[j, i] = evec[0, idx_sort[0]]
#         minor_v[j, i] = evec[1, idx_sort[0]]
#         minor_norm[j, i] = ev[idx_sort[0]]
#
#         norm[j, i] =np.sqrt(np.absolute(ev[0] * ev[1]))
#
#
# plt.subplot(2, 2, 1)
#
# # plt.streamplot(X, Y, major_u, major_v, color="black", arrowstyle="-")
# # plt.streamplot(X, Y, minor_u, minor_v, color=norm / norm.max(), cmap="inferno_r", norm=colors.Normalize(vmin=0., vmax=1.0), arrowstyle="-")
# plt.grid()
# plt.quiver(X, Y, minor_u, minor_v, norm / norm.max(), cmap="jet", pivot="mid")
# plt.colorbar()
#
# plt.xlim(xl[0], xl[-1])
# plt.ylim(yl[0], yl[-1])
#
# plt.xlabel(r"$\phi$")
# plt.ylabel(r"$\theta$")
# plt.title("Sphere")
#
#
# # torus
# R = 2.
# r = 1.
#
# xl = np.linspace(0, 2. * np.pi, res)
# yl = np.linspace(0, 2. * np.pi, res)
#
# X, Y = np.meshgrid(xl, yl)
#
# major_u = np.zeros_like(X)
# major_v = np.zeros_like(X)
# minor_u = np.zeros_like(X)
# minor_v = np.zeros_like(X)
#
# major_norm = np.zeros_like(X)
# minor_norm = np.zeros_like(X)
# norm = np.zeros_like(X)
#
# for i, x in enumerate(xl):
#     for j, y in enumerate(yl):
#
#         g = np.zeros((2, 2))
#         g[0, 0] = (R + r * np.cos(y) ** 2)
#         g[0, 1] = 0
#         g[1, 0] = g[0, 1]
#         g[1, 1] = r ** 2
#
#         ev, evec = np.linalg.eigh(g)
#         idx_sort = np.argsort(np.absolute(ev))
#
#         major_u[j, i] = evec[0, idx_sort[1]]
#         major_v[j, i] = evec[1, idx_sort[1]]
#         major_norm[j, i] = ev[idx_sort[1]]
#
#         minor_u[j, i] = evec[0, idx_sort[0]]
#         minor_v[j, i] = evec[1, idx_sort[0]]
#         minor_norm[j, i] = ev[idx_sort[0]]
#
#         norm[j, i] =np.sqrt(np.absolute(ev[0] * ev[1]))
#
#
# plt.subplot(2, 2, 2)
#
# # plt.streamplot(X, Y, major_u, major_v, color="black", arrowstyle="-")
# # plt.streamplot(X, Y, minor_u, minor_v, color=norm / norm.max(), cmap="inferno_r", norm=colors.Normalize(vmin=0., vmax=1.0), arrowstyle="-")
# plt.grid()
# plt.quiver(X, Y, minor_u, minor_v, norm / norm.max(), cmap="jet", pivot="mid")
# plt.colorbar()
#
# plt.xlim(xl[0], xl[-1])
# plt.ylim(yl[0], yl[-1])
# plt.xlabel(r"$\phi$")
# plt.ylabel(r"$\theta$")
# plt.title("Torus")
#
#
# # elliptic
# a = 1.
# b = 1.
#
# xl = np.linspace(-10, 10, res)
# yl = np.linspace(-10, 10, res)
#
# X, Y = np.meshgrid(xl, yl)
#
# major_u = np.zeros_like(X)
# major_v = np.zeros_like(X)
# minor_u = np.zeros_like(X)
# minor_v = np.zeros_like(X)
#
# major_norm = np.zeros_like(X)
# minor_norm = np.zeros_like(X)
# norm = np.zeros_like(X)
#
# for i, x in enumerate(xl):
#     for j, y in enumerate(yl):
#
#         g = np.zeros((2, 2))
#         g[0, 0] = 1 + (2 * a * x) ** 2
#         g[0, 1] = 4 * a * b * x * y
#         g[1, 0] = g[0, 1]
#         g[1, 1] = 1 + (2 * b * y) ** 2
#
#         ev, evec = np.linalg.eigh(g)
#         idx_sort = np.argsort(np.absolute(ev))
#
#         major_u[j, i] = evec[0, idx_sort[1]]
#         major_v[j, i] = evec[1, idx_sort[1]]
#         major_norm[j, i] = ev[idx_sort[1]]
#
#         minor_u[j, i] = evec[0, idx_sort[0]]
#         minor_v[j, i] = evec[1, idx_sort[0]]
#         minor_norm[j, i] = ev[idx_sort[0]]
#
#         norm[j, i] =np.sqrt(np.absolute(ev[0] * ev[1]))
#
# weight_major = major_norm / major_norm.max()
# weight_minor = minor_norm / major_norm.max()
# v_min = weight_minor.min()
#
# plt.subplot(2, 2, 3)
#
# # plt.streamplot(X, Y, major_u, major_v, color="black", arrowstyle="-")
# # plt.streamplot(X, Y, minor_u, minor_v, color=norm / norm.max(), cmap="inferno_r", norm=colors.Normalize(vmin=0., vmax=1.0), arrowstyle="-")
# plt.grid()
# plt.quiver(X, Y, minor_u, minor_v, norm / norm.max(), cmap="jet", pivot="mid")
# plt.colorbar()
#
# plt.xlim(xl[0], xl[-1])
# plt.ylim(yl[0], yl[-1])
# plt.xlabel(r"$x$")
# plt.ylabel(r"$y$")
# plt.title("Elliptic Paraboloid")
#
# # hyperbolic
# a = 1.
# b = -1.
#
# xl = np.linspace(-10, 10, res)
# yl = np.linspace(-10, 10, res)
#
# X, Y = np.meshgrid(xl, yl)
#
# major_u = np.zeros_like(X)
# major_v = np.zeros_like(X)
# minor_u = np.zeros_like(X)
# minor_v = np.zeros_like(X)
#
# major_norm = np.zeros_like(X)
# minor_norm = np.zeros_like(X)
# norm = np.zeros_like(X)
#
# for i, x in enumerate(xl):
#     for j, y in enumerate(yl):
#
#         g = np.zeros((2, 2))
#         g[0, 0] = 1 + (2 * a * x) ** 2
#         g[0, 1] = 4 * a * b * x * y
#         g[1, 0] = g[0, 1]
#         g[1, 1] = 1 + (2 * b * y) ** 2
#
#         ev, evec = np.linalg.eigh(g)
#         idx_sort = np.argsort(np.absolute(ev))
#
#         major_u[j, i] = evec[0, idx_sort[1]]
#         major_v[j, i] = evec[1, idx_sort[1]]
#
#         minor_u[j, i] = evec[0, idx_sort[0]]
#         minor_v[j, i] = evec[1, idx_sort[0]]
#
#         norm[j, i] =np.sqrt(np.absolute(ev[0] * ev[1]))
#
#
# plt.subplot(2, 2, 4)
#
# # plt.streamplot(X, Y, major_u, major_v, color="black", arrowstyle="-")
# # plt.streamplot(X, Y, minor_u, minor_v, color=norm / norm.max(), cmap="inferno_r", norm=colors.Normalize(vmin=0., vmax=1.0), arrowstyle="-")
# plt.grid()
# plt.quiver(X, Y, minor_u, minor_v, norm / norm.max(), cmap="jet", pivot="mid")
# plt.colorbar()
#
# plt.xlim(xl[0], xl[-1])
# plt.ylim(yl[0], yl[-1])
# plt.xlabel(r"$x$")
# plt.ylabel(r"$y$")
# plt.title("Hyperbolic Paraboloid")
#
#
# plt.subplots_adjust(wspace=0.3, hspace=0.3)
# plt.suptitle("Metrics of Various Surfaces")

# plt.show()
# plt.savefig("example_metrics.pdf", format="pdf", dpi=300)


# elliptic
a = 1.
b = 1.

xl = np.linspace(-10, 10, res)
yl = np.linspace(-10, 10, res)

X, Y = np.meshgrid(xl, yl)

major_u = np.zeros_like(X)
major_v = np.zeros_like(X)
minor_u = np.zeros_like(X)
minor_v = np.zeros_like(X)

major_norm = np.zeros_like(X)
minor_norm = np.zeros_like(X)
norm = np.zeros_like(X)

for i, x in enumerate(xl):
    for j, y in enumerate(yl):

        g = np.zeros((2, 2))
        g[0, 0] = 1 + (2 * a * x) ** 2
        g[0, 1] = 4 * a * b * x * y
        g[1, 0] = g[0, 1]
        g[1, 1] = 1 + (2 * b * y) ** 2

        ev, evec = np.linalg.eigh(g)
        idx_sort = np.argsort(np.absolute(ev))

        major_u[j, i] = evec[0, idx_sort[1]]
        major_v[j, i] = evec[1, idx_sort[1]]
        major_norm[j, i] = ev[idx_sort[1]]

        minor_u[j, i] = evec[0, idx_sort[0]]
        minor_v[j, i] = evec[1, idx_sort[0]]
        minor_norm[j, i] = ev[idx_sort[0]]

        norm[j, i] =np.sqrt(np.absolute(ev[0] * ev[1]))

weight_major = major_norm / major_norm.max()
weight_minor = minor_norm / major_norm.max()
v_min = weight_minor.min()

# plt.streamplot(X, Y, major_u, major_v, color="black", arrowstyle="-")
# plt.streamplot(X, Y, minor_u, minor_v, color=norm / norm.max(), cmap="inferno_r", norm=colors.Normalize(vmin=0., vmax=1.0), arrowstyle="-")
plt.grid()
plt.quiver(X, Y, minor_u, minor_v, norm / norm.max(), cmap="jet", pivot="mid")
plt.colorbar()

plt.xlim(xl[0], xl[-1])
plt.ylim(yl[0], yl[-1])
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title("Elliptic Paraboloid")
plt.show()

def streamplot_metric_tensor(xl, yl, gl, cmap):

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

            norm[j, i] =np.sqrt(np.absolute(ev[0] * ev[1]))

    # plt.streamplot(X, Y, major_u, major_v, color="black", arrowstyle="-")
    plt.streamplot(X, Y, minor_u, minor_v, color=norm / norm.max(), cmap=cmap,
                   norm=colors.Normalize(vmin=0., vmax=1.0), arrowstyle="-")
    plt.colorbar()


def quiver_metric_tensor(xl, yl, gl, cmap):

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

            norm[j, i] =np.sqrt(np.absolute(ev[0] * ev[1]))

    plt.grid()
    plt.quiver(X, Y, minor_u, minor_v, norm / norm.max(), cmap=cmap, pivot="mid")
    plt.colorbar()
