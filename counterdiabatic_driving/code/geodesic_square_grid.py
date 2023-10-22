import numpy as np
import scipy.sparse.csgraph as csg
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math


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

    if cmap_major == "None":
        p_min = plt.quiver(X, Y, minor_u, minor_v, norm, norm=colors.LogNorm(vmin=0.01, vmax=1.0), cmap=cmap_minor, pivot="mid")
        cbar = plt.colorbar(p_min)
        cbar.ax.tick_params(labelsize=5)

    elif cmap_minor == "None":
        p_maj = plt.quiver(X, Y, major_u, major_v, norm, norm=colors.LogNorm(vmin=0.01, vmax=1.0), cmap=cmap_major, pivot="mid")
        cbar = plt.colorbar(p_maj)
        cbar.ax.tick_params(labelsize=5)

    else:
        p_maj = plt.quiver(X, Y, major_u, major_v, norm, norm=colors.LogNorm(vmin=0.01, vmax=1.0), cmap=cmap_major, pivot="mid")
        p_min = plt.quiver(X, Y, minor_u, minor_v, norm, norm=colors.LogNorm(vmin=0.01, vmax=1.0), cmap=cmap_minor, pivot="mid")

        cbar_min = plt.colorbar(p_min)
        cbar_min.ax.tick_params(labelsize=5)

        cbar_maj = plt.colorbar(p_maj)
        cbar_maj.ax.tick_params(labelsize=5)


def neighbors(point, order, dx, dy):

    list_of_neighbors = []
    list_of_neighbor_idx = []
    if order == 1:

        # right
        neighbor = [point[0] + dx, point[1]]
        list_of_neighbors.append(neighbor)
        list_of_neighbor_idx.append([1, 0])

        # up
        neighbor = [point[0], point[1] + dy]
        list_of_neighbors.append(neighbor)
        list_of_neighbor_idx.append([0, 1])

    else:
        for n in np.arange(1, order, 1):
            m = order - n

            # right direction
            neighbor = [point[0] + n * dx, point[1] + m * dy]
            list_of_neighbors.append(neighbor)
            list_of_neighbor_idx.append([n, m])

            # left direction
            neighbor = [point[0] - n * dx, point[1] + m * dy]
            list_of_neighbors.append(neighbor)
            list_of_neighbor_idx.append([-n, m])

    return list_of_neighbors, list_of_neighbor_idx


def intersections(point1, steps_x, steps_y, dx, dy):

    angle = np.arctan2(steps_y * dy, steps_x * dx)
    tan = np.tan(angle)
    list_of_intersections = []
    list_of_lengths = []
    list_of_steps = []

    if steps_x >= 0:
        for n in range(steps_x):

            point_x = 0.5 * dx + n * dx
            point_y = tan * point_x

            list_of_intersections.append([point_x + point1[0], point_y + point1[1]])
            list_of_lengths.append(math.sqrt(point_x ** 2 + point_y ** 2))

            if np.absolute(np.rint((point_y + 0.5 * dy) / dy) - (point_y + 0.5 * dy) / dy) < 1.e-2:
                list_of_steps.append("diag")
            else:
                list_of_steps.append("right")

        for n in range(steps_y):

            point_y = 0.5 * dy + n * dy
            point_x = point_y / tan

            # check if point is on an intersection of x, y grid, then it is already covered by the x-loop
            if np.absolute(np.rint((point_x + 0.5 * dx) / dx) - (point_x + 0.5 * dx) / dx) < 1.e-2:
                continue

            else:
                list_of_intersections.append([point_x + point1[0], point_y + point1[1]])
                list_of_lengths.append(math.sqrt(point_x ** 2 + point_y ** 2))
                list_of_steps.append("up")

        list_of_intersections.append([steps_x * dx + point1[0], steps_y * dy + point1[1]])
        list_of_lengths.append(math.sqrt((steps_x * dx) ** 2 + (steps_y * dy) ** 2))
        list_of_steps.append("end")

    else:
        for n in range(-steps_x):

            point_x = -0.5 * dx - n * dx
            point_y = tan * point_x

            list_of_intersections.append([point_x + point1[0], point_y + point1[1]])
            list_of_lengths.append(math.sqrt(point_x ** 2 + point_y ** 2))

            if np.absolute(np.rint((point_y + 0.5 * dy) / dy) - (point_y + 0.5 * dy) / dy) < 1.e-2:
                list_of_steps.append("antidiag")
            else:
                list_of_steps.append("left")

        for n in range(steps_y):

            point_y = 0.5 * dy + n * dy
            point_x = point_y / tan

            # check if point is on an intersection of x, y grid, then it is already covered by the x-loop
            if np.absolute(np.rint((point_x + 0.5 * dx) / dx) - (point_x + 0.5 * dx) / dx) < 1.e-2:
                continue

            else:
                list_of_intersections.append([point_x + point1[0], point_y + point1[1]])
                list_of_lengths.append(math.sqrt(point_x ** 2 + point_y ** 2))
                list_of_steps.append("up")

        list_of_intersections.append([steps_x * dx + point1[0], steps_y * dy + point1[1]])
        list_of_lengths.append(math.sqrt((steps_x * dx) ** 2 + (steps_y * dy) ** 2))
        list_of_steps.append("end")

    # sort by distance
    list_of_lengths = np.array(list_of_lengths)
    list_of_intersections = np.array(list_of_intersections)
    list_of_steps = np.array(list_of_steps)

    argsort = np.argsort(list_of_lengths)
    list_of_lengths_sorted = list_of_lengths[argsort]
    list_of_intersections_sorted = list_of_intersections[argsort, :]
    list_of_steps = list_of_steps[argsort]

    list_of_distances = np.zeros(len(list_of_lengths_sorted))
    list_of_distances[0] = list_of_lengths_sorted[0]
    list_of_distances[1:] = list_of_lengths_sorted[1:] - list_of_lengths_sorted[0:-1]
    return list_of_intersections_sorted, list_of_distances, list_of_steps


# compute distance between two points based on metric and segments
def distance(angle, lengths, steps, metric, initial_i, initial_j):

    i = initial_i
    j = initial_j

    cos = math.cos(angle)
    sin = math.sin(angle)
    d = 0.

    # print(i, j)
    d += lengths[0] * math.sqrt(metric[i, j, 0, 0] * cos ** 2 + metric[i, j, 1, 1] * sin ** 2 + 2. * metric[i, j, 1, 0] * cos * sin)
    for k, step in enumerate(steps):

        if step == "right":

            i += 1
            # print("right", i, j)

            d += lengths[k + 1] * math.sqrt(
                metric[i, j, 0, 0] * cos ** 2 + metric[i, j, 1, 1] * sin ** 2 + 2. * metric[i, j, 1, 0] * cos * sin)

        if step == "left":

            i -= 1
            # print("left", i, j)

            d += lengths[k + 1] * math.sqrt(
                metric[i, j, 0, 0] * cos ** 2 + metric[i, j, 1, 1] * sin ** 2 + 2. * metric[i, j, 1, 0] * cos * sin)

        elif step == "up":

            j += 1
            # print("up", i, j)

            d += lengths[k + 1] * math.sqrt(
                metric[i, j, 0, 0] * cos ** 2 + metric[i, j, 1, 1] * sin ** 2 + 2. * metric[i, j, 1, 0] * cos * sin)

        elif step == "diag":

            i += 1
            j += 1
            # print("diag", i, j)

            d += lengths[k + 1] * math.sqrt(
                metric[i, j, 0, 0] * cos ** 2 + metric[i, j, 1, 1] * sin ** 2 + 2. * metric[i, j, 1, 0] * cos * sin)

        elif step == "antidiag":

            i -= 1
            j += 1
            # print("antidiag", i, j)

            d += lengths[k + 1] * math.sqrt(
                metric[i, j, 0, 0] * cos ** 2 + metric[i, j, 1, 1] * sin ** 2 + 2. * metric[i, j, 1, 0] * cos * sin)

        else:

            continue

    return d


# compute distance between two points based on metric and segments
def distance_euclidean(angle, lengths, steps):

    metric = np.identity(2)

    cos = math.cos(angle)
    sin = math.sin(angle)
    d = 0.

    d += lengths[0] * math.sqrt(metric[0, 0] * cos ** 2 + metric[1, 1] * sin ** 2 + 2. * metric[1, 0] * cos * sin)
    for k, step in enumerate(steps):

        if step == "right":

            d += lengths[k + 1] * math.sqrt(
                metric[0, 0] * cos ** 2 + metric[1, 1] * sin ** 2 + 2. * metric[1, 0] * cos * sin)

        elif step == "left":

            d += lengths[k + 1] * math.sqrt(
                metric[0, 0] * cos ** 2 + metric[1, 1] * sin ** 2 + 2. * metric[1, 0] * cos * sin)

        elif step == "up":

            d += lengths[k + 1] * math.sqrt(
                metric[0, 0] * cos ** 2 + metric[1, 1] * sin ** 2 + 2. * metric[1, 0] * cos * sin)

        elif step == "diag":

            d += lengths[k + 1] * math.sqrt(
                metric[0, 0] * cos ** 2 + metric[1, 1] * sin ** 2 + 2. * metric[1, 0] * cos * sin)

        elif step == "antidiag":

            d += lengths[k + 1] * math.sqrt(
                metric[0, 0] * cos ** 2 + metric[1, 1] * sin ** 2 + 2. * metric[1, 0] * cos * sin)

        else:

            continue

    return d


def create_graph_square_grid(N_x, N_y, gl, dx, dy, order):

    # there are N^2 nodes (for a square grid)
    # each can in principle be connected to each other

    distance_graph = np.zeros((N_x * N_y, N_x * N_y))

    # we consider all connections up to the given order
    # here we loop through every node and check all the neighbors in the upper half-space
    # this has some redundancy but should be fast enough for our purposes

    for i in range(N_x * N_y):

            idx_x = i % N_x
            idx_y = i // N_x
            point = [idx_x * dx, idx_y * dy]
            # print("node:", i, idx_x, idx_y)

            for deg in np.arange(1, order + 1, 1):

                # print("degree:", deg)
                # find neighbors
                nb, nb_idx = neighbors(point, deg, dx, dy)

                # add connection between each neighbor
                for j, neigh in enumerate(nb):

                    steps_x = nb_idx[j][0]
                    steps_y = nb_idx[j][1]
                    node_neighbor = i + N_x * steps_y + steps_x
                    # print("proposed neighbor:", node_neighbor, idx_x + steps_x, idx_y + steps_y)

                    # check if proposed neighbor is part of grid
                    if 0 <= idx_x + steps_x < N_x and 0 <= idx_y + steps_y < N_y:

                        # print("existing neighbor:", node_neighbor)
                        # print("steps:", steps_x, steps_y)

                        # compute intersections and distance
                        isec, dist, steps = intersections(point, steps_x, steps_y, dx, dy)
                        # print("steps:", steps)
                        angle = np.arctan2(steps_y * dy, steps_x * dx)
                        d = distance(angle, dist, steps, gl, idx_x, idx_y)

                        # add the coonection
                        distance_graph[i, node_neighbor] = d
                        distance_graph[node_neighbor, i] = d

    return csg.csgraph_from_dense(distance_graph)


# returns the length of a defined path
def pathlength(nodepath, distgraph):

    length = 0.
    for i in range(len(nodepath) - 1):

        # print(nodepath[i], nodepath[i + 1], distgraph[nodepath[i], nodepath[i + 1]])
        length += distgraph[nodepath[i], nodepath[i + 1]]

    return length


# known metrics on surfaces, the surface is defined as a function that returns the metric (matrix) at point x, y
def surface_metric(surface, size_x, size_y, dx, dy):

    metric = np.zeros((size_x, size_y, 2, 2))
    for i in range(size_x):
        for j in range(size_y):

            x = i * dx
            y = j * dy

            metric[i, j, :, :] = surface(x, y)

    return metric


def euclidean_metric(x, y):

    return np.identity(2)


# computes the length of the path based on the output from nx shortest path, which is a list of nodes
def pathlength_nx(nodepath, graph):

    pathlength = 0.
    for i in range(len(nodepath) - 1):

        pathlength += graph.edges[nodepath[i], nodepath[i + 1]]["weight"]

    return pathlength


# parameters
L = 6
res_x = 50
res_y = 50
l = 4
order = 15
S = 2

hl = np.linspace(1.e-6, 1.5, res_x)
gl = np.linspace(1.e-6, 1.5, res_y)
dh = hl[1] - hl[0]
dg = gl[1] - gl[0]

cmap_maj = "None"
cmap_min = "jet"
computations = ["subspace_coherent"]
save = "no"                                         # yes if plot should be saved

# prefix = "/home/artem/Dropbox/bu_research/data/"
prefix = "D:/Dropbox/data/agp/"
# prefix = "C:/Users/ARakc/Dropbox/bu_research/data/"
# prefix = "/media/artem/TOSHIBA EXT/Dropbox/data/agp/"
name = prefix + "metrics_subspace_coherent_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order) \
       + "_S" + str(S) + "_res_x" + str(res_x) + "_res_y" + str(res_y) + ".npz"
data = np.load(name)
metric_grid = data["metric"]

idx_h_start = 6
idx_g_start = 0
node_start = idx_g_start * res_x + idx_h_start

idx_h_end = 13
idx_g_end = 13
node_end = idx_g_end * res_x + idx_h_end

plt.figure(1, figsize=(12, 4))
orders = [1, 3, 5]
for k, ord in enumerate(orders):


    graph_csg = create_graph_square_grid(res_x, res_y, metric_grid, dh, dg, ord)
    graph = nx.convert_matrix.from_scipy_sparse_matrix(graph_csg)

    pos = {}
    for i in range(res_x):
        for j in range(res_y):
            node = j * res_x + i
            pos[node] = np.array([i * dh, j * dg], dtype=np.float32)

    # find shortest path
    path = nx.algorithms.shortest_paths.dijkstra_path(graph, node_start, node_end)

    # plot
    plt.subplot(1, len(orders), k + 1)

    quiver_metric_tensor(hl, gl, metric_grid / L, cmap_major=cmap_maj, cmap_minor=cmap_min)
    plt.xticks([0., 0.5, 1., 1.5], fontsize=5)
    plt.yticks([0., 0.5, 1., 1.5], fontsize=5)
    plt.minorticks_on()
    plt.xlabel(r"$h$", fontsize=6)
    plt.ylabel(r"$g$", fontsize=6)
    # plt.title("Coherent Subspace Metric Density of the LTFI with S=" + str(S) + r", $l=" + str(l) + "$", fontsize=5.5)

    # nx.draw_networkx(test_graph, pos, node_color="white", node_size=400)
    nx.draw_networkx_nodes(graph, pos, nodelist=path, node_color='gray', node_size=5, alpha=0.4)
    plt.title("Connectivity: " + str(ord) + ", Geodesic Length: " + str(round(pathlength_nx(path, graph), 3)), fontsize=8)

plt.savefig("connectivity_demo_ising.pdf", format="pdf")
plt.show()
