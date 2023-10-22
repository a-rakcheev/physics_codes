import numpy as np
import matplotlib.pyplot as plt
import math


def neighbors(point, order, dx, dy):

    list_of_neighbors = []
    list_of_neighbor_idx = []
    if order == 1:

        neighbor = [point[0] + dx, point[1]]
        list_of_neighbors.append(neighbor)
        list_of_neighbor_idx.append([1, 0])

        neighbor = [point[0], point[1] + dy]
        list_of_neighbors.append(neighbor)
        list_of_neighbor_idx.append([0, 1])

    else:
        for n in np.arange(1, order, 1):
            m = order - n

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
                print("True")
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

    d += lengths[0] * math.sqrt(metric[i, j, 0, 0] * cos ** 2 + metric[i, j, 1, 1] * sin ** 2 + 2. * metric[i, j, 1, 0] * cos * sin)
    for k, step in enumerate(steps):

        if step == "right":

            i += 1
            d += lengths[k + 1] * math.sqrt(
                metric[i, j, 0, 0] * cos ** 2 + metric[i, j, 1, 1] * sin ** 2 + 2. * metric[i, j, 1, 0] * cos * sin)

        if step == "left":

            i -= 1
            d += lengths[k + 1] * math.sqrt(
                metric[i, j, 0, 0] * cos ** 2 + metric[i, j, 1, 1] * sin ** 2 + 2. * metric[i, j, 1, 0] * cos * sin)

        elif step == "up":

            j += 1
            d += lengths[k + 1] * math.sqrt(
                metric[i, j, 0, 0] * cos ** 2 + metric[i, j, 1, 1] * sin ** 2 + 2. * metric[i, j, 1, 0] * cos * sin)

        elif step == "diag":

            i += 1
            j += 1
            d += lengths[k + 1] * math.sqrt(
                metric[i, j, 0, 0] * cos ** 2 + metric[i, j, 1, 1] * sin ** 2 + 2. * metric[i, j, 1, 0] * cos * sin)

        elif step == "antidiag":

            i -= 1
            j += 1
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


# delta_x = 1.
# delta_y = 1.
#
# steps = 5
# points_x = np.arange(steps) * delta_x
# points_y = np.arange(steps) * delta_y
#
# grid_x, grid_y = np.meshgrid(points_x, points_y)
# plt.scatter(grid_x, grid_y, 7, marker="s", color="black")
#
# for x in points_x:
#     plt.axvline(x + 0.5 * delta_x, ls="--")
#
# for y in points_y:
#     plt.axhline(y + 0.5 * delta_y, ls="--")
#
# in_point = [3 * delta_x, 0 * delta_y]
#
# n1, n_idx1 = neighbors(in_point, 1, delta_x, delta_y)
# n2, n_idx2 = neighbors(in_point, 2, delta_x, delta_y)
# n3, n_idx3 = neighbors(in_point, 3, delta_x, delta_y)
# n4, n_idx4 = neighbors(in_point, 4, delta_x, delta_y)
# n5, n_idx5 = neighbors(in_point, 5, delta_x, delta_y)
#
# plt.scatter(in_point[0], in_point[1], 15, marker="o", color="red")
#
# for neigh in n1:
#     plt.scatter(neigh[0], neigh[1], 15, marker="o", color="blue")
#
# for neigh in n2:
#     plt.scatter(neigh[0], neigh[1], 15, marker="o", color="green")
#
# for neigh in n3:
#     plt.scatter(neigh[0], neigh[1], 15, marker="o", color="purple")
#
# for neigh in n4:
#     plt.scatter(neigh[0], neigh[1], 15, marker="o", color="yellow")
#
# for neigh in n5:
#     plt.scatter(neigh[0], neigh[1], 15, marker="o", color="brown")
#
# print(n4)
# print(n_idx4)
# print(n_idx4[1][0], n_idx4[1][1])
#
# # intersections
# isec, dist, steps = intersections(in_point, -1, 3, delta_x, delta_y)
# for sec in isec:
#
#     plt.scatter(sec[0], sec[1], 20, marker="x", color="black")
#
# print(isec)
# print(dist)
# print(steps)
# print(distance_euclidean(0.3, dist, steps) ** 2)
#
# plt.show()

isec, dist, steps = intersections([0., 0.], -1, 1, 0.5, 0.5)
print(isec)
print(dist)
print(steps)