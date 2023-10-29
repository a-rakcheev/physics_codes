# define functions for the dipole-dipole + conductor Hamiltonian

import numpy as np
import numba as nb


# some small convenience functions
@nb.njit(parallel=False)
def random_vector_unit_sphere(dimension, length):

    rv = np.random.normal(0, 1, dimension)
    rv /= np.linalg.norm(rv, 2)
    return length * rv


@nb.njit(parallel=False)
# cross matrix of a vector
def cross_matrix(vector):

    mat = np.zeros((3, 3))
    mat[0, 1] = -vector[2]
    mat[0, 2] = vector[1]
    mat[1, 2] = -vector[0]

    return mat - mat.T


@nb.njit(parallel=False)
# rotate vector around given axis (unnormalized vector)
def rotate_vector(vector, rotation_axis, rotation_angle):

    rotation_axis /= np.linalg.norm(rotation_axis, 2)

    rotation_matrix = np.cos(rotation_angle) * np.eye(3) + np.sin(rotation_angle) * cross_matrix(rotation_axis) \
                      + (1. - np.cos(rotation_angle)) * np.outer(rotation_axis, rotation_axis)

    vector = np.dot(rotation_matrix, vector)
    return vector


# define qudratures
# both take a vectorized function as input
# nodes and weights are read from file
def gauss_chebyshev_nodes_and_weights(order):

    table = np.zeros((order, 2))
    filename = "gauss-chebyshev_nodes_and_weights_order" + str(order) + ".txt"
    with open(filename, "r") as readfile:

        for idx, line in enumerate(readfile):

            vals = line.split(";")
            table[idx, 0] = vals[0]
            table[idx, 1] = vals[1]

    return table


def gauss_laguerre_nodes_and_weights(order):

    table = np.zeros((order, 2))
    filename = "gauss-laguerre_nodes_and_weights_order" + str(order) + ".txt"
    with open(filename, "r") as readfile:

        for idx, line in enumerate(readfile):

            vals = line.split(";")
            table[idx, 0] = vals[0]
            table[idx, 1] = vals[1]

    return table


@nb.njit(parallel=False)
def gauss_chebyshev_quadrature(func, nodes_and_weights):

    nodes = nodes_and_weights[:, 0]
    weights = nodes_and_weights[:, 1]

    values = func(nodes)
    quadrature = np.sum(weights * values)

    return quadrature


@nb.njit(parallel=False)
def gauss_laguerre_quadrature(func, nodes_and_weights):

    nodes = nodes_and_weights[:, 0]
    weights = nodes_and_weights[:, 1]

    values = func(nodes)
    quadrature = np.sum(weights * values)

    return quadrature


@nb.njit(parallel=False)
def trapezoid_integral(values, lower, upper):

    N = len(values)
    if N == 1:
        return values[0]

    if N == 2:

        h = (upper - lower)
        integ = 0.5 * (values[0] + values[N - 1])
        return h * integ

    else:

        h = (upper - lower) / (N - 1)
        integ = np.sum(values[1:N - 1])
        integ += 0.5 * (values[0] + values[N - 1])
        return h * integ


@nb.njit(parallel=False)
def romberg_integral(values, lower, upper):

    # check that the number of points is a power of 2 + 1
    N = len(values)
    m = np.log2(N - 1)
    M = int(m)
    if m != M:

        raise ValueError("The input length is not 2^k + 1")

    else:

        integral_table = np.zeros((M + 1, M + 1), dtype=np.complex128)

        # create trapezoidal integrals
        for i in range(M + 1):
            integral_table[i, 0] = trapezoid_integral(values[::2 ** (M - i)], lower, upper)

        # extrapolation
        for j in np.arange(1, M + 1, 1):
            factor = 4 ** j - 1
            for i in range(M + 1 - j):

                integral_table[i, j] = integral_table[i + 1, j - 1] \
                                       + (integral_table[i + 1, j - 1] - integral_table[i, j - 1]) / factor
    return integral_table[0, M]


# coupling constants based on adaptive sampling (recommended)
@nb.njit(parallel=False)
def dipole_field_matrix(difference_vector):

    distance = np.linalg.norm(difference_vector, 2)
    normalized_vector = difference_vector / distance
    factor = 8. * np.pi / (distance ** 3)
    matrix = 3. * np.outer(normalized_vector, normalized_vector) - np.identity(3)
    matrix *= factor

    return matrix


# weights for angular integral
@nb.njit(parallel=False)
def weight_Jx(u):
    return u ** 2


@nb.njit(parallel=False)
def weight_Jz(u):
    return 1.


@nb.njit(parallel=False)
def weight_Jxy(u):
    return u * np.sqrt(1 - u ** 2)


@nb.njit(parallel=False)
def weight_Dx(u):
    return np.sqrt(1 - u ** 2)


@nb.njit(parallel=False)
def weight_Dy(u):
    return -u


# frequency for radial integral
@nb.njit(parallel=False)
def omega(u, q, r, theta):
    return 0.5 * q * r * (u * np.cos(theta) + np.sqrt(1 - u ** 2) * np.sin(theta))


@nb.njit(parallel=False)
def scattering_function(x, u):
    val = (x - np.sqrt(x ** 2 - 1.j * 2. * u * x)) / (x + np.sqrt(x ** 2 - 1.j * 2. * u * x))
    return val.real


@nb.njit(parallel=False)
def radial_integrand(x, u, q, r, z_0, theta):
    val = (scattering_function(x, u) * x ** 2 * np.exp(-q * z_0 * x)).astype(np.complex128)
    freq = omega(u, q, r, theta)
    val *= np.exp(1.j * freq * x)
    return val


@nb.njit(parallel=False)
def radial_integrand_plate(x, u, q, r, z_0, theta):
    val = (-1 * x ** 2 * np.exp(-q * z_0 * x)).astype(np.complex128)
    freq = omega(u, q, r, theta)
    val *= np.exp(1.j * freq * x)
    return val


@nb.njit(parallel=False)
def adaptive_radial_integral(u, q, z_0, r, theta, sampling_factor, boundary_factor):
    x_max = 2. / (q * z_0)
    w = omega(u, q, r, theta)
    samples = boundary_factor * x_max * sampling_factor * np.absolute(w) / (2. * np.pi)
    if samples <= 10000:
        N_samp = 8193

    else:
        k_samp = int(np.ceil(np.log2(samples)))
        N_samp = 2 ** k_samp + 1

    xl = np.linspace(1.e-10, boundary_factor * x_max, N_samp)
    fl = radial_integrand(xl, u, q, r, z_0, theta)
    val = romberg_integral(fl, 0., boundary_factor * x_max)

    return val * q ** 3

@nb.njit(parallel=False)
def adaptive_radial_integral_plate(u, q, z_0, r, theta, sampling_factor, boundary_factor):
    x_max = 2. / (q * z_0)
    w = omega(u, q, r, theta)
    samples = boundary_factor * x_max * sampling_factor * np.absolute(w) / (2. * np.pi)
    if samples <= 10000:
        N_samp = 8193

    else:
        k_samp = int(np.ceil(np.log2(samples)))
        N_samp = 2 ** k_samp + 1

    xl = np.linspace(1.e-10, boundary_factor * x_max, N_samp)
    fl = radial_integrand_plate(xl, u, q, r, z_0, theta)
    val = romberg_integral(fl, 0., boundary_factor * x_max)

    return val * q ** 3


@nb.njit(parallel=False)
def radial_integral_quadrature(u, q, z_0, nodes_and_weights):

    R = q * z_0
    nodes = nodes_and_weights[:, 0]
    weights = nodes_and_weights[:, 1]

    values = scattering_function(nodes / R, u)
    quadrature = np.sum(weights * values)

    return quadrature / (R ** 3)


@nb.njit(parallel=False)
def sia_couplings(q, z_0, nodes_and_weights_gc, nodes_and_weights_gl):

    nodes = nodes_and_weights_gc[:, 0]
    weights = nodes_and_weights_gc[:, 1]

    radial_val = np.zeros_like(nodes)
    for i in nb.prange(len(nodes)):

        radial_val[i] = radial_integral_quadrature(nodes[i], q, z_0, nodes_and_weights_gl)

    fx = weight_Jx(nodes)
    fz = weight_Jz(nodes)

    Ax = np.sum(weights * fx * radial_val)
    Az = np.sum(weights * fz * radial_val)
    Ax *= 0.5 * (q ** 3) 
    Az *= 0.5 * (q ** 3)
    Ay = Az - Ax
    return Ax, Ay, Az


@nb.njit(parallel=False)
def radial_integral_smallR(u, q, z_0, r, theta):

    val = -0.25 * (u ** 2) / (q * z_0 - 1.j * omega(u, q, r, theta))
    return val * q ** 3


@nb.njit(parallel=False)
def radial_integral_largeR(u, q, z_0, r, theta):
    val = -2. / ((q * z_0 - 1.j * omega(u, q, r, theta)) ** 3)
    return val * q ** 3


@nb.njit(parallel=False, cache=True)
def adaptive_integrals(q, z_0, r, theta, sampling_factor, boundary_factor, nodes, weights):
    radial_val = np.zeros_like(nodes, dtype=np.complex128)
    for i, u in enumerate(nodes):
        radial_val[i] = adaptive_radial_integral(u, q, z_0, r, theta, sampling_factor, boundary_factor)

    radial_real = radial_val.real
    radial_imag = radial_val.imag

    val_Jx = np.sum(radial_real * weight_Jx(nodes) * weights)
    val_Jz = np.sum(radial_real * weight_Jz(nodes) * weights)
    val_Jxy = np.sum(radial_real * weight_Jxy(nodes) * weights)
    val_Dx = np.sum(radial_imag * weight_Dx(nodes) * weights)
    val_Dy = np.sum(radial_imag * weight_Dy(nodes) * weights)
    val_Jy = val_Jz - val_Jx

    return val_Jx, val_Jy, val_Jz, val_Jxy, val_Dx, val_Dy

@nb.njit(parallel=False, cache=True)
def adaptive_integrals_plate(q, z_0, r, theta, sampling_factor, boundary_factor, nodes, weights):
    radial_val = np.zeros_like(nodes, dtype=np.complex128)
    for i, u in enumerate(nodes):
        radial_val[i] = adaptive_radial_integral_plate(u, q, z_0, r, theta, sampling_factor, boundary_factor)

    radial_real = radial_val.real
    radial_imag = radial_val.imag

    val_Jx = np.sum(radial_real * weight_Jx(nodes) * weights)
    val_Jz = np.sum(radial_real * weight_Jz(nodes) * weights)
    val_Jxy = np.sum(radial_real * weight_Jxy(nodes) * weights)
    val_Dx = np.sum(radial_imag * weight_Dx(nodes) * weights)
    val_Dy = np.sum(radial_imag * weight_Dy(nodes) * weights)
    val_Jy = val_Jz - val_Jx

    return val_Jx, val_Jy, val_Jz, val_Jxy, val_Dx, val_Dy


@nb.njit(parallel=False, cache=True)
def semianalytic_integrals_plate(z_0, r, theta, nodes, weights):
    radial_val = np.zeros_like(nodes, dtype=np.complex128)
    for i, u in enumerate(nodes):
        radial_val[i] = -2 / (z_0 - 1.j * 0.5 * r * (u * np.cos(theta) + np.sqrt(1 - u ** 2) * np.sin(theta))) ** 3

    radial_real = radial_val.real
    radial_imag = radial_val.imag

    val_Jx = np.sum(radial_real * weight_Jx(nodes) * weights)
    val_Jz = np.sum(radial_real * weight_Jz(nodes) * weights)
    val_Jxy = np.sum(radial_real * weight_Jxy(nodes) * weights)
    val_Dx = np.sum(radial_imag * weight_Dx(nodes) * weights)
    val_Dy = np.sum(radial_imag * weight_Dy(nodes) * weights)
    val_Jy = val_Jz - val_Jx

    return val_Jx, val_Jy, val_Jz, val_Jxy, val_Dx, val_Dy



@nb.njit(parallel=False)
def integrals_smallR(q, z_0, r, theta, nodes, weights):
    radial_val = np.zeros_like(nodes, dtype=np.complex128)
    for i, u in enumerate(nodes):
        radial_val[i] = radial_integral_smallR(u, q, z_0, r, theta)

    radial_real = radial_val.real
    radial_imag = radial_val.imag

    val_Jx = np.sum(radial_real * weight_Jx(nodes) * weights)
    val_Jz = np.sum(radial_real * weight_Jz(nodes) * weights)
    val_Jxy = np.sum(radial_real * weight_Jxy(nodes) * weights)
    val_Dx = np.sum(radial_imag * weight_Dx(nodes) * weights)
    val_Dy = np.sum(radial_imag * weight_Dy(nodes) * weights)
    val_Jy = val_Jz - val_Jx

    return val_Jx, val_Jy, val_Jz, val_Jxy, val_Dx, val_Dy


@nb.njit(parallel=False)
def integrals_largeR(q, z_0, r, theta, nodes, weights):
    radial_val = np.zeros_like(nodes, dtype=np.complex128)
    for i, u in enumerate(nodes):
        radial_val[i] = radial_integral_largeR(u, q, z_0, r, theta)

    radial_real = radial_val.real
    radial_imag = radial_val.imag

    val_Jx = np.sum(radial_real * weight_Jx(nodes) * weights)
    val_Jz = np.sum(radial_real * weight_Jz(nodes) * weights)
    val_Jxy = np.sum(radial_real * weight_Jxy(nodes) * weights)
    val_Dx = np.sum(radial_imag * weight_Dx(nodes) * weights)
    val_Dy = np.sum(radial_imag * weight_Dy(nodes) * weights)
    val_Jy = val_Jz - val_Jx

    return val_Jx, val_Jy, val_Jz, val_Jxy, val_Dx, val_Dy


# convenience functions for optimization


# convert angles to dipole moments
@nb.njit(parallel=False)
def angle_to_dipole(angles, number_of_dipoles):
    moments = np.zeros(3 * number_of_dipoles)
    sin = np.sin(angles)
    cos = np.cos(angles)
    for i in range(number_of_dipoles):
        moments[3 * i] = sin[number_of_dipoles + i] * cos[i]
        moments[3 * i + 1] = sin[number_of_dipoles + i] * sin[i]
        moments[3 * i + 2] = cos[number_of_dipoles + i]

    return moments


@nb.njit(parallel=True)
def hamiltonians_1d_chain(q, z_0, theta, number_of_dipoles, sampling_factor, boundary_factor, nodes_and_weights):
    # number of dipoles
    N = number_of_dipoles

    # lattice vector
    lattice_vector = np.array([np.cos(theta), np.sin(theta), 0.])

    # create the Hamiltonians
    h_x = np.zeros((3 * N, 3 * N))
    h_y = np.zeros((3 * N, 3 * N))
    h_z = np.zeros((3 * N, 3 * N))
    h_xy = np.zeros((3 * N, 3 * N))
    h_Dx = np.zeros((3 * N, 3 * N))
    h_Dy = np.zeros((3 * N, 3 * N))
    h_dpl = np.zeros((3 * N, 3 * N))

    # create Hamiltonian by looping through all difference vectors and finding
    # the corresponding pairs of dipoles
    # the looping limits are such that no calculation is repeated
    # and only dipole pairs with indices where the second one is larger are considered
    # this corresponds to filling the upper (block) triangle in the Hamiltonian

    for i in nb.prange(N):
        for j_shifted in range(N - i - 1):

            j = j_shifted + i + 1

            # get difference vector
            vector = (j - i) * lattice_vector
            r = (j - i)

            mat_x = np.zeros((3, 3))
            mat_y = np.zeros((3, 3))
            mat_z = np.zeros((3, 3))
            mat_xy = np.zeros((3, 3))
            mat_Dx = np.zeros((3, 3))
            mat_Dy = np.zeros((3, 3))
            Jx, Jy, Jz, Jxy, Dx, Dy = adaptive_integrals(q, z_0, r, theta, sampling_factor, boundary_factor,
                                                         nodes_and_weights[:, 0], nodes_and_weights[:, 1])

            mat_x[0, 0] = Jx
            mat_y[1, 1] = Jy
            mat_z[2, 2] = Jz
            mat_xy[1, 0] = Jxy
            mat_xy[0, 1] = Jxy
            mat_Dy[0, 2] = Dy
            mat_Dx[1, 2] = -Dx
            mat_Dy[2, 0] = -Dy
            mat_Dx[2, 1] = Dx

            dipole_mat = dipole_field_matrix(vector)

            h_x[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_x
            h_y[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_y
            h_z[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_z
            h_xy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_xy
            h_Dx[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dx
            h_Dy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dy
            h_dpl[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = dipole_mat

    return h_x, h_y, h_z, h_xy, h_Dx, h_Dy, h_dpl


@nb.njit(parallel=False)
# return coupling matrices for 2d square lattice
def hamiltonians_2d_square_lattice(q, z_0, orientation_factor, N1, N2, sampling_factor,
                                                          boundary_factor, nodes_and_weights):
    # number of dipoles
    N = N1 * N2

    # angles of the chain rel to x-axis
    orientation_angle1 = orientation_factor * np.pi
    orientation_angle2 = (orientation_factor + 0.5) * np.pi

    # lattice vector
    lattice_vector1 = np.array([np.cos(orientation_angle1), np.sin(orientation_angle1), 0.])
    lattice_vector2 = np.array([np.cos(orientation_angle2), np.sin(orientation_angle2), 0.])

    # create the Hamiltonians
    h_x = np.zeros((3 * N, 3 * N))
    h_y = np.zeros((3 * N, 3 * N))
    h_z = np.zeros((3 * N, 3 * N))
    h_xy = np.zeros((3 * N, 3 * N))
    h_Dx = np.zeros((3 * N, 3 * N))
    h_Dy = np.zeros((3 * N, 3 * N))
    h_dpl = np.zeros((3 * N, 3 * N))

    # create Hamiltonian by looping through all difference vectors and finding
    # the corresponding pairs of dipoles
    # the looping limits are such that no calculation is repeated
    # and only dipole pairs with indices where the second one is larger are considered
    # this corresponds to filling the upper (block) triangle in the Hamiltonian

    for m in nb.prange(N2):
        for n_shift in range(2 * N1 - 1):

            n = n_shift - N1 + 1
            if m == 0 and n <= 0:

                continue

            else:

                # get difference vector
                vector = n * lattice_vector1 + m * lattice_vector2

                # length and angle wrt x-axis
                r = np.sqrt(np.sum(vector ** 2))
                theta = np.arctan2(vector[1], vector[0])

                mat_x = np.zeros((3, 3))
                mat_y = np.zeros((3, 3))
                mat_z = np.zeros((3, 3))
                mat_xy = np.zeros((3, 3))
                mat_Dx = np.zeros((3, 3))
                mat_Dy = np.zeros((3, 3))
                Jx, Jy, Jz, Jxy, Dx, Dy = adaptive_integrals(q, z_0, r, theta, sampling_factor, boundary_factor,
                                                             nodes_and_weights[:, 0], nodes_and_weights[:, 1])

                mat_x[0, 0] = Jx
                mat_y[1, 1] = Jy
                mat_z[2, 2] = Jz
                mat_xy[1, 0] = Jxy
                mat_xy[0, 1] = Jxy
                mat_Dy[0, 2] = Dy
                mat_Dx[1, 2] = -Dx
                mat_Dy[2, 0] = -Dy
                mat_Dx[2, 1] = Dx

                dipole_mat = dipole_field_matrix(vector)

                # find pairs of dipoles with corresponding indices
                for l in range(N2 - m):
                    if n < 0:
                        for k in np.arange(-n, N1, 1):
                            # indices of dipoles
                            i = k + l * N1
                            j = (k + n) + (l + m) * N1

                            h_x[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_x
                            h_y[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_y
                            h_z[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_z
                            h_xy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_xy
                            h_Dx[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dx
                            h_Dy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dy
                            h_dpl[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = dipole_mat

                    else:
                        for k in range(N1 - n):
                            # indices of dipoles
                            i = k + l * N1
                            j = (k + n) + (l + m) * N1

                            h_x[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_x
                            h_y[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_y
                            h_z[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_z
                            h_xy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_xy
                            h_Dx[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dx
                            h_Dy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dy
                            h_dpl[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = dipole_mat

    return h_x, h_y, h_z, h_xy, h_Dx, h_Dy, h_dpl


@nb.njit(parallel=True)
# return coupling matrices for 2d square lattice
# set maximal connectivity for truncation
def hamiltonians_2d_square_lattice_truncated_connectivity(q, z_0, orientation_factor, N1, N2, sampling_factor, boundary_factor, nodes_and_weights, connectivity):

    # number of dipoles
    N = N1 * N2

    # angles of the chain rel to x-axis
    orientation_angle1 = orientation_factor * np.pi
    orientation_angle2 = (orientation_factor + 0.5) * np.pi

    # lattice vector
    lattice_vector1 = np.array([np.cos(orientation_angle1), np.sin(orientation_angle1), 0.])
    lattice_vector2 = np.array([np.cos(orientation_angle2), np.sin(orientation_angle2), 0.])

    # create the Hamiltonians
    h_x = np.zeros((3 * N, 3 * N))
    h_y = np.zeros((3 * N, 3 * N))
    h_z = np.zeros((3 * N, 3 * N))
    h_xy = np.zeros((3 * N, 3 * N))
    h_Dx = np.zeros((3 * N, 3 * N))
    h_Dy = np.zeros((3 * N, 3 * N))
    h_dpl = np.zeros((3 * N, 3 * N))

    # create Hamiltonian by looping through all difference vectors and finding
    # the corresponding pairs of dipoles
    # the looping limits are such that no calculation is repeated
    # and only dipole pairs with indices where the second one is larger are considered
    # this corresponds to filling the upper (block) triangle in the Hamiltonian

    for m in nb.prange(N2):
        for n_shift in range(2 * N1 - 1):

            n = n_shift - N1 + 1
            if m == 0 and n <= 0:

                continue

            else:
                if (n ** 2 + m ** 2) < connectivity ** 2:

                    # get difference vector
                    vector = n * lattice_vector1 + m * lattice_vector2

                    # length and angle wrt x-axis
                    r = np.sqrt(np.sum(vector ** 2))
                    theta = np.arctan2(vector[1], vector[0])

                    mat_x = np.zeros((3, 3))
                    mat_y = np.zeros((3, 3))
                    mat_z = np.zeros((3, 3))
                    mat_xy = np.zeros((3, 3))
                    mat_Dx = np.zeros((3, 3))
                    mat_Dy = np.zeros((3, 3))
                    Jx, Jy, Jz, Jxy, Dx, Dy = adaptive_integrals(q, z_0, r, theta, sampling_factor, boundary_factor,
                                                                     nodes_and_weights[:, 0], nodes_and_weights[:, 1])

                    mat_x[0, 0] = Jx
                    mat_y[1, 1] = Jy
                    mat_z[2, 2] = Jz
                    mat_xy[1, 0] = Jxy
                    mat_xy[0, 1] = Jxy
                    mat_Dy[0, 2] = Dy
                    mat_Dx[1, 2] = -Dx
                    mat_Dy[2, 0] = -Dy
                    mat_Dx[2, 1] = Dx

                    dipole_mat = dipole_field_matrix(vector)

                    # find pairs of dipoles with corresponding indices
                    for l in range(N2 - m):
                        if n < 0:
                            for k in np.arange(-n, N1, 1):

                                # indices of dipoles
                                i = k + l * N1
                                j = (k + n) + (l + m) * N1

                                h_x[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_x
                                h_y[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_y
                                h_z[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_z
                                h_xy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_xy
                                h_Dx[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dx
                                h_Dy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dy
                                h_dpl[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = dipole_mat

                        else:
                            for k in range(N1 - n):

                                # indices of dipoles
                                i = k + l * N1
                                j = (k + n) + (l + m) * N1

                                h_x[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_x
                                h_y[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_y
                                h_z[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_z
                                h_xy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_xy
                                h_Dx[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dx
                                h_Dy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dy
                                h_dpl[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = dipole_mat

    return h_x, h_y, h_z, h_xy, h_Dx, h_Dy, h_dpl


@nb.njit(parallel=True)
# return coupling matrices for 2d square lattice
def hamiltonians_2d_triangular_lattice(q, z_0, orientation_factor, N1, N2, sampling_factor, boundary_factor, nodes_and_weights):

    # number of dipoles
    N = N1 * N2

    # angles of the chain rel to x-axis
    orientation_angle1 = orientation_factor * np.pi
    orientation_angle2 = (orientation_factor + 1./3) * np.pi

    # lattice vector
    lattice_vector1 = np.array([np.cos(orientation_angle1), np.sin(orientation_angle1), 0.])
    lattice_vector2 = np.array([np.cos(orientation_angle2), np.sin(orientation_angle2), 0.])

    # create the Hamiltonians
    h_x = np.zeros((3 * N, 3 * N))
    h_y = np.zeros((3 * N, 3 * N))
    h_z = np.zeros((3 * N, 3 * N))
    h_xy = np.zeros((3 * N, 3 * N))
    h_Dx = np.zeros((3 * N, 3 * N))
    h_Dy = np.zeros((3 * N, 3 * N))
    h_dpl = np.zeros((3 * N, 3 * N))

    # create Hamiltonian by looping through all difference vectors and finding
    # the corresponding pairs of dipoles
    # the looping limits are such that no calculation is repeated
    # and only dipole pairs with indices where the second one is larger are considered
    # this corresponds to filling the upper (block) triangle in the Hamiltonian

    for m in nb.prange(N2):
        for n_shift in range(2 * N1 - 1):

            n = n_shift - N1 + 1
            if m == 0 and n <= 0:

                continue

            else:

                # get difference vector
                vector = n * lattice_vector1 + m * lattice_vector2

                # length and angle wrt x-axis
                r = np.sqrt(np.sum(vector ** 2))
                theta = np.arctan2(vector[1], vector[0])

                mat_x = np.zeros((3, 3))
                mat_y = np.zeros((3, 3))
                mat_z = np.zeros((3, 3))
                mat_xy = np.zeros((3, 3))
                mat_Dx = np.zeros((3, 3))
                mat_Dy = np.zeros((3, 3))
                Jx, Jy, Jz, Jxy, Dx, Dy = adaptive_integrals(q, z_0, r, theta, sampling_factor, boundary_factor,
                                                                 nodes_and_weights[:, 0], nodes_and_weights[:, 1])

                mat_x[0, 0] = Jx
                mat_y[1, 1] = Jy
                mat_z[2, 2] = Jz
                mat_xy[1, 0] = Jxy
                mat_xy[0, 1] = Jxy
                mat_Dy[0, 2] = Dy
                mat_Dx[1, 2] = -Dx
                mat_Dy[2, 0] = -Dy
                mat_Dx[2, 1] = Dx

                dipole_mat = dipole_field_matrix(vector)

                # find pairs of dipoles with corresponding indices
                for l in range(N2 - m):
                    if n < 0:
                        for k in np.arange(-n, N1, 1):

                            # indices of dipoles
                            i = k + l * N1
                            j = (k + n) + (l + m) * N1

                            h_x[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_x
                            h_y[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_y
                            h_z[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_z
                            h_xy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_xy
                            h_Dx[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dx
                            h_Dy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dy
                            h_dpl[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = dipole_mat

                    else:
                        for k in range(N1 - n):

                            # indices of dipoles
                            i = k + l * N1
                            j = (k + n) + (l + m) * N1

                            h_x[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_x
                            h_y[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_y
                            h_z[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_z
                            h_xy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_xy
                            h_Dx[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dx
                            h_Dy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dy
                            h_dpl[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = dipole_mat

    return h_x, h_y, h_z, h_xy, h_Dx, h_Dy, h_dpl


@nb.njit(parallel=True)
def hamiltonians_fixed_points(coordinates, q, z_0, sampling_factor, boundary_factor, nodes_and_weights):

    # number of dipoles
    N = len(coordinates[0, :])

    # create the Hamiltonians
    h_x = np.zeros((3 * N, 3 * N))
    h_y = np.zeros((3 * N, 3 * N))
    h_z = np.zeros((3 * N, 3 * N))
    h_xy = np.zeros((3 * N, 3 * N))
    h_Dx = np.zeros((3 * N, 3 * N))
    h_Dy = np.zeros((3 * N, 3 * N))
    h_dpl = np.zeros((3 * N, 3 * N))

    # create Hamiltonian by looping through all difference vectors and finding
    # the corresponding pairs of dipoles
    # the looping limits are such that no calculation is repeated
    # and only dipole pairs with indices where the second one is larger are considered
    # this corresponds to filling the upper (block) triangle in the Hamiltonian

    for i in nb.prange(N):
        for j_shifted in range(N - i - 1):

            j = j_shifted + i + 1

            # get difference vector
            vector = np.array([coordinates[0, j] - coordinates[0, i], coordinates[1, j] - coordinates[1, i], 0.])
            r = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
            theta = np.arctan2(vector[1], vector[0])

            mat_x = np.zeros((3, 3))
            mat_y = np.zeros((3, 3))
            mat_z = np.zeros((3, 3))
            mat_xy = np.zeros((3, 3))
            mat_Dx = np.zeros((3, 3))
            mat_Dy = np.zeros((3, 3))
            Jx, Jy, Jz, Jxy, Dx, Dy = adaptive_integrals(q, z_0, r, theta, sampling_factor, boundary_factor,
                                                         nodes_and_weights[:, 0], nodes_and_weights[:, 1])

            mat_x[0, 0] = Jx
            mat_y[1, 1] = Jy
            mat_z[2, 2] = Jz
            mat_xy[1, 0] = Jxy
            mat_xy[0, 1] = Jxy
            mat_Dy[0, 2] = Dy
            mat_Dx[1, 2] = -Dx
            mat_Dy[2, 0] = -Dy
            mat_Dx[2, 1] = Dx

            dipole_mat = dipole_field_matrix(vector)

            h_x[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_x
            h_y[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_y
            h_z[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_z
            h_xy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_xy
            h_Dx[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dx
            h_Dy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dy
            h_dpl[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = dipole_mat

    return h_x, h_y, h_z, h_xy, h_Dx, h_Dy, h_dpl


def hamiltonians_1d_chain_from_couplings(q, z_0, theta, number_of_dipoles, R, radial_res, angle):
    # number of dipoles
    N = number_of_dipoles

    # lattice vector
    lattice_vector = np.array([np.cos(theta), np.sin(theta), 0.])

    # create the Hamiltonians
    h_x = np.zeros((3 * N, 3 * N))
    h_y = np.zeros((3 * N, 3 * N))
    h_z = np.zeros((3 * N, 3 * N))
    h_xy = np.zeros((3 * N, 3 * N))
    h_Dx = np.zeros((3 * N, 3 * N))
    h_Dy = np.zeros((3 * N, 3 * N))
    h_dpl = np.zeros((3 * N, 3 * N))

    # create Hamiltonian by looping through all difference vectors and finding
    # the corresponding pairs of dipoles
    # the looping limits are such that no calculation is repeated
    # and only dipole pairs with indices where the second one is larger are considered
    # this corresponds to filling the upper (block) triangle in the Hamiltonian

    name = "data/couplings_polar_q=" + str(q).replace(".", "-") + "_z0=" + str(z_0).replace(".", "-") \
           + "_rmax=" + str(R) + "_rres=" + str(radial_res)

    data = np.load(name + ".npz")
    int_Jx = data["Jx"]
    int_Jy = data["Jy"]
    int_Jz = data["Jz"]
    int_Jxy = data["Jxy"]
    int_Dx = data["Dx"]
    int_Dy = data["Dy"]

    for i in range(N):
        for j_shifted in range(N - i - 1):

            j = j_shifted + i + 1

            # get difference vector
            vector = (j - i) * lattice_vector
            r = (j - i)

            mat_x = np.zeros((3, 3))
            mat_y = np.zeros((3, 3))
            mat_z = np.zeros((3, 3))
            mat_xy = np.zeros((3, 3))
            mat_Dx = np.zeros((3, 3))
            mat_Dy = np.zeros((3, 3))

            Jx = int_Jx[r, angle]
            Jy = int_Jy[r, angle]
            Jz = int_Jz[r, angle]
            Jxy = int_Jxy[r, angle]
            Dx = int_Dx[r, angle]
            Dy = int_Dy[r, angle]

            mat_x[0, 0] = Jx
            mat_y[1, 1] = Jy
            mat_z[2, 2] = Jz
            mat_xy[1, 0] = Jxy
            mat_xy[0, 1] = Jxy
            mat_Dy[0, 2] = Dy
            mat_Dx[1, 2] = -Dx
            mat_Dy[2, 0] = -Dy
            mat_Dx[2, 1] = Dx

            dipole_mat = dipole_field_matrix(vector)

            h_x[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_x
            h_y[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_y
            h_z[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_z
            h_xy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_xy
            h_Dx[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dx
            h_Dy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dy
            h_dpl[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = dipole_mat

    return h_x, h_y, h_z, h_xy, h_Dx, h_Dy, h_dpl


def hamiltonians_1d_chain_plate(z_0, theta, angle, number_of_dipoles):

    # number of dipoles
    N = number_of_dipoles

    # create the Hamiltonians
    h_x = np.zeros((3 * N, 3 * N))
    h_y = np.zeros((3 * N, 3 * N))
    h_z = np.zeros((3 * N, 3 * N))
    h_xy = np.zeros((3 * N, 3 * N))
    h_Dx = np.zeros((3 * N, 3 * N))
    h_Dy = np.zeros((3 * N, 3 * N))
    h_dpl = np.zeros((3 * N, 3 * N))

    for i in range(N):
        for j_shifted in range(N - i - 1):

            j = j_shifted + i + 1

            # get difference vector
            r = (j - i)

            mat_x = np.zeros((3, 3))
            mat_y = np.zeros((3, 3))
            mat_z = np.zeros((3, 3))
            mat_xy = np.zeros((3, 3))
            mat_Dx = np.zeros((3, 3))
            mat_Dy = np.zeros((3, 3))

            Jx = 4 * np.pi * (3 * r ** 2 * np.cos(2 * theta) + r ** 2 - 8 * z_0 ** 2) / (r ** 2 + 4 * z_0 ** 2) ** (5 / 2) + (8 * np.pi / r ** 3) * (3 * np.cos(theta) ** 2 - 1)
            Jy = 4 * np.pi * (-3 * r ** 2 * np.cos(2 * theta) + r ** 2 - 8 * z_0 ** 2) / (r ** 2 + 4 * z_0 ** 2) ** (5 / 2) + (8 * np.pi / r ** 3) * (3 * np.sin(theta) ** 2 - 1)
            Jz = 8 * np.pi * (r ** 2 - 8 * z_0 ** 2) / (r ** 2 + 4 * z_0 ** 2) ** (5 / 2) - 8 * np.pi / r ** 3
            Jxy = 12 * np.pi * (r ** 2 * np.sin(2 * theta)) / (r ** 2 + 4 * z_0 ** 2) ** (5 / 2) + (8 * np.pi / r ** 3) * 3 * np.sin(theta) * np.cos(theta)
            Dx = -48 * np.pi * r * z_0 * np.sin(theta) / (r ** 2 + 4 * z_0 ** 2) ** (5 / 2)
            Dy =  48 * np.pi * r * z_0 * np.cos(theta) / (r ** 2 + 4 * z_0 ** 2) ** (5 / 2)

            mat_x[0, 0] = Jx
            mat_y[1, 1] = Jy
            mat_z[2, 2] = Jz
            mat_xy[1, 0] = Jxy
            mat_xy[0, 1] = Jxy
            mat_Dy[0, 2] = Dy
            mat_Dx[1, 2] = -Dx
            mat_Dy[2, 0] = -Dy
            mat_Dx[2, 1] = Dx

            h_x[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_x
            h_y[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_y
            h_z[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_z
            h_xy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_xy
            h_Dx[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dx
            h_Dy[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] = mat_Dy

    return h_x, h_y, h_z, h_xy, h_Dx, h_Dy