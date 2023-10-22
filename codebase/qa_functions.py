import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


# operators and states

def entanglement_entropy(state, spins_right, spins_left):

    size_right = 2 ** spins_right
    size_left = 2 ** spins_left

    matrix = np.reshape(state, (size_left, size_right))
    sval = np.linalg.svd(matrix, compute_uv=False)
    entropy = 0
    for s in sval:

        entropy -= 2. * s ** 2 * np.log(s)

    return entropy


def x_state_down(number_of_spins):

    state = np.array([-1./np.sqrt(2), 1./np.sqrt(2)], dtype=complex)

    for i in range(number_of_spins-1):

        state = np.kron(state, np.array([1./np.sqrt(2), -1./np.sqrt(2)]))

    return state


def x_state_up(number_of_spins):

    state = np.array([1./np.sqrt(2), 1./np.sqrt(2)], dtype=complex)

    for i in range(number_of_spins-1):

        state = np.kron(state, np.array([1./np.sqrt(2), 1./np.sqrt(2)]))

    return state


def y_state_down(number_of_spins):

    state = np.array([-1.j / np.sqrt(2), 1./np.sqrt(2)], dtype=complex)

    for i in range(number_of_spins-1):

        state = np.kron(state, np.array([-1.j / np.sqrt(2), 1./np.sqrt(2)]))

    return state


def y_state_up(number_of_spins):

    state = np.array([1.j / np.sqrt(2), 1./np.sqrt(2)], dtype=complex)

    for i in range(number_of_spins-1):

        state = np.kron(state, np.array([1.j / np.sqrt(2), 1./np.sqrt(2)]))

    return state


def afm_state_1(number_of_spins):

    state = np.array([1., 0.], dtype=complex)

    for i in range(number_of_spins-1):

        if i % 2 == 0:

            state = np.kron(state, np.array([0., 1.]))

        else:

            state = np.kron(state, np.array([1., 0.]))

    return state


def afm_state_2(number_of_spins):
    state = np.array([0., 1.], dtype=complex)

    for i in range(number_of_spins - 1):

        if i % 2 == 0:

            state = np.kron(state, np.array([1., 0.]))

        else:

            state = np.kron(state, np.array([0., 1.]))

    return state


def x_and_z_state(number_of_spins, local_fields):

    if len(local_fields) != number_of_spins:

        print("Local fields not specified for each site")
        return 0

    def state_xz(field):

        state = np.array([-field - np.sqrt(1 + field**2), 1.], dtype=complex)

        # normalize
        # minus sign to have the correct x-state if h=0

        state /= -np.linalg.norm(state, 2)
        return state

    initial_state = state_xz(local_fields[0])
    for h in local_fields[1:]:

        initial_state = np.kron(initial_state, state_xz(h))

    return initial_state


def spin_z(size, position):                # gives matrix for sigma^z_i but as list since only diagonal

    mat = np.zeros(size)                        # Note that spins are counted from right due to binary counting

    for c in range(size):

        d = 2 ** position
        s = c & d
        mat[c] = float(s/d) - 0.5

    return mat


def spin_matrix(number_of_spins):

    size = 2 ** number_of_spins
    matrix = np.zeros((number_of_spins, size))

    for n in range(number_of_spins):

        matrix[n, :] = spin_z(size, number_of_spins - n - 1)

    return matrix


def hamiltonian_z_sparse(number_of_spins, sigma, field):                        # field hamiltonian h*S^z

    size = 2 ** number_of_spins
    diag = field * np.einsum("ij -> j", sigma)

    return sp.diags(diag, offsets=0, shape=(size, size), format="csr")


def hamiltonian_j_sparse(number_of_spins, sigma, boundary_condition):

    size = 2 ** number_of_spins
    if boundary_condition == "obc":

        diag = np.einsum("ij, ij -> j", sigma[0:number_of_spins - 1, :], sigma[1:, :])

    elif boundary_condition == "pbc":

        diag = np.einsum("ij, ij -> j", sigma, np.roll(sigma, 1, axis=0))

    else:

        raise ValueError("Unknown Boundary Conditions")

    return sp.diags(diag, offsets=0, shape=(size, size), format="csr")


def hamiltonian_j_sparse_distance(distance, number_of_spins, sigma, boundary_condition, field):

    size = 2 ** number_of_spins
    if boundary_condition == "obc":

        diag = field * np.einsum("ij, ij -> j", sigma[0:number_of_spins - distance, :], sigma[distance:, :])

    elif boundary_condition == "pbc":

        diag = field * np.einsum("ij, ij -> j", sigma, np.roll(sigma, distance, axis=0))

    else:

        raise ValueError("Unknown Boundary Conditions")

    return sp.diags(diag, offsets=0, shape=(size, size), format="csr")


def spin_x_index_array(spin_index, number_of_spins):

    index = number_of_spins - spin_index
    array = np.arange(0, 2 ** number_of_spins, 1)
    array2 = (2 ** index) * np.ones(2 ** number_of_spins, dtype=int)

    # flip bit corresponding to the index

    array = np.bitwise_xor(array, array2)

    return array


def spin_x_index_matrix(number_of_spins):

    matrix = np.zeros((number_of_spins, 2 ** number_of_spins), dtype=int)
    for j in range(number_of_spins):
        matrix[j, :] = spin_x_index_array(j + 1, number_of_spins)
    return matrix


def spin_x_diagonal(spin_index, number_of_spins):

    index = number_of_spins - spin_index
    pattern = np.concatenate((np.ones(2 ** index), np.zeros(2 ** index)), axis=0)
    diag = np.tile(pattern, 2 ** (number_of_spins - index - 1))
    lenght_of_diag = 2 ** number_of_spins - 2 ** index

    return diag[0:lenght_of_diag]


def spin_x_matrix_sparse(spin_index, number_of_spins, field):

    index = number_of_spins - spin_index
    values = 0.5 * field * spin_x_diagonal(spin_index, number_of_spins)
    matrix = sp.diags([values, values], offsets=[2 ** index, -(2 ** index)], format="csr")

    return matrix


def hamiltonian_x_sparse_from_diag(number_of_spins, field):

    val_array = []
    offset_array = []
    for j in np.arange(1, number_of_spins + 1, 1):

        index = number_of_spins - j
        offset_array.append(2 ** index)
        offset_array.append(-(2 ** index))

        values = 0.5 * field * spin_x_diagonal(j, number_of_spins)
        val_array.append(values)
        val_array.append(values)

    matrix = sp.diags(val_array, offset_array, format="csr")
    return matrix


def hamiltonian_x_sparse(number_of_spins, field):

    mat = sp.lil_matrix((2 ** number_of_spins, 2 ** number_of_spins))

    for n in np.arange(0, number_of_spins, 1):

        k = 2 ** n
        q = 2 ** (number_of_spins - n - 1)
        for i in np.arange(0, k, 1):
            for j in np.arange(0, q, 1):

                mat[i + 2 * j * k, i + 2 * j * k + k] = 0.5 * field
                mat[i + 2 * j * k + k, i + 2 * j * k] = 0.5 * field

    return mat.tocsr()


def magnus_hamiltonian(matrix_1, matrix_2, time, final_time):

    return matrix_1 + (time/final_time) * matrix_2


# matrix exponential

taylor_fifth_order_coefficients = np.array([0.458587880860234998, -0.024360697582562145 - 1.j * 0.317791457983284570,
                                            -0.024360697582562145 + 1.j * 0.317791457983284570,
                                            0.295066757152444645 - 1.j * 0.303014597390897350,
                                            0.295066757152444645 + 1.j * 0.303014597390897350])


def taylor_exponential_matrix(matrix, size, time_step, coefficients):

    mat = sp.identity(size, dtype="complex128", format="csr")
    for m in range(len(coefficients)):

        mat += -1.j * time_step * coefficients[m] * matrix * mat

    return mat


def taylor_evolution(matrix, vector, time_step, coefficients):

    for m in range(len(coefficients)):

        vector += -1.j * time_step * coefficients[m] * matrix.dot(vector)

    return vector


def matrix_exponential_symplectic_splitting(matrix, vector, time_step, coefficients_1, coefficients_2):

    m = len(coefficients_2)     # rank of method
    vec_real = vector.real
    vec_imag = vector.imag
    vector = None

    # main loop
    # q <- q + a_k * t * M * p
    # p <- p - b_k * t * M * q

    for k in range(m):

        vec_real = vec_real + coefficients_1[k] * time_step * matrix.dot(vec_imag)
        vec_imag = vec_imag - coefficients_2[k] * time_step * matrix.dot(vec_real)

    # half stage

    vec_real = vec_real + coefficients_1[m] * time_step * matrix.dot(vec_imag)

    return vec_real + 1.j * vec_imag

# coefficients three stages


a_three_stages = np.zeros(4, dtype=complex)
a_three_stages[0] = (1./96.) * (13. + 1.j * np.sqrt(23.))
a_three_stages[1] = (5./96.) * (7. + 1.j * np.sqrt(23.))
a_three_stages[2] = (5./96.) * (7. - 1.j * np.sqrt(23.))
a_three_stages[3] = (1./96.) * (13. - 1.j * np.sqrt(23.))

b_three_stages = np.zeros(3, dtype=complex)
b_three_stages[0] = (1./30.) * (11. + 1.j * np.sqrt(23.))
b_three_stages[1] = 4./15.
b_three_stages[2] = (1./30.) * (11. - 1.j * np.sqrt(23.))


def matrix_exponential_krylov_sparse(matrix, vector, order, time_step, norm_limit, size):

    qmat = np.zeros((size, order), dtype=complex)
    tmat = np.zeros((order, order))
    qmat[:, 0] = vector
    counter = 0

    for c in range(order - 1):

        r = matrix.dot(qmat[:, c])
        tmat[c, c] = np.vdot(r, qmat[:, c]).real

        if c == 0:

            s = r - tmat[c, c] * qmat[:, c]

        else:

            s = r - tmat[c, c] * qmat[:, c] - tmat[c - 1, c] * qmat[:, c - 1]

        beta = np.linalg.norm(s)

        if beta >= norm_limit:

            tmat[c, c + 1] = beta
            tmat[c + 1, c] = beta
            qmat[:, c + 1] = s/beta
            counter += 1

        else:

            tmat = tmat[0: c + 1, 0: c + 1]
            qmat = qmat[:, 0: c + 1]
            break

    if counter == (order - 1):

        r = matrix.dot(qmat[:, order - 1])
        tmat[order - 1, order - 1] = np.vdot(r, qmat[:, order - 1]).real

    exp = la.expm(-1.j * time_step * tmat)
    mat = np.dot(qmat, exp)
    return mat[:, 0]

# time evolution


def time_evolution_sparse(h_0, h_1, sigma, state, time_step, final_time, write_step,
                          number_of_spins, order, norm_limit, filename1, filename2):

    size = 2 ** number_of_spins
    writefile_spin = open(filename1, "wb")
    writefile_correlation = open(filename2, "wb")
    c = int(round(final_time/time_step))
    counter = 0

    while counter <= c:

        t = counter * time_step

        if counter % write_step == 0:

            print(t)

            spin = np.einsum("j, ij -> i ", np.absolute(state) ** 2, sigma)
            spin.tofile(writefile_spin)

            correlation = np.einsum("ik, jk, k -> ij", sigma - spin[:, None],
                                    sigma - spin[:, None], np.absolute(state) ** 2)
            correlation.tofile(writefile_correlation)

            h_tot = magnus_hamiltonian(h_0, h_1, t, final_time)
            state = matrix_exponential_krylov_sparse(h_tot, state, order, time_step, norm_limit, size)

        else:

            h_tot = magnus_hamiltonian(h_0, h_1, t, final_time)
            state = matrix_exponential_krylov_sparse(h_tot, state, order, time_step, norm_limit, size)

        counter += 1

    return 0


def time_evolution_symplectic(h_0, h_1, sigma, state, time_step, final_time, write_step, filename1, filename2):

    writefile_spin = open(filename1, "wb")
    writefile_correlation = open(filename2, "wb")
    c = int(round(final_time/time_step))
    counter = 0

    while counter <= c:

        t = counter * time_step

        if counter % write_step == 0:

            print(t)

            spin = np.einsum("j, ij -> i ", np.absolute(state) ** 2, sigma)
            spin.tofile(writefile_spin)

            correlation = np.einsum("ik, jk, k -> ij", sigma - spin[:, None],
                                    sigma - spin[:, None], np.absolute(state) ** 2)
            correlation.tofile(writefile_correlation)

            h_tot = magnus_hamiltonian(h_0, h_1, t, final_time)
            state = matrix_exponential_symplectic_splitting(h_tot, state, time_step, a_three_stages, b_three_stages)

        else:

            h_tot = magnus_hamiltonian(h_0, h_1, t, final_time)
            state = matrix_exponential_symplectic_splitting(h_tot, state, time_step, a_three_stages, b_three_stages)

        counter += 1

    return 0


def time_evolution_taylor(h_0, h_1, sigma, state, time_step, final_time,
                          write_step, filename1, filename2):

    writefile_spin = open(filename1, "wb")
    writefile_correlation = open(filename2, "wb")
    c = int(round(final_time/time_step))
    counter = 0

    while counter <= c:

        t = counter * time_step

        if counter % write_step == 0:

            print(t)

            spin = np.einsum("j, ij -> i ", np.absolute(state) ** 2, sigma)
            spin.tofile(writefile_spin)

            correlation = np.einsum("ik, jk, k -> ij", sigma - spin[:, None],
                                    sigma - spin[:, None], np.absolute(state) ** 2)
            correlation.tofile(writefile_correlation)

            h_tot = magnus_hamiltonian(h_0, h_1, t, final_time)
            state = taylor_evolution(h_tot, state, time_step, taylor_fifth_order_coefficients)

        else:

            h_tot = magnus_hamiltonian(h_0, h_1, t, final_time)
            state = taylor_evolution(h_tot, state, time_step, taylor_fifth_order_coefficients)

        counter += 1

    return 0


def entropy_time_evolution_taylor(h_0, h_1, number_of_spins, state, time_step, final_time, write_step, filename):

    writefile = open(filename, "wb")
    c = int(round(final_time/time_step))
    counter = 0

    while counter <= c:

        t = counter * time_step

        if counter % write_step == 0:

            print(t)

            entropy_list = []
            lengths = np.arange(1, number_of_spins, 1)
            for L in lengths:

                entropy_list.append(entanglement_entropy(state, L, number_of_spins - L))

            entropy = np.array(entropy_list)
            entropy.tofile(writefile)

            h_tot = magnus_hamiltonian(h_0, h_1, t, final_time)
            state = taylor_evolution(h_tot, state, time_step, taylor_fifth_order_coefficients)

        else:

            h_tot = magnus_hamiltonian(h_0, h_1, t, final_time)
            state = taylor_evolution(h_tot, state, time_step, taylor_fifth_order_coefficients)

        counter += 1

    return 0


def time_evolution_taylor_full(h_0, h_1, sigma, sigma_x_idx, state, number_of_spins, time_step, final_time,
                          write_step, filename1, filename2, filename3, filename4):

    writefile_spin = open(filename1, "wb")
    writefile_correlation = open(filename2, "wb")
    writefile_entropy = open(filename3, "wb")
    writefile_spin_x = open(filename4, "wb")

    c = int(round(final_time/time_step))
    counter = 0

    while counter <= c:

        t = counter * time_step

        if counter % write_step == 0:

            print(t)

            spin = np.einsum("j, ij -> i ", np.absolute(state) ** 2, sigma)
            spin.tofile(writefile_spin)

            correlation = np.einsum("ik, jk, k -> ij", sigma - spin[:, None],
                                    sigma - spin[:, None], np.absolute(state) ** 2)
            correlation.tofile(writefile_correlation)

            entropy_list = []
            lengths = np.arange(1, number_of_spins, 1)
            for L in lengths:
                entropy_list.append(entanglement_entropy(state, L, number_of_spins - L))

            entropy = np.array(entropy_list)
            entropy.tofile(writefile_entropy)

            spin_x = np.zeros(number_of_spins)
            for i in range(number_of_spins):
                spin_x[i] = 0.5 * np.einsum("i, i -> ", state.conj(), state[sigma_x_idx[i, :]]).real

            spin_x.tofile(writefile_spin_x)

            h_tot = magnus_hamiltonian(h_0, h_1, t, final_time)
            state = taylor_evolution(h_tot, state, time_step, taylor_fifth_order_coefficients)

        else:

            h_tot = magnus_hamiltonian(h_0, h_1, t, final_time)
            state = taylor_evolution(h_tot, state, time_step, taylor_fifth_order_coefficients)

        counter += 1

    return 0


def entropy_semichain_time_evolution_taylor(h_0, h_1, number_of_spins, state, time_step, final_time, write_step,
                                            filename):

    writefile = open(filename, "w")
    c = int(round(final_time/time_step))
    counter = 0

    while counter <= c:

        t = counter * time_step

        if counter % write_step == 0:

            print(t)

            if number_of_spins % 2 == 0:

                L = number_of_spins / 2

            else:

                L = (number_of_spins + 1) / 2

            entropy = entanglement_entropy(state, L, number_of_spins - L)
            writefile.write(str(entropy) + "\n")

            h_tot = magnus_hamiltonian(h_0, h_1, t, final_time)
            state = taylor_evolution(h_tot, state, time_step, taylor_fifth_order_coefficients)

        else:

            h_tot = magnus_hamiltonian(h_0, h_1, t, final_time)
            state = taylor_evolution(h_tot, state, time_step, taylor_fifth_order_coefficients)

        counter += 1

    return 0


def time_evolution_free(hamiltonian, sigma, state, time_step, final_time, write_step, filename1, filename2):

    writefile_spin = open(filename1, "wb")
    writefile_correlation = open(filename2, "wb")
    c = int(round(final_time / time_step))
    counter = 0

    while counter <= c:

        t = counter * time_step

        if counter % write_step == 0:

            print(t)

            spin = np.einsum("j, ij -> i ", np.absolute(state) ** 2, sigma)
            spin.tofile(writefile_spin)

            correlation = np.einsum("ik, jk, k -> ij", sigma - spin[:, None],
                                    sigma - spin[:, None], np.absolute(state) ** 2)
            correlation.tofile(writefile_correlation)

            state = taylor_evolution(hamiltonian, state, time_step, taylor_fifth_order_coefficients)

        else:

            state = taylor_evolution(hamiltonian, state, time_step, taylor_fifth_order_coefficients)

        counter += 1

    return 0


def entropy_time_evolution_free(hamiltonian, state, number_of_spins, time_step, final_time, write_step, filename):

    writefile = open(filename, "wb")
    c = int(round(final_time / time_step))
    counter = 0

    while counter <= c:

        t = counter * time_step

        if counter % write_step == 0:

            print(t)

            entropy_list = []
            lengths = np.arange(1, number_of_spins, 1)
            for L in lengths:

                entropy_list.append(entanglement_entropy(state, L, number_of_spins - L))

            entropy = np.array(entropy_list)
            entropy.tofile(writefile)

            state = taylor_evolution(hamiltonian, state, time_step, taylor_fifth_order_coefficients)

        else:

            state = taylor_evolution(hamiltonian, state, time_step, taylor_fifth_order_coefficients)

        counter += 1

    return 0


def time_evolution_free_full(hamiltonian, sigma, sigma_x_idx, state, number_of_spins, time_step, final_time,
                          write_step, filename1, filename2, filename3, filename4):

    writefile_spin = open(filename1, "wb")
    writefile_correlation = open(filename2, "wb")
    writefile_entropy = open(filename3, "wb")
    writefile_spin_x = open(filename4, "wb")

    c = int(round(final_time / time_step))
    counter = 0

    while counter <= c:

        t = counter * time_step

        if counter % write_step == 0:

            print(t)

            spin = np.einsum("j, ij -> i ", np.absolute(state) ** 2, sigma)
            spin.tofile(writefile_spin)

            correlation = np.einsum("ik, jk, k -> ij", sigma - spin[:, None],
                                    sigma - spin[:, None], np.absolute(state) ** 2)
            correlation.tofile(writefile_correlation)

            entropy_list = []
            lengths = np.arange(1, number_of_spins, 1)
            for L in lengths:
                entropy_list.append(entanglement_entropy(state, L, number_of_spins - L))

            entropy = np.array(entropy_list)
            entropy.tofile(writefile_entropy)

            spin_x = np.zeros(number_of_spins)
            for i in range(number_of_spins):
                spin_x[i] = 0.5 * np.einsum("i, i -> ", state.conj(), state[sigma_x_idx[i, :]]).real

            spin_x.tofile(writefile_spin_x)

            state = taylor_evolution(hamiltonian, state, time_step, taylor_fifth_order_coefficients)

        else:

            state = taylor_evolution(hamiltonian, state, time_step, taylor_fifth_order_coefficients)

        counter += 1

    return 0


# plot classes and functions


class SpinValue:

    def __init__(self, filename, number_of_spins, number_of_time_steps):

        self.file = filename
        self.N = number_of_spins
        self.times = number_of_time_steps

    def information(self):

        print("Spin expectation values from file " + str(self.file))
        print("N = " + str(self.N))
        print(str(self.times) + " time steps")

    def single_spin(self, index, time_steps):

        with open(self.file, "rb") as readfile:

            readfile.seek((time_steps * self.N + index) * 8, 0)
            value = np.fromfile(readfile, dtype=float, count=1)

        return value

    def single_spin_time_series(self, index):

        vector = np.zeros(self.times)

        with open(self.file, "rb") as readfile:
            for t in range(self.times):

                readfile.seek((t * self.N + index) * 8, 0)
                vector[t] = np.fromfile(readfile, dtype=float, count=1)

        return vector

    def single_time_spatial_series(self, time):

        with open(self.file, "rb") as readfile:

            readfile.seek((time * self.N) * 8, 0)
            vector = np.fromfile(readfile, dtype=float, count=self.N)

        return vector

    def space_and_time_series(self):

        matrix = np.zeros((self.times, self.N))

        with open(self.file, "rb") as readfile:
            for i in range(self.times):

                readfile.seek((i * self.N) * 8, 0)
                matrix[i, :] = np.fromfile(readfile, dtype=float, count=self.N)

        return matrix


class CorrelationFunction:

    def __init__(self, filename, number_of_spins, number_of_time_steps):

        self.file = filename
        self.N = number_of_spins
        self.times = number_of_time_steps

    def information(self):

        print("Correlation Function from file " + str(self.file))
        print("N = " + str(self.N))
        print(str(self.times) + " time steps")

    def single_correlation(self, index1, index2, time_steps):

        with open(self.file, "rb") as readfile:

            readfile.seek((time_steps * self.N ** 2 + index1 * self.N + index2) * 8, 0)
            value = np.fromfile(readfile, dtype=float, count=1)

        return value

    def single_correlation_time_series(self, index1, index2):

        vector = np.zeros(self.times)

        with open(self.file, "rb") as readfile:
            for t in range(self.times):

                readfile.seek((t * self.N ** 2 + index1 * self.N + index2) * 8, 0)
                vector[t] = np.fromfile(readfile, dtype=float, count=1)

        return vector

    def single_axis_time_series(self, index1):

        matrix = np.zeros((self.times, self.N))

        with open(self.file, "rb") as readfile:
            for t in range(self.times):

                readfile.seek((t * self.N ** 2 + index1 * self.N) * 8, 0)
                matrix[t, :] = np.fromfile(readfile, dtype=float, count=self.N)

        return matrix

    def single_time_all_space(self, time_steps):

        matrix = np.zeros((self.N, self.N))

        with open(self.file, "rb") as readfile:
            for index1 in range(self.N):

                readfile.seek((time_steps * self.N ** 2 + index1 * self.N) * 8, 0)
                matrix[index1, :] = np.fromfile(readfile, dtype=float, count=self.N)

        return matrix


class StateProbabilities:

    def __init__(self, filename, number_of_spins, number_of_time_steps):

        self.file = filename
        self.N = number_of_spins
        self.times = number_of_time_steps
        self.size = 2 ** self.N

    def information(self):

        print("State Probabilities from file " + str(self.file))
        print("N = " + str(self.N))
        print(str(self.times) + " time steps")

    def single_probability(self, index, time_steps):

        with open(self.file, "rb") as readfile:

            readfile.seek((time_steps * self.size + index) * 8, 0)
            value = np.fromfile(readfile, dtype=float, count=1)

        return value

    def single_state_time_series(self, index):

        vector = np.zeros(self.times)

        with open(self.file, "rb") as readfile:
            for t in range(self.times):

                readfile.seek((t * self.size + index) * 8, 0)
                vector[t] = np.fromfile(readfile, dtype=float, count=1)

        return vector

    def single_time_series(self, time):

        with open(self.file, "rb") as readfile:

            readfile.seek((time * self.size) * 8, 0)
            vector = np.fromfile(readfile, dtype=float, count=self.size)

        return vector

    def space_and_time_series(self):

        matrix = np.zeros((self.times, self.size))

        with open(self.file, "rb") as readfile:
            for i in range(self.times):

                readfile.seek((i * self.size) * 8, 0)
                matrix[i, :] = np.fromfile(readfile, dtype=float, count=self.size)

        return matrix


class EntanglementEntropy:

    def __init__(self, filename, number_of_spins, number_of_time_steps):

        self.file = filename
        self.N = number_of_spins
        self.times = number_of_time_steps

    def information(self):

        print("Entanglement Entropies from file " + str(self.file))
        print("N = " + str(self.N))
        print(str(self.times) + " time steps")

    def single_value(self, index, time_steps):

        with open(self.file, "rb") as readfile:

            readfile.seek((time_steps * (self.N - 1) + index) * 8, 0)
            value = np.fromfile(readfile, dtype=float, count=1)

        return value

    def single_partition_time_series(self, index):

        vector = np.zeros(self.times)

        with open(self.file, "rb") as readfile:
            for t in range(self.times):

                readfile.seek((t * (self.N - 1) + index) * 8, 0)
                vector[t] = np.fromfile(readfile, dtype=float, count=1)

        return vector

    def single_time_spatial_series(self, time):

        with open(self.file, "rb") as readfile:

            readfile.seek((time * (self.N - 1)) * 8, 0)
            vector = np.fromfile(readfile, dtype=float, count=(self.N - 1))

        return vector

    def space_and_time_series(self):

        matrix = np.zeros((self.times, (self.N - 1)))

        with open(self.file, "rb") as readfile:
            for i in range(self.times):

                readfile.seek((i * (self.N - 1)) * 8, 0)
                matrix[i, :] = np.fromfile(readfile, dtype=float, count=(self.N - 1))

        return matrix
