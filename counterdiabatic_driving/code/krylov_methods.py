# krylov methods of various degree of sophistication
# all compute the matrix exponential applied to a vector for hermitian matrices
# most take an initial vector and a sparse matrix as input, but could be made matrix free

import random
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from scipy.integrate import romb
import matplotlib.pyplot as plt
import time
random.seed(7)

def lanczos_pro(matrix, vector, order, eps=np.finfo(float).eps, eta_power=0.75, acc_power=0.5):

    acc = eps ** acc_power                      # accuracy for semi-orthogonality
    eta = eps ** eta_power                      # accuracy for reorthogonalization batch
    size = len(vector)

    omega = [[1.]]
    omega_est = [[1.]]

    qmat = np.zeros((size, order), dtype=np.complex128)
    tmat = np.zeros((order, order))
    qmat[:, 0] = vector

    ro_idx_start = 0
    ro_idx_end = 0
    ro_next = 0

    reo_count = []

    for i in range(order - 1):

        if ro_next == 0:

            r = matrix.dot(qmat[:, i])
            tmat[i, i] = (r.T.conj() @ qmat[:, i]).real

            if i == 0:

                s = r - tmat[i, i] * qmat[:, i]

            else:

                s = r - tmat[i, i] * qmat[:, i] - tmat[i - 1, i] * qmat[:, i - 1]

            beta = np.sqrt(s.T.conj() @ s).real
            tmat[i, i + 1] = beta
            tmat[i + 1, i] = beta
            qmat[:, i + 1] = s / beta

            # estimate dot products based on H. Simon paper
            dot_est = []

            # print(i)
            if i > 0:
                # inner loop
                for j in range(i):

                    # print("j", j)
                    theta = random.normalvariate(0, 0.3) * eps * (beta + tmat[j + 1, j + 2])          # random number
                    term1 = (tmat[j, j] - tmat[i, i]) * omega_est[i][j]
                    term2 = omega_est[i][j + 1] * tmat[j, j + 1]
                    term3 = -omega_est[i - 1][j] * tmat[i - 1, i]
                    recursion_sum = term1 + term2 + term3

                    if j > 0:
                        term4 = omega_est[i][j - 1] * tmat[j - 1, j]
                        recursion_sum += term4

                    dot_est.append(recursion_sum / beta + theta)

            # check if reorthogonalization needed
            print(i)
            for w_check in dot_est:

                if np.absolute(w_check) >= acc:

                    print("reorthogonalizing")
                    ro_next = 1

                    # search indices for reorthogonalization
                    # start
                    for k, w in enumerate(dot_est):
                        if np.absolute(w) >= eta:
                            ro_idx_start = k
                            break

                    # end
                    for k, w in enumerate(dot_est[::-1]):
                        if np.absolute(w) >= eta:
                            ro_idx_end = len(dot_est) - k
                            break

                    reo_count.append([i, ro_idx_start, ro_idx_end])

                    # reorthogonalize selected vectors and update beta and scalar products
                    for l in range(ro_idx_start, ro_idx_end, 1):

                        s = s - qmat[:, l] * (qmat[:, l].T.conj() @ s)
                        dot_est[l] = eps * random.normalvariate(0., 1.5)

                    beta = np.sqrt(s.T.conj() @ s).real
                    tmat[i, i + 1] = beta
                    tmat[i + 1, i] = beta
                    qmat[:, i + 1] = s / beta

                break

            # outer appends
            # inner product with previous vector
            psi = random.normalvariate(0, 0.6)  # random number
            dot_est.append(size * eps * psi * (tmat[0, 1] / beta))

            # inner product with self
            dot_est.append(1.)
            omega_est.append(dot_est)

            dot_exact = []
            for j in range(i + 1):
                q = qmat[:, j]
                dot_exact.append(np.absolute(qmat[:, i + 1].T.conj() @ q))

            dot_exact.append(1.)
            omega.append(dot_exact)

        # second step of reorthogonalization
        else:

            ro_next = 0
            r = matrix.dot(qmat[:, i])
            tmat[i, i] = (r.T.conj() @ qmat[:, i]).real
            s = r - tmat[i, i] * qmat[:, i] - tmat[i - 1, i] * qmat[:, i - 1]

            # reorthogonalize selected vectors and update beta and scalar products
            for l in range(ro_idx_start, ro_idx_end, 1):
                s = s - qmat[:, l] * (qmat[:, l].T.conj() @ s)

            beta = np.sqrt(s.T.conj() @ s).real
            tmat[i, i + 1] = beta
            tmat[i + 1, i] = beta
            qmat[:, i + 1] = s / beta


            # estimate dot products based on H. Simon paper
            dot_est = []

            # print(i)
            if i > 0:
                # inner loop
                for j in range(i):

                    if j in range(ro_idx_start, ro_idx_end, 1):

                        dot_est.append(eps * random.normalvariate(0., 1.5))

                    else:

                        # print("j", j)
                        theta = random.normalvariate(0, 0.3) * eps * (beta + tmat[j + 1, j + 2])  # random number
                        term1 = (tmat[j, j] - tmat[i, i]) * omega_est[i][j]
                        term2 = omega_est[i][j + 1] * tmat[j, j + 1]
                        term3 = -omega_est[i - 1][j] * tmat[i - 1, i]
                        recursion_sum = term1 + term2 + term3

                        if j > 0:
                            term4 = omega_est[i][j - 1] * tmat[j - 1, j]
                            recursion_sum += term4

                        dot_est.append(recursion_sum / beta + theta)

            # outer appends
            # inner product with previous vector
            psi = random.normalvariate(0, 0.6)  # random number
            dot_est.append(size * eps * psi * (tmat[0, 1] / beta))

            # inner product with self
            dot_est.append(1.)
            omega_est.append(dot_est)
            ro_idx_start = 0
            ro_idx_end = 0

            dot_exact = []
            for j in range(i + 1):
                q = qmat[:, j]
                dot_exact.append(np.absolute(qmat[:, i + 1].T.conj() @ q))

            dot_exact.append(1.)
            omega.append(dot_exact)

    # last step
    r = matrix.dot(qmat[:, order - 1])
    tmat[order - 1, order - 1] = np.vdot(r, qmat[:, order - 1]).real
    return qmat, tmat, omega, omega_est, reo_count


# the simple m-step Krylov method - no convergence, memory, iteration criteria
# no re-orthogonalization

def matrix_exponential_krylov_m_step(matrix, vector, order, time):

    size = len(vector)
    qmat = np.zeros((size, order), dtype=np.complex128)
    tmat = np.zeros((order, order))
    qmat[:, 0] = vector
    for i in range(order - 1):

        r = matrix.dot(qmat[:, i])
        tmat[i, i] = (r.T.conj() @ qmat[:, i]).real

        if i == 0:

            s = r - tmat[i, i] * qmat[:, i]

        else:

            s = r - tmat[i, i] * qmat[:, i] - tmat[i - 1, i] * qmat[:, i - 1]

        beta = np.linalg.norm(s)

        tmat[i, i + 1] = beta
        tmat[i + 1, i] = beta
        qmat[:, i + 1] = s / beta

    # last step
    r = matrix.dot(qmat[:, order - 1])
    tmat[order - 1, order - 1] = np.vdot(r, qmat[:, order - 1]).real

    # compute exponential of small matrix
    ev, evec = la.eigh_tridiagonal(np.diag(tmat), np.diag(tmat, 1))

    exp_ev = np.exp(-1.j * time * ev)
    exp = evec @ np.diag(exp_ev) @ evec.T.conj()
    mat = qmat @ exp
    return mat[:, 0]


# krylov algorithm with error bound see arxiv:1603.07358v1
# and maximal iteration bound, full matrix for all iterations will be pre-created
# no re-orthogonalization

def error_integral(ev, evec, final_time, integration_order):

    vals = np.zeros(integration_order)
    tl = np.linspace(0., final_time, integration_order)
    for j, t in enumerate(tl):
        exp_ev = np.exp(-1.j * t * ev)
        exp_mat = evec @ np.diag(exp_ev) @ evec.T.conj()
        vals[j] = np.absolute(exp_mat[-1, 0])

    integral = romb(vals, tl[1] - tl[0])
    return integral


def matrix_exponential_krylov_error_estimate(matrix, vector, error, order, time):

    size = len(vector)
    qmat = np.zeros((size, order), dtype=np.complex128)
    tmat = np.zeros((order, order))
    qmat[:, 0] = vector
    counter = 0
    error_est = 0.
    converged = False

    for i in range(order - 1):

        r = matrix.dot(qmat[:, i])
        tmat[i, i] = (r.T.conj() @ qmat[:, i]).real

        if i == 0:

            s = r - tmat[i, i] * qmat[:, i]

        else:

            s = r - tmat[i, i] * qmat[:, i] - tmat[i - 1, i] * qmat[:, i - 1]

        beta = np.linalg.norm(s)

        # compute error estimate
        ev, evec = la.eigh_tridiagonal(np.diag(tmat[0: i + 1, 0: i + 1]), np.diag(tmat[0: i + 1, 0: i + 1], 1))
        error_est = beta * error_integral(ev, evec, time, 2 ** 10 + 1)

        if error_est > error:

            tmat[i, i + 1] = beta
            tmat[i + 1, i] = beta
            qmat[:, i + 1] = s / beta
            counter += 1

        else:

            converged = True
            tmat = tmat[0: i + 1, 0: i + 1]
            qmat = qmat[:, 0: i + 1]
            break

    # last step, if not converged
    if counter == order - 1:
        r = matrix.dot(qmat[:, order - 1])
        tmat[order - 1, order - 1] = np.vdot(r, qmat[:, order - 1]).real

        ev, evec = la.eigh_tridiagonal(np.diag(tmat), np.diag(tmat, 1))

    # compute exponential of small matrix
    exp_ev = np.exp(-1.j * time * ev)
    exp = evec @ np.diag(exp_ev) @ evec.T.conj()
    mat = qmat @ exp
    return mat[:, 0], error_est, converged


def matrix_exponential_krylov_error_pro(matrix, vector, order, time, error,
                                        eps=np.finfo(float).eps, eta_power=0.75, acc_power=0.5):


    converged = False
    converged_early = False
    error_est = 1.
    acc = eps ** acc_power                      # accuracy for semi-orthogonality
    eta = eps ** eta_power                      # accuracy for reorthogonalization batch
    size = len(vector)

    omega = [[1.]]
    omega_est = [[1.]]

    qmat = np.zeros((size, order), dtype=np.complex128)
    tmat = np.zeros((order, order))
    qmat[:, 0] = vector

    ro_idx_start = 0
    ro_idx_end = 0
    ro_next = 0

    reo_count = []

    for i in range(order - 1):

        if converged:
            tmat = tmat[0: i, 0: i]
            qmat = qmat[:, 0: i]
            converged_early = True
            break

        if ro_next == 0:

            r = matrix.dot(qmat[:, i])
            tmat[i, i] = (r.T.conj() @ qmat[:, i]).real

            if i == 0:

                s = r - tmat[i, i] * qmat[:, i]

            else:

                s = r - tmat[i, i] * qmat[:, i] - tmat[i - 1, i] * qmat[:, i - 1]

            beta = np.sqrt(s.T.conj() @ s).real
            tmat[i, i + 1] = beta
            tmat[i + 1, i] = beta
            qmat[:, i + 1] = s / beta

            # estimate dot products based on H. Simon paper
            dot_est = []

            # print(i)
            if i > 0:
                # inner loop
                for j in range(i):

                    # print("j", j)
                    theta = random.normalvariate(0, 0.3) * eps * (beta + tmat[j + 1, j + 2])          # random number
                    term1 = (tmat[j, j] - tmat[i, i]) * omega_est[i][j]
                    term2 = omega_est[i][j + 1] * tmat[j, j + 1]
                    term3 = -omega_est[i - 1][j] * tmat[i - 1, i]
                    recursion_sum = term1 + term2 + term3

                    if j > 0:
                        term4 = omega_est[i][j - 1] * tmat[j - 1, j]
                        recursion_sum += term4

                    dot_est.append(recursion_sum / beta + theta)

            # check if reorthogonalization needed
            # print(i)
            for w_check in dot_est:

                if np.absolute(w_check) >= acc:

                    # print("reorthogonalizing")
                    ro_next = 1

                    # search indices for reorthogonalization
                    # start
                    for k, w in enumerate(dot_est):
                        if np.absolute(w) >= eta:
                            ro_idx_start = k
                            break

                    # end
                    for k, w in enumerate(dot_est[::-1]):
                        if np.absolute(w) >= eta:
                            ro_idx_end = len(dot_est) - k
                            break

                    reo_count.append([i, ro_idx_start, ro_idx_end])

                    # reorthogonalize selected vectors and update beta and scalar products
                    for l in range(ro_idx_start, ro_idx_end, 1):

                        s = s - qmat[:, l] * (qmat[:, l].T.conj() @ s)
                        dot_est[l] = eps * random.normalvariate(0., 1.5)

                    beta = np.sqrt(s.T.conj() @ s).real
                    tmat[i, i + 1] = beta
                    tmat[i + 1, i] = beta
                    qmat[:, i + 1] = s / beta

                break

            # outer appends
            # inner product with previous vector
            psi = random.normalvariate(0, 0.6)  # random number
            dot_est.append(size * eps * psi * (tmat[0, 1] / beta))

            # inner product with self
            dot_est.append(1.)
            omega_est.append(dot_est)

            dot_exact = []
            for j in range(i + 1):
                q = qmat[:, j]
                dot_exact.append(np.absolute(qmat[:, i + 1].T.conj() @ q))

            dot_exact.append(1.)
            omega.append(dot_exact)

        # second step of reorthogonalization
        else:

            ro_next = 0
            r = matrix.dot(qmat[:, i])
            tmat[i, i] = (r.T.conj() @ qmat[:, i]).real
            s = r - tmat[i, i] * qmat[:, i] - tmat[i - 1, i] * qmat[:, i - 1]

            # reorthogonalize selected vectors and update beta and scalar products
            for l in range(ro_idx_start, ro_idx_end, 1):
                s = s - qmat[:, l] * (qmat[:, l].T.conj() @ s)

            beta = np.sqrt(s.T.conj() @ s).real
            tmat[i, i + 1] = beta
            tmat[i + 1, i] = beta
            qmat[:, i + 1] = s / beta


            # estimate dot products based on H. Simon paper
            dot_est = []

            # print(i)
            if i > 0:
                # inner loop
                for j in range(i):

                    if j in range(ro_idx_start, ro_idx_end, 1):

                        dot_est.append(eps * random.normalvariate(0., 1.5))

                    else:

                        # print("j", j)
                        theta = random.normalvariate(0, 0.3) * eps * (beta + tmat[j + 1, j + 2])  # random number
                        term1 = (tmat[j, j] - tmat[i, i]) * omega_est[i][j]
                        term2 = omega_est[i][j + 1] * tmat[j, j + 1]
                        term3 = -omega_est[i - 1][j] * tmat[i - 1, i]
                        recursion_sum = term1 + term2 + term3

                        if j > 0:
                            term4 = omega_est[i][j - 1] * tmat[j - 1, j]
                            recursion_sum += term4

                        dot_est.append(recursion_sum / beta + theta)

            # outer appends
            # inner product with previous vector
            psi = random.normalvariate(0, 0.6)  # random number
            dot_est.append(size * eps * psi * (tmat[0, 1] / beta))

            # inner product with self
            dot_est.append(1.)
            omega_est.append(dot_est)
            ro_idx_start = 0
            ro_idx_end = 0

            dot_exact = []
            for j in range(i + 1):
                q = qmat[:, j]
                dot_exact.append(np.absolute(qmat[:, i + 1].T.conj() @ q))

            dot_exact.append(1.)
            omega.append(dot_exact)

        # compute error estimate
        ev, evec = la.eigh_tridiagonal(np.diag(tmat[0: i + 1, 0: i + 1]), np.diag(tmat[0: i + 1, 0: i + 1], 1))
        error_est = beta * error_integral(ev, evec, time, 2 ** 10 + 1)
        if error_est <= error:
            converged = True

    if converged_early == False:
        # last step
        r = matrix.dot(qmat[:, order - 1])
        tmat[order - 1, order - 1] = np.vdot(r, qmat[:, order - 1]).real
        ev, evec = la.eigh_tridiagonal(np.diag(tmat), np.diag(tmat, 1))

    # compute exponential of small matrix
    exp_ev = np.exp(-1.j * time * ev)
    exp = evec @ np.diag(exp_ev) @ evec.T.conj()
    mat = qmat @ exp

    return mat[:, 0], error_est, converged


# test with random matrix and vector
N = 1000
steps = 100
norm = 1000
tl = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
error = []
error_estimate = []
convergence = []

rand_vec = np.random.rand(N) + 1.j * np.random.rand(N)
rand_vec /= np.linalg.norm(rand_vec)

rand_mat = np.random.rand(N, N) + 1.j * np.random.rand(N, N)
rand_mat = 0.5 * (rand_mat + rand_mat.T.conj())
rand_mat *= norm / np.linalg.norm(rand_mat, 2)

# sparsify matrix
rand_mat = sp.csc_matrix(rand_mat)


for t in tl:

    print(t)
    start_time = time.time()
    rand_exp = la.expm(-1.j * t * rand_mat)
    vec_evolved = rand_exp @ rand_vec
    end_time = time.time()
    print("Diag:", end_time - start_time)


    start_time = time.time()

    # vec_evolved_krylov = matrix_exponential_krylov_m_step(rand_mat, rand_vec, steps, t)
    # vec_evolved_krylov, err, conv = matrix_exponential_krylov_error_estimate(rand_mat, rand_vec, 1.e-3, steps, t)
    vec_evolved_krylov, err, conv = matrix_exponential_krylov_error_pro(rand_mat, rand_vec, steps, t, 1.e-6)

    end_time = time.time()

    error.append(np.linalg.norm(vec_evolved - vec_evolved_krylov))
    error_estimate.append(err)
    convergence.append(conv)

    print("Krylov:", end_time - start_time)

plt.plot(tl, error, marker="o", color="navy", ls="--", label="True")
plt.plot(tl, error_estimate, marker="s", color="darkred", ls="-.", label="Estimate")

plt.xlabel(r"$t$", fontsize=12)
plt.ylabel=("Error")
plt.grid()
plt.legend()
plt.yscale("log")
plt.show()

# m = 1000
# n = m + 1
#
# np.set_printoptions(linewidth=150)
#
# # test matrices
#
# # squares
# # test_mat = np.diag(np.arange(1, n + 1, 1) ** 2)
#
# test_mat = np.zeros((n, n))
#
# # # fractions
# # for i in range(n):
# #
# #     if i == 0:
# #         test_mat[i, i] = 0.
# #     else:
# #         test_mat[i, i] = n / i
#
# # linear
# for i in range(n):
#
#     if i == 0:
#         test_mat[i, i] = 100
#     else:
#         test_mat[i, i] = 50.0 - 0.5 * i
#
# test_vec = np.ones(n) / np.sqrt(n)
# Q, T, w, w_est, rcount = lanczos_pro(test_mat, test_vec, m)
#
# print(rcount)
# # print(np.diag(T).astype(int))
# # print(np.diag(T, 1).astype(int))
# # print(np.absolute(Q[:, 0].T.conj() @ Q[:, -1]))
# meps = np.finfo(float).eps
# dots = []
# dots_est = []
# for ar in w:
#     dots.append(ar[0])
#
# for ar in w_est:
#     dots_est.append(np.absolute(ar[0]))
#
# plt.plot(dots[1:], marker="o", ls="--")
# plt.plot(dots_est[1:], marker="s", ls="--")
#
# plt.grid()
# plt.yscale("log")
# plt.ylim(1.e-16)
# plt.show()