# QA of SK Ising with +/- J (bimodal) bonds with degeneracy in the ground state
# use a commutator-free magnus expansion (cfme) for quantum annealing
# here fourth (ans second?) order is implemented based on https://doi.org/10.1016/j.physrep.2008.11.001 [1] and https://www.sciencedirect.com/science/article/pii/S0021999111002300?via%3Dihub [2]
# there are two possibilities depending on whether one wants to split the Hamiltonian in the evaluation or not
# splitting can be useful to exploit the structure, for instance if one term is diagonal, but leads to more matrix exponentials that need to be evaluated

import sys
sys.path.append("/data/Dropbox/codebase/")
import numpy as np
import scipy.sparse.linalg as spla
import hamiltonians_32 as ham32
import math
import krylov_methods as kryl
import scipy.sparse as sp
import time
import json

# entropy
def entropy(probabilities):
    return -np.sum(np.where(probabilities < 1.e-8, 0., probabilities * np.log(probabilities)))


# measure evolved state
# measure all two-spin correlation functions
# measure all single spin expectation values
# ground state fidelity
# instantaneous state fidelity
# due to symmetry only m_x, G_xx, G_yy, G_zz, G_yz, G_zy are non-vanishing
def measure_state_zz(state, size, ground_state_idx):

    corr_zz = np.zeros(size * (size - 1) // 2)

    # correlation
    for i in range(size * (size - 1) // 2):

        c = np.sum(ops_zz[i, :] * np.absolute(state) ** 2)
        corr_zz[i] = c.real

    # fidelities
    gs_fidelity = np.absolute(state[ground_state_idx]) ** 2

    # entropy
    ent = entropy(np.absolute(state) ** 2)

    # most probable state
    idx_max = np.argmax(np.absolute(state) ** 2)
    p_max = np.absolute(state[idx_max]) ** 2

    return corr_zz, gs_fidelity, ent, idx_max, p_max



# evolve state without splitting second order (midpoint rule)
def second_order_cfme(ham_x, ham_ising, vec, t, T, dt, order, err):

    # time for gaussian quadrature
    t1 = t + 0.5 * dt

    # corresponding s value
    s1 = t1 / T

    # coefficients for exponential
    c_1_x = (1. - s1)
    c_1_z = s1

    # evolve with exponential
    # vec = spla.expm_multiply(-1.j * dt * (c_1_x * ham_x + c_1_z * ham_ising), vec)
    vec, error_est, converged, step_count = kryl.matrix_exponential_krylov_error_pro((c_1_x * ham_x + c_1_z * ham_ising), vec, order, dt,
                                                                                     err, 0.0, eps=np.finfo(float).eps, eta_power=0.75, acc_power=0.5)

    return vec, step_count


# evolve state without splitting fourth order [2] eq 61 (the expressions were expanded to avoid summing the matrices often)
def fourth_order_cfme(ham_x, ham_ising, vec, t, T, dt, order, err):

    # times for gaussian quadrature
    t1 = t + dt / 6
    t2 = t + 5 * dt / 6

    # corresponding s values
    s1 = t1 / T
    s2 = t2 / T

    # evolve with first exponential
    # vec = spla.expm_multiply(-1.j * 0.5 * dt * ((1 - s1) * ham_x + s1 * ham_ising), vec)

    vec, error_est1, converged1, step_count1 = kryl.matrix_exponential_krylov_error_pro(0.5 * ((1 - s1) * ham_x + s1 * ham_ising), vec,
                                                                                     order, dt, err, 0.0, eps=np.finfo(float).eps, eta_power=0.75, acc_power=0.5)
    vec /= np.linalg.norm(vec)

    # evolve with second exponential
    # vec = spla.expm_multiply(-1.j * 0.5 * dt * ((1 - s2) * ham_x + s2 * ham_ising), vec)
    vec, error_est2, converged2, step_count2 = kryl.matrix_exponential_krylov_error_pro(0.5 * ((1 - s2) * ham_x + s2 * ham_ising), vec,
                                                                                     order, dt, err, 0.0, eps=np.finfo(float).eps, eta_power=0.75, acc_power=0.5)
    vec /= np.linalg.norm(vec)
    return vec, step_count1, step_count2, error_est1, error_est2


# parameters
N = int(sys.argv[1])                        # system size
T_min = float(sys.argv[2])                      # duration
T_max = float(sys.argv[3]) 
dt = float(sys.argv[4])                     # time step (ignored later)
steps = int(sys.argv[5])                    # num of measurements
inst = int(sys.argv[6])                     # instance
device = str(sys.argv[7])                   # where the computation was performed (home, office, regulus, leo3, leo3e, leo4)

kryl_err = 1.e-8
kryl_order = 10
prefix = ""

Tl = np.arange(T_min, T_max, dt)

# correlation operators
ops_zz = np.zeros((N * (N - 1) // 2, 2 ** (N - 1)))

# create operators
count = 0
for i in range(N):
    for j in range(i + 1, N):

        positions = np.array([[i + 1, j + 1]], dtype=np.int64)
        labels = np.array([[3, 3]], dtype=np.int8)

        ops_zz[count, :] = (ham32.operator_sum_real_diag(N, 2, np.ones(1), positions, labels).diagonal())[0:2 ** (N - 1)]

        count += 1

# create Hamiltonians and states
# input for ZZ hamiltonian
positions_zz = []
labels_zz = []

for i in range(1, N + 1):
    for j in range(i + 1, N + 1):

        positions_zz.append([i, j])
        labels_zz.append([3, 3])

positions_zz = np.array(positions_zz, dtype=np.int64)
labels_zz = np.array(labels_zz, dtype=np.int8)

# bonds from instance file
name_bonds = prefix + "sk_ising_bimodal_N" + str(N) + "_unique_inst" + str(inst) + ".txt"
bonds = np.loadtxt(name_bonds)

# transverse field and AFM
positions_x, labels_x = ham32.input_h_x(N)
fields_x = np.ones(N)

h_x = -ham32.operator_sum_real(N, 1, fields_x, positions_x, labels_x)
h_ising = ham32.operator_sum_real_diag(N, 2, bonds, positions_zz, labels_zz)

# transform to spin flip invariant space
h_ising = h_ising[0:2 ** (N - 1), 0:2 ** (N - 1)]
h_x = h_x[0:2 ** (N - 1), 0:2 ** (N - 1)]

# set anti-diagonal for h_x
rows = []
cols = []
vals = []
for i in range(2 ** (N - 1)):
    rows.append(i)
    cols.append(2 ** (N - 1) - 1 - i)
    vals.append(-1.)

h_x_anti = sp.csr_matrix((vals, (rows, cols)), shape=(2 ** (N - 1), 2 ** (N - 1)))

# full h_x
h_x = h_x + h_x_anti
h_x_anti = None

# find ground state index
gs_idx = np.argmin(h_ising.diagonal())


for T in Tl:

    # set dt as T / 100
    dt = T / 100

    # ground state (x, +)
    ground_state = np.full(2 ** (N - 1), 1. / np.sqrt(2) ** (N - 1))

    # observables
    corrs_zz = np.zeros((steps + 1, (N * (N - 1)) // 2))
    fids_gs = np.zeros(steps + 1)
    entropies = np.zeros(steps + 1)
    idx_max_meas = np.zeros(steps + 1)
    p_max_meas = np.zeros(steps + 1)

    corr_zz, fid_gs, ent, idx_max, p_max = measure_state_zz(ground_state, N, gs_idx)
    corrs_zz[0, :] = corr_zz
    fids_gs[0] = fid_gs / 2
    entropies[0] = ent + np.log(2)
    idx_max_meas[0] = idx_max
    p_max_meas[0] = p_max / 2

    # time evolution
    t = 0.
    count = 0
    step_counts1_krylov = []
    step_counts2_krylov = []
    err1_krylov = []
    err2_krylov = []

    start_time = time.time()
    for i in range(steps):

        ground_state, step_count1_4th_order, step_count2_4th_order, err1_4th_order, err2_fourth_order = fourth_order_cfme(h_x, h_ising, ground_state, t, T, dt, order=kryl_order, err=kryl_err)

        step_counts1_krylov.append(step_count1_4th_order)
        step_counts2_krylov.append(step_count2_4th_order)
        err1_krylov.append(err1_4th_order)
        err2_krylov.append(err2_fourth_order)

        # measure state at fixed intervals
        count += 1

        corr_zz, fid_gs, ent, idx_max, p_max = measure_state_zz(ground_state, N, gs_idx)
        corrs_zz[count, :] = corr_zz
        fids_gs[count] = fid_gs / 2
        entropies[count] = ent + np.log(2)
        idx_max_meas[count] = idx_max
        p_max_meas[count] = p_max / 2

        t += dt

    end_time = time.time()
    runtime = end_time - start_time

    # np.set_printoptions(3, linewidth=150)
    # print(T, runtime)
    # print(corrs_zz[:, 0])
    # print(corrs_zz[:, 10])
    # print(fids_gs)
    # print(entropies)
    # print(p_max_meas)

    step_counts1_krylov = np.array(step_counts1_krylov)
    step_counts2_krylov = np.array(step_counts2_krylov)
    step_counts = np.concatenate((step_counts1_krylov, step_counts2_krylov))

    err1_krylov = np.array(err1_krylov)
    err2_krylov = np.array(err2_krylov)
    err_krylov = np.concatenate((err1_krylov, err2_krylov))

    # filenames
    name = prefix + "sk_ising_bimodal_nofield_N" + str(N) + "_inst" + str(inst) + "_T=" + str(T).replace(".", "-") + "_dt=" + str(dt).replace(".", "-") + "_steps=" + str(steps)
    np.savez_compressed(name + ".npz", corrs=corrs_zz, fids_gs=fids_gs, ent=entropies, idx_max=idx_max_meas, p_max=p_max_meas)

    # # metadata
    # attributes = dict()
    # attributes["device"] = device
    # attributes["runtime"] = runtime
    # attributes["measurements"] = int(steps)
    # attributes["duration"] = int(T)
    # attributes["inst"] = int(inst)
    # attributes["time_step"] = dt
    # attributes["krylov_order"] = int(kryl_order)
    # attributes["krylov_error"] = int(kryl_err)
    # attributes["krylov_step_max"] = int(np.max(step_counts))
    # attributes["krylov_err_max"] = float(np.max(err_krylov))

    # with open(name + ".json", "w") as jsfile:
    #     json.dump(attributes, jsfile, sort_keys=True, indent=4)
