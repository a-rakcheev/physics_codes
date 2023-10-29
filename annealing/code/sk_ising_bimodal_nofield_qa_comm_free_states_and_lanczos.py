# QA of SK Ising with +/- J (bimodal) bonds with degeneracy in the ground state
# use a commutator-free magnus expansion (cfme) for quantum annealing
# here fourth (ans second?) order is implemented based on https://doi.org/10.1016/j.physrep.2008.11.001 [1] and https://www.sciencedirect.com/science/article/pii/S0021999111002300?via%3Dihub [2]
# there are two possibilities depending on whether one wants to split the Hamiltonian in the evaluation or not
# splitting can be useful to exploit the structure, for instance if one term is diagonal, but leads to more matrix exponentials that need to be evaluated

import sys
import numpy as np
import scipy.sparse.linalg as spla
import hamiltonians_32 as ham32
import krylov_methods as kryl
import scipy.sparse as sp
import time
import json


# measure evolved state
# obtain n states with largest probability and save their indices and their amplitudes
def measure_state(state, number_of_states):

    # most probable state
    idx_max = np.argsort(np.absolute(state) ** 2)[-number_of_states:]
    amps_max = state[idx_max]

    return idx_max, amps_max


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
T = float(sys.argv[2])                      # duration
dt = float(sys.argv[3])                     # time step
steps = int(sys.argv[4])                    # num of measurements
inst = int(sys.argv[5])                     # instance
device = str(sys.argv[6])                   # where the computation was performed (home, office, regulus, leo3, leo3e, leo4)
states = int(sys.argv[7])

kryl_err = 1.e-8
kryl_order = 10
lanczos_order = 40
step_size = int((T / dt) / float(steps))

prefix = "./data/unique/"

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


# ground state (x, +)
state = np.full(2 ** (N - 1), 1. / np.sqrt(2) ** (N - 1))

# observables
idx_states = np.zeros((steps + 1, states))
amps_states = np.zeros((steps + 1, states), dtype=np.complex128)
amps_states_inst = np.zeros((steps + 1, lanczos_order), dtype=np.complex128)
ev_inst = np.zeros((steps + 1, lanczos_order))

idx_max, amps_max = measure_state(state, states)
idx_states[0, :] = idx_max
amps_states[0, :] = amps_max


# instantaneous basis
# lanczos decomposition
ev, prob = kryl.lanczos_pro_decomposition_amps(h_x, state, lanczos_order)

# measure
comp_order = len(prob)
if comp_order < lanczos_order:

    amps_states_inst[0, 0:comp_order] = prob
    ev_inst[0, 0:comp_order] = ev
    ev_inst[0, comp_order:] = np.full(lanczos_order - comp_order, np.inf)

else:
    amps_states_inst[0, :] = prob
    ev_inst[0, :] = ev


# time evolution
t = 0.
count = 0
step_counts1_krylov = []
step_counts2_krylov = []
err1_krylov = []
err2_krylov = []

start_time = time.time()
while t < T:

    state, step_count1_4th_order, step_count2_4th_order, err1_4th_order, err2_fourth_order = fourth_order_cfme(h_x, h_ising, state, t, T, dt, order=kryl_order, err=kryl_err)

    step_counts1_krylov.append(step_count1_4th_order)
    step_counts2_krylov.append(step_count2_4th_order)
    err1_krylov.append(err1_4th_order)
    err2_krylov.append(err2_fourth_order)

    # measure state at fixed intervals
    count += 1
    if count % step_size == 0:

        idx_max, amps_max = measure_state(state, states)
        idx_states[count // step_size, :] = idx_max
        amps_states[count // step_size, :] = amps_max

        # instantaneous basis
        # lanczos decomposition
        ev, prob = kryl.lanczos_pro_decomposition_amps((1 - t / T) * h_x + (t / T) * h_ising, state, lanczos_order)

        # measure
        comp_order = len(prob)
        if comp_order < lanczos_order:

            amps_states_inst[count // step_size, 0:comp_order] = prob
            ev_inst[count // step_size, 0:comp_order] = ev
            ev_inst[count // step_size, comp_order:] = np.full(lanczos_order - comp_order, np.inf)

        else:
            amps_states_inst[count // step_size, :] = prob
            ev_inst[count // step_size, :] = ev
    t += dt

end_time = time.time()
runtime = end_time - start_time

step_counts1_krylov = np.array(step_counts1_krylov)
step_counts2_krylov = np.array(step_counts2_krylov)
step_counts = np.concatenate((step_counts1_krylov, step_counts2_krylov))

err1_krylov = np.array(err1_krylov)
err2_krylov = np.array(err2_krylov)
err_krylov = np.concatenate((err1_krylov, err2_krylov))


name = prefix + "quantum_annealing/states_and_lanczos/N" + str(N) + "/sk_ising_bimodal_nofield_states_and_lanczos_N" + str(N) + "_inst" + str(inst) + "_T=" + str(T).replace(".", "-") + "_dt=" + str(dt).replace(".", "-") + "_steps=" + str(steps) + "_states=" + str(states) + "_lanczos_order=" + str(lanczos_order)
np.savez_compressed(name + ".npz", idx_comp=idx_states, amps_comp=amps_states, amps_inst=amps_states_inst, ev_inst=ev_inst)

# metadata
attributes = dict()
attributes["device"] = device
attributes["runtime"] = runtime
attributes["measurements"] = steps
attributes["duration"] = T
attributes["inst"] = inst
attributes["time_step"] = dt
attributes["krylov_order"] = kryl_order
attributes["krylov_error"] = kryl_err
attributes["krylov_step_max"] = int(np.max(step_counts))
attributes["krylov_err_max"] = float(np.max(err_krylov))

with open(name + ".json", "w") as jsfile:
    json.dump(attributes, jsfile, sort_keys=True, indent=4)