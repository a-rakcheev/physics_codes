# SQA of SK Ising with +/- J (bimodal) bonds
# here based on single spin flips

import sys
import numpy as np
import numba as nb
import math
import time
import json


# help functions
# annealing schedule - here time-like coupling as a function of s
@nb.jit(nopython=True)
def J_t(s, beta_sqa):
    return math.log(math.tanh((1 - s) * beta_sqa)) / (2 * beta_sqa)


# acceptance criterion - glauber heat bath rule
@nb.jit(nopython=True, cache=True)
def accept_glauber(energy_diff, beta):
    return 0.5 * (1 - math.tanh(0.5 * beta * energy_diff))


# energy from state and bond matrix - needed since numba does not support int matvec natively
@nb.jit(nopython=True)
def energy_nb(state, bond_matrix):
    en = 0
    for i in range(len(state)):
        for j in range(i + 1, len(state)):

            en += bond_matrix[i, j] * state[i] * state[j]

    return en

# returns index of state, due to spin flip symmetry the smaller index of the state and the spin flipped state is returned
# the bit coding associates the first spin states[0] with the lowest bit (first bit from the right) 
@nb.jit(nopython=True)
def state_index(state, number_of_spins):

    weights = 2 ** np.arange(0, number_of_spins, 1)[::-1]
    idx = np.sum(((state + 1) // 2) * weights)

    return min(idx, 2 ** (number_of_spins) - 1 - idx)

# MC simulation to create all data
@nb.jit(nopython=True)
def mc_simulation(number_of_spins, trotter_steps, beta_sqa, runs, measurements, mc_steps, bond_matrix, seeds):

    # derived params
    sl = np.linspace(0., 1., mc_steps + 1)
    sl = 0.5 * (sl[1:] + sl[0:-1])
    measure_step = mc_steps // measurements

    # count occurence of every state
    data = np.zeros((measurements, 2 ** (number_of_spins - 1), trotter_steps), dtype=np.uint32)

    for run in range(runs):

        # create all random numbers in advance
        # initial state is random
        np.random.seed(seeds[run])

        # random spin flip locations
        # for each step we propose to flip MCS spins at random locations
        spin_flip_indices_real = np.random.randint(0, number_of_spins, size=mc_steps)
        spin_flip_indices_imag = np.random.randint(0, trotter_steps, size=mc_steps)

        # random acceptance probabilities for each flip between 0 and 1
        acceptance_probabilities = np.random.rand(mc_steps)

        # save initial state
        state = 2 * np.random.randint(0, 2, size=(number_of_spins, trotter_steps)).astype(np.int8) - 1

        # MC sampling
        for i in range(mc_steps):

            # update state based on single spin flip
            spin_flip_idx_real = spin_flip_indices_real[i]
            spin_flip_idx_imag = spin_flip_indices_imag[i]

            # compute energy diff
            # ising coupling
            deltaE_ising = -2 * sl[i] * state[spin_flip_idx_real, spin_flip_idx_imag] * np.sum(bond_matrix[spin_flip_idx_real, :] * state[:, spin_flip_idx_imag])

            # sqa coupling
            deltaE_sqa = -2 * J_t(sl[i], beta_sqa) * state[spin_flip_idx_real, spin_flip_idx_imag] * \
                         (state[spin_flip_idx_real, (spin_flip_idx_imag + 1) % trotter_steps] + state[spin_flip_idx_real, (spin_flip_idx_imag - 1) % trotter_steps])

            deltaE = deltaE_ising + deltaE_sqa

            # print("Delta E:", deltaE)
            # print("acceptance prob:", accept_glauber(deltaE, beta_step))

            # flip spin if move accepted
            if accept_glauber(deltaE, beta_sqa) >= acceptance_probabilities[i]:
                state[spin_flip_idx_real, spin_flip_idx_imag] = -state[spin_flip_idx_real, spin_flip_idx_imag]

            # save state at measurement interval
            if i % measure_step == 0:
                for j in range(trotter_steps):
                    data[i // measure_step, state_index(state[:, j], number_of_spins), j] += 1


    return data


# parameters
M = 100                                   # number of measurements (+ initial state) / 100 per default - if changed new dataset in group needs to be created manually before
N = int(sys.argv[1])                      # system size of sk model
n = int(sys.argv[2])                      # trotter steps
R = int(sys.argv[3])                      # how many runs are performed for averaging
device = str(sys.argv[4])                 # where the computation was performed (home, office, regulus, leo3, leo3e, leo4)
inst = int(sys.argv[5])                   # single instance - or change code below
beta = float(sys.argv[6])                 # (inverse) temperature of sk model during the schedule - SQA computes the observables wrt this temperature
MCS = int(sys.argv[7])                    # number of Monte Carlo steps
states = int(sys.argv[8])                 # save probabilities of the most probable states for each replica


# entire run for an instance - as a function for profiling
prefix = "./data/unique/"

# bonds from instance file
name_bonds = prefix + "sk_ising_bimodal_N" + str(N) + "_unique_inst" + str(inst) + ".txt"
bonds = np.loadtxt(name_bonds)

# bond matrix
J = np.zeros((N, N), dtype=np.int8)
count = 0
for i in range(N):
    for j in range(i + 1, N):
        J[i, j] = int(bonds[count])
        count += 1
J += J.T

# create random seeds for the random number generator in the numba code - necessary since np.random.seed() requires an argument in numba, so the real "random" seeding happens here
np.random.seed()
rand_seeds = np.random.randint(low=0, high=1000000000, size=R)

# MC Simulation
start_time = time.time()
data = mc_simulation(number_of_spins=N, trotter_steps=n, beta_sqa=beta / n, runs=R, measurements=M, mc_steps=MCS, bond_matrix=J, seeds=rand_seeds)

end_time = time.time()
runtime = end_time - start_time

# save most probable states
idx_max = np.zeros((M, states, n), dtype=np.uint32)
count_max = np.zeros((M, states, n))

# get n most probable indices and counts
for i in range(n):
    idx_max[:, :, i] = np.argsort(data[:, :, i], axis=1)[:, -states:]
    count_max[:, :, i] = np.take_along_axis(data[:, :, i], idx_max[:, :, i], axis=1)

# save in directory
name = prefix + "simulated_quantum_annealing/states/N" + str(N) + "/sqa_states_inst" + str(inst) + "_MCS" + str(MCS) + "_R=" + str(R) + "_n=" + str(n) + "_beta=" + str(beta).replace(".", "-")
np.savez_compressed(name + ".npz", idx=idx_max, prob=count_max / R)

# metadata
attributes = dict()
attributes["device"] = device
attributes["runtime"] = runtime
attributes["mcs"] = int(MCS)
attributes["measurements"] = M
attributes["trotter_steps"] = n
attributes["beta"] = beta
attributes["runs"] = R
attributes["inst"] = inst
attributes["cluster_size"] = 1

with open(name + ".json", "w") as jsfile:
    json.dump(attributes, jsfile, sort_keys=True, indent=4)
