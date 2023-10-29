# investigate low-energy spectrum
import sys
sys.path.append("/home/artem/Dropbox/codebase/")
sys.path.append("D:/Dropbox/codebase/")
sys.path.append("E:/Dropbox/codebase/")
sys.path.append("/run/media/artem/d64a440d-7b6c-45ed-aa6c-c9d991212d84/Dropbox/codebase/")

import numpy as np
import hamiltonians_32 as ham32
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import tqdm

# parameters
N = 18                                                          # system size
steps = 100                                                     # num of measurements
instances = np.arange(101, 201, 1)                                # instance
states = 10000                                                 # number of low-energy states
device = "office"                                                # computer name

if device == "tower":
    prefix = "/run/media/artem/d64a440d-7b6c-45ed-aa6c-c9d991212d84/Dropbox/data/quantum_sweeps/sk_ising_bimodal/data/unique/"
elif device == "office":
    prefix = "E:/Dropbox/data/quantum_sweeps/sk_ising_bimodal/data/unique/"
else:
    prefix = "./data/unique/"


for k, inst in tqdm.tqdm(enumerate(instances)):
    
    # hamiltonian
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

    h_ising = ham32.operator_sum_real_diag(N, 2, bonds, positions_zz, labels_zz).diagonal()

    # find k lowest states
    idx_sort = np.argsort(h_ising)[0:states]
    spec = h_ising[idx_sort]

    # compute correlations
    mags = np.zeros((states, N), dtype=np.int8)
    corrs = np.zeros((states, N * (N - 1) // 2), dtype=np.int8)

    for s in range(states):

        # extract magnetization
        mag = np.zeros(N)
        for i in range(N):
            mag[N - 1 - i] = np.sign(float((idx_sort[s] & (1 << i)) >> i) - 0.5)

        mags[s, :] = mag

        # obtain correlations
        count = 0
        for i in range(N):
            for j in range(i + 1, N):
                corrs[s, count] = mag[i] * mag[j]
                count += 1


    np.savez_compressed(prefix + "/spectrum/N" + str(N) + "/sk_ising_bimodal_spectrum_analysis_N=" + str(N) + "_inst=" + str(inst) + ".npz", mags=mags, en=spec)
