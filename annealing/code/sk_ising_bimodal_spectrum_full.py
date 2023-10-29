# investigate low-energy spectrum
import sys
sys.path.append("/home/artem/Dropbox/codebase/")
sys.path.append("D:/Dropbox/codebase/")
sys.path.append("C:/Users/rakche_a-adm/Dropbox/codebase/")
sys.path.append("/mnt/d64a440d-7b6c-45ed-aa6c-c9d991212d84/Dropbox/codebase/")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import hamiltonians_32 as ham32
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import tqdm
import time

# entropy
def entropy(probabilities):
    return -np.sum(np.where(probabilities < 1.e-8, 0., probabilities * np.log2(probabilities)))

# parameters
N = 22                                                         # system size
steps = 101                                                     # num of measurements

# instances = np.arange(1, 201, 1)
# instances = [238, 74, 148, 155, 171, 173, 193, 235, 263]
instances_marked = [238, 74]

instances = []
for i in np.arange(260, 261, 1):

    if i in instances_marked:
        continue
    else:
        instances.append(i)
                                                # instance
device = "lenovo"                                                # computer name

if device == "tower":
    prefix = "/mnt/d64a440d-7b6c-45ed-aa6c-c9d991212d84/Dropbox/data/quantum_sweeps/sk_ising_bimodal/data/unique/"
elif device == "office":
    prefix = "C:/Users/rakche_a-adm/Dropbox/data/quantum_sweeps/sk_ising_bimodal/data/unique/"
elif device == "lenovo":
    prefix = "/home/artem/Dropbox/data/quantum_sweeps/sk_ising_bimodal/data/unique/"
else:
    prefix = "./data/unique/spectrum/N" + str(N)

sl = np.linspace(0., 1.0, steps)

for k, inst in enumerate(instances):
    
    # bonds from instance file
    name_bonds = prefix + "sk_ising_bimodal_N" + str(N) + "_unique_inst" + str(inst) + ".txt"
    bonds = np.loadtxt(name_bonds)

    # input for ZZ hamiltonian
    positions_zz = []
    labels_zz = []

    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            positions_zz.append([i, j])
            labels_zz.append([3, 3])

    positions_zz = np.array(positions_zz, dtype=np.int64)
    labels_zz = np.array(labels_zz, dtype=np.int8)

    # transverse field and AFM
    positions_x, labels_x = ham32.input_h_x(N)
    fields_x = np.ones(N)

    h_x = -ham32.operator_sum_real(N, 1, fields_x, positions_x, labels_x)
    h_ising = ham32.operator_sum_real_diag(N, 2, bonds, positions_zz, labels_zz)

    # symmerty sector
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


    # find number of states in first two excited sectors
    idx_sort = np.argsort(h_ising.diagonal())
    spec = h_ising.diagonal()[idx_sort]
    E = spec[0]
    boundary = []

    for k, en in enumerate(spec):

        if en > E:
            boundary.append(k)
            E = en

        if len(boundary) >= 3:
            break

    boundary = np.array(boundary)

    # total number of degenerate states (including the ground state)
    states = boundary[-1]
    print(inst, states)
    
    # find magnetizations of these states
    mags = np.zeros((states, N), dtype=np.int8)
    for n in range(states):

        # extract magnetization
        mag = np.zeros(N)
        for i in range(N):
            mag[N - 1 - i] = np.sign(float((idx_sort[n] & (1 << i)) >> i) - 0.5)

        mags[n, :] = mag


    idx_sort = idx_sort[0:states]
    spec = np.zeros((steps, states))

    ent = np.zeros((steps, states))
    amps = np.zeros((steps, states, states), dtype=np.complex128)
    matel_ising = np.zeros((steps, states, states), dtype=np.complex128)
    matel_x = np.zeros((steps, states, states), dtype=np.complex128)

    # find n lowest states
    for i, s in enumerate(tqdm.tqdm(sl)):
        
        # start = time.time()
        
        if i == steps or i == (steps - 1):
            ev = np.sort(h_ising.diagonal())[0:states]

        else:
            h_tot = (1 - s) * h_x + s * h_ising
            ev, evec = spla.eigsh(h_tot, k=states, which="SA")

        spec[i, :] = ev

        # compute matel and amplitudes
        for n in range(states):

            amps[i, n, :] = evec[idx_sort, n]
            ent[i, n] = entropy(np.absolute(evec[:, n]) ** 2)

            for m in range(n, states):                             
                ml_ising = evec[:, n].T.conj() @ h_ising @ evec[:, m]
                ml_x = evec[:, n].T.conj() @ h_x @ evec[:, m]

                matel_ising[i, n, m] = ml_ising
                matel_ising[i, m, n] = ml_ising.conjugate()

                matel_x[i, n, m] = ml_x
                matel_x[i, m, n] = ml_x.conjugate()

        # end = time.time()
        # print("Time:", end - start)


    # np.save(prefix + "spectrum/N" + str(N) + "/sk_ising_bimodal_low_energy_spectrum_N=" + str(N) + ".npy", spec)
    np.savez_compressed(prefix + "spectrum/N" + str(N) + "/sk_ising_bimodal_second_sector_spectrum_and_matel_N=" + str(N) + "_inst=" + str(inst) + ".npz", spec=spec, sl=sl, mags=mags, matel_x=matel_x, matel_ising=matel_ising, amps=amps, boundary=boundary, ent=ent)
