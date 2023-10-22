import numpy as np
import scipy.linalg as la
import time

# pauli matrices
sigma_x = np.array([[0., 1.], [1., 0.]])
sigma_y = np.array([[0., -1.j], [1.j, 0.]])
sigma_z = np.array([[1., 0.], [0., -1.]])


def landau_zener_hamiltonian(t, gamma, T):

    h_eff = sigma_x + (gamma * t / T) * sigma_z

    return h_eff


def effective_hamiltonian_first_order(t, dt, f, gamma, T):

    h_eff = dt * sigma_x + (gamma * (2 * t * dt + dt ** 2) / (2 * T)) * sigma_z \
            + 0.5 * f * (np.arctan(gamma * t / T) - np.arctan(gamma * (t + dt) / T)) * sigma_y

    return h_eff


# parameters
gamma = 20
dt = 5.e-4
fl = np.linspace(0., 1., 21)
T = 25


# measurements
fidelity = np.zeros_like(fl)
energy_variance = np.zeros_like(fl)

for j, f in enumerate(fl):

    print("f:", f)

    tl = np.arange(-T/2, T/2, dt)
    steps = int(T / dt)

    # initial state (ground state of initial ham)
    ev, evec = np.linalg.eigh(sigma_x - 0.5 * gamma * sigma_z)
    state = evec[:, 1]

    # how close to z, - ?
    print("Initial State:", np.absolute(state) ** 2)


    # evolve state
    start_time = time.time()
    for i, t in enumerate(tl):

        h_eff = effective_hamiltonian_first_order(t, dt, f, gamma, T)
        state = la.expm(-1.j * h_eff) @ state

        h_lz = landau_zener_hamiltonian(t + dt, gamma, T)
        ev, evec = np.linalg.eigh(h_lz)

    end_time = time.time()

    h_lz = landau_zener_hamiltonian(T/2, gamma, T)
    fidelity[j] = np.absolute(state.T.conj() @ evec[:, 1]) ** 2
    energy_variance[j] = (state.T.conj() @ np.linalg.matrix_power(h_lz, 2) @ state
                          - (state.T.conj() @ h_lz @ state) ** 2).real / gamma ** 2

    print("Time for " + str(steps) + " steps:", end_time - start_time, flush=True)
    print(np.absolute(state) ** 2, 1. - fidelity, energy_variance)
    # print("Final State:", np.absolute(state) ** 2)

# np.savez_compressed("lz_scaling_f_gamma=" + str(gamma) + "_T=" + str(T).replace(".", "-")
#                     + "_dt=" + str(dt).replace(".", "-") + ".npz", fidelity=fidelity, energy_variance=energy_variance)



