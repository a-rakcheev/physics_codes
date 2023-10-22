import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
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
gamma = 10
f = 0.0
T = 100

dt = 1.e-4
tl = np.arange(-T/2, T/2, dt)
steps = int(T / dt)

# initial state (ground state of initial ham)
ev, evec = np.linalg.eigh(sigma_x - 0.5 * gamma * sigma_z)
state = evec[:, 1]

# how close to z, - ?
print("Initial State:", np.absolute(state) ** 2)

# measurements
fidelity = np.zeros(steps + 1)
energy_variance = np.zeros(steps + 1)

h_lz = landau_zener_hamiltonian(-T/2, gamma, T)
ev, evec = np.linalg.eigh(h_lz)

fidelity[0] = np.absolute(state.T.conj() @ evec[:, 1]) ** 2
energy_variance[0] = np.sqrt((state.T.conj() @ np.linalg.matrix_power(h_lz, 2) @ state
                      - (state.T.conj() @ h_lz @ state) ** 2).real / gamma ** 2)


# evolve state
start_time = time.time()
for i, t in enumerate(tl):

    h_eff = effective_hamiltonian_first_order(t, dt, f, gamma, T)
    state = la.expm(-1.j * h_eff) @ state

    h_lz = landau_zener_hamiltonian(t + dt, gamma, T)
    ev, evec = np.linalg.eigh(h_lz)

    fidelity[i + 1] = np.absolute(state.T.conj() @ evec[:, 1]) ** 2
    energy_variance[i + 1] = np.sqrt((state.T.conj() @ np.linalg.matrix_power(h_lz, 2) @ state
                          - (state.T.conj() @ h_lz @ state) ** 2).real / gamma ** 2)



end_time = time.time()
print("Time for " + str(steps) + " steps:", end_time - start_time)
print("Final State:", np.absolute(state) ** 2)

plt.figure(1, (6, 4.5))

plt.subplot(1, 2, 1)
plt.plot(np.linspace(-0.5, 0.5, steps + 1), fidelity, color="navy", label=r"$f=" + str(f) + r"$")


plt.grid()
plt.legend()

plt.xlabel(r"$t / T$", fontsize=14)
plt.ylabel(r"$\mathcal{F}$", fontsize=14)

plt.ylim(0., 1.05)


plt.subplot(1, 2, 2)
plt.plot(np.linspace(-0.5, 0.5, steps + 1), energy_variance, color="darkred", label=r"$f=" + str(f) + r"$")


plt.grid()
plt.legend()

plt.xlabel(r"$t / T$", fontsize=14)
plt.ylabel(r"$\Delta E / \gamma$", fontsize=14)

plt.subplots_adjust(wspace=0.5)
plt.show()


