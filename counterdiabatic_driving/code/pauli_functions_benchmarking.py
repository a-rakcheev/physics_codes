import pauli_string_functions as pauli_func
import numpy as np
import time

L = 10
id_string = np.zeros(L, dtype=np.int8)
x_string = np.ones(L, dtype=np.int8)
y_string = np.full(L, 2, dtype=np.int8)
z_string = np.full(L, 3, dtype=np.int8)

# perform operations for compiling
lab, coeff = pauli_func.multiply_pauli_strings(x_string, y_string, 1.0, 1.0)
lab, coeff = pauli_func.commute_pauli_strings(x_string, y_string, 1.0, 1.0)

start = time.time()
for i in range(100000):
    lab, coeff = pauli_func.multiply_pauli_strings(x_string, z_string, 1.0, 1.0)
end = time.time()
print("Multiply:", end - start)

start = time.time()
for i in range(100000):
    lab, coeff = pauli_func.commute_pauli_strings(x_string, z_string, 1.0, 1.0)
end = time.time()
print("Commute:", end - start)