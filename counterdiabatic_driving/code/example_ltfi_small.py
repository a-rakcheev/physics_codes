import pauli_string_functions as pauli_func
import numpy as np
import scipy.linalg as la
from scipy.sparse import save_npz, csr_matrix
import zipfile
import io


def trace_of_product_TP(label_1, coeff_1, label_2, coeff_2):

    label_3, coeff_3 = pauli_func.multiply_operators_TP(label_1, label_2, coeff_1, coeff_2)
    return pauli_func.trace_operator_TP(label_3, coeff_3)

def Pmatrix(h, g):

    mat = np.zeros((3, 3))
    mat[0, 0] = 2048 + 512 * (h ** 2 + g ** 2)
    mat[1, 1] = 512 * g ** 2 + 2048 * h ** 2
    mat[2, 2] = 2048 + 512 * (h ** 2 + 4 * g ** 2)
    mat[0, 1] = 1024 * g
    mat[1, 0] = mat[0, 1]
    mat[0, 2] = -2048 * h
    mat[2, 0] = mat[0, 2]
    mat[1, 2] = -1536 * h * g
    mat[2, 1] = mat[1, 2]

    return mat


def RvectorZ(h, g):
    vec = np.zeros(3)
    vec[0] = -256 * g
    return vec

def RvectorX(h, g):
    vec = np.zeros(3)
    vec[0] = 256 * h
    vec[2] = -512
    return vec

def commutator_matrix(mat1, mat2):

    return mat1 @ mat2 - mat2 @ mat1

def check_matrices(h, g):

    P = Pmatrix(h, g)
    R_Z = RvectorZ(h, g)
    R_X = RvectorX(h, g)

    P_num = np.zeros((3, 3))
    R_Z_num = np.zeros(3)
    R_X_num = np.zeros(3)

    # fill R matrix

    # Z direction
    term_zz_lab, term_zz_c = pauli_func.multiply_operators_TP(lab_z, comm_lab_zz_y, c_z, comm_c_zz_y)
    term_z_lab, term_z_c = pauli_func.multiply_operators_TP(lab_z, comm_lab_z_y, c_z, comm_c_z_y)
    term_x_lab, term_x_c = pauli_func.multiply_operators_TP(lab_z, comm_lab_x_y, c_z, comm_c_x_y)

    R_Z_num[0] = (-2.j * (pauli_func.trace_operator_TP(term_zz_lab, term_zz_c)
                    - h * pauli_func.trace_operator_TP(term_z_lab, term_z_c)
                    - g * pauli_func.trace_operator_TP(term_x_lab, term_x_c))).real

    term_zz_lab, term_zz_c = pauli_func.multiply_operators_TP(lab_z, comm_lab_zz_xy, c_z, comm_c_zz_xy)
    term_z_lab, term_z_c = pauli_func.multiply_operators_TP(lab_z, comm_lab_z_xy, c_z, comm_c_z_xy)
    term_x_lab, term_x_c = pauli_func.multiply_operators_TP(lab_z, comm_lab_x_xy, c_z, comm_c_x_xy)

    R_Z_num[1] = (-2.j * (pauli_func.trace_operator_TP(term_zz_lab, term_zz_c)
                    - h * pauli_func.trace_operator_TP(term_z_lab, term_z_c)
                    - g * pauli_func.trace_operator_TP(term_x_lab, term_x_c))).real

    term_zz_lab, term_zz_c = pauli_func.multiply_operators_TP(lab_z, comm_lab_zz_yz, c_z, comm_c_zz_yz)
    term_z_lab, term_z_c = pauli_func.multiply_operators_TP(lab_z, comm_lab_z_yz, c_z, comm_c_z_yz)
    term_x_lab, term_x_c = pauli_func.multiply_operators_TP(lab_z, comm_lab_x_yz, c_z, comm_c_x_yz)

    R_Z_num[2] = (-2.j * (pauli_func.trace_operator_TP(term_zz_lab, term_zz_c)
                    - h * pauli_func.trace_operator_TP(term_z_lab, term_z_c)
                    - g * pauli_func.trace_operator_TP(term_x_lab, term_x_c))).real

    print("R Matrix Z:")
    print("Analytical:")
    print(R_Z)
    print("Numerical:")
    print(R_Z_num)

    # Z direction
    term_zz_lab, term_zz_c = pauli_func.multiply_operators_TP(lab_x, comm_lab_zz_y, c_x, comm_c_zz_y)
    term_z_lab, term_z_c = pauli_func.multiply_operators_TP(lab_x, comm_lab_z_y, c_x, comm_c_z_y)
    term_x_lab, term_x_c = pauli_func.multiply_operators_TP(lab_x, comm_lab_x_y, c_x, comm_c_x_y)

    R_X_num[0] = (-2.j * (pauli_func.trace_operator_TP(term_zz_lab, term_zz_c)
                        - h * pauli_func.trace_operator_TP(term_z_lab, term_z_c)
                        - g * pauli_func.trace_operator_TP(term_x_lab, term_x_c))).real

    term_zz_lab, term_zz_c = pauli_func.multiply_operators_TP(lab_x, comm_lab_zz_xy, c_x, comm_c_zz_xy)
    term_z_lab, term_z_c = pauli_func.multiply_operators_TP(lab_x, comm_lab_z_xy, c_x, comm_c_z_xy)
    term_x_lab, term_x_c = pauli_func.multiply_operators_TP(lab_x, comm_lab_x_xy, c_x, comm_c_x_xy)

    R_X_num[1] = (-2.j * (pauli_func.trace_operator_TP(term_zz_lab, term_zz_c)
                        - h * pauli_func.trace_operator_TP(term_z_lab, term_z_c)
                        - g * pauli_func.trace_operator_TP(term_x_lab, term_x_c))).real

    term_zz_lab, term_zz_c = pauli_func.multiply_operators_TP(lab_x, comm_lab_zz_yz, c_x, comm_c_zz_yz)
    term_z_lab, term_z_c = pauli_func.multiply_operators_TP(lab_x, comm_lab_z_yz, c_x, comm_c_z_yz)
    term_x_lab, term_x_c = pauli_func.multiply_operators_TP(lab_x, comm_lab_x_yz, c_x, comm_c_x_yz)

    R_X_num[2] = (-2.j * (pauli_func.trace_operator_TP(term_zz_lab, term_zz_c)
                        - h * pauli_func.trace_operator_TP(term_z_lab, term_z_c)
                        - g * pauli_func.trace_operator_TP(term_x_lab, term_x_c))).real

    print("R Matrix X:")
    print("Analytical:")
    print(R_X)
    print("Numerical:")
    print(R_X_num)

    # P Matrix
    term_zz_zz_lab, term_zz_zz_c = pauli_func.multiply_operators_TP(comm_lab_zz_y, comm_lab_zz_y, comm_c_zz_y, comm_c_zz_y)
    term_x_zz_lab, term_x_zz_c = pauli_func.multiply_operators_TP(comm_lab_x_y, comm_lab_zz_y, comm_c_x_y, comm_c_zz_y)
    term_z_zz_lab, term_z_zz_c = pauli_func.multiply_operators_TP(comm_lab_z_y, comm_lab_zz_y, comm_c_z_y, comm_c_zz_y)
    term_zz_x_lab, term_zz_x_c = pauli_func.multiply_operators_TP(comm_lab_zz_y, comm_lab_x_y, comm_c_zz_y, comm_c_x_y)
    term_x_x_lab, term_x_x_c = pauli_func.multiply_operators_TP(comm_lab_x_y, comm_lab_x_y, comm_c_x_y, comm_c_x_y)
    term_z_x_lab, term_z_x_c = pauli_func.multiply_operators_TP(comm_lab_z_y, comm_lab_x_y, comm_c_z_y, comm_c_x_y)
    term_zz_z_lab, term_zz_z_c = pauli_func.multiply_operators_TP(comm_lab_zz_y, comm_lab_z_y, comm_c_zz_y, comm_c_z_y)
    term_x_z_lab, term_x_z_c = pauli_func.multiply_operators_TP(comm_lab_x_y, comm_lab_z_y, comm_c_x_y, comm_c_z_y)
    term_z_z_lab, term_z_z_c = pauli_func.multiply_operators_TP(comm_lab_z_y, comm_lab_z_y, comm_c_z_y, comm_c_z_y)
    P_num[0, 0] = - (pauli_func.trace_operator_TP(term_zz_zz_lab, term_zz_zz_c)
                    - g * pauli_func.trace_operator_TP(term_x_zz_lab, term_x_zz_c)
                    - h * pauli_func.trace_operator_TP(term_z_zz_lab, term_z_zz_c)
                    - g * pauli_func.trace_operator_TP(term_zz_x_lab, term_zz_x_c)
                    + g ** 2 * pauli_func.trace_operator_TP(term_x_x_lab, term_x_x_c)
                    + h * g * pauli_func.trace_operator_TP(term_z_x_lab, term_z_x_c)
                    - h * pauli_func.trace_operator_TP(term_zz_z_lab, term_zz_z_c)
                    + h * g * pauli_func.trace_operator_TP(term_x_z_lab, term_x_z_c)
                    + h ** 2 * pauli_func.trace_operator_TP(term_z_z_lab, term_z_z_c)
                    ).real


    term_zz_zz_lab, term_zz_zz_c = pauli_func.multiply_operators_TP(comm_lab_zz_y, comm_lab_zz_xy, comm_c_zz_y, comm_c_zz_xy)
    term_x_zz_lab, term_x_zz_c = pauli_func.multiply_operators_TP(comm_lab_x_y, comm_lab_zz_xy, comm_c_x_y, comm_c_zz_xy)
    term_z_zz_lab, term_z_zz_c = pauli_func.multiply_operators_TP(comm_lab_z_y, comm_lab_zz_xy, comm_c_z_y, comm_c_zz_xy)
    term_zz_x_lab, term_zz_x_c = pauli_func.multiply_operators_TP(comm_lab_zz_y, comm_lab_x_xy, comm_c_zz_y, comm_c_x_xy)
    term_x_x_lab, term_x_x_c = pauli_func.multiply_operators_TP(comm_lab_x_y, comm_lab_x_xy, comm_c_x_y, comm_c_x_xy)
    term_z_x_lab, term_z_x_c = pauli_func.multiply_operators_TP(comm_lab_z_y, comm_lab_x_xy, comm_c_z_y, comm_c_x_xy)
    term_zz_z_lab, term_zz_z_c = pauli_func.multiply_operators_TP(comm_lab_zz_y, comm_lab_z_xy, comm_c_zz_y, comm_c_z_xy)
    term_x_z_lab, term_x_z_c = pauli_func.multiply_operators_TP(comm_lab_x_y, comm_lab_z_xy, comm_c_x_y, comm_c_z_xy)
    term_z_z_lab, term_z_z_c = pauli_func.multiply_operators_TP(comm_lab_z_y, comm_lab_z_xy, comm_c_z_y, comm_c_z_xy)
    P_num[0, 1] = - (pauli_func.trace_operator_TP(term_zz_zz_lab, term_zz_zz_c)
                    - g * pauli_func.trace_operator_TP(term_x_zz_lab, term_x_zz_c)
                    - h * pauli_func.trace_operator_TP(term_z_zz_lab, term_z_zz_c)
                    - g * pauli_func.trace_operator_TP(term_zz_x_lab, term_zz_x_c)
                    + g ** 2 * pauli_func.trace_operator_TP(term_x_x_lab, term_x_x_c)
                    + h * g * pauli_func.trace_operator_TP(term_z_x_lab, term_z_x_c)
                    - h * pauli_func.trace_operator_TP(term_zz_z_lab, term_zz_z_c)
                    + h * g * pauli_func.trace_operator_TP(term_x_z_lab, term_x_z_c)
                    + h ** 2 * pauli_func.trace_operator_TP(term_z_z_lab, term_z_z_c)
                    ).real


    term_zz_zz_lab, term_zz_zz_c = pauli_func.multiply_operators_TP(comm_lab_zz_y, comm_lab_zz_yz, comm_c_zz_y, comm_c_zz_yz)
    term_x_zz_lab, term_x_zz_c = pauli_func.multiply_operators_TP(comm_lab_x_y, comm_lab_zz_yz, comm_c_x_y, comm_c_zz_yz)
    term_z_zz_lab, term_z_zz_c = pauli_func.multiply_operators_TP(comm_lab_z_y, comm_lab_zz_yz, comm_c_z_y, comm_c_zz_yz)
    term_zz_x_lab, term_zz_x_c = pauli_func.multiply_operators_TP(comm_lab_zz_y, comm_lab_x_yz, comm_c_zz_y, comm_c_x_yz)
    term_x_x_lab, term_x_x_c = pauli_func.multiply_operators_TP(comm_lab_x_y, comm_lab_x_yz, comm_c_x_y, comm_c_x_yz)
    term_z_x_lab, term_z_x_c = pauli_func.multiply_operators_TP(comm_lab_z_y, comm_lab_x_yz, comm_c_z_y, comm_c_x_yz)
    term_zz_z_lab, term_zz_z_c = pauli_func.multiply_operators_TP(comm_lab_zz_y, comm_lab_z_yz, comm_c_zz_y, comm_c_z_yz)
    term_x_z_lab, term_x_z_c = pauli_func.multiply_operators_TP(comm_lab_x_y, comm_lab_z_yz, comm_c_x_y, comm_c_z_yz)
    term_z_z_lab, term_z_z_c = pauli_func.multiply_operators_TP(comm_lab_z_y, comm_lab_z_yz, comm_c_z_y, comm_c_z_yz)
    P_num[0, 2] = - (pauli_func.trace_operator_TP(term_zz_zz_lab, term_zz_zz_c)
                    - g * pauli_func.trace_operator_TP(term_x_zz_lab, term_x_zz_c)
                    - h * pauli_func.trace_operator_TP(term_z_zz_lab, term_z_zz_c)
                    - g * pauli_func.trace_operator_TP(term_zz_x_lab, term_zz_x_c)
                    + g ** 2 * pauli_func.trace_operator_TP(term_x_x_lab, term_x_x_c)
                    + h * g * pauli_func.trace_operator_TP(term_z_x_lab, term_z_x_c)
                    - h * pauli_func.trace_operator_TP(term_zz_z_lab, term_zz_z_c)
                    + h * g * pauli_func.trace_operator_TP(term_x_z_lab, term_x_z_c)
                    + h ** 2 * pauli_func.trace_operator_TP(term_z_z_lab, term_z_z_c)
                    ).real


    term_zz_zz_lab, term_zz_zz_c = pauli_func.multiply_operators_TP(comm_lab_zz_xy, comm_lab_zz_y, comm_c_zz_xy, comm_c_zz_y)
    term_x_zz_lab, term_x_zz_c = pauli_func.multiply_operators_TP(comm_lab_x_xy, comm_lab_zz_y, comm_c_x_xy, comm_c_zz_y)
    term_z_zz_lab, term_z_zz_c = pauli_func.multiply_operators_TP(comm_lab_z_xy, comm_lab_zz_y, comm_c_z_xy, comm_c_zz_y)
    term_zz_x_lab, term_zz_x_c = pauli_func.multiply_operators_TP(comm_lab_zz_xy, comm_lab_x_y, comm_c_zz_xy, comm_c_x_y)
    term_x_x_lab, term_x_x_c = pauli_func.multiply_operators_TP(comm_lab_x_xy, comm_lab_x_y, comm_c_x_xy, comm_c_x_y)
    term_z_x_lab, term_z_x_c = pauli_func.multiply_operators_TP(comm_lab_z_xy, comm_lab_x_y, comm_c_z_xy, comm_c_x_y)
    term_zz_z_lab, term_zz_z_c = pauli_func.multiply_operators_TP(comm_lab_zz_xy, comm_lab_z_y, comm_c_zz_xy, comm_c_z_y)
    term_x_z_lab, term_x_z_c = pauli_func.multiply_operators_TP(comm_lab_x_xy, comm_lab_z_y, comm_c_x_xy, comm_c_z_y)
    term_z_z_lab, term_z_z_c = pauli_func.multiply_operators_TP(comm_lab_z_xy, comm_lab_z_y, comm_c_z_xy, comm_c_z_y)
    P_num[1, 0] = - (pauli_func.trace_operator_TP(term_zz_zz_lab, term_zz_zz_c)
                    - g * pauli_func.trace_operator_TP(term_x_zz_lab, term_x_zz_c)
                    - h * pauli_func.trace_operator_TP(term_z_zz_lab, term_z_zz_c)
                    - g * pauli_func.trace_operator_TP(term_zz_x_lab, term_zz_x_c)
                    + g ** 2 * pauli_func.trace_operator_TP(term_x_x_lab, term_x_x_c)
                    + h * g * pauli_func.trace_operator_TP(term_z_x_lab, term_z_x_c)
                    - h * pauli_func.trace_operator_TP(term_zz_z_lab, term_zz_z_c)
                    + h * g * pauli_func.trace_operator_TP(term_x_z_lab, term_x_z_c)
                    + h ** 2 * pauli_func.trace_operator_TP(term_z_z_lab, term_z_z_c)
                    ).real


    term_zz_zz_lab, term_zz_zz_c = pauli_func.multiply_operators_TP(comm_lab_zz_xy, comm_lab_zz_xy, comm_c_zz_xy, comm_c_zz_xy)
    term_x_zz_lab, term_x_zz_c = pauli_func.multiply_operators_TP(comm_lab_x_xy, comm_lab_zz_xy, comm_c_x_xy, comm_c_zz_xy)
    term_z_zz_lab, term_z_zz_c = pauli_func.multiply_operators_TP(comm_lab_z_xy, comm_lab_zz_xy, comm_c_z_xy, comm_c_zz_xy)
    term_zz_x_lab, term_zz_x_c = pauli_func.multiply_operators_TP(comm_lab_zz_xy, comm_lab_x_xy, comm_c_zz_xy, comm_c_x_xy)
    term_x_x_lab, term_x_x_c = pauli_func.multiply_operators_TP(comm_lab_x_xy, comm_lab_x_xy, comm_c_x_xy, comm_c_x_xy)
    term_z_x_lab, term_z_x_c = pauli_func.multiply_operators_TP(comm_lab_z_xy, comm_lab_x_xy, comm_c_z_xy, comm_c_x_xy)
    term_zz_z_lab, term_zz_z_c = pauli_func.multiply_operators_TP(comm_lab_zz_xy, comm_lab_z_xy, comm_c_zz_xy, comm_c_z_xy)
    term_x_z_lab, term_x_z_c = pauli_func.multiply_operators_TP(comm_lab_x_xy, comm_lab_z_xy, comm_c_x_xy, comm_c_z_xy)
    term_z_z_lab, term_z_z_c = pauli_func.multiply_operators_TP(comm_lab_z_xy, comm_lab_z_xy, comm_c_z_xy, comm_c_z_xy)
    P_num[1, 1] = - (pauli_func.trace_operator_TP(term_zz_zz_lab, term_zz_zz_c)
                    - g * pauli_func.trace_operator_TP(term_x_zz_lab, term_x_zz_c)
                    - h * pauli_func.trace_operator_TP(term_z_zz_lab, term_z_zz_c)
                    - g * pauli_func.trace_operator_TP(term_zz_x_lab, term_zz_x_c)
                    + g ** 2 * pauli_func.trace_operator_TP(term_x_x_lab, term_x_x_c)
                    + h * g * pauli_func.trace_operator_TP(term_z_x_lab, term_z_x_c)
                    - h * pauli_func.trace_operator_TP(term_zz_z_lab, term_zz_z_c)
                    + h * g * pauli_func.trace_operator_TP(term_x_z_lab, term_x_z_c)
                    + h ** 2 * pauli_func.trace_operator_TP(term_z_z_lab, term_z_z_c)
                    ).real


    term_zz_zz_lab, term_zz_zz_c = pauli_func.multiply_operators_TP(comm_lab_zz_xy, comm_lab_zz_yz, comm_c_zz_xy, comm_c_zz_yz)
    term_x_zz_lab, term_x_zz_c = pauli_func.multiply_operators_TP(comm_lab_x_xy, comm_lab_zz_yz, comm_c_x_xy, comm_c_zz_yz)
    term_z_zz_lab, term_z_zz_c = pauli_func.multiply_operators_TP(comm_lab_z_xy, comm_lab_zz_yz, comm_c_z_xy, comm_c_zz_yz)
    term_zz_x_lab, term_zz_x_c = pauli_func.multiply_operators_TP(comm_lab_zz_xy, comm_lab_x_yz, comm_c_zz_xy, comm_c_x_yz)
    term_x_x_lab, term_x_x_c = pauli_func.multiply_operators_TP(comm_lab_x_xy, comm_lab_x_yz, comm_c_x_xy, comm_c_x_yz)
    term_z_x_lab, term_z_x_c = pauli_func.multiply_operators_TP(comm_lab_z_xy, comm_lab_x_yz, comm_c_z_xy, comm_c_x_yz)
    term_zz_z_lab, term_zz_z_c = pauli_func.multiply_operators_TP(comm_lab_zz_xy, comm_lab_z_yz, comm_c_zz_xy, comm_c_z_yz)
    term_x_z_lab, term_x_z_c = pauli_func.multiply_operators_TP(comm_lab_x_xy, comm_lab_z_yz, comm_c_x_xy, comm_c_z_yz)
    term_z_z_lab, term_z_z_c = pauli_func.multiply_operators_TP(comm_lab_z_xy, comm_lab_z_yz, comm_c_z_xy, comm_c_z_yz)
    P_num[1, 2] = - (pauli_func.trace_operator_TP(term_zz_zz_lab, term_zz_zz_c)
                    - g * pauli_func.trace_operator_TP(term_x_zz_lab, term_x_zz_c)
                    - h * pauli_func.trace_operator_TP(term_z_zz_lab, term_z_zz_c)
                    - g * pauli_func.trace_operator_TP(term_zz_x_lab, term_zz_x_c)
                    + g ** 2 * pauli_func.trace_operator_TP(term_x_x_lab, term_x_x_c)
                    + h * g * pauli_func.trace_operator_TP(term_z_x_lab, term_z_x_c)
                    - h * pauli_func.trace_operator_TP(term_zz_z_lab, term_zz_z_c)
                    + h * g * pauli_func.trace_operator_TP(term_x_z_lab, term_x_z_c)
                    + h ** 2 * pauli_func.trace_operator_TP(term_z_z_lab, term_z_z_c)
                    ).real


    term_zz_zz_lab, term_zz_zz_c = pauli_func.multiply_operators_TP(comm_lab_zz_yz, comm_lab_zz_y, comm_c_zz_yz, comm_c_zz_y)
    term_x_zz_lab, term_x_zz_c = pauli_func.multiply_operators_TP(comm_lab_x_yz, comm_lab_zz_y, comm_c_x_yz, comm_c_zz_y)
    term_z_zz_lab, term_z_zz_c = pauli_func.multiply_operators_TP(comm_lab_z_yz, comm_lab_zz_y, comm_c_z_yz, comm_c_zz_y)
    term_zz_x_lab, term_zz_x_c = pauli_func.multiply_operators_TP(comm_lab_zz_yz, comm_lab_x_y, comm_c_zz_yz, comm_c_x_y)
    term_x_x_lab, term_x_x_c = pauli_func.multiply_operators_TP(comm_lab_x_yz, comm_lab_x_y, comm_c_x_yz, comm_c_x_y)
    term_z_x_lab, term_z_x_c = pauli_func.multiply_operators_TP(comm_lab_z_yz, comm_lab_x_y, comm_c_z_yz, comm_c_x_y)
    term_zz_z_lab, term_zz_z_c = pauli_func.multiply_operators_TP(comm_lab_zz_yz, comm_lab_z_y, comm_c_zz_yz, comm_c_z_y)
    term_x_z_lab, term_x_z_c = pauli_func.multiply_operators_TP(comm_lab_x_yz, comm_lab_z_y, comm_c_x_yz, comm_c_z_y)
    term_z_z_lab, term_z_z_c = pauli_func.multiply_operators_TP(comm_lab_z_yz, comm_lab_z_y, comm_c_z_yz, comm_c_z_y)
    P_num[2, 0] = - (pauli_func.trace_operator_TP(term_zz_zz_lab, term_zz_zz_c)
                    - g * pauli_func.trace_operator_TP(term_x_zz_lab, term_x_zz_c)
                    - h * pauli_func.trace_operator_TP(term_z_zz_lab, term_z_zz_c)
                    - g * pauli_func.trace_operator_TP(term_zz_x_lab, term_zz_x_c)
                    + g ** 2 * pauli_func.trace_operator_TP(term_x_x_lab, term_x_x_c)
                    + h * g * pauli_func.trace_operator_TP(term_z_x_lab, term_z_x_c)
                    - h * pauli_func.trace_operator_TP(term_zz_z_lab, term_zz_z_c)
                    + h * g * pauli_func.trace_operator_TP(term_x_z_lab, term_x_z_c)
                    + h ** 2 * pauli_func.trace_operator_TP(term_z_z_lab, term_z_z_c)
                    ).real


    term_zz_zz_lab, term_zz_zz_c = pauli_func.multiply_operators_TP(comm_lab_zz_yz, comm_lab_zz_xy, comm_c_zz_yz, comm_c_zz_xy)
    term_x_zz_lab, term_x_zz_c = pauli_func.multiply_operators_TP(comm_lab_x_yz, comm_lab_zz_xy, comm_c_x_yz, comm_c_zz_xy)
    term_z_zz_lab, term_z_zz_c = pauli_func.multiply_operators_TP(comm_lab_z_yz, comm_lab_zz_xy, comm_c_z_yz, comm_c_zz_xy)
    term_zz_x_lab, term_zz_x_c = pauli_func.multiply_operators_TP(comm_lab_zz_yz, comm_lab_x_xy, comm_c_zz_yz, comm_c_x_xy)
    term_x_x_lab, term_x_x_c = pauli_func.multiply_operators_TP(comm_lab_x_yz, comm_lab_x_xy, comm_c_x_yz, comm_c_x_xy)
    term_z_x_lab, term_z_x_c = pauli_func.multiply_operators_TP(comm_lab_z_yz, comm_lab_x_xy, comm_c_z_yz, comm_c_x_xy)
    term_zz_z_lab, term_zz_z_c = pauli_func.multiply_operators_TP(comm_lab_zz_yz, comm_lab_z_xy, comm_c_zz_yz, comm_c_z_xy)
    term_x_z_lab, term_x_z_c = pauli_func.multiply_operators_TP(comm_lab_x_yz, comm_lab_z_xy, comm_c_x_yz, comm_c_z_xy)
    term_z_z_lab, term_z_z_c = pauli_func.multiply_operators_TP(comm_lab_z_yz, comm_lab_z_xy, comm_c_z_yz, comm_c_z_xy)
    P_num[2, 1] = - (pauli_func.trace_operator_TP(term_zz_zz_lab, term_zz_zz_c)
                    - g * pauli_func.trace_operator_TP(term_x_zz_lab, term_x_zz_c)
                    - h * pauli_func.trace_operator_TP(term_z_zz_lab, term_z_zz_c)
                    - g * pauli_func.trace_operator_TP(term_zz_x_lab, term_zz_x_c)
                    + g ** 2 * pauli_func.trace_operator_TP(term_x_x_lab, term_x_x_c)
                    + h * g * pauli_func.trace_operator_TP(term_z_x_lab, term_z_x_c)
                    - h * pauli_func.trace_operator_TP(term_zz_z_lab, term_zz_z_c)
                    + h * g * pauli_func.trace_operator_TP(term_x_z_lab, term_x_z_c)
                    + h ** 2 * pauli_func.trace_operator_TP(term_z_z_lab, term_z_z_c)
                    ).real


    term_zz_zz_lab, term_zz_zz_c = pauli_func.multiply_operators_TP(comm_lab_zz_yz, comm_lab_zz_yz, comm_c_zz_yz, comm_c_zz_yz)
    term_x_zz_lab, term_x_zz_c = pauli_func.multiply_operators_TP(comm_lab_x_yz, comm_lab_zz_yz, comm_c_x_yz, comm_c_zz_yz)
    term_z_zz_lab, term_z_zz_c = pauli_func.multiply_operators_TP(comm_lab_z_yz, comm_lab_zz_yz, comm_c_z_yz, comm_c_zz_yz)
    term_zz_x_lab, term_zz_x_c = pauli_func.multiply_operators_TP(comm_lab_zz_yz, comm_lab_x_yz, comm_c_zz_yz, comm_c_x_yz)
    term_x_x_lab, term_x_x_c = pauli_func.multiply_operators_TP(comm_lab_x_yz, comm_lab_x_yz, comm_c_x_yz, comm_c_x_yz)
    term_z_x_lab, term_z_x_c = pauli_func.multiply_operators_TP(comm_lab_z_yz, comm_lab_x_yz, comm_c_z_yz, comm_c_x_yz)
    term_zz_z_lab, term_zz_z_c = pauli_func.multiply_operators_TP(comm_lab_zz_yz, comm_lab_z_yz, comm_c_zz_yz, comm_c_z_yz)
    term_x_z_lab, term_x_z_c = pauli_func.multiply_operators_TP(comm_lab_x_yz, comm_lab_z_yz, comm_c_x_yz, comm_c_z_yz)
    term_z_z_lab, term_z_z_c = pauli_func.multiply_operators_TP(comm_lab_z_yz, comm_lab_z_yz, comm_c_z_yz, comm_c_z_yz)
    P_num[2, 2] = - (pauli_func.trace_operator_TP(term_zz_zz_lab, term_zz_zz_c)
                    - g * pauli_func.trace_operator_TP(term_x_zz_lab, term_x_zz_c)
                    - h * pauli_func.trace_operator_TP(term_z_zz_lab, term_z_zz_c)
                    - g * pauli_func.trace_operator_TP(term_zz_x_lab, term_zz_x_c)
                    + g ** 2 * pauli_func.trace_operator_TP(term_x_x_lab, term_x_x_c)
                    + h * g * pauli_func.trace_operator_TP(term_z_x_lab, term_z_x_c)
                    - h * pauli_func.trace_operator_TP(term_zz_z_lab, term_zz_z_c)
                    + h * g * pauli_func.trace_operator_TP(term_x_z_lab, term_x_z_c)
                    + h ** 2 * pauli_func.trace_operator_TP(term_z_z_lab, term_z_z_c)
                    ).real

    print("P Matrices:")
    print("Analytical:")
    print(P)
    print("Numerical:")
    print(P_num)
    return 0

def check_commutator(h, g):

    P = Pmatrix(h, g)
    R_Z = RvectorZ(h, g)
    R_X = RvectorX(h, g)

    print("Eigenvalues:", np.linalg.eigvalsh(P))

    alpha_Z = 0.5 * np.linalg.inv(P) @ R_Z
    alpha_X = 0.5 * np.linalg.inv(P) @ R_X

    # hamiltonian
    lab_ham, c_ham = pauli_func.add_operators(lab_zz, lab_x, c_zz, -g * c_x)
    lab_ham, c_ham = pauli_func.add_operators(lab_ham, lab_z, c_ham, -h * c_z)

    # AGP
    # A_Z
    lab_A_Z_1 = lab_y
    c_A_Z_1 = alpha_Z[0] * c_y

    lab_A_Z_2 = lab_xy
    c_A_Z_2 = alpha_Z[1] * c_xy

    lab_A_Z_3 = lab_yz
    c_A_Z_3 = alpha_Z[2] * c_yz

    lab_A_Z, c_A_Z = pauli_func.add_operators(lab_A_Z_1, lab_A_Z_2, c_A_Z_1, c_A_Z_2)
    lab_A_Z, c_A_Z = pauli_func.add_operators(lab_A_Z, lab_A_Z_3, c_A_Z, c_A_Z_3)

    # A_X
    lab_A_X_1 = lab_y
    c_A_X_1 = alpha_X[0] * c_y

    lab_A_X_2 = lab_xy
    c_A_X_2 = alpha_X[1] * c_xy

    lab_A_X_3 = lab_yz
    c_A_X_3 = alpha_X[2] * c_yz

    lab_A_X, c_A_X = pauli_func.add_operators(lab_A_X_1, lab_A_X_2, c_A_X_1, c_A_X_2)
    lab_A_X, c_A_X = pauli_func.add_operators(lab_A_X, lab_A_X_3, c_A_X, c_A_X_3)


    # commutators of AGP
    lab_comm_A_Z, c_comm_A_Z = pauli_func.commute_operators_TP(lab_ham, lab_A_Z, c_ham, c_A_Z)
    lab_comm_A_X, c_comm_A_X = pauli_func.commute_operators_TP(lab_ham, lab_A_X, c_ham, c_A_X)

    # double commutator
    lab_comm_A_Z, c_comm_A_Z = pauli_func.commute_operators_TP(lab_ham, lab_comm_A_Z, c_ham, c_comm_A_Z)
    lab_comm_A_X, c_comm_A_X = pauli_func.commute_operators_TP(lab_ham, lab_comm_A_X, c_ham, c_comm_A_X)

    lab_comm_A_Z, c_comm_A_Z = pauli_func.operator_cleanup(lab_comm_A_Z, c_comm_A_Z)
    lab_comm_A_X, c_comm_A_X= pauli_func.operator_cleanup(lab_comm_A_X, c_comm_A_X)

    # commutator with partial derivative
    lab_comm_part_Z, c_comm_part_Z = pauli_func.commute_operators_TP(lab_ham, lab_z, c_ham, -c_z)
    lab_comm_part_X, c_comm_part_X = pauli_func.commute_operators_TP(lab_ham, lab_x, c_ham, -c_x)

    lab_comm_part_Z, c_comm_part_Z = pauli_func.operator_cleanup(lab_comm_part_Z, c_comm_part_Z)
    lab_comm_part_X, c_comm_part_X = pauli_func.operator_cleanup(lab_comm_part_X, c_comm_part_X)

    # show results
    print("Z Direction:")
    print("-[H, Z]:")
    pauli_func.print_operator_full(lab_comm_part_Z, c_comm_part_Z)
    print("[H, [H, A_Z]]:")
    pauli_func.print_operator_full(lab_comm_A_Z, c_comm_A_Z)

    print("X Direction:")
    print("-[H, X]:")
    pauli_func.print_operator_full(lab_comm_part_X, c_comm_part_X)
    print("[H, [H, A_X]]:")
    pauli_func.print_operator_full(lab_comm_A_X, c_comm_A_X)


# compute full AGP in moving frame as matrix from numerical derivative
# CHECK SIGN
def compute_agp_from_spectrum(h, g, delta=1.e-6):

    ham_tot = mat_zz - h * mat_z - g * mat_x
    ev_tot, evec_tot = np.linalg.eigh(ham_tot)

    # direction 1
    ham_plus = mat_zz - (h - delta) * mat_z - g * mat_x
    ev_plus, evec_plus = np.linalg.eigh(ham_plus)

    ham_minus = mat_zz - (h + delta) * mat_z - g * mat_x
    ev_minus, evec_minus = np.linalg.eigh(ham_minus)

    # derivative (2nd order central finite difference)
    evec_diff = 0.5 * (evec_plus - evec_minus) / delta
    A_mat_1 = 1.j * evec_tot.T.conj() @ evec_diff

    # direction 2
    ham_plus = mat_zz - h * mat_z - (g - delta) * mat_x
    ev_plus, evec_plus = np.linalg.eigh(ham_plus)

    ham_minus = mat_zz - h * mat_z - (g + delta) * mat_x
    ev_minus, evec_minus = np.linalg.eigh(ham_minus)

    # derivative (2nd order central finite difference)
    evec_diff = 0.5 * (evec_plus - evec_minus) / delta
    A_mat_2 = 1.j * evec_tot.T.conj() @ evec_diff

    return A_mat_1, A_mat_2


# now to optimization in the ground state sector
def compute_agp_gs(h, g):
    ham_tot = mat_zz - h * mat_z - g * mat_x
    ev, evec = np.linalg.eigh(ham_tot)
    gs = evec[:, 0]

    # hamiltonian
    lab_ham, c_ham = pauli_func.add_operators(lab_zz, lab_x, c_zz, -g * c_x)
    lab_ham, c_ham = pauli_func.add_operators(lab_ham, lab_z, c_ham, -h * c_z)

    comm_lab_0, comm_c_0 = pauli_func.commute_operators_TP(lab_ham, lab_y, c_ham, c_y)
    comm_lab_1, comm_c_1 = pauli_func.commute_operators_TP(lab_ham, lab_xy, c_ham, c_xy)
    comm_lab_2, comm_c_2 = pauli_func.commute_operators_TP(lab_ham, lab_yz, c_ham, c_yz)

    # compute expectation values
    expval_partial_z = -gs.T.conj() @ mat_z @ gs
    expval_partial_x = -gs.T.conj() @ mat_x @ gs

    return 0


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

L = 2
k_name = "0"
par = 1                             # parity sector
tr_I = 2 ** L

# create commutators
# create TPY operators from file
lab_x, c_x = pauli_func.TP_operator_from_string_compact("x", L)
lab_y, c_y = pauli_func.TP_operator_from_string_compact("y", L)
lab_z, c_z = pauli_func.TP_operator_from_string_compact("z", L)
lab_xx, c_xx = pauli_func.TP_operator_from_string_compact("xx", L)
lab_xy, c_xy = pauli_func.TP_operator_from_string_compact("xy", L)
lab_xz, c_xz = pauli_func.TP_operator_from_string_compact("xz", L)
lab_yy, c_yy = pauli_func.TP_operator_from_string_compact("yy", L)
lab_yz, c_yz = pauli_func.TP_operator_from_string_compact("yz", L)
lab_zz, c_zz = pauli_func.TP_operator_from_string_compact("zz", L)

comm_lab_zz_y, comm_c_zz_y = pauli_func.commute_operators_TP(lab_zz, lab_y, c_zz, c_y)
comm_lab_zz_xy, comm_c_zz_xy = pauli_func.commute_operators_TP(lab_zz, lab_xy, c_zz, c_xy)
comm_lab_zz_yz, comm_c_zz_yz = pauli_func.commute_operators_TP(lab_zz, lab_yz, c_zz, c_yz)

comm_lab_z_y, comm_c_z_y = pauli_func.commute_operators_TP(lab_z, lab_y, c_z, c_y)
comm_lab_z_xy, comm_c_z_xy = pauli_func.commute_operators_TP(lab_z, lab_xy, c_z, c_xy)
comm_lab_z_yz, comm_c_z_yz = pauli_func.commute_operators_TP(lab_z, lab_yz, c_z, c_yz)

comm_lab_x_y, comm_c_x_y = pauli_func.commute_operators_TP(lab_x, lab_y, c_x, c_y)
comm_lab_x_xy, comm_c_x_xy = pauli_func.commute_operators_TP(lab_x, lab_xy, c_x, c_xy)
comm_lab_x_yz, comm_c_x_yz = pauli_func.commute_operators_TP(lab_x, lab_yz, c_x, c_yz)

comm, coeff = pauli_func.operator_cleanup(comm_lab_zz_y, comm_c_zz_y)
print("Commutator: ZZ, Y")
pauli_func.print_operator_full(comm, coeff)

comm, coeff = pauli_func.operator_cleanup(comm_lab_zz_xy, comm_c_zz_xy)
print("Commutator: ZZ, XY")
pauli_func.print_operator_full(comm, coeff)

comm, coeff = pauli_func.operator_cleanup(comm_lab_zz_yz, comm_c_zz_yz)
print("Commutator: ZZ, YZ")
pauli_func.print_operator_full(comm, coeff)

comm, coeff = pauli_func.operator_cleanup(comm_lab_z_y, comm_c_z_y)
print("Commutator: Z, Y")
pauli_func.print_operator_full(comm, coeff)

comm, coeff = pauli_func.operator_cleanup(comm_lab_z_xy, comm_c_z_xy)
print("Commutator: Z, XY")
pauli_func.print_operator_full(comm, coeff)

comm, coeff = pauli_func.operator_cleanup(comm_lab_z_yz, comm_c_z_yz)
print("Commutator: Z, YZ")
pauli_func.print_operator_full(comm, coeff)

comm, coeff = pauli_func.operator_cleanup(comm_lab_x_y, comm_c_x_y)
print("Commutator: X, Y")
pauli_func.print_operator_full(comm, coeff)

comm, coeff = pauli_func.operator_cleanup(comm_lab_x_xy, comm_c_x_xy)
print("Commutator: X, XY")
pauli_func.print_operator_full(comm, coeff)

comm, coeff = pauli_func.operator_cleanup(comm_lab_x_yz, comm_c_x_yz)
print("Commutator: X, YZ")
pauli_func.print_operator_full(comm, coeff)

# print("P_11:")
# print(trace_of_product_TP(comm_lab_zz_y, comm_c_zz_y, comm_lab_zz_y, comm_c_zz_y))
# print(trace_of_product_TP(comm_lab_x_y, comm_c_x_y, comm_lab_zz_y, comm_c_zz_y))
# print(trace_of_product_TP(comm_lab_z_y, comm_c_z_y, comm_lab_zz_y, comm_c_zz_y))
# print(trace_of_product_TP(comm_lab_zz_y, comm_c_zz_y, comm_lab_x_y, comm_c_x_y))
# print(trace_of_product_TP(comm_lab_x_y, comm_c_x_y, comm_lab_x_y, comm_c_x_y))
# print(trace_of_product_TP(comm_lab_z_y, comm_c_z_y, comm_lab_x_y, comm_c_x_y))
# print(trace_of_product_TP(comm_lab_zz_y, comm_c_zz_y, comm_lab_z_y, comm_c_z_y))
# print(trace_of_product_TP(comm_lab_x_y, comm_c_x_y, comm_lab_z_y, comm_c_z_y))
# print(trace_of_product_TP(comm_lab_z_y, comm_c_z_y, comm_lab_z_y, comm_c_z_y))
# print("P_22:")
# print(trace_of_product_TP(comm_lab_zz_xy, comm_c_zz_xy, comm_lab_zz_xy, comm_c_zz_xy))
# print(trace_of_product_TP(comm_lab_x_xy, comm_c_x_xy, comm_lab_zz_xy, comm_c_zz_xy))
# print(trace_of_product_TP(comm_lab_z_xy, comm_c_z_xy, comm_lab_zz_xy, comm_c_zz_xy))
# print(trace_of_product_TP(comm_lab_zz_xy, comm_c_zz_xy, comm_lab_x_xy, comm_c_x_xy))
# print(trace_of_product_TP(comm_lab_x_xy, comm_c_x_xy, comm_lab_x_xy, comm_c_x_xy))
# print(trace_of_product_TP(comm_lab_z_xy, comm_c_z_xy, comm_lab_x_xy, comm_c_x_xy))
# print(trace_of_product_TP(comm_lab_zz_xy, comm_c_zz_xy, comm_lab_z_xy, comm_c_z_xy))
# print(trace_of_product_TP(comm_lab_x_xy, comm_c_x_xy, comm_lab_z_xy, comm_c_z_xy))
# print(trace_of_product_TP(comm_lab_z_xy, comm_c_z_xy, comm_lab_z_xy, comm_c_z_xy))
# print("P_33:")
# print(trace_of_product_TP(comm_lab_zz_yz, comm_c_zz_yz, comm_lab_zz_yz, comm_c_zz_yz))
# print(trace_of_product_TP(comm_lab_x_yz, comm_c_x_yz, comm_lab_zz_yz, comm_c_zz_yz))
# print(trace_of_product_TP(comm_lab_z_yz, comm_c_z_yz, comm_lab_zz_yz, comm_c_zz_yz))
# print(trace_of_product_TP(comm_lab_zz_yz, comm_c_zz_yz, comm_lab_x_yz, comm_c_x_yz))
# print(trace_of_product_TP(comm_lab_x_yz, comm_c_x_yz, comm_lab_x_yz, comm_c_x_yz))
# print(trace_of_product_TP(comm_lab_z_yz, comm_c_z_yz, comm_lab_x_yz, comm_c_x_yz))
# print(trace_of_product_TP(comm_lab_zz_yz, comm_c_zz_yz, comm_lab_z_yz, comm_c_z_yz))
# print(trace_of_product_TP(comm_lab_x_yz, comm_c_x_yz, comm_lab_z_yz, comm_c_z_yz))
# print(trace_of_product_TP(comm_lab_z_yz, comm_c_z_yz, comm_lab_z_yz, comm_c_z_yz))
# print("P_12:")
# print(trace_of_product_TP(comm_lab_zz_y, comm_c_zz_y, comm_lab_zz_xy, comm_c_zz_xy))
# print(trace_of_product_TP(comm_lab_x_y, comm_c_x_y, comm_lab_zz_xy, comm_c_zz_xy))
# print(trace_of_product_TP(comm_lab_z_y, comm_c_z_y, comm_lab_zz_xy, comm_c_zz_xy))
# print(trace_of_product_TP(comm_lab_zz_y, comm_c_zz_y, comm_lab_x_xy, comm_c_x_xy))
# print(trace_of_product_TP(comm_lab_x_y, comm_c_x_y, comm_lab_x_xy, comm_c_x_xy))
# print(trace_of_product_TP(comm_lab_z_y, comm_c_z_y, comm_lab_x_xy, comm_c_x_xy))
# print(trace_of_product_TP(comm_lab_zz_y, comm_c_zz_y, comm_lab_z_xy, comm_c_z_xy))
# print(trace_of_product_TP(comm_lab_x_y, comm_c_x_y, comm_lab_z_xy, comm_c_z_xy))
# print(trace_of_product_TP(comm_lab_z_y, comm_c_z_y, comm_lab_z_xy, comm_c_z_xy))
# print("P_13:")
# print(trace_of_product_TP(comm_lab_zz_y, comm_c_zz_y, comm_lab_zz_yz, comm_c_zz_yz))
# print(trace_of_product_TP(comm_lab_x_y, comm_c_x_y, comm_lab_zz_yz, comm_c_zz_yz))
# print(trace_of_product_TP(comm_lab_z_y, comm_c_z_y, comm_lab_zz_yz, comm_c_zz_yz))
# print(trace_of_product_TP(comm_lab_zz_y, comm_c_zz_y, comm_lab_x_yz, comm_c_x_yz))
# print(trace_of_product_TP(comm_lab_x_y, comm_c_x_y, comm_lab_x_yz, comm_c_x_yz))
# print(trace_of_product_TP(comm_lab_z_y, comm_c_z_y, comm_lab_x_yz, comm_c_x_yz))
# print(trace_of_product_TP(comm_lab_zz_y, comm_c_zz_y, comm_lab_z_yz, comm_c_z_yz))
# print(trace_of_product_TP(comm_lab_x_y, comm_c_x_y, comm_lab_z_yz, comm_c_z_yz))
# print(trace_of_product_TP(comm_lab_z_y, comm_c_z_y, comm_lab_z_yz, comm_c_z_yz))
# print("P_23:")
# print(trace_of_product_TP(comm_lab_zz_xy, comm_c_zz_xy, comm_lab_zz_yz, comm_c_zz_yz))
# print(trace_of_product_TP(comm_lab_x_xy, comm_c_x_xy, comm_lab_zz_yz, comm_c_zz_yz))
# print(trace_of_product_TP(comm_lab_z_xy, comm_c_z_xy, comm_lab_zz_yz, comm_c_zz_yz))
# print(trace_of_product_TP(comm_lab_zz_xy, comm_c_zz_xy, comm_lab_x_yz, comm_c_x_yz))
# print(trace_of_product_TP(comm_lab_x_xy, comm_c_x_xy, comm_lab_x_yz, comm_c_x_yz))
# print(trace_of_product_TP(comm_lab_z_xy, comm_c_z_xy, comm_lab_x_yz, comm_c_x_yz))
# print(trace_of_product_TP(comm_lab_zz_xy, comm_c_zz_xy, comm_lab_z_yz, comm_c_z_yz))
# print(trace_of_product_TP(comm_lab_x_xy, comm_c_x_xy, comm_lab_z_yz, comm_c_z_yz))
# print(trace_of_product_TP(comm_lab_z_xy, comm_c_z_xy, comm_lab_z_yz, comm_c_z_yz))


# # multiply operators
# mult_lab_x_1, mult_c_x_1 = pauli_func.multiply_operators_TP(lab_x, comm_lab_x_y, c_zz, c_y)
#
#
# # fill operators
# R_X_X = np.zeros(3)
# R_X_Z = np.zeros(3)
# R_X_ZZ = np.zeros(3)
#
# R_Z_X = np.zeros(3)
# R_Z_Z = np.zeros(3)
# R_Z_ZZ = np.zeros(3)
#
# P_X_X = np.zeros((3, 3))
# P_X_Z = np.zeros((3, 3))
# P_X_ZZ = np.zeros((3, 3))
# P_Z_ZZ = np.zeros((3, 3))
# P_Z_Z = np.zeros((3, 3))
# P_ZZ_ZZ = np.zeros((3, 3))

# create all operators as matrices
# create operators up to range r and save them in memoryoperators = []
name_zip = "operators/1d_chain_indices_and_periods.zip"
with zipfile.ZipFile(name_zip) as zipper:

    name = "1d_chain_TP_indices_and_periods_L=" + str(L) + "_k=" + k_name + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:

        data = np.load(f)
        periods = data["period"]
        parities = data["parity"]
        size = len(periods)

name_zip_op = "operators/operators_TP_L=" + str(L) + ".zip"
with zipfile.ZipFile(name_zip_op) as zipper_op:


    mat_name = "x_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    mat_x = csr_matrix((val, indices, indptr), shape=(size, size)).todense()

    mat_name = "y_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    mat_y = csr_matrix((val, indices, indptr), shape=(size, size)).todense()

    mat_name = "z_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    mat_z = csr_matrix((val, indices, indptr), shape=(size, size)).todense()

    mat_name = "xy_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    mat_xy = csr_matrix((val, indices, indptr), shape=(size, size)).todense()

    mat_name = "xz_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    mat_xz = csr_matrix((val, indices, indptr), shape=(size, size)).todense()

    mat_name = "yz_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    mat_yz = csr_matrix((val, indices, indptr), shape=(size, size)).todense()

    mat_name = "xx_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    mat_xx = csr_matrix((val, indices, indptr), shape=(size, size)).todense()

    mat_name = "yy_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    mat_yy = csr_matrix((val, indices, indptr), shape=(size, size)).todense()

    mat_name = "zz_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    mat_zz = csr_matrix((val, indices, indptr), shape=(size, size)).todense()


print("[ZZ, Y]")
print(-8j * mat_xz - pauli_func.create_TP_operator_matrix(comm_lab_zz_y, comm_c_zz_y, L).todense())

print("[ZZ, XY]")
print(pauli_func.create_TP_operator_matrix(comm_lab_zz_xy, comm_c_zz_xy, L).todense())

print("[ZZ, YZ]")
print(-8j * mat_x - pauli_func.create_TP_operator_matrix(comm_lab_zz_yz, comm_c_zz_yz, L).todense())

#

print("[Z, Y]")
print(-4j * mat_x - pauli_func.create_TP_operator_matrix(comm_lab_z_y, comm_c_z_y, L).todense())

print("[Z, XY]")
print(4j * (mat_yy - mat_xx) - pauli_func.create_TP_operator_matrix(comm_lab_z_xy, comm_c_z_xy, L).todense())

print("[Z, YZ]")
print(-4j * mat_xz - pauli_func.create_TP_operator_matrix(comm_lab_z_yz, comm_c_z_yz, L).todense())

#

print("[X, Y]")
print(4j * mat_z - pauli_func.create_TP_operator_matrix(comm_lab_x_y, comm_c_x_y, L).todense())

print("[X, XY]")
print(4j * mat_xz - pauli_func.create_TP_operator_matrix(comm_lab_x_xy, comm_c_x_xy, L).todense())

print("[X, YZ]")
print(4j * (mat_zz - mat_yy) - pauli_func.create_TP_operator_matrix(comm_lab_x_yz, comm_c_x_yz, L).todense())

# tests
# np.set_printoptions(3, linewidth=200)
#
# h = 0.2
# g = 0.5

# check_matrices(h, g)
# check_commutator(h, g)

# A_1, A_2 = compute_agp_from_spectrum(h, g, 0.00001)
#
# ham_tot = mat_zz - h * mat_z - g * mat_x
# ham_deriv_1 = -mat_z
# ham_deriv_2 = -mat_x
#
# ev, evec = np.linalg.eigh(ham_tot)

# # test off-diagonal equality
# # transform into moving frame
#
# ham_tot = evec.T.conj() @ ham_tot @ evec
# ham_deriv_1 = evec.T.conj() @ ham_deriv_1 @ evec
# ham_deriv_2 = evec.T.conj() @ ham_deriv_2 @ evec
#
# mat_x = evec.T.conj() @ mat_x @ evec
# mat_y = evec.T.conj() @ mat_y @ evec
# mat_z = evec.T.conj() @ mat_z @ evec
#
# mat_xy = evec.T.conj() @ mat_xy @ evec
# mat_xz = evec.T.conj() @ mat_xz @ evec
# mat_yz = evec.T.conj() @ mat_yz @ evec
#
# mat_xx = evec.T.conj() @ mat_xx @ evec
# mat_yy = evec.T.conj() @ mat_yy @ evec
# mat_zz = evec.T.conj() @ mat_zz @ evec

# print("X")
# print(mat_x)
# print("Y")
# print(mat_y)
# print("Z")
# print(mat_z)
# print("XY")
# print(mat_xy)
# print("XZ")
# print(mat_xz)
# print("YZ")
# print(mat_yz)
# print("XX")
# print(mat_xx)
# print("YY")
# print(mat_yy)
# print("ZZ")
# print(mat_zz)
#
# print("H")
# print(ham_tot)
# print("H_z")
# print(ham_deriv_1)
# print("-i[A_z, H]")
# print(-1.j * commutator_matrix(A_1, ham_tot))
#
# print("H_x")
# print(ham_deriv_2)
# print("-i[A_x, H]")
# print(-1.j * commutator_matrix(A_2, ham_tot))
#
#
# # test commutator
# print("||[H, H_i + 1.j[A_i, H]||")
# print(np.linalg.norm(commutator_matrix(ham_tot, (ham_deriv_1 + 1.j * commutator_matrix(A_1, ham_tot)))))
# print(np.linalg.norm(commutator_matrix(ham_tot, (ham_deriv_2 + 1.j * commutator_matrix(A_2, ham_tot)))))

# # decompose into operators
# coeff_y_1 = np.trace(mat_y @ A_1) / np.trace(mat_y @ mat_y)
# coeff_xy_1 = np.trace(mat_xy @ A_1) / np.trace(mat_xy @ mat_xy)
# coeff_yz_1 = np.trace(mat_yz @ A_1) / np.trace(mat_yz @ mat_yz)
#
# A_sum_1 = coeff_y_1 * mat_y + coeff_xy_1 * mat_xy + coeff_yz_1 * mat_yz

# print(A_1)
# print(A_sum_1)
# print(np.linalg.norm(A_1 - A_sum_1))

# coeff_y_2 = np.trace(mat_y @ A_2) / np.trace(mat_y @ mat_y)
# coeff_xy_2 = np.trace(mat_xy @ A_2) / np.trace(mat_xy @ mat_xy)
# coeff_yz_2 = np.trace(mat_yz @ A_2) / np.trace(mat_yz @ mat_yz)
#
# A_sum_2 = coeff_y_2 * mat_y + coeff_xy_2 * mat_xy + coeff_yz_2 * mat_yz


# print(A_2)
# print(A_sum_2)
# print(np.linalg.norm(A_2 - A_sum_2))
# P = Pmatrix(h, g)
# R_Z = RvectorZ(h, g)
# R_X = RvectorX(h, g)
#
# print("coefficients:")
# alpha_Z = 0.5 * np.linalg.inv(P) @ R_Z
# print(coeff_y_1, coeff_xy_1, coeff_yz_1)
# print(alpha_Z)
#
# alpha_X = 0.5 * np.linalg.inv(P) @ R_X
# print(coeff_y_2, coeff_xy_2, coeff_yz_2)
# print(alpha_X)








