import sys
sys.path.append("/home/artem/Dropbox/codebase/")
sys.path.append("C:/Users/rakche_a-adm/Dropbox/codebase/")
sys.path.append("/run/media/artem/d64a440d-7b6c-45ed-aa6c-c9d991212d84/Dropbox/codebase/")

import pauli_string_functions as pauli_func
import numpy as np

L = 2
  
# create TPY operators from file
tr_I = 2 ** L

# operators
lab_x, c_x = pauli_func.TP_operator_from_string_compact("x", L)
lab_z, c_z = pauli_func.TP_operator_from_string_compact("z", L)
lab_zz, c_zz = pauli_func.TP_operator_from_string_compact("zz", L)

lab_y, c_y = pauli_func.TP_operator_from_string_compact("y", L)
lab_xy, c_xy = pauli_func.TP_operator_from_string_compact("xy", L)
lab_yz, c_yz = pauli_func.TP_operator_from_string_compact("yz", L)

print("Operators:")
pauli_func.print_operator_balanced(lab_x, c_x)
pauli_func.print_operator_balanced(lab_z, c_z)
pauli_func.print_operator_balanced(lab_zz, c_zz)

pauli_func.print_operator_balanced(lab_y, c_y)
pauli_func.print_operator_balanced(lab_xy, c_xy)
pauli_func.print_operator_balanced(lab_yz, c_yz)

# commutators
lab_comm_x_y, c_comm_x_y = pauli_func.commute_operators_TP(lab_x, lab_y, c_x, c_y)
lab_comm_z_y, c_comm_z_y = pauli_func.commute_operators_TP(lab_z, lab_y, c_z, c_y)
lab_comm_zz_y, c_comm_zz_y = pauli_func.commute_operators_TP(lab_zz, lab_y, c_zz, c_y)

lab_comm_x_xy, c_comm_x_xy = pauli_func.commute_operators_TP(lab_x, lab_xy, c_x, c_xy)
lab_comm_z_xy, c_comm_z_xy = pauli_func.commute_operators_TP(lab_z, lab_xy, c_z, c_xy)
lab_comm_zz_xy, c_comm_zz_xy = pauli_func.commute_operators_TP(lab_zz, lab_xy, c_zz, c_xy)

lab_comm_x_yz, c_comm_x_yz = pauli_func.commute_operators_TP(lab_x, lab_yz, c_x, c_yz)
lab_comm_z_yz, c_comm_z_yz = pauli_func.commute_operators_TP(lab_z, lab_yz, c_z, c_yz)
lab_comm_zz_yz, c_comm_zz_yz = pauli_func.commute_operators_TP(lab_zz, lab_yz, c_zz, c_yz)

print("Comm with Y:")
pauli_func.print_operator_balanced(lab_comm_x_y, c_comm_x_y)
pauli_func.print_operator_balanced(lab_comm_z_y, c_comm_z_y)
pauli_func.print_operator_balanced(lab_comm_zz_y, c_comm_zz_y)

print("Comm with XY:")
pauli_func.print_operator_balanced(lab_comm_x_xy, c_comm_x_xy)
pauli_func.print_operator_balanced(lab_comm_z_xy, c_comm_z_xy)
pauli_func.print_operator_balanced(lab_comm_zz_xy, c_comm_zz_xy)

print("Comm with YZ:")
pauli_func.print_operator_balanced(lab_comm_x_yz, c_comm_x_yz)
pauli_func.print_operator_balanced(lab_comm_z_yz, c_comm_z_yz)
pauli_func.print_operator_balanced(lab_comm_zz_yz, c_comm_zz_yz)
