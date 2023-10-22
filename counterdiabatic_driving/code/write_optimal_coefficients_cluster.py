# write script to submit jobs and scripts for the jobs, each job passes
# variables to a python program

import numpy as np

# parameters
l = 3
number_of_processes = 2
res_1 = 5
res_2 = 5

start1 = 1.e-6
start2 = 1.e-6
end_1 = 2.
end_2 = 2.

s_0 = 0
s_1 = 0
s_2 = 0

op_name_0 = "xx"
op_name_1 = "yy"
op_name_2 = "zz"


sym_name = "TPFXY"
# file names
qsub = "qsub_opt_coeff.sh"
name_of_job = "opt_coeff_l" + str(l) + "_"

namestring = op_name_0.upper() + "_" + op_name_1.upper() + "_" + op_name_2.upper()

with open("script_opt_coeff.sh", "w") as scriptfile:

    scriptfile.write("#!/bin/bash \n")
    scriptfile.write("\n")
    scriptfile.write("#$ -q std.q \n")
    scriptfile.write("#$ -N " + name_of_job + namestring + "\n")
    scriptfile.write("#$ -o " + name_of_job + namestring + ".out \n")
    scriptfile.write("#$ -M Artem.Rakcheev@uibk.ac.at \n")
    scriptfile.write("#$ -m ea \n")
    scriptfile.write("#$ -j yes \n")
    scriptfile.write("#$ -wd /scratch/c7051098/agp \n")
    scriptfile.write("#$ -l h_vmem=1G \n")
    scriptfile.write("#$ -pe openmpi-fillup " + str(number_of_processes) + "\n")
    scriptfile.write("\n")
    scriptfile.write("module load openmpi \n")
    scriptfile.write("\n")
    scriptfile.write("mpiexec -n $NSLOTS $CONDA_PREFIX/bin/python ")
    scriptfile.write("optimal_coefficients_" + sym_name + "_cluster.py "
                     + str(l) + " " + str(number_of_processes) + " " + str(res_1) + " " + str(res_2)
                     + " " + str(start1) + " " + str(start2) + " " + str(end_1) + " " + str(end_2)
                     + " " + str(s_0) + " " + str(s_1) + " " + str(s_2) + " " + str(op_name_0)
                     + " " + str(op_name_1) + " " + str(op_name_2))

with open(qsub, "a") as jobfile:

    jobfile.write("qsub script_opt_coeff.sh \n")