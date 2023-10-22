# write script to submit jobs and scripts for the jobs, each job passes
# variables to a python program

import numpy as np

# parameters
l = 6
L = 11
processes = [4, 8]
op_names_1 = ["x", "x"]
op_names_2 = ["z", "zz"]

# file names
qsub = "qsub_ltfi_opt.sh"
name_of_job = "ltfi_opt"

for idx, proc in enumerate(processes):

    job = "_P_" + op_names_1[idx].upper() + "_" + op_names_2[idx].upper()
    namestring = "_L" + str(L) + "_l" + str(l) + job

    with open("script_opt_" + str(idx) + ".sh", "w") as scriptfile:

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
        scriptfile.write("#$ -pe openmpi-fillup " + str(proc) + "\n")
        scriptfile.write("\n")

        scriptfile.write("module load openmpi \n")
        scriptfile.write("\n")

        scriptfile.write("mpiexec -np $NSLOTS $CONDA_PREFIX/bin/python ")

        if op_names_1[idx] == op_names_2[idx]:
            scriptfile.write("optimization_matrices_ltfi_symmetric_mpi.py "
                             + str(l) + " " + str(L) + " " + op_names_1[idx])

        else:
            scriptfile.write("optimization_matrices_ltfi_full_mpi.py "
                             + str(l) + " " + str(L) + " " + op_names_1[idx] + " " + op_names_2[idx])

    with open(qsub, "a") as jobfile:
        jobfile.write("qsub script_opt_" + str(idx) + ".sh \n")