# write script to submit jobs and scripts for the jobs, each job passes
# variables to a python program

import numpy as np

# parameters
lengths = [4]
processes = [8]
symmetry = "TPFXY"
program = "symmetric"

# file names
qsub = "qsub_opt.sh"
name_of_job = "opt_"

for idx, l in enumerate(lengths):

    job = symmetry + "_" + program
    namestring = "_l" + str(l) + job

    with open("script_opt_" + str(idx) + ".sh", "w") as scriptfile:

        scriptfile.write("#!/bin/bash \n")
        scriptfile.write("\n")
        scriptfile.write("#$ -q short.q \n")
        scriptfile.write("#$ -N " + name_of_job + namestring + "\n")
        scriptfile.write("#$ -o " + name_of_job + namestring + ".out \n")
        scriptfile.write("#$ -M Artem.Rakcheev@uibk.ac.at \n")
        scriptfile.write("#$ -m ea \n")
        scriptfile.write("#$ -j yes \n")
        scriptfile.write("#$ -wd /scratch/c7051098/agp \n")
        scriptfile.write("#$ -l h_vmem=250M \n")
        scriptfile.write("#$ -l h_rt=10:00 \n")
        scriptfile.write("#$ -pe openmpi-4perhost " + str(processes[idx]) + "\n")
        scriptfile.write("\n")

        scriptfile.write("module load openmpi \n")
        scriptfile.write("\n")

        scriptfile.write("mpiexec -np $NSLOTS $CONDA_PREFIX/bin/python ")

        if program == "symmetric":
            scriptfile.write("optimize_matrix_" + symmetry + "_exact_symmetric.py " + str(l))

        else:
            scriptfile.write("optimize_matrix_" + symmetry + "_exact_full.py " + str(l))

    with open(qsub, "a") as jobfile:
        jobfile.write("qsub script_opt_" + str(idx) + ".sh \n")