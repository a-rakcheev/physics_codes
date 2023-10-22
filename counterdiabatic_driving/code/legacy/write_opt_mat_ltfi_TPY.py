# write script to submit jobs and scripts for the jobs, each job passes
# variables to a python program

import numpy as np

# parameters

L = 12
processes = 32
l = 5


# file names
qsub = "qsub_opt_TPY.sh"
name_of_job = "opt_TPY_"


with open("script_opt_TPY.sh", "w") as scriptfile:

    scriptfile.write("#!/bin/bash \n")
    scriptfile.write("\n")
    scriptfile.write("#$ -q std.q \n")
    scriptfile.write("#$ -N " + name_of_job + "\n")
    scriptfile.write("#$ -o " + name_of_job + ".out \n")
    scriptfile.write("#$ -M Artem.Rakcheev@uibk.ac.at \n")
    scriptfile.write("#$ -m ea \n")
    scriptfile.write("#$ -j yes \n")
    scriptfile.write("#$ -wd /scratch/c7051098/agp \n")
    scriptfile.write("#$ -l h_vmem=2G \n")
    scriptfile.write("#$ -pe openmp " + str(processes) + "\n")
    scriptfile.write("\n")
    scriptfile.write("export OMP_NUM_THREADS=1 \n")
    scriptfile.write("$CONDA_PREFIX/envs/Py27/bin/python ")
    scriptfile.write("write_optimization_matrices_ltfi_TPY_cluster.py " + str(L) + " " + str(l) + " " + str(processes))

with open(qsub, "a") as jobfile:

    jobfile.write("qsub script_opt_TPY.sh \n")