# write script to submit jobs and scripts for the jobs, each job passes
# variables to a python program

import numpy as np

# parameters

L = 6
res_x = 50
res_y = 50
order = 15
l = 2
S = 2
beta = 10.0
processes = 32

computations = ["subspace", "finite_temperature_coherent", "finite_temperature_incoherent", "infinite_temperature_coherent", "infinite_temperature_incoherent", "chi_metric"]
# file names

qsub = "qsub_process_agp.sh"
name_of_job = "process_agp_"

for idx, comp in enumerate(computations):

    namestring = str(idx)

    with open("script_process_agp_" + str(idx) + ".sh", "w") as scriptfile:

        scriptfile.write("#!/bin/bash \n")
        scriptfile.write("\n")
        scriptfile.write("#$ -q std.q \n")
        scriptfile.write("#$ -N " + name_of_job + namestring + "\n")
        scriptfile.write("#$ -o " + name_of_job + namestring + ".out \n")
        scriptfile.write("#$ -M Artem.Rakcheev@uibk.ac.at \n")
        scriptfile.write("#$ -m ea \n")
        scriptfile.write("#$ -j yes \n")
        scriptfile.write("#$ -wd /scratch/c7051098/agp \n")
        scriptfile.write("#$ -l h_vmem=500M \n")
        scriptfile.write("#$ -pe openmp " + str(processes) + "\n")
        scriptfile.write("\n")
        scriptfile.write("export OMP_NUM_THREADS=1 \n")
        scriptfile.write("$CONDA_PREFIX/envs/Py27/bin/python ")
        scriptfile.write("process_optimized_agp.py " + str(L) + " " + str(res_x) + " " + str(res_y)
                         + " " + str(order) + " " + str(processes) + " " + str(l) + " " + str(S) + " " + str(beta) + " " + comp)

    with open(qsub, "a") as jobfile:

        jobfile.write("qsub script_process_agp_" + str(idx) + ".sh \n")