# write script to submit jobs and scripts for the jobs, each job passes
# variables to a python program

import numpy as np

# parameters

L = 26
k_idx = 0
k_name = "0"

# file names
qsub = "qsub_idx.sh"
name_of_job = "dqs_idx_"


with open("script_idx.sh", "w") as scriptfile:

    scriptfile.write("#!/bin/bash \n")
    scriptfile.write("\n")
    scriptfile.write("#$ -q std.q \n")
    scriptfile.write("#$ -N " + name_of_job + "\n")
    scriptfile.write("#$ -o " + name_of_job + ".out \n")
    scriptfile.write("#$ -M Artem.Rakcheev@uibk.ac.at \n")
    scriptfile.write("#$ -m ea \n")
    scriptfile.write("#$ -j yes \n")
    scriptfile.write("#$ -wd /scratch/c7051098/dqs/operators_TPY \n")
    scriptfile.write("#$ -l h_vmem=4G \n")
    scriptfile.write("#$ -pe openmp 1 \n")
    scriptfile.write("\n")
    scriptfile.write("export OMP_NUM_THREADS=$NSLOTS \n")
    scriptfile.write("$CONDA_PREFIX/bin/python \n")
    scriptfile.write("python create_idx_table_T.py " + str(L) + " " + str(k_idx) + " "
                     + k_name)
    
    # scriptfile.write("python create_idx_table_TP.py " + str(L) + " " + str(k_idx) + " "
    #                  + k_name)
   # scriptfile.write("python create_TP_operators.py " + str(L) + " " + str(k_idx) + " "
   #                   + k_name)
   # scriptfile.write("python create_TP_operators_diagonal.py " + str(L) + " " + str(k_idx) + " "
   #                   + k_name)

with open(qsub, "a") as jobfile:

    jobfile.write("qsub script_idx.sh \n")
