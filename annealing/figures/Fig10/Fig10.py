from shutil import which
import sys
sys.path.append("/home/artem/Dropbox/codebase/")
sys.path.append("D:/Dropbox/codebase/")
sys.path.append("C:/Users/ARakc/Dropbox/codebase/")
sys.path.append("C:/Users/rakche_a-adm/Dropbox/codebase/")


import zipfile
import io
import json
import numpy as np

from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmasher as cmr

# parameters

# # Fig 10a
# N = 10
# instances = np.arange(1, 101, 1)

# Fig 10b
N = 20
instances = np.arange(1, 201, 1)

# real-time
dt = 0.01
steps = 100
Tl = np.arange(1., 51., 1)

# simulated annealing
R = 1000000
MCSl_sa = np.arange(100, 10100, 100)

# simulated quantum annealing
n = 8
beta = 10.0
MCSl_sqa = np.arange(100, 10100, 100)

prefix = "C:/Users/rakche_a-adm/Downloads/data_sk/data/"

plt.figure(1, figsize=(7.5, 2.5))
labelsize = 11
ticksize = 9

counts_qa = np.zeros((len(instances), len(Tl)))
counts_sa = np.zeros((len(instances), len(MCSl_sa)))
counts_sqa = np.zeros((len(instances), len(MCSl_sqa)))


# real-time
zip_file = prefix + "instance_data_qa_N" + str(N) + ".zip"
with zipfile.ZipFile(zip_file) as zipper:

    for l, inst in enumerate(instances):
        for k, T in enumerate(Tl):
            # read data
            json_file = "sk_ising_bimodal_nofield_instance_data_N" + str(N) + "_inst" + str(inst) + "_T=" + str(T).replace(".", "-") + "_dt=" \
                + str(dt).replace(".", "-") + "_steps=" + str(steps) + ".json"

            data = []
            with io.BufferedReader(zipper.open(json_file, mode='r')) as file:
                data.append(json.load(file))


            x = []
            y = []
            count_x = []
            count_y = []

            for i in range(len(data[0]) - 1):

                count_x.append(data[0][i][0])
                count_y.append(len(data[0][i][1]))

                for j in range(len(data[0][i][1])):

                    x.append(data[0][i][1][j])
                    y.append(data[0][i][0])

            x = np.array(x)
            y = np.array(y)
            count_x = np.array(count_x)
            count_y = np.array(count_y)

            counts_qa[l, k] = len(x)


# simulated_annealing
zip_file = prefix + "instance_data_sa_N" + str(N) + ".zip"
with zipfile.ZipFile(zip_file) as zipper:

    for l, inst in enumerate(instances):

        for k, MCS_sa in enumerate(MCSl_sa):

            # read data
            json_file = "sa_nofield_instance_data_N" + str(N) + "_inst" + str(inst) + "_MCS" + str(MCS_sa) + "_R=" + str(R) + ".json"

            data = []
            with io.BufferedReader(zipper.open(json_file, mode='r')) as file:
                data.append(json.load(file))


            x = []
            y = []
            count_x = []
            count_y = []

            for i in range(len(data[0]) - 1):

                count_x.append(data[0][i][0])
                count_y.append(len(data[0][i][1]))

                for j in range(len(data[0][i][1])):

                    x.append(data[0][i][1][j])
                    y.append(data[0][i][0])

            x = np.array(x)
            y = np.array(y)
            count_x = np.array(count_x)
            count_y = np.array(count_y)

            counts_sa[l, k] = len(x)


# simulated_annealing
zip_file = prefix + "instance_data_sqa_N" + str(N) + ".zip"
with zipfile.ZipFile(zip_file) as zipper:
    
    for l, inst in enumerate(instances):
        for k, MCS_sqa in enumerate(MCSl_sqa):

            # read data
            json_file = "sqa_nofield_instance_data_N" + str(N) + "_n" + str(n) + "_inst" + str(inst) + "_MCS" + str(MCS_sqa) + "_beta=" + str(beta).replace(".", "-") + "_R=" + str(R) + ".json"
            data = []
            with io.BufferedReader(zipper.open(json_file, mode='r')) as file:
                data.append(json.load(file))


            x = []
            y = []
            count_x = []
            count_y = []

            for i in range(len(data[0]) - 1):

                count_x.append(data[0][i][0])
                count_y.append(len(data[0][i][1]))

                for j in range(len(data[0][i][1])):

                    x.append(data[0][i][1][j])
                    y.append(data[0][i][0])

            x = np.array(x)
            y = np.array(y)
            count_x = np.array(count_x)
            count_y = np.array(count_y)

            counts_sqa[l, k] = len(x)


mean_qa = np.mean(counts_qa, axis=0)
std_qa = np.std(counts_qa, axis=0)

mean_sa = np.mean(counts_sa, axis=0)
std_sa = np.std(counts_sa, axis=0)

mean_sqa = np.mean(counts_sqa, axis=0)
std_sqa = np.std(counts_sqa, axis=0)

maxcount = max(np.max(mean_qa + std_qa), np.max(mean_sa + std_sa), np.max(mean_sqa + std_sqa))

plt.subplot(1, 3, 1)

plt.fill_between(Tl, mean_qa - std_qa ,mean_qa + std_qa, color="lightblue", zorder=2)
plt.plot(Tl, mean_qa, ls="-", marker="", color="black", lw=1, markersize=3, zorder=3)

plt.grid(lw=1, color="grey", zorder=1)

plt.ylim(0, maxcount)
plt.xlim(0, Tl[-1])

plt.xlabel(r"$T$", fontsize=labelsize)
plt.ylabel(r"$\textrm{number of sign changes}$", fontsize=labelsize)
plt.title(r"$\mathrm{QA}$", fontsize=10)
plt.tick_params("both", which="major", labelsize=ticksize)

if N == 10:
    plt.yticks([0, 5, 10, 15])

plt.subplot(1, 3, 2)
plt.fill_between(MCSl_sa, mean_sa - std_sa ,mean_sa + std_sa, color="lightblue", zorder=2)
plt.plot(MCSl_sa, mean_sa, ls="-", marker="", color="black", lw=1, markersize=3, zorder=3)

plt.grid(lw=1, color="grey", zorder=1)

plt.gca().set_yticklabels([])
plt.ylim(0, maxcount)
plt.xlim(0, MCSl_sa[-1])

plt.xlabel(r"$\mathrm{MCS}$", fontsize=labelsize)
plt.title(r"$\mathrm{SA}$", fontsize=10)
plt.tick_params("both", which="major", labelsize=ticksize)
if N == 10:
    plt.yticks([0, 5, 10, 15])

plt.subplot(1, 3, 3)
plt.fill_between(MCSl_sqa, mean_sqa - std_sqa ,mean_sqa + std_sqa, color="lightblue", zorder=2)
plt.plot(MCSl_sqa, mean_sqa, ls="-", marker="", color="black", lw=1, markersize=3, zorder=3)

plt.grid(lw=1, color="grey", zorder=1)

plt.gca().set_yticklabels([])
plt.ylim(0, maxcount)
plt.xlim(0, MCSl_sqa[-1])

plt.xlabel(r"$\mathrm{MCS}$", fontsize=labelsize)
plt.title(r"$\mathrm{SQA}$", fontsize=10)
plt.tick_params("both", which="major", labelsize=ticksize)
if N == 10:
    plt.yticks([0, 5, 10, 15])
    
plt.subplots_adjust(wspace=0.2, hspace=0.35, top=0.925, bottom=0.15, left=0.07, right=0.975)
plt.savefig("Fig10b.pdf", format="pdf")
plt.show()




