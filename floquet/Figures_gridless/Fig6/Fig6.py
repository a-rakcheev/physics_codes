import numpy as np
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def cm_to_inch(value):
    return value/2.54

# parameters
g = 1.0
m = 1.0
h = 1.0
start = 0.0
res = 1000000

fig = plt.figure(1, figsize=(cm_to_inch(17), cm_to_inch(6)), constrained_layout=False)

for l, L in enumerate([6, 8, 10, 12]):

       plt.subplot(1, 4, l + 1)

       if L == 6:
              size = 13
              idx_final_0 = 28
              idx_initial_pi = 28
              tau_initial = 0.0
              tau_final = 0.0
              end = 3.141592653589793
              idx_hr = 0
              idx_res_upper = 28
              annotation = r"$(\mathrm{a}) \; L=6$"

              name = "hhz_scaled_gap_widths_and_energies_L" + str(L) \
              + "_tau_start" + str(start).replace(".", "-") + "_tau_end" + str(end).replace(".", "-") \
              + "_tau_res" + str(res) + ".npz"

              data = np.load(name)
              indices = data["idx"]
              widths = data["widths"]

              widths_x_0 = data["widths_x_0"]
              widths_z_0 = data["widths_z_0"]
              widths_zz_0 = data["widths_zz_0"]
              widths_avg_0 = 0.5 * (widths_x_0 + widths_z_0 + widths_zz_0)

              widths_x_pi = data["widths_x_pi"]
              widths_z_pi = data["widths_z_pi"]
              widths_zz_pi = data["widths_zz_pi"]
              widths_avg_pi = 0.5 * (widths_x_pi + widths_z_pi + widths_zz_pi)

              tau_min = start + indices[0] * (end - start) / res

              # plot results
              markersize = 7
              plt.axvline(2. * np.pi / size, color="black", ls="--", lw=0.75, zorder=4)

              # these lines are for all L
              scat = plt.scatter(widths[idx_hr:idx_final_0 + 1], tau_min[idx_hr:idx_final_0 + 1],  marker="o", s=markersize, c=np.absolute(widths_avg_0[idx_hr:idx_final_0 + 1]), cmap="viridis_r", vmin=0., vmax=5., zorder=3)
              scat = plt.scatter(widths[idx_initial_pi: idx_res_upper + 1], tau_min[idx_initial_pi: idx_res_upper + 1],  marker="o", s=markersize, c=np.absolute(widths_avg_pi[idx_initial_pi:idx_res_upper + 1]), cmap="viridis_r", vmin=0., vmax=5., zorder=3)

              for i in range(idx_res_upper, len(tau_min)):

                     if widths[i] < 1.e-5:
                            scat = plt.scatter(widths[i:i + 1], tau_min[i:i + 1], marker="o", s=markersize,
                                                        c="silver", cmap="viridis_r", vmin=0., vmax=5., zorder=3)

                     else:
                            scat = plt.scatter(widths[i:i + 1], tau_min[i:i + 1], marker="o", s=markersize,
                                                        c=np.absolute(widths_avg_pi[i:i + 1]), cmap="viridis_r", vmin=0.,
                                                        vmax=5., zorder=3)


              scat = plt.scatter(widths[idx_final_0 + 1: idx_initial_pi], tau_min[idx_final_0 + 1: idx_initial_pi],  marker="o", s=markersize, color="gray", zorder=3)

              plt.axhline(tau_initial, color="gray", lw=1, zorder=2)
              plt.axhline(tau_final, color="gray", lw=1, zorder=2)
              plt.annotate(r"$\bar{\Delta}$", (0.25 * np.pi / size, 0.1), fontsize=12, zorder=5)


              # plt.grid(color="black", lw=0.5, zorder=1)
              plt.xlabel(r"$\Delta_{c}$", fontsize=12)

              # # ylabel L = 6 only
              plt.ylabel(r"$\tau$", fontsize=12)

              plt.xscale("log")
              plt.xlim(1.e-8, 2.0)
              plt.xticks([1.e-6, 1.e-2], fontsize=12)


              plt.ylim(0.0, np.pi)

              # # L = 6
              plt.yticks([0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi], [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"], fontsize=12)

              # annotate with system size
              # plt.gcf().text(0.02, 0.02, annotation, fontsize=8)
              plt.title(r"$L=" + str(L) + r"$", fontsize=12)


       elif L == 8:

              size = 30
              end = 3.141
              idx_final_0 = 97
              idx_initial_pi = 155
              tau_initial = 1.49
              tau_final = 2.105
              idx_res_upper = 190
              annotation = r"$(\mathrm{b}) \; L=8$"

              name = "hhz_scaled_gap_widths_and_energies_L" + str(L) \
              + "_tau_start" + str(start).replace(".", "-") + "_tau_end" + str(end).replace(".", "-") \
              + "_tau_res" + str(res) + ".npz"

              data = np.load(name)
              indices = data["idx"]
              widths = data["widths"]

              widths_x_0 = data["widths_x_0"]
              widths_z_0 = data["widths_z_0"]
              widths_zz_0 = data["widths_zz_0"]
              widths_avg_0 = 0.5 * (widths_x_0 + widths_z_0 + widths_zz_0)

              widths_x_pi = data["widths_x_pi"]
              widths_z_pi = data["widths_z_pi"]
              widths_zz_pi = data["widths_zz_pi"]
              widths_avg_pi = 0.5 * (widths_x_pi + widths_z_pi + widths_zz_pi)

              tau_min = start + indices[0] * (end - start) / res

              # high res (only for  L=8, 10, 12 - for L=6 comment all lines including high res data)
              start_hr = 0.25
              end_hr = 1.0
              res_hr = 15000000
              name = "hhz_scaled_gap_widths_and_energies_from_gaps_L" + str(L) \
                     + "_tau_start" + str(start_hr).replace(".", "-") + "_tau_end" + str(end_hr).replace(".", "-") \
                     + "_tau_res" + str(res_hr) + ".npz"

              data = np.load(name)
              tau_min_hr = data["times"]
              widths_hr = data["widths"]
              widths_avg_hr = data["widths_avg_0"]

              # fibd index between high res and low res
              idx_hr = np.searchsorted(tau_min, 1.0, side="right")

              # find index between resolution
              idx_res = np.searchsorted(tau_min_hr, 0.7, side="right")


              # plot results
              markersize = 7
              plt.axvline(2. * np.pi / size, color="black", ls="--", lw=0.75, zorder=4)


              # high res (only for  L=8, 10, 12 - for L=6 comment all lines including high res data)
              for i in range(idx_res):

                     if widths_hr[i] < 1.e-5:

                            scat_hr = plt.scatter(widths_hr[i:i + 1], tau_min_hr[i:i + 1],  marker="o", s=markersize, c="silver", cmap="viridis_r", vmin=0., vmax=5., zorder=3)

                     else:
                            scat_hr = plt.scatter(widths_hr[i:i + 1], tau_min_hr[i:i + 1],  marker="o", s=markersize, c=np.absolute(widths_avg_hr[i:i + 1]), cmap="viridis_r", vmin=0., vmax=5., zorder=3)


              scat_hr = plt.scatter(widths_hr[idx_res:], tau_min_hr[idx_res:],  marker="o", s=markersize, c=np.absolute(widths_avg_hr[idx_res:]), cmap="viridis_r", vmin=0., vmax=5., zorder=3)

              # these lines are for all L
              scat = plt.scatter(widths[idx_hr:idx_final_0 + 1], tau_min[idx_hr:idx_final_0 + 1],  marker="o", s=markersize, c=np.absolute(widths_avg_0[idx_hr:idx_final_0 + 1]), cmap="viridis_r", vmin=0., vmax=5., zorder=3)
              scat = plt.scatter(widths[idx_initial_pi: idx_res_upper + 1], tau_min[idx_initial_pi: idx_res_upper + 1],  marker="o", s=markersize, c=np.absolute(widths_avg_pi[idx_initial_pi:idx_res_upper + 1]), cmap="viridis_r", vmin=0., vmax=5., zorder=3)

              for i in range(idx_res_upper, len(tau_min)):

                     if widths[i] < 1.e-5:
                            scat = plt.scatter(widths[i:i + 1], tau_min[i:i + 1], marker="o", s=markersize,
                                                        c="silver", cmap="viridis_r", vmin=0., vmax=5., zorder=3)

                     else:
                            scat = plt.scatter(widths[i:i + 1], tau_min[i:i + 1], marker="o", s=markersize,
                                                        c=np.absolute(widths_avg_pi[i:i + 1]), cmap="viridis_r", vmin=0.,
                                                        vmax=5., zorder=3)


              scat = plt.scatter(widths[idx_final_0 + 1: idx_initial_pi], tau_min[idx_final_0 + 1: idx_initial_pi],  marker="o", s=markersize, color="gray", zorder=3)

              plt.axhline(tau_initial, color="gray", lw=1, zorder=2)
              plt.axhline(tau_final, color="gray", lw=1, zorder=2)
              plt.annotate(r"$\bar{\Delta}$", (3 * np.pi / size, 0.1), fontsize=12, zorder=5)


              # plt.grid(color="black", lw=0.5, zorder=1)
              plt.xlabel(r"$\Delta_{c}$", fontsize=12)

              plt.xscale("log")
              plt.xlim(1.e-8, 2.0)
              plt.xticks([1.e-6, 1.e-2], fontsize=12)


              plt.ylim(0.0, np.pi)

              # L=8, 10, 12
              plt.yticks([0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi], [], fontsize=12)

              # annotate with system size
              # plt.gcf().text(0.02, 0.02, annotation, fontsize=8)
              plt.title(r"$L=" + str(L) + r"$", fontsize=12)


       elif L == 10:
              size = 78
              idx_final_0 = 675
              idx_initial_pi = 1125
              end = 3.141592653589793
              idx_res_upper = 1320
              tau_initial = 1.36
              tau_final = 2.18
              annotation = r"$(\mathrm{c}) \; L=10$"

              name = "hhz_scaled_gap_widths_and_energies_L" + str(L) \
              + "_tau_start" + str(start).replace(".", "-") + "_tau_end" + str(end).replace(".", "-") \
              + "_tau_res" + str(res) + ".npz"

              data = np.load(name)
              indices = data["idx"]
              widths = data["widths"]

              widths_x_0 = data["widths_x_0"]
              widths_z_0 = data["widths_z_0"]
              widths_zz_0 = data["widths_zz_0"]
              widths_avg_0 = 0.5 * (widths_x_0 + widths_z_0 + widths_zz_0)

              widths_x_pi = data["widths_x_pi"]
              widths_z_pi = data["widths_z_pi"]
              widths_zz_pi = data["widths_zz_pi"]
              widths_avg_pi = 0.5 * (widths_x_pi + widths_z_pi + widths_zz_pi)

              tau_min = start + indices[0] * (end - start) / res

              # high res (only for  L=8, 10, 12 - for L=6 comment all lines including high res data)
              start_hr = 0.25
              end_hr = 1.0
              res_hr = 15000000
              name = "hhz_scaled_gap_widths_and_energies_from_gaps_L" + str(L) \
                     + "_tau_start" + str(start_hr).replace(".", "-") + "_tau_end" + str(end_hr).replace(".", "-") \
                     + "_tau_res" + str(res_hr) + ".npz"

              data = np.load(name)
              tau_min_hr = data["times"]
              widths_hr = data["widths"]
              widths_avg_hr = data["widths_avg_0"]

              # fibd index between high res and low res
              idx_hr = np.searchsorted(tau_min, 1.0, side="right")

              # find index between resolution
              idx_res = np.searchsorted(tau_min_hr, 0.7, side="right")


              # plot results
              markersize = 7
              plt.axvline(2. * np.pi / size, color="black", ls="--", lw=0.75, zorder=4)


              # high res (only for  L=8, 10, 12 - for L=6 comment all lines including high res data)
              for i in range(idx_res):

                     if widths_hr[i] < 1.e-5:

                            scat_hr = plt.scatter(widths_hr[i:i + 1], tau_min_hr[i:i + 1],  marker="o", s=markersize, c="silver", cmap="viridis_r", vmin=0., vmax=5., zorder=3)

                     else:
                            scat_hr = plt.scatter(widths_hr[i:i + 1], tau_min_hr[i:i + 1],  marker="o", s=markersize, c=np.absolute(widths_avg_hr[i:i + 1]), cmap="viridis_r", vmin=0., vmax=5., zorder=3)


              scat_hr = plt.scatter(widths_hr[idx_res:], tau_min_hr[idx_res:],  marker="o", s=markersize, c=np.absolute(widths_avg_hr[idx_res:]), cmap="viridis_r", vmin=0., vmax=5., zorder=3)

              # these lines are for all L
              scat = plt.scatter(widths[idx_hr:idx_final_0 + 1], tau_min[idx_hr:idx_final_0 + 1],  marker="o", s=markersize, c=np.absolute(widths_avg_0[idx_hr:idx_final_0 + 1]), cmap="viridis_r", vmin=0., vmax=5., zorder=3)
              scat = plt.scatter(widths[idx_initial_pi: idx_res_upper + 1], tau_min[idx_initial_pi: idx_res_upper + 1],  marker="o", s=markersize, c=np.absolute(widths_avg_pi[idx_initial_pi:idx_res_upper + 1]), cmap="viridis_r", vmin=0., vmax=5., zorder=3)

              for i in range(idx_res_upper, len(tau_min)):

                     if widths[i] < 1.e-5:
                            scat = plt.scatter(widths[i:i + 1], tau_min[i:i + 1], marker="o", s=markersize,
                                                        c="silver", cmap="viridis_r", vmin=0., vmax=5., zorder=3)

                     else:
                            scat = plt.scatter(widths[i:i + 1], tau_min[i:i + 1], marker="o", s=markersize,
                                                        c=np.absolute(widths_avg_pi[i:i + 1]), cmap="viridis_r", vmin=0.,
                                                        vmax=5., zorder=3)


              scat = plt.scatter(widths[idx_final_0 + 1: idx_initial_pi], tau_min[idx_final_0 + 1: idx_initial_pi],  marker="o", s=markersize, color="gray", zorder=3)

              plt.axhline(tau_initial, color="gray", lw=1, zorder=2)
              plt.axhline(tau_final, color="gray", lw=1, zorder=2)
              plt.annotate(r"$\bar{\Delta}$", (3 * np.pi / size, 0.1), fontsize=12, zorder=5)


              # plt.grid(color="black", lw=0.5, zorder=1)
              plt.xlabel(r"$\Delta_{c}$", fontsize=12)

              plt.xscale("log")
              plt.xlim(1.e-8, 2.0)
              plt.xticks([1.e-6, 1.e-2], fontsize=12)


              plt.ylim(0.0, np.pi)

              # L=8, 10, 12
              plt.yticks([0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi], [], fontsize=12)

              # annotate with system size
              # plt.gcf().text(0.02, 0.02, annotation, fontsize=8)
              plt.title(r"$L=" + str(L) + r"$", fontsize=12)

       else:
              L = 12
              size = 224
              idx_final_0 = 5500
              idx_initial_pi = 9500
              idx_res_upper = 11195
              tau_initial = 1.28
              tau_final = 2.22
              end = 3.141592653589793
              annotation = r"$(\mathrm{d}) \; L=12$"

              name = "hhz_scaled_gap_widths_and_energies_L" + str(L) \
              + "_tau_start" + str(start).replace(".", "-") + "_tau_end" + str(end).replace(".", "-") \
              + "_tau_res" + str(res) + ".npz"

              data = np.load(name)
              indices = data["idx"]
              widths = data["widths"]

              widths_x_0 = data["widths_x_0"]
              widths_z_0 = data["widths_z_0"]
              widths_zz_0 = data["widths_zz_0"]
              widths_avg_0 = 0.5 * (widths_x_0 + widths_z_0 + widths_zz_0)

              widths_x_pi = data["widths_x_pi"]
              widths_z_pi = data["widths_z_pi"]
              widths_zz_pi = data["widths_zz_pi"]
              widths_avg_pi = 0.5 * (widths_x_pi + widths_z_pi + widths_zz_pi)

              tau_min = start + indices[0] * (end - start) / res

              # high res (only for  L=8, 10, 12 - for L=6 comment all lines including high res data)
              start_hr = 0.25
              end_hr = 1.0
              res_hr = 15000000
              name = "hhz_scaled_gap_widths_and_energies_from_gaps_L" + str(L) \
                     + "_tau_start" + str(start_hr).replace(".", "-") + "_tau_end" + str(end_hr).replace(".", "-") \
                     + "_tau_res" + str(res_hr) + ".npz"

              data = np.load(name)
              tau_min_hr = data["times"]
              widths_hr = data["widths"]
              widths_avg_hr = data["widths_avg_0"]

              # fibd index between high res and low res
              idx_hr = np.searchsorted(tau_min, 1.0, side="right")

              # find index between resolution
              idx_res = np.searchsorted(tau_min_hr, 0.7, side="right")


              # plot results
              markersize = 7
              plt.axvline(2. * np.pi / size, color="black", ls="--", lw=0.75, zorder=4)


              # high res (only for  L=8, 10, 12 - for L=6 comment all lines including high res data)
              for i in range(idx_res):

                     if widths_hr[i] < 1.e-5:

                            scat_hr = plt.scatter(widths_hr[i:i + 1], tau_min_hr[i:i + 1],  marker="o", s=markersize, c="silver", cmap="viridis_r", vmin=0., vmax=5., zorder=3)

                     else:
                            scat_hr = plt.scatter(widths_hr[i:i + 1], tau_min_hr[i:i + 1],  marker="o", s=markersize, c=np.absolute(widths_avg_hr[i:i + 1]), cmap="viridis_r", vmin=0., vmax=5., zorder=3)


              scat_hr = plt.scatter(widths_hr[idx_res:], tau_min_hr[idx_res:],  marker="o", s=markersize, c=np.absolute(widths_avg_hr[idx_res:]), cmap="viridis_r", vmin=0., vmax=5., zorder=3)

              # these lines are for all L
              scat = plt.scatter(widths[idx_hr:idx_final_0 + 1], tau_min[idx_hr:idx_final_0 + 1],  marker="o", s=markersize, c=np.absolute(widths_avg_0[idx_hr:idx_final_0 + 1]), cmap="viridis_r", vmin=0., vmax=5., zorder=3)
              scat = plt.scatter(widths[idx_initial_pi: idx_res_upper + 1], tau_min[idx_initial_pi: idx_res_upper + 1],  marker="o", s=markersize, c=np.absolute(widths_avg_pi[idx_initial_pi:idx_res_upper + 1]), cmap="viridis_r", vmin=0., vmax=5., zorder=3)

              for i in range(idx_res_upper, len(tau_min)):

                     if widths[i] < 1.e-5:
                            scat = plt.scatter(widths[i:i + 1], tau_min[i:i + 1], marker="o", s=markersize,
                                                        c="silver", cmap="viridis_r", vmin=0., vmax=5., zorder=3)

                     else:
                            scat = plt.scatter(widths[i:i + 1], tau_min[i:i + 1], marker="o", s=markersize,
                                                        c=np.absolute(widths_avg_pi[i:i + 1]), cmap="viridis_r", vmin=0.,
                                                        vmax=5., zorder=3)


              scat = plt.scatter(widths[idx_final_0 + 1: idx_initial_pi], tau_min[idx_final_0 + 1: idx_initial_pi],  marker="o", s=markersize, color="gray", zorder=3)

              plt.axhline(tau_initial, color="gray", lw=1, zorder=2)
              plt.axhline(tau_final, color="gray", lw=1, zorder=2)
              plt.annotate(r"$\bar{\Delta}$", (3 * np.pi / size, 0.1), fontsize=12, zorder=5)


              # plt.grid(color="black", lw=0.5, zorder=1)
              plt.xlabel(r"$\Delta_{c}$", fontsize=12)

              plt.xscale("log")
              plt.xlim(1.e-8, 2.0)
              plt.xticks([1.e-6, 1.e-2], fontsize=12)


              plt.ylim(0.0, np.pi)

              # L=8, 10, 12
              plt.yticks([0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi], [], fontsize=12)

              # annotate with system size
              # plt.gcf().text(0.02, 0.02, annotation, fontsize=8)
              plt.title(r"$L=" + str(L) + r"$", fontsize=12)




# colorbar
# colorbar L = 12 only
cax = plt.axes([0.93, 0.2, 0.15, 0.7])
cax.set_xticks([], [])
cax.set_yticks([], [])
clb = plt.colorbar(scat, ax=cax, pad=0.0, fraction=1, orientation="vertical", aspect=15)
clb.ax.tick_params(labelsize=9.5)
clb.ax.set_title(r"$\Delta E_{c}$", fontsize=12)

plt.subplots_adjust(hspace=0.2, left=0.07, right=0.9, top=0.9, bottom=0.2)
plt.savefig("Fig6.pdf", format="pdf")
# plt.show()