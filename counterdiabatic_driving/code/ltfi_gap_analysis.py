import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

L = 16
data = np.load("ltfi_spectrum_L" + str(L) + ".npz")
hl = data['hl']
gl = data['gl']
ev = data["ev"]

# plt.plot(ev[0, :, 0], label="0")
# plt.plot(ev[0, :, 1], label="1")
# plt.plot(ev[0, :, 2], label="2")
# plt.plot(ev[0, :, 3], label="3")
# plt.legend()

gap = ev[:, :, 1] - ev[:, :, 3]

plt.figure(1, figsize=(3.375, 3.))
plt.pcolormesh(hl, gl, 1. / gap.T, cmap="inferno", vmin=1., vmax=15.)

plt.colorbar()

plt.xlabel(r"$h$", fontsize=10)
plt.ylabel(r"$g$", fontsize=10)
plt.title("Inverse Gap - Correlation Length", fontsize=10)

plt.tight_layout()
plt.savefig("ltfi_inverse_gap_L" + str(L) + ".pdf", format="pdf")
plt.show()