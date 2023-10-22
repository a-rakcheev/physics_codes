import numpy as np
res = 50                                    # resolution
x_start = 20                                # start bin of x
x_end = 10                                  # end bin of x
P = abs(x_end - x_start)                    # pathlength

xl = np.linspace(1.e-6, 1.5, res)

x_path = xl[x_start:x_end]                  # x-coordinates
path = np.empty(P)                          # y-coordinates













