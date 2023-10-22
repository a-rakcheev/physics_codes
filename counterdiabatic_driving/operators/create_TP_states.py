import numpy as np
for L in range(2, 3, 1):

    print(L)
    # L = 2                       # number of sites
    k_idx = 0                   # momentum index (full momentum is (2 pi / L) * index), should be 0, 1, ..., L - 1
    k_name = "0"                # currently 0 or pi (k_idx = L // 2) available
    par = 1

    if k_idx >= L or k_idx < 0:
        raise (ValueError, "k index needs to be between 0 and L - 1")

    name = "1d_chain_TP_indices_and_periods_L=" + str(L) + "_k=" + k_name + ".npz"
    data = np.load(name)
    periods = data["period"]
    parities = data["parity"]

    x_plus = np.zeros(len(periods))
    for i in range(len(periods)):

        period = periods[i]
        parity = parities[i]
        norm = (L ** 2) / (period * (1. + parity % 2))
        x_plus[i] = np.sqrt(norm) * np.sqrt(2) ** L
        
    print(x_plus)