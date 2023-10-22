import numpy as np
import multiprocessing as mp
import time

def fill_metric_subspace_coherent(index_tuple):

    # print("i, j:", index_tuple[0], index_tuple[1])
    metric_grid[(index_tuple[0] * res_y + index_tuple[1])] = 1.


if __name__ == '__main__':

    res_x = 2 ** 11                                          # number of grid points on x axis
    res_y = res_x                                          # number of grid points on y axis
    number_of_processes = 4                             # number of parallel processes (should be equal to number of (logical) cores
    testarray = np.ones(res_x * res_y, dtype=np.float64)
    time.sleep(3)
    testarray = None
    time.sleep(3)
    metric_grid = mp.Array('d', res_x * res_y)
    time.sleep(3)
    pool = mp.Pool(processes=number_of_processes)
    time.sleep(3)
    computation = pool.map_async(fill_metric_subspace_coherent, [(i, j) for i in range(res_x) for j in range(res_y)])

