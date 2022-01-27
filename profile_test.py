from itertools import product

import torch
from afori_utils.debug_utils import TimesAgg

from Parzen import weight_shape_decomp, count_clusters_by_z_line, get_data, x_to_energy, calc_sum_3d, \
    get_gaussian_kernel, get_L, conv3d
from utils.data_utils import get_data_by_maximal_number_of_particles
import pandas as pd
import numpy as np
import torch.nn.functional as F

if __name__ == '__main__':

    times_agg = TimesAgg()
    times_agg.get_all_stats(req_list=['min', 'max', 'avg', 'std'])

    min_n, max_n = 2, 2
    z_list = [10]

    print('Extracting events')
    calo, energies = get_data_by_maximal_number_of_particles(max_n, min_n)
    print(f'Extracted {len(list(calo.keys()))} events')

    extracted_data_dict = dict()
    for event_id in calo.keys():
        print(event_id)
        with times_agg('calc_sum_3d'):
            sum_3d = calc_sum_3d((calo, None), event_id)
        with times_agg('get_data'):
            raw_data, e_list = get_data(calo, energies, t=2)

        d, K_DIM, sigma = raw_data, 9, (1, 2, 2)

        with times_agg('get_gaussian_kernel'):
            k_space, K = get_gaussian_kernel(dim=K_DIM, sigma=sigma)

        with times_agg('get_L'):
            L = get_L(K, k_space, disp=False)

        with times_agg('conv3d'):
            C_2 = conv3d(torch.Tensor(d), L)
            psi = conv3d(torch.Tensor(d), K)

        with times_agg('exp'):
            S = np.exp(-C_2)

    print(times_agg)

    print(times_agg.get_all_stats_string(req_list=['min', 'max', 'avg', 'std']))
