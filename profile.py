from itertools import product

from Parzen import weight_shape_decomp, count_clusters_by_z_line, get_data, x_to_energy, calc_sum_3d
from utils.data_utils import get_data_by_maximal_number_of_particles
import pandas as pd
import numpy as np

min_n, max_n = 2, 5
z_list = [10]

print('Extracting events')
calo, energies = get_data_by_maximal_number_of_particles(None, 1)
print(f'Extracted {len(list(calo.keys()))} events')

extracted_data_dict = dict()
for event_id in calo.keys():
    sum_3d = calc_sum_3d((calo, None), event_id)
    raw_data, e_list = get_data((calo, energies), event_id, t=2)
    P, _, _, S = weight_shape_decomp(raw_data, 9, (1, 2, 2))