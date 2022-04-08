from quantum_clustering.ws_decomposition import weight_shape_decomp, count_clusters_by_z_line, convert_to_array_and_expand, x_to_energy, calc_sum_3d
import pandas as pd
import numpy as np

if __name__ == '__main__':
    min_n, max_n = 1, np.inf
    z_list = [10]
    name_save_prefix = 'mono10k'

    print('Extracting events')
    calo, energies = get_data_by_maximal_number_of_particles(max_n, min_n)
    print(f'Extracted {len(list(calo.keys()))} events')

    extracted_data_dict = dict()
    for event_id in calo.keys():
        sum_3d = calc_sum_3d((calo, None), event_id)
        raw_data, e_list = convert_to_array_and_expand((calo, energies), event_id, t=2)
        P, V, S, W = weight_shape_decomp(raw_data, 9, (1, 2, 2))
        PxS = P * S
        extracted_data_dict[event_id] = (PxS, e_list, sum_3d)
    print('Finished getting raw data')

    row_list = list()
    for z in z_list:
        print(f'Starting z={z}')
        for event_id, (PxS, e_list, sum_3d) in extracted_data_dict.items():
            peaks_x, _, _ = count_clusters_by_z_line(PxS, z_value=z)

            try:
                min_delta_E_Pred = min(-np.diff(x_to_energy(peaks_x)))
            except ValueError:
                min_delta_E_Pred = 0
            try:
                sorted_e_list = sorted(e_list)
                minimal_delta_E_True = min(np.diff(sorted_e_list))
            except ValueError:
                minimal_delta_E_True = 0

            row = {'id': event_id,
                   'Z': z,
                   'True_N': len(e_list),
                   'Pred_N': len(peaks_x),
                   'minimal_delta_E_True': minimal_delta_E_True,
                   'min_delta_E_Pred': min_delta_E_Pred,
                   'sum_3d': sum_3d}
            row_list.append(row)

    columns = row_list[0].keys()
    df = pd.DataFrame(columns=columns)
    df = df.append(row_list)
    df.to_csv(rf'./data_processed/{name_save_prefix}_{min_n}to{max_n}')