import EcalDataIO


def get_data_by_maximal_number_of_particles(max_n, min_n=1, data_and_name_tuple_list=None):
    if data_and_name_tuple_list is None:
        data_and_name_tuple_list = [
            (r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data\signal.al.elaser.IP05.edeplist.mat',
             r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data\signal.al.elaser.IP05.energy.mat',
             '05'),
            (r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data\signal.al.elaser.IP08.edeplist.mat',
             r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data\signal.al.elaser.IP08.energy.mat',
             '08')
        ]

    calo_dict = dict()
    energies_dict = dict()

    for ecal_path, energy_path, name in data_and_name_tuple_list:
        calo = EcalDataIO.ecalmatio(ecal_path)
        energies = EcalDataIO.energymatio(energy_path)

        for event_id in calo.keys():
            cur_n = len(energies[event_id])
            if min_n <= cur_n <= max_n:
                new_event_id = f'{name}_{event_id}'
                calo_dict[new_event_id] = calo[event_id]
                energies_dict[new_event_id] = energies[event_id]

    return calo_dict, energies_dict
