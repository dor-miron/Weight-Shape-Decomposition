import pandas as pd
from afori_utils.debug_utils import TimesAgg
import plotly.express as px
from utils import EcalDataIO
from collections import OrderedDict, defaultdict
import random
from os import path

class Dataset:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.calo_dict = OrderedDict()
        self.energies_dict = OrderedDict()
        self.n_datasets = 0

    def __len__(self):
        return len(self.calo_dict.keys())

    @staticmethod
    def event_full_key(dataset_name, event_id):
        return f'{dataset_name}_{event_id}'

    def add_dataset(self, ecal_path, energy_path, dataset_name=None, to_print=False):

        if dataset_name is None:
            dataset_name = self.n_datasets
        self.n_datasets += 1

        if self.data_dir is not None:
            calo = EcalDataIO.ecalmatio(path.join(self.data_dir, ecal_path))
            energies = EcalDataIO.energymatio(path.join(self.data_dir, energy_path))
        else:
            calo = EcalDataIO.ecalmatio(ecal_path)
            energies = EcalDataIO.energymatio(energy_path)

        for event_id in calo.keys():
            new_event_id = self.event_full_key(dataset_name, event_id)
            self.calo_dict[new_event_id] = calo[event_id]
            self.energies_dict[new_event_id] = energies[event_id]

        if to_print:
            print(f'Added dataset "{dataset_name}" with {len(list(calo.keys()))} events')

    def filter_by_number_of_particles(self, min_n=1, max_n=float('inf')):
        event_id_to_filter = list()
        for event_id in self.calo_dict.keys():
            cur_n = len(self.energies_dict[event_id])
            if not min_n <= cur_n <= max_n:
                event_id_to_filter.append(event_id)

        for event_id in event_id_to_filter:
            self.calo_dict.pop(event_id)
            self.energies_dict.pop(event_id)

    def filter_stratified_percentage(self, percentage):
        assert 0 < percentage < 1
        n2events = self.get_n_to_events_dict()
        for n in n2events:
            total_events_for_n = len(n2events[n])
            k_events_to_pop = round(percentage * total_events_for_n)
            events_to_pop = random.sample(n2events[n], k_events_to_pop)
            for event_id in events_to_pop:
                self.calo_dict.pop(event_id)
                self.energies_dict.pop(event_id)

    def get_event_true_n(self, event):
        return len(self.energies_dict[event])

    def get_n_to_events_dict(self):
        n2events = defaultdict(list)
        for event in self.keys():
            n2events[self.get_event_true_n(event)].append(event)
        return n2events

    def plot_n_particles_histogram(self):
        n_list = list()
        for event_id, energies in self.energies_dict.items():
            n_list.append(len(energies))

        df = pd.DataFrame({'n': n_list})
        fig = px.histogram(df, x='n')
        fig.show()

    def validate_dataset(self):
        assert set(self.calo_dict.keys()) == set(self.energies_dict.keys())

    def keys(self):
        return self.calo_dict.keys()

    def values(self):
        for event_id in self.calo_dict:
            yield self.calo_dict[event_id], self.energies_dict[event_id]


if __name__ == '__main__':
    times_agg = TimesAgg()

    data_and_name_tuple_list = [
        (r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data\signal.al.elaser.IP05.edeplist.mat',
         r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data\signal.al.elaser.IP05.energy.mat',
         '05'),
        (r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data\signal.al.elaser.IP08.edeplist.mat',
         r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data\signal.al.elaser.IP08.energy.mat',
         '08')
    ]

    with times_agg('A'):
        dataset = Dataset()
        for tup in data_and_name_tuple_list:
            dataset.add_dataset(*tup)

    dataset.plot_n_particles_histogram()

    with times_agg('B'):
        dataset.filter_by_number_of_particles(2, 4)

    dataset.plot_n_particles_histogram()

    print(times_agg.__str__())
