from __future__ import annotations
from copy import deepcopy, copy
import pandas as pd
from afori_utils.debug_utils import TimesAgg
import plotly.express as px
import plotly.graph_objects as go
from utils import EcalDataIO
from collections import OrderedDict, defaultdict
import random
from os import path
import numpy as np

FACTOR_TO_MEV = 1000
PIXEL_DIST_MM = 5

def energy_to_x(e, x_id=True):
    e = np.array(e) / FACTOR_TO_MEV
    a = (1.0 / PIXEL_DIST_MM) if x_id else 1
    b, c = 684.2, 41.63
    return a * (b / e - c)

def x_to_energy(x, x_id=True):
    x = np.array(x)
    a = (1 / PIXEL_DIST_MM) if x_id else 1
    b, c = 684.2, 41.63
    return a*b / (x + a*c) * FACTOR_TO_MEV


class Dataset:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.calo_dict = OrderedDict()
        self.energies_dict = OrderedDict()
        self.n_datasets = 0

    def __len__(self):
        return len(self.calo_dict.keys())

    def __copy__(self):
        new = Dataset()
        new.data_dir = self.data_dir
        new.calo_dict = copy(self.calo_dict)
        new.energies_dict = copy(self.energies_dict)
        new.n_datasets = self.n_datasets
        return new

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
            for loc, value in calo[event_id].items():
                # Convert GeV to MeV
                calo[event_id][loc] = value * FACTOR_TO_MEV
            energies[event_id] = tuple(energy * FACTOR_TO_MEV for energy in energies[event_id])

        for event_id in calo.keys():
            new_event_id = self.event_full_key(dataset_name, event_id)
            self.calo_dict[new_event_id] = calo[event_id]
            self.energies_dict[new_event_id] = energies[event_id]

        if to_print:
            print(f'Added dataset "{dataset_name}" with {len(list(calo.keys()))} events')

    """ FILTERS """

    def filter_events(self, event_list, inplace=True) -> Dataset:
        dataset_to_filter = self if inplace else copy(self)

        for event_id in event_list:
            dataset_to_filter.calo_dict.pop(event_id)
            dataset_to_filter.energies_dict.pop(event_id)

        return dataset_to_filter

    def filter_by_number_of_particles(self, min_n=1, max_n=float('inf'), inplace=True) -> Dataset:
        event_id_list_to_filter = list()
        for event_id in self.calo_dict.keys():
            cur_n = len(self.energies_dict[event_id])
            if not min_n <= cur_n <= max_n:
                event_id_list_to_filter.append(event_id)
        return self.filter_events(event_id_list_to_filter, inplace=inplace)

    def filter_by_distance(self, min_threshold, inplace=True) -> Dataset:
        min_dist_dict = self.get_min_dist_dict()
        events_to_filter = [event_id for event_id, min_dist in min_dist_dict.items()
                            if min_dist <= min_threshold]
        return self.filter_events(events_to_filter, inplace=inplace)

    def filter_stratified_percentage(self, percentage) -> Dataset:
        assert 0 < percentage < 1
        n2events = self.get_n_to_events_dict()
        events_to_filter = list()
        for n in n2events:
            total_events_for_n = len(n2events[n])
            k_events_to_pop = round(percentage * total_events_for_n)
            events_to_filter.extend(random.sample(n2events[n], k_events_to_pop))
        return self.filter_events(events_to_filter)

    """ GETTERS """

    def get_random_event_id(self):
        return np.random.choice(list(self.keys()))

    def get_event_true_n(self, event):
        return len(self.energies_dict[event])

    def get_max_true_n(self):
        return max([len(lst) for lst in self.energies_dict.values()])

    def get_n_to_events_dict(self):
        n2events = defaultdict(list)
        for event in self.keys():
            n2events[self.get_event_true_n(event)].append(event)
        return n2events

    def get_min_dist_dict(self):
        min_dist = dict()
        for event_id, e_list in self.energies_dict.items():
            sorted_x = sorted(energy_to_x(e_list, x_id=False))
            if len(sorted_x) == 1:
                min_dist[event_id] = float('inf')
            else:
                min_dist[event_id] = np.min(np.diff(sorted_x))

        return min_dist

    def get_true_n_dict(self):
        true_n = dict()
        for event_id, e_list in self.energies_dict.items():
            true_n[event_id] = len(e_list)

        return true_n

    """ PLOT """

    def get_n_particles_histogram(self, max_n=None):
        n_list = list()
        for event_id, energies in self.energies_dict.items():
            cur_n = len(energies)
            if max_n is None or cur_n <= max_n:
                n_list.append(cur_n)

        #TODO add numbering for each bin
        # fig = go.Figure()
        # fig.add_trace(go.Histogram(x=n_list, mode="markers+text", textposition="top center"))

        df = pd.DataFrame({'n': n_list})
        fig = px.histogram(df, x='n', nbins=max(n_list))
        return fig

    def get_min_distance_histogram(self):
        min_dist_dict = self.get_min_dist_dict()
        fig = px.histogram(x=min_dist_dict, title='Minimum distance histogram')
        fig.update_layout(dict(xaxis_title='Min dx (mm)', yaxis_title='Count'))
        return fig

    def get_min_dist_vs_n_scatter(self):
        fig = px.scatter(x=self.get_true_n_dict(), y=self.get_min_dist_dict(), title='Minimum distance VS. True N')
        fig.update_layout(dict(xaxis_title='True N', yaxis_title='Min dx (mm)'))
        return fig

    """ OTHER """

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

    dataset.get_min_distance_histogram().show()

    print(times_agg.__str__())
