from copy import deepcopy

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator
from afori_utils.debug_utils import TimesAgg
from sigfig import sigfig

from quantum_clustering.ws_decomposition import weight_shape_decomp, convert_to_array_and_expand
from quantum_clustering.gradient_descent import cluster_by_gradient_descent
from utils.data_utils import Dataset
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.other_utils import sigfiground


class IsolationEstimator(BaseEstimator):
    def __init__(self, expansion_factor=1, raw_threshold=0.001, sigma=(0.5, 1.0, 1.0),
                 kernel_size=7, v_threshold=33, window_size=4):
        self.expansion_factor = expansion_factor
        self.raw_threshold = raw_threshold
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.v_threshold = v_threshold
        self.window_size = window_size

    def fit(self, x, y):
        pass

    def predict(self, x):
        pred_n_list = list()
        for calo_event in x:
            raw_data = convert_to_array_and_expand(calo_event, t=self.expansion_factor,
                                                   threshold=self.raw_threshold)

            sigma_effective = self.expansion_factor * np.array(self.sigma)
            kernel_size_effective = (self.kernel_size - 1) * 2 + 1
            P, V = weight_shape_decomp(raw_data, kernel_size_effective, sigma_effective, get_S=False)

            raw_2d = raw_data.sum(axis=1)
            v_2d = V.sum(axis=1)
            v_mask = np.asarray(v_2d > self.v_threshold, dtype=float)
            raw_2d_filtered = raw_2d * v_mask

            data_to_fit = raw_2d_filtered.sum(axis=1)

            conved = np.convolve(data_to_fit, [1 / float(self.window_size)] * self.window_size)
            sum_list, _ = get_sum_per_island(conved, np.finfo(conved[0]).eps * 5)
            pred_n_list.append(len(sum_list))

        return pred_n_list

    @staticmethod
    def score(y_true, y_pred):
        total = len(y_true)
        correct = np.sum(np.array(y_true) == np.array(y_pred))
        return correct / float(total)


class IslandEstimator(BaseEstimator):
    def __init__(self, dep_2d_threshold=0, tower_threshold=0, cluster_threshold=0):
        self.dep_2d_threshold = dep_2d_threshold
        self.tower_threshold = tower_threshold
        self.cluster_threshold = cluster_threshold

        self.NO_CLUSTER = -1

        self.raw_1d_ = None
        self.filtered_data_1d_ = None
        self.ind2cluster_ = None
        self.cluster_count_ = None

        self._sum_per_cluster = None

    def fit(self, x, y):
        pass

    def predict_one(self, event):
        raw_3d = convert_to_array_and_expand(event, t=1)
        raw_2d = raw_3d.sum(axis=2)
        self.raw_1d_ = raw_2d.sum(axis=1)

        filtered_data_2d = np.copy(raw_2d)
        filtered_data_2d[raw_2d < self.dep_2d_threshold] = 0
        self.filtered_data_1d_ = filtered_data_2d.sum(axis=1)

        self.ind2cluster_ = np.full_like(self.filtered_data_1d_, self.NO_CLUSTER)
        self.cluster_count_ = 0
        indices_sorted_by_value = [ind for ind, val in sorted(enumerate(self.filtered_data_1d_), key=lambda x: x[1],
                                                              reverse=True)]
        for cur_ind in indices_sorted_by_value:
            if self.filtered_data_1d_[cur_ind] <= self.tower_threshold:
                continue
            if self.ind2cluster_[cur_ind] == self.NO_CLUSTER:
                self.ind2cluster_[cur_ind] = self.cluster_count_
                self.cluster_count_ += 1
            for nei_ind in [cur_ind - 1, cur_ind + 1]:
                try:
                    if self.ind2cluster_[nei_ind] == self.NO_CLUSTER:
                        self.ind2cluster_[nei_ind] = self.ind2cluster_[cur_ind]
                except IndexError:
                    continue

        for cluster_num in self.get_unique_clusters():
            cur_cluster_mask = self.ind2cluster_ == cluster_num
            cur_sum = self.raw_1d_[cur_cluster_mask].sum()
            if cur_sum < self.cluster_threshold:
                self.ind2cluster_[cur_cluster_mask] = self.NO_CLUSTER
                self.cluster_count_ -= 1

    @staticmethod
    def score(y_true, y_pred):
        total = len(y_true)
        correct = np.sum(np.array(y_true) == np.array(y_pred))
        return correct / float(total)

    @property
    def sum_per_cluster(self):
        if self._sum_per_cluster is None:
            sum_list = [np.sum(self.raw_1d_[unique_val == self.ind2cluster_])
                        for unique_val in self.get_unique_clusters()]
            self._sum_per_cluster = sorted(sum_list, reverse=True)
        return self._sum_per_cluster

    def get_unique_clusters(self):
        return np.unique(self.ind2cluster_[self.ind2cluster_ != self.NO_CLUSTER])


def main_sklearn():
    times_agg = TimesAgg()

    with times_agg('A'):
        dataset = Dataset()
        dataset.add_dataset(
            r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data\signal.al.elaser.IP05.edeplist.mat',
            r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data\signal.al.elaser.IP05.energy.mat',
            '05', to_print=True
        )
        dataset.add_dataset(
            r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data\signal.al.elaser.IP08.edeplist.mat',
            r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data\signal.al.elaser.IP08.energy.mat',
            '08', to_print=True
        )
        dataset.filter_by_number_of_particles(min_n=1, max_n=20)
        # dataset.filter_stratified_percentage(0.5)
        print(f'#events: {len(dataset)}')

    with times_agg('B'):
        pred_n_array = IslandEstimator().predict(dataset.calo_dict.values())
        true_n_array = np.array([len(energies_event) for energies_event in dataset.energies_dict.values()])
        error_array = true_n_array - pred_n_array
        true_n_unique = np.unique(true_n_array)

        mean_error_per_true_n, std_error_per_true_n, accuracy_per_true_n = list(), list(), list()
        multiplicity_per_true_n = list()
        for cur_val in true_n_unique:
            mask = true_n_array == cur_val
            cur_error_array = error_array[mask]
            mean_error_per_true_n.append(np.mean(cur_error_array))
            std_error_per_true_n.append(np.std(cur_error_array))
            accuracy_per_true_n.append(np.sum(cur_error_array == 0) / len(cur_error_array))
            multiplicity_per_true_n.append(len(cur_error_array))

        p = np.argsort(true_n_array)
        events_df = pd.DataFrame({
            'True': true_n_array[p],
            'pred_n_TB': pred_n_array[p],
            'error_n_TB': error_array[p],
        })

        hist_df = pd.DataFrame({
            'True N': true_n_unique,
            'Mean': sigfiground(mean_error_per_true_n, 2),
            'Std': sigfiground(std_error_per_true_n, 2),
            'Accuracy': sigfiground(accuracy_per_true_n, 2),
            'Multiplicity': sigfiground(multiplicity_per_true_n, 2)
        })

        fig = px.scatter(hist_df, x='True N', y='Mean', error_y='Std', title=f'#Events considered = {len(dataset)}',
                         labels={'mean': 'mean(True - Rec)'})
        fig.show()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=hist_df['True N'], y=hist_df['Accuracy'], name='Accuracy',
                                 mode='lines+markers'), secondary_y=False)
        fig.add_trace(go.Scatter(x=hist_df['True N'], y=hist_df['Multiplicity'], name='Multiplicity',
                                 mode='lines+markers'), secondary_y=True,)
        fig.update_layout(title_text="Accuracy and multiplicity")
        fig.update_xaxes(title_text="True N")
        fig.update_yaxes(title_text="<b>Accuracy</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>Multiplicity</b>", secondary_y=True)
        fig.show()

    print(times_agg.__str__())


def main_grid_search():
    times_agg = TimesAgg()

    with times_agg('A'):
        dataset = Dataset()
        dataset.add_dataset(
            r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data\signal.al.elaser.IP05.edeplist.mat',
            r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data\signal.al.elaser.IP05.energy.mat',
            '05'
        )
        dataset.add_dataset(
            r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data\signal.al.elaser.IP08.edeplist.mat',
            r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data\signal.al.elaser.IP08.energy.mat',
            '08'
        )
        dataset.filter_by_number_of_particles(min_n=1, max_n=6)
        dataset.filter_stratified_percentage(0.85)
        print(f'#events: {len(dataset)}')

    with times_agg('B'):
        estimator = MyEstimator()
        param_grid = [
            {'window_size': [1, 2, 3, 4]}
        ]
        gs = GridSearchCV(estimator, param_grid=param_grid)
        x = list(dataset.calo_dict.values())
        y = [len(energies) for energies in dataset.energies_dict.values()]
        gs.fit(x, y)

    print(times_agg.__str__())


if __name__ == '__main__':
    main_sklearn()
