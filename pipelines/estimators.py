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
    def __init__(self, raw_threshold=0.001, cluster_threshold=1e-5):
        self.raw_threshold = raw_threshold
        self.cluster_threshold = cluster_threshold

    def fit(self, x, y):
        pass

    def predict_one(self, event, return_raw_1d=False, return_filtered_1d=False):
        NO_CLUSTER = -1
        raw_data = convert_to_array_and_expand(event, t=1)
        filtered_data = deepcopy(raw_data)
        filtered_data[raw_data < self.raw_threshold] = 0
        filtered_data_1d = np.array(filtered_data).sum(axis=2).sum(axis=1)

        ind2cluster = np.full_like(filtered_data_1d, NO_CLUSTER)
        cluster_count = 0
        indices_sorted_by_value = [ind for ind, val in sorted(enumerate(filtered_data_1d), key=lambda x: x[1],
                                                              reverse=True)]
        for cur_ind in indices_sorted_by_value:
            if filtered_data_1d[cur_ind] < self.cluster_threshold:
                continue
            if ind2cluster[cur_ind] == NO_CLUSTER:
                ind2cluster[cur_ind] = cluster_count
                cluster_count += 1
            for nei_ind in [cur_ind - 1, cur_ind + 1]:
                try:
                    if ind2cluster[nei_ind] == NO_CLUSTER:
                        ind2cluster[nei_ind] = ind2cluster[cur_ind]
                except IndexError:
                    continue

        outputs = [ind2cluster, cluster_count]
        if return_raw_1d:
            outputs.append(np.array(raw_data).sum(axis=2).sum(axis=1))
        if return_filtered_1d:
            outputs.append(filtered_data_1d)
        return outputs

    def predict(self, x):
        pred_n_list = list()
        for ind, calo_event in enumerate(x):
            _, cluster_count = self.predict_one(calo_event)
            pred_n_list.append(cluster_count)
        return np.array(pred_n_list)

    @staticmethod
    def score(y_true, y_pred):
        total = len(y_true)
        correct = np.sum(np.array(y_true) == np.array(y_pred))
        return correct / float(total)


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
