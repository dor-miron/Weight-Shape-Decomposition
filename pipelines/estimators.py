from copy import deepcopy

import pandas as pd
import plotly
import sklearn.linear_model
from lmfit import Minimizer, Parameters, Parameter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator
from afori_utils.debug_utils import TimesAgg
from sigfig import sigfig
from pipelines.shan_correction import ShiftingXArray, shift_2d_matrix
from quantum_clustering.ws_decomposition import weight_shape_decomp, convert_to_array_and_expand
from quantum_clustering.gradient_descent import cluster_by_gradient_descent
from streamlit_helper import get_plotly_layout
from utils.data_utils import Dataset, x_to_energy
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.other_utils import sigfiground


class IslandEstimator(BaseEstimator):
    def __init__(self, position_reconstruction_method='angle',
                 en_dep_threshold=0, tower_threshold=0, cluster_threshold=0):
        self.en_dep_threshold = en_dep_threshold
        self.tower_threshold = tower_threshold
        self.cluster_threshold = cluster_threshold
        self.position_reconstruction_method = position_reconstruction_method

        self.NO_CLUSTER = -1

        self.raw_2d_ = None
        self.raw_1d_ = None
        self.filtered_data_2d_ = None
        self.filtered_data_1d_ = None
        self.ind2cluster_ = None
        self.cluster_count_ = None
        self.sum_per_cluster_ = None

        self.calibrated_energies_ = None
        self.energy_ratio_dict_ = None
        self.energies_ = None

    def fit(self, x, y):
        pass

    def predict_one(self, event):
        raw_3d = convert_to_array_and_expand(event, t=1)
        self.raw_2d_ = raw_3d.sum(axis=1)
        self.raw_1d_ = self.raw_2d_.sum(axis=1)

        filtered_data_3d = np.copy(raw_3d)
        filtered_data_3d[raw_3d < self.en_dep_threshold] = 0
        self.filtered_data_2d_ = filtered_data_3d.sum(axis=1)
        self.filtered_data_1d_ = self.filtered_data_2d_.sum(axis=1)

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
        if self.sum_per_cluster_ is None:
            sum_list = [np.sum(self.raw_1d_[unique_val == self.ind2cluster_])
                        for unique_val in self.get_unique_clusters()]
            self.sum_per_cluster_ = sorted(sum_list, reverse=True)
        return self.sum_per_cluster_

    def calc_calibrated_energies(self, func):
        # make sure this method is called only once
        assert self.calibrated_energies_ is None

        np_func = np.vectorize(func)
        self.calibrated_energies_ = np_func(self.sum_per_cluster)
        return self.calibrated_energies_

    @property
    def energy_ratio_dict(self):
        if self.energy_ratio_dict_ is None:
            energy_ratio_dict = dict()
            for ind, calibrated_energy in enumerate(self.calibrated_energies_):
                predicted_x = self.position_reconstruction(ind)
                estimated_energy = x_to_energy(predicted_x)
                energy_ratio = calibrated_energy / estimated_energy
                energy_ratio_dict[estimated_energy] = energy_ratio
            self.energy_ratio_dict_ = energy_ratio_dict
        return self.energy_ratio_dict_

    def position_reconstruction(self, ind):
        name_to_method = {
            'regular_com': self.cluster_to_position_naive,
            'shift_com': self.cluster_to_position_2d_shift,
            'angle': self.cluster_to_position_angle_fit
        }
        chosen_method = name_to_method[self.position_reconstruction_method]
        return chosen_method(ind)

    @property
    def energies_list(self):
        if self.energies_ is None:
            energies = list()
            for energy, ratio in self.energy_ratio_dict.items():
                energies.extend([energy] * round(ratio))
            self.energies_ = sorted(energies, reverse=True)
        return self.energies_

    def get_unique_clusters(self):
        return np.unique(self.ind2cluster_[self.ind2cluster_ != self.NO_CLUSTER])

    def cluster_to_position_naive(self, cluster_ind):
        weighted_sum = np.sum([en_dep * ind
                               for ind, en_dep in enumerate(self.raw_1d_)
                               if (self.ind2cluster_[ind] == cluster_ind) and en_dep > 1e-4])
        return weighted_sum / self.sum_per_cluster_[cluster_ind]

    def cluster_to_position_2d_shift(self, cluster_ind):
        # THRESHOLD_3D = 90e-03
        # shift_array = ShiftingXArray(dimension=2).T
        shift_array = shift_2d_matrix
        weights = shift_array.T * self.filtered_data_2d_
        array_shape = self.filtered_data_2d_.shape
        x_value_array = np.tile(np.arange(array_shape[0]), (array_shape[1], 1)).T
        weighted_position = weights * x_value_array
        cluster_pixel_mask = np.tile(self.ind2cluster_ == cluster_ind, (self.raw_2d_.shape[1], 1)).T
        weighted_sum = weighted_position[cluster_pixel_mask].sum()
        sum_of_weights = weights[cluster_pixel_mask].sum()

        return weighted_sum / sum_of_weights

    def plot_cluster_to_position_2d_shift_process(self, cluster_ind):
        THRESHOLD_3D = 90e-03
        shift_array = ShiftingXArray(dimension=2).T
        weights = shift_array * self.filtered_data_2d_
        array_shape = self.filtered_data_2d_.shape
        x_value_array = np.tile(np.arange(array_shape[0]), (array_shape[1], 1)).T
        weighted_position = weights * x_value_array
        cluster_pixel_mask = np.tile(self.ind2cluster_ == cluster_ind, (self.raw_2d_.shape[1], 1)).T
        weighted_sum = weighted_position[cluster_pixel_mask].sum()
        sum_of_weights = weights[cluster_pixel_mask].sum()

        fig_list = list()
        fig = px.imshow(cluster_pixel_mask.T, title="Cluster Mask")
        fig_list.append(fig)

        fig = px.imshow(shift_array.T, title="Shift Array")
        fig.add_trace(go.Contour(z=shift_array.T))
        fig_list.append(fig)

        fig = px.imshow(weights.T, title="Weights")
        fig_list.append(fig)

        fig = px.imshow(self.filtered_data_2d_.T, title="Result")
        fig.add_vline(weighted_sum / sum_of_weights, line_color='white')
        fig_list.append(fig)

        return fig_list

    def get_fig_cluster_to_position_compare(self):

        fig = px.imshow(self.filtered_data_2d_.T)

        for cluster_ind in range(self.cluster_count_):
            shift_2d_est = self.cluster_to_position_2d_shift(cluster_ind)
            angle_fit_est, angle_fit_scatter = self.cluster_to_position_angle_fit(cluster_ind, return_fit_scatter=True)
            x, y= angle_fit_scatter
            fig.add_vline(x=shift_2d_est, line_color='green', line_dash='dash')
            fig.add_trace(go.Scatter(x=x, y=y, marker=dict(color='orange')))
            fig.add_vline(x=angle_fit_est, line_color='orange', line_dash='dash')

        return fig

    def cluster_to_position_angle_fit(self, cluster_ind, return_fit_scatter=False, return_fig=False):
        used_array = self.filtered_data_2d_
        array_shape = used_array.shape
        mask = np.tile(self.ind2cluster_ == cluster_ind, (self.raw_2d_.shape[1], 1)).T
        x_value_array = np.tile(np.arange(array_shape[0]), (array_shape[1], 1)).T
        y_value_array = np.tile(np.arange(array_shape[1]), (array_shape[0], 1))
        weights = self.filtered_data_2d_[mask]
        coordinates = list(zip(x_value_array[mask], y_value_array[mask]))

        def fcn(pars, x, y, w):
            a = pars['a']
            b = pars['b']
            distance_array = np.abs(a * x + b * y + 1) / np.sqrt(a ** 2 + b ** 2 + 1e-7)
            return distance_array * w
        params = Parameters()
        params['a'] = Parameter(name='a', value=1)
        params['b'] = Parameter(name='b', value=1)
        model = Minimizer(fcn, params=params,
                          fcn_args=(x_value_array[mask], y_value_array[mask], weights))
        result = model.minimize()
        a, b = result.params['a'].value, result.params['b'].value

        to_return = [- 1/ a]

        x_first, x_last = np.where(mask)[0][[0, -1]]
        line_x = np.linspace(x_first, x_last, 100)
        line_y = -1 / b - a / b * line_x
        line_y_mask = (line_y <= 20) & (line_y >= 0)

        if return_fit_scatter:
            to_return.append((line_x[line_y_mask], line_y[line_y_mask]))
        if return_fig:
            fig = px.imshow((used_array * mask).T)
            fig.add_trace(go.Scatter(x=line_x[line_y_mask], y=line_y[line_y_mask]))
            fig.add_vline(x=-1 / a, line_color='white')
            to_return.append(fig)

        if len(to_return) == 1:
            to_return = to_return[0]
        return to_return


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
                                 mode='lines+markers'), secondary_y=True, )
        fig.update_layout(title_text="Accuracy and multiplicity")
        fig.update_xaxes(title_text="True N")
        fig.update_yaxes(title_text="<b>Accuracy</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>Multiplicity</b>", secondary_y=True)
        fig.show()

    print(times_agg.__str__())


def one_event():
    times_agg = TimesAgg()

    with times_agg('A'):
        dataset = Dataset()
        dataset.add_dataset(
            r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data\signal.al.elaser.IP05.edeplist.mat',
            r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data\signal.al.elaser.IP05.energy.mat',
            '05'
        )

    with times_agg('estimator check'):
        estimator = IslandEstimator()
        estimator.predict_one(dataset.calo_dict[dataset.get_random_event_id()])
        estimator.calc_calibrated_energies(lambda x: x * 85)
        print(estimator.energies_list)


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

    with times_agg('Search'):
        estimator = IslandEstimator()
        param_grid = [
            {'window_size': [1, 2, 3, 4]}
        ]
        gs = GridSearchCV(estimator, param_grid=param_grid)
        x = list(dataset.calo_dict.values())
        y = [len(energies) for energies in dataset.energies_dict.values()]
        gs.fit(x, y)

    print(times_agg.__str__())


if __name__ == '__main__':
    one_event()
