from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import product

import lmfit.models
import streamlit as st
from plotly.subplots import make_subplots
from astroML.sum_of_norms import sum_of_norms, norm
from fit_to_gaussians_1d.fit_to_gaussians_1d import get_n_1d_gaussians_model
from quantum_clustering.ws_decomposition import weight_shape_decomp, convert_to_array_and_expand, \
    count_clusters_by_z_line, energy_to_x
import numpy as np
from afori_utils.debug_utils import TimesAgg
import altair as alt
import pandas as pd
from plotly import graph_objects as go
from plotly import express as px
from streamlit_helper import LINE_SEPARATION, my_plotly_chart, get_plotly_layout
from pipelines.estimators import IslandEstimator
from utils.data_utils import Dataset
from utils.other_utils import sigfiground, calibrate_energies, get_random_string, get_sum_per_cluster

DS5 = 'FL=5'
DS8 = 'FL=8'
DS_SINGLE_10GEV = 'Single - 10 Gev'

MENU_OPTIONS = ['Nothing', 'Dataset exploration', 'Event exploration']
DEFAULT_MENU_INDEX = 1

DEFAULT_KERNEL_VALUE = 7
RAW_THRESHOLD_DEFAULT = 0.000
default_sigma_list = [0.5, 1.0, 1.0]

st.set_page_config(
    page_title='QC Dashboard',
    page_icon=":shark:",
    layout='wide',
    initial_sidebar_state="expanded"
)

top_container = st.container()
top_container.text("Loading...")


def event2label(event_id):
    return f'{event_id.split("_")[-1]}'


times_agg = TimesAgg()

# """ GET DATA """
dataset_key = st.sidebar.radio('Data Set', [DS5, DS8, DS_SINGLE_10GEV], index=0)
st.sidebar.text(LINE_SEPARATION)
key2path = {
    DS5: ("signal.al.elaser.IP05.edeplist.mat", "signal.al.elaser.IP05.energy.mat"),
    DS8: ("signal.al.elaser.IP08.edeplist.mat", "signal.al.elaser.IP08.energy.mat"),
    DS_SINGLE_10GEV: ("alw_10k_10GeV_edep.mat", "alw_10k_10GeV_trueXY.mat"),
}

with times_agg('Get data'), st.spinner('Get data'):
    cache_key = f"data_{dataset_key}"
    if cache_key in st.session_state:
        dataset = st.session_state[cache_key]
    else:
        dataset = Dataset(r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data')
        dataset.add_dataset(*key2path[dataset_key], dataset_name=dataset_key)
        st.session_state[cache_key] = dataset

    dataset.filter_by_number_of_particles(max_n=20)
    calo_dict, energies_dict = dataset.calo_dict, dataset.energies_dict

st.text(f'Saved session keys: {list(st.session_state.keys())}')


def dataset_exploration():
    def get_events_array(stride=1):
        event_id_list = list(calo_dict.keys())
        events_list = list()
        true_n_list = list()
        for event_id in event_id_list[::stride]:
            calo_event, energy_event = calo_dict[event_id], energies_dict[event_id]
            raw_data = convert_to_array_and_expand(calo_event, t=1)
            events_list.append(raw_data.sum(axis=1))
            true_n_list.append(len(energy_event))

        events_array = np.stack(events_list, axis=0)

        return events_array, true_n_list

    def main():
        stride = 1
        with times_agg('get_all'):
            events_array, true_n_list = get_events_array(stride)

        st.header(f'Showing {events_array.shape[0]} events')

        with times_agg('Statistics'):
            st.header('Statistics')
            is_show_stats = st.sidebar.checkbox('Show Statistics', value=True)
            if is_show_stats:
                stat_cols = st.columns(2)
                fig = px.imshow(events_array.mean(axis=0).T, origin='lower', title='Mean')
                fig.update_layout(get_plotly_layout())
                stat_cols[0].plotly_chart(fig)

                fig = px.imshow(events_array.std(axis=0).T, origin='lower', title='Std')
                fig.update_layout(get_plotly_layout())
                stat_cols[1].plotly_chart(fig)

                df = pd.DataFrame({'true_n': true_n_list})
                fig = px.histogram(df, x='true_n', title='Multiplicity')
                fig.update_layout(get_plotly_layout())
                st.plotly_chart(fig)

        with times_agg('Islands predictor'):
            progress_placeholder = st.empty()
            progress_placeholder.header('Islands predictor')
            with st.expander('See explanation'):
                st.write(
                    "The island predictor works as follows:\n"
                    " - Take a 3D image\n"
                    " - Sum over Z and Y axes\n"
                    " - Over the 1D array:\n"
                    "   - Take maximal value and assign it to new cluster, then remove from array\n"
                    "   - Take new maximal value. \n"
                    "       - If it neighbors and existing cluster, then assign to existing cluster\n"
                    "       - Else, assign to new cluster\n"
                    "   - Remove assigned value from array and repeat last stage\n"
                )

            ISLANDS_DEFAULT = True
            is_show_islands = st.sidebar.checkbox('Show Islands predictor', value=ISLANDS_DEFAULT)
            is_show_statistics = st.sidebar.checkbox(
                'Show Statistics', value=True, disabled=not is_show_islands, key=get_random_string(10))
            is_show_calibration = st.sidebar.checkbox(
                'Show Calibration', value=ISLANDS_DEFAULT, disabled=not is_show_islands, key=get_random_string(10))

            if is_show_islands:
                islands_cols = st.columns(2)
                island_estimator = IslandEstimator()
                n_events = len(dataset)
                pred_n_dict = dict()
                pred_n_array = np.full((n_events,), 0)
                true_n_array = np.full((n_events,), 0)
                true_energies_dict = dict()
                ind2cluster_dict = dict()
                raw_1d_dict = dict()
                id_list = list()
                for ind, event_id in enumerate(dataset.keys()):
                    progress_placeholder.header(f'Islands predictor -  {ind + 1}/{n_events}')
                    ind2cluster, cluster_count, raw_1d = island_estimator.predict_one(
                        calo_dict[event_id], return_filtered_1d=True)
                    id_list.append(event_id)
                    pred_n_array[ind] = cluster_count
                    pred_n_dict[event_id] = cluster_count
                    true_n_array[ind] = len(dataset.energies_dict[event_id])
                    true_energies_dict[event_id] = sorted(energies_dict[event_id], reverse=True)
                    ind2cluster_dict[event_id] = ind2cluster
                    raw_1d_dict[event_id] = raw_1d

            if is_show_statistics:
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
                    accuracy_per_true_n.append(np.sum(cur_error_array == 0) / float(len(cur_error_array)))
                    multiplicity_per_true_n.append(len(cur_error_array))

                hist_df = pd.DataFrame({
                    'True N': true_n_unique,
                    'Mean Error': sigfiground(mean_error_per_true_n, 2),
                    'Std': sigfiground(std_error_per_true_n, 2),
                    'Accuracy': sigfiground(accuracy_per_true_n, 2),
                    'Multiplicity': sigfiground(multiplicity_per_true_n, 2)
                })

                fig = px.scatter(hist_df, x='True N', y='Mean Error', error_y='Std',
                                 title=f'#Events considered = {n_events}',
                                 labels={'mean': 'mean(True - Rec)'})
                fig.update_layout(get_plotly_layout())
                my_plotly_chart(islands_cols[0], fig)

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=hist_df['True N'], y=hist_df['Accuracy'], name='Accuracy',
                                         mode='lines+markers'), secondary_y=False)
                fig.add_trace(go.Scatter(x=hist_df['True N'], y=hist_df['Multiplicity'], name='Multiplicity',
                                         mode='lines+markers'), secondary_y=True, )
                fig.update_layout(title_text="Accuracy and multiplicity")
                fig.update_xaxes(title_text="True N")
                fig.update_yaxes(title_text="<b>Accuracy</b>", secondary_y=False)
                fig.update_yaxes(title_text="<b>Multiplicity</b>", secondary_y=True)
                fig.update_layout(get_plotly_layout())
                my_plotly_chart(islands_cols[1], fig)

            if is_show_calibration:
                true_n_dict = {event_id: len(energies_event) for event_id, energies_event in
                               dataset.energies_dict.items()}
                pred_energies_dict = dict()
                for event_id in true_energies_dict:
                    cur_pred_energies = get_sum_per_cluster(raw_1d_dict[event_id], ind2cluster_dict[event_id])
                    pred_energies_dict[event_id] = cur_pred_energies

                max_energy = np.max([val for _, lst in pred_energies_dict.items() for val in lst])
                max_pred_energy_threshold = st.sidebar.number_input(
                    label='True energy max threshold', min_value=0.0, max_value=max_energy, value=max_energy,
                    step=0.001,
                )
                min_pred_energy_threshold = st.sidebar.number_input(
                    label='True energy min threshold', min_value=0.0, max_value=max_energy, value=0.0,
                    step=0.001,
                )

                filtered_true_energies = list()
                filtered_pred_energies = list()
                filtered_true_n = list()
                for event_id, cur_pred_list in pred_energies_dict.items():
                    if true_n_dict[event_id] == pred_n_dict[event_id]:
                        for ind, cur_pred_energy in enumerate(cur_pred_list):
                            if min_pred_energy_threshold < cur_pred_energy < max_pred_energy_threshold:
                                filtered_true_energies.append(true_energies_dict[event_id][ind])
                                filtered_pred_energies.append(pred_energies_dict[event_id][ind])
                                filtered_true_n.append(true_n_dict[event_id])
                filtered_true_energies = np.array(filtered_true_energies)
                filtered_pred_energies = np.array(filtered_pred_energies)
                filtered_true_n = np.array(filtered_true_n)

                calibration_cols1 = st.columns(2)
                result, fig = calibrate_energies(filtered_true_energies, filtered_pred_energies,
                                                 return_plot=True, intercept=0, slope=85)
                fig.update_layout(get_plotly_layout())
                calibration_cols1[0].plotly_chart(fig)
                calibration_cols1[0].header(f'{sigfiground(result.best_values, 3)}, Chi2={result.chisqr}')
                correct_true_n_unique = np.unique(true_n_array)
                filtered_calibrated_energies = result.best_fit

                mean_error_per_true_n, std_error_per_true_n, accuracy_per_true_n = list(), list(), list()
                multiplicity_per_true_n = list()
                for cur_n in correct_true_n_unique:
                    cur_mask = np.array(filtered_true_n) == cur_n
                    cur_calibrated_energies = filtered_calibrated_energies[cur_mask]
                    cur_true_energies = filtered_true_energies[cur_mask]
                    cur_error_array = cur_calibrated_energies - cur_true_energies
                    mean_error_per_true_n.append(np.mean(cur_error_array))
                    std_error_per_true_n.append(np.std(cur_error_array))
                    accuracy_per_true_n.append(np.sum(cur_error_array == 0) / float(len(cur_error_array)))
                    multiplicity_per_true_n.append(len(cur_error_array))

                hist_df2 = pd.DataFrame({
                    'True N': correct_true_n_unique,
                    'Mean Error': sigfiground(mean_error_per_true_n, 2),
                    'Std': sigfiground(std_error_per_true_n, 2),
                    'Accuracy': sigfiground(accuracy_per_true_n, 2),
                    'Multiplicity': sigfiground(multiplicity_per_true_n, 2)
                })
                islands_cols2 = st.columns(2)
                fig = px.scatter(hist_df2, x='True N', y='Mean Error', error_y='Std',
                                 title=f'Mean error per N',
                                 labels={'mean': 'mean(True - Rec)'})
                fig.update_layout(get_plotly_layout())
                my_plotly_chart(islands_cols2[0], fig)

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=hist_df2['True N'], y=hist_df2['Accuracy'], name='Accuracy',
                                         mode='lines+markers'), secondary_y=False)
                fig.add_trace(go.Scatter(x=hist_df2['True N'], y=hist_df2['Multiplicity'], name='Multiplicity',
                                         mode='lines+markers'), secondary_y=True, )
                fig.update_layout(title_text="Accuracy and multiplicity")
                fig.update_xaxes(title_text="True N")
                fig.update_yaxes(title_text="<b>Accuracy</b>", secondary_y=False)
                fig.update_yaxes(title_text="<b>Multiplicity</b>", secondary_y=True)
                fig.update_layout(get_plotly_layout())
                my_plotly_chart(islands_cols2[1], fig)

                error_array = filtered_calibrated_energies - filtered_true_energies
                total_error = np.std(error_array[~np.isnan(error_array)])
                st.header(f'Mean error for all events = {total_error}')

    main()


def event_exploration(dataset_key):
    from quantum_clustering.ws_decomposition import weight_shape_decomp, convert_to_array_and_expand
    from quantum_clustering.gradient_descent import cluster_by_gradient_descent

    # INIT WIDGETS
    with times_agg("init widgets"):
        columns1 = st.sidebar.columns(2)
        expansion_factor = columns1[0].radio('Expansion Factor', [1, 2, 3], index=0)

        axis_name_list = ['X', 'Y', 'Z']
        sigma = [st.sidebar.slider(f'Sigma-{axis_name_list[i]}', min_value=0.1, max_value=5.0, step=0.1,
                                   value=default_sigma_list[i])
                 for i in range(3)]
        kernel_size = st.sidebar.slider('Kernel Size', min_value=3, max_value=21, step=2, value=DEFAULT_KERNEL_VALUE)

        event_id_array = np.array(list(dataset.keys()))
        energy_list = list(dataset.energies_dict.values())
        sorting_permutation = np.argsort(energy_list)
        ordered_event_id_list = event_id_array[sorting_permutation]
        n_list = [len(energy) for energy in energy_list]
        unique_n_list = np.unique(n_list)
        event_cols = st.sidebar.columns(2)
        chosen_n = event_cols[0].selectbox('Select N', options=unique_n_list, index=3)
        chosen_n_event_list = event_id_array[n_list == chosen_n]
        event_id = event_cols[1].selectbox('Select event', options=chosen_n_event_list,
                                           format_func=event2label)
        calo_event, energy_event = calo_dict[event_id], energies_dict[event_id]
        true_x_values = energy_to_x(energy_event)

    convert_to_array_and_expand = st.cache(convert_to_array_and_expand)

    is_show_plot_2d = st.sidebar.checkbox('Show 2D plot', value=True)
    if is_show_plot_2d:
        raw_data = convert_to_array_and_expand(calo_event, t=expansion_factor)
        raw_2d = raw_data.sum(axis=1)
        fig = px.imshow(raw_2d.T, origin='lower', title='Raw data')
        for x_val in true_x_values:
            fig.add_vline(x=x_val, line_width=2, line_color='black', line_dash='dash')
        fig.update_layout(get_plotly_layout(1, 1))
        st.plotly_chart(fig)

    is_show_1d_tbc = st.sidebar.checkbox('Show Quantum clustering', value=False)
    if is_show_1d_tbc:
        with times_agg('Quantum clustering'):
            st.header('Quantum clustering')
            raw_threshold = st.sidebar.number_input('Raw Threshold', min_value=0.000, max_value=0.006,
                                                    value=RAW_THRESHOLD_DEFAULT, step=0.0001, format="%f")

            raw_data = convert_to_array_and_expand(calo_event, t=expansion_factor,
                                                   threshold=raw_threshold)

            sigma_effective = expansion_factor * np.array(sigma)
            kernel_size_effective = (kernel_size - 1) * 2 + 1
            weight_shape_decomp = st.cache(weight_shape_decomp)
            P, V, S, W = weight_shape_decomp(raw_data, kernel_size_effective, sigma_effective)
            PxS = P * S

            raw_2d = raw_data.sum(axis=1)
            v_2d = V.sum(axis=1)
            v_threshold = st.sidebar.number_input('V Threshold', min_value=0.0,
                                                  max_value=float(np.max(v_2d)), step=0.01, value=33.0)
            v_mask = np.asarray(v_2d > v_threshold, dtype=float)
            raw_2d_filtered = raw_2d * v_mask
            plot_list = [
                raw_2d.T,
                # P.sum(axis=1).T,
                v_2d.T,
                v_mask.T,
                raw_2d_filtered.T,
                # PxS.sum(axis=1).T
            ]

            for i, z in enumerate(plot_list):
                cur_col = st.columns(2)
                cur_is_log_scale = cur_col[1].checkbox('Log Scale', value=False, key=f'Log scale {i}')
                if cur_is_log_scale:
                    cur_z = np.log(z)
                else:
                    cur_z = z
                fig = px.imshow(cur_z, origin='lower')
                for x_val in true_x_values:
                    fig.add_vline(x=x_val, line_width=2, line_color='black', line_dash='dash')
                fig.update_layout(get_plotly_layout(1, 1))
                cur_col[0].plotly_chart(fig)

            data_to_fit = raw_2d_filtered.sum(axis=1)
            x_axis = np.arange(data_to_fit.shape[0])
            true_n = len(energy_event)

            line_df = pd.DataFrame({'raw_1d': data_to_fit})
            fig = px.line(line_df, x=line_df.index, y='raw_1d')
            fig.update_layout(get_plotly_layout(1, 1))
            st.plotly_chart(fig)

    # """ PLOT """
    st.sidebar.text(LINE_SEPARATION)

    with times_agg('Islands clustering'):
        st.header('Islands clustering')
        is_show_1d_tbc = st.sidebar.checkbox('Show TB clustering', value=True)
        if is_show_1d_tbc:
            island_estimator = IslandEstimator()
            ind2cluster, _, data_to_fit = island_estimator.predict_one(calo_event, return_filtered_1d=True)
            plotly_df = pd.DataFrame({'data': data_to_fit, 'ind2cluster': ind2cluster})
            fig = px.bar(plotly_df, x=plotly_df.index, y='data', color='ind2cluster')
            fig.update_layout(get_plotly_layout(1, 1))
            st.plotly_chart(fig)

    with times_agg('1D Gauss fit'):
        st.header('1D Gauss fit')
        is_show_1d_fit = st.sidebar.checkbox('Show 1D fit', value=False)
        if is_show_1d_fit:
            gauss_model, params = get_n_1d_gaussians_model(data_to_fit, x_axis, true_n, guess=True)
            # gauss_single, params = GaussianModel()
            model_result = gauss_model.fit(
                data_to_fit, params=params, x=x_axis, method='leastsq'
            )
            best_fit = pd.DataFrame({'best_fit': model_result.best_fit,
                                     'data_2_fit': data_to_fit})
            fig = px.line(best_fit, x=best_fit.index, y=['data_2_fit', 'best_fit'])
            fig.update_layout(get_plotly_layout(2, 2))
            st.plotly_chart(fig)

    with times_agg('Show gradient Clustering'):
        show_gd_clustering = st.sidebar.checkbox('Show GD Clustering', value=False)
        if show_gd_clustering:
            cluster_by_gradient_descent = st.cache(cluster_by_gradient_descent)
            clustered_array = cluster_by_gradient_descent(-V.sum(axis=1))

            hm = partial(go.Heatmap, coloraxis='coloraxis1')
            fig = make_subplots(1, 1)
            fig.add_trace(hm(z=clustered_array.T), 1, 1)
            fig.update_layout(get_plotly_layout())
            st.plotly_chart(fig)

    with times_agg('Show Z Line'):
        show_z_line = st.sidebar.checkbox('Show Z Line', value=False)
        if show_z_line:
            z_line = st.sidebar.slider('Z Line', min_value=0, max_value=30, step=1, value=10)
            peaks_x, peaks_y, chosen_line = count_clusters_by_z_line(PxS, z_value=z_line)
            line_df = pd.DataFrame({'line': chosen_line}).reset_index()
            peaks_df = pd.DataFrame({'peaks_x': peaks_x, 'peaks_y': peaks_y})
            line_chart = alt.Chart(line_df).mark_line().encode(
                x='index',
                y='line'
            )
            peaks_chart = alt.Chart(peaks_df).mark_point().encode(x='peaks_x', y='peaks_y')
            chart = line_chart + peaks_chart
            st.altair_chart(chart)


choice = st.sidebar.radio('Choice', MENU_OPTIONS, index=DEFAULT_MENU_INDEX)
st.sidebar.text(LINE_SEPARATION)
if choice == 'Dataset exploration':
    dataset_exploration()
elif choice == 'Event exploration':
    event_exploration(dataset_key)

st.text(times_agg.__str__())

top_container.text("Finished Loading")
