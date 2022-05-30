import lmfit.models
import streamlit as st
from plotly.subplots import make_subplots
from astroML.sum_of_norms import sum_of_norms, norm
from fit_to_gaussians_1d.fit_to_gaussians_1d import get_n_1d_gaussians_model
from quantum_clustering.ws_decomposition import weight_shape_decomp, convert_to_array_and_expand, \
    count_clusters_by_z_line
import numpy as np
from afori_utils.debug_utils import TimesAgg
import altair as alt
import pandas as pd
from plotly import graph_objects as go
from plotly import express as px
from streamlit_helper import LINE_SEPARATION, my_plotly_chart, get_plotly_layout, ISLANDS_EXPLANATION
import streamlit_helper as sth
from pipelines.estimators import IslandEstimator
from utils.data_utils import Dataset
from utils.other_utils import sigfiground, calibrate_energies, log_histogram

CLUSTER_THRESHOLD_VALUE = 15.0

def dataset_exploration(dataset, times_agg):
    calo_dict, energies_dict = dataset.calo_dict, dataset.energies_dict

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

        with times_agg('Dataset Statistics'):
            st.header('Dataset Statistics')
            is_show_stats = st.sidebar.checkbox('Show Dataset Statistics', value=True)
            if is_show_stats:
                stat_cols = st.columns(2)
                fig = px.imshow(events_array.mean(axis=0).T, origin='lower', title='Mean')
                fig.update_layout(get_plotly_layout())
                stat_cols[0].plotly_chart(fig)

                fig = px.imshow(events_array.std(axis=0).T, origin='lower', title='Std')
                fig.update_layout(get_plotly_layout())
                stat_cols[1].plotly_chart(fig)

                hist_cols = st.columns(2)
                fig = dataset.get_n_particles_histogram()
                fig.update_layout(
                    get_plotly_layout(xaxis_title='True N', yaxis_title='Count', title='Multiplicity - All'))
                hist_cols[0].plotly_chart(fig)

                fig = dataset.get_n_particles_histogram(max_n=20)
                fig.update_layout(
                    get_plotly_layout(xaxis_title='True N', yaxis_title='Count', title='Multiplicity - Partial'))
                hist_cols[1].plotly_chart(fig)

        ISLANDS_DEFAULT = True
        is_show_islands = st.sidebar.checkbox('Show Islands predictor', value=ISLANDS_DEFAULT)

        def show_islands():
            progress_placeholder = st.empty()
            progress_placeholder.header('Islands predictor')
            with st.expander('See explanation'):
                st.write(ISLANDS_EXPLANATION)

            dep_thresh = st.sidebar.number_input('2D Deposition Threshold', min_value=0.0, value=0.0, step=0.001)
            tower_thresh = st.sidebar.number_input('Tower Threshold', min_value=0.0, value=0.0, step=0.01)
            cluster_thresh = st.sidebar.number_input('Cluster Threshold', min_value=0.0, value=CLUSTER_THRESHOLD_VALUE,
                                                     step=0.01)

            islands_cols = st.columns(2)
            n_events = len(dataset)
            rec_n_dict = dict()
            rec_n_array = np.full((n_events,), 0)
            true_n_array = np.full((n_events,), 0)
            true_energies_dict = dict()
            ind2cluster_dict = dict()
            raw_1d_dict = dict()
            estimators = dict()
            id_list = list()
            for ind, event_id in enumerate(dataset.keys()):
                progress_placeholder.header(f'Islands predictor -  {ind + 1}/{n_events} Loaded')
                island_estimator = IslandEstimator(dep_2d_threshold=dep_thresh, tower_threshold=tower_thresh,
                                                   cluster_threshold=cluster_thresh)
                island_estimator.predict_one(calo_dict[event_id])
                estimators[event_id] = island_estimator
                id_list.append(event_id)
                rec_n_array[ind] = island_estimator.cluster_count_
                rec_n_dict[event_id] = island_estimator.cluster_count_
                true_n_array[ind] = len(dataset.energies_dict[event_id])
                true_energies_dict[event_id] = sorted(energies_dict[event_id], reverse=True)
                ind2cluster_dict[event_id] = island_estimator.ind2cluster_
                raw_1d_dict[event_id] = island_estimator.raw_1d_

            def show_statistics():
                true_n_array = np.array([len(energies_event) for energies_event in dataset.energies_dict.values()])
                error_array = true_n_array - rec_n_array
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
                                 labels={'Mean Error': 'mean(True - Rec)'})
                fig.update_layout(get_plotly_layout())
                my_plotly_chart(islands_cols[0], fig)

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=hist_df['True N'], y=hist_df['Accuracy'], name='Accuracy',
                                         mode='lines+markers'), secondary_y=False)
                fig.add_trace(go.Scatter(x=hist_df['True N'], y=hist_df['Multiplicity'], name='Multiplicity',
                                         mode='lines+markers'), secondary_y=True, )
                fig.update_xaxes(title_text="True N")
                fig.update_yaxes(title_text="<b>Accuracy</b>", secondary_y=False)
                fig.update_yaxes(title_text="<b>Multiplicity</b>", secondary_y=True)
                fig.update_layout(get_plotly_layout())
                islands_cols[1].header(f'Correct events: {(error_array == 0).sum()}')
                my_plotly_chart(islands_cols[1], fig)

                histogram_cols = st.columns(2)
                all_clusters = np.concatenate([estimator.sum_per_cluster for estimator in estimators.values()])
                fig = log_histogram(x=all_clusters, title="Histogram of all clusters")
                fig.update_layout(get_plotly_layout(xaxis_title='Cluster Energies (MeV)'))
                my_plotly_chart(histogram_cols[0], fig)

            is_show_statistics = st.sidebar.checkbox(
                'Show Statistics', value=True, disabled=not is_show_islands, key='ISLANDS STATISTICS CHECKBOX')
            if is_show_statistics:
                show_statistics()

            true_n_dict = {event_id: len(energies_event) for event_id, energies_event in
                           dataset.energies_dict.items()}
            cluster_energies_dict = dict()
            for event_id in true_energies_dict:
                cur_cluster_energies = estimators[event_id].sum_per_cluster
                cluster_energies_dict[event_id] = cur_cluster_energies

            if st.sidebar.checkbox('Filter by Rec energy', value=False):
                max_energy = np.max([val for _, lst in cluster_energies_dict.items() for val in lst])
                max_cluster_energy_threshold = st.sidebar.number_input(
                    label='Cluster energy max threshold', min_value=0.0, max_value=max_energy, value=max_energy,
                    step=0.001,
                )
                min_cluster_energy_threshold = st.sidebar.number_input(
                    label='Cluster energy min threshold', min_value=0.0, max_value=max_energy, value=0.0,
                    step=0.001,
                )
            else:
                min_cluster_energy_threshold, max_cluster_energy_threshold = 0, np.inf

            filtered_true_energies = list()
            filtered_cluster_energies = list()
            filtered_true_n = list()
            for event_id, cur_cluster_e_list in cluster_energies_dict.items():
                if true_n_dict[event_id] == rec_n_dict[event_id]:
                    for ind, cur_cluster_energy in enumerate(cur_cluster_e_list):
                        if min_cluster_energy_threshold <= cur_cluster_energy <= max_cluster_energy_threshold:
                            filtered_true_energies.append(true_energies_dict[event_id][ind])
                            filtered_cluster_energies.append(cluster_energies_dict[event_id][ind])
                            filtered_true_n.append(true_n_dict[event_id])
            filtered_true_energies = np.array(filtered_true_energies)
            filtered_cluster_energies = np.array(filtered_cluster_energies)

            filtered_true_n = np.array(filtered_true_n)
            filtered_true_n_unique = np.unique(filtered_true_n)

            if len(filtered_true_n_unique) == 0:
                st.header("You filtered out all energies")
                return

            auto_calibration = st.sidebar.checkbox(
                'Auto Calibration', value=False, disabled=not is_show_islands, key='ISLANDS CALIBRATION CHECKBOX')
            if auto_calibration:
                result, fig = calibrate_energies(filtered_true_energies, filtered_cluster_energies,
                                                 return_plot=True, intercept=0)
            else:
                chosen_slope = st.sidebar.number_input('Choose Slope', value=85, step=1)
                result, fig = calibrate_energies(filtered_true_energies, filtered_cluster_energies,
                                                 return_plot=True, intercept=0, slope=chosen_slope)
            fig.update_layout(get_plotly_layout(xaxis_title='True', yaxis_title='Reconstructed'))
            calibration_cols1 = st.columns(2)
            calibration_cols1[0].plotly_chart(fig)
            calibration_cols1[0].header(f'{sigfiground(result.best_values)}, '
                                        f'Chi2red={sigfiground(result.redchi)}')
            filtered_calibrated_energies = result.best_fit

            mean_error_per_true_n, std_error_per_true_n, accuracy_per_true_n = list(), list(), list()
            multiplicity_per_true_n = list()
            for cur_n in filtered_true_n_unique:
                cur_mask = np.array(filtered_true_n) == cur_n
                cur_calibrated_energies = filtered_calibrated_energies[cur_mask]
                cur_true_energies = filtered_true_energies[cur_mask]
                cur_error_array = cur_calibrated_energies - cur_true_energies
                mean_error_per_true_n.append(np.mean(cur_error_array))
                std_error_per_true_n.append(np.std(cur_error_array))
                multiplicity_per_true_n.append(len(cur_error_array))

            hist_df2 = pd.DataFrame({
                'True N': filtered_true_n_unique,
                'Mean Error': sigfiground(mean_error_per_true_n, 2),
                'Std': sigfiground(std_error_per_true_n, 2),
                'Multiplicity': sigfiground(multiplicity_per_true_n, 2)
            })
            islands_cols2 = st.columns(2)
            fig = px.scatter(hist_df2, x='True N', y='Mean Error', error_y='Std',
                             title=f'Mean error per N',
                             labels={'Mean Error': 'mean(True - Rec)'})
            fig.update_layout(get_plotly_layout())
            my_plotly_chart(islands_cols2[0], fig)

            fig = px.bar(hist_df2, x='True N', y='Multiplicity', text_auto=True)
            fig.update_layout(get_plotly_layout(title_text="Multiplicity"))
            my_plotly_chart(islands_cols2[1], fig)

            error_array = filtered_calibrated_energies - filtered_true_energies
            total_error = np.std(error_array[~np.isnan(error_array)])
            st.header(f'Mean error for all events = {sigfiground(total_error)}')

        if is_show_islands:
            with times_agg('Islands predictor'):
                show_islands()

    main()