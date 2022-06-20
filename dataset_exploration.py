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

    stride = 1
    with times_agg('get_all'):
        events_array, true_n_list = get_events_array(stride)

    def dataset_statistics():
        stat_cols = st.columns(2)
        fig = px.imshow(events_array.mean(axis=0).T, origin='lower', title='Mean')
        fig.update_layout(get_plotly_layout())
        stat_cols[0].plotly_chart(fig)

        fig = px.imshow(events_array.std(axis=0).T, origin='lower', title='Std')
        fig.update_layout(get_plotly_layout())
        stat_cols[1].plotly_chart(fig)

        all_energies = np.concatenate(list(energies_dict.values()))
        fig = px.histogram(all_energies, title='True energies distribution')
        fig.update_layout((get_plotly_layout(xaxis_title='Energy (MeV)', yaxis_title='Count')))
        st.plotly_chart(fig)

        min_dist_cols = st.columns(2)
        fig = dataset.get_min_distance_histogram()
        fig.update_layout(get_plotly_layout())
        min_dist_cols[0].plotly_chart(fig)

        fig = dataset.get_min_dist_vs_n_scatter()
        fig.update_layout(get_plotly_layout())
        min_dist_cols[1].plotly_chart(fig)

        hist_cols = st.columns(2)
        fig = dataset.get_n_particles_histogram()
        fig.update_layout(
            get_plotly_layout(xaxis_title='True N', yaxis_title='Count', title='Multiplicity - All'))
        hist_cols[0].plotly_chart(fig)

        fig = dataset.get_n_particles_histogram(max_n=20)
        fig.update_layout(
            get_plotly_layout(xaxis_title='True N', yaxis_title='Count', title='Multiplicity - Partial'))
        hist_cols[1].plotly_chart(fig)

    sth.wrap_streamlit_function(dataset_statistics, 'Dataset Statistics', value=True, times_agg=times_agg)

    def show_islands():
        progress_placeholder = st.empty()
        progress_placeholder.header('Islands predictor')
        with st.expander('See explanation'):
            st.write(ISLANDS_EXPLANATION)

        mode = st.sidebar.radio('Data Set', ['regular_com', 'shift_com', 'angle'], index=2)
        en_dep_thresh = st.sidebar.number_input('Energy Deposition Threshold', min_value=0.0, value=0.0, step=0.001)
        tower_thresh = st.sidebar.number_input('Tower Threshold', min_value=0.0, value=0.0, step=0.01)
        cluster_thresh = st.sidebar.number_input('Cluster Threshold', min_value=0.0, value=CLUSTER_THRESHOLD_VALUE,
                                                 step=0.01)
        n_events = len(dataset)
        true_energies_dict = dict()
        estimators = dict()
        for ind, event_id in enumerate(dataset.keys()):
            progress_placeholder.header(f'{ind + 1}/{n_events} Events Loaded')
            island_estimator = IslandEstimator(position_reconstruction_method=mode,
                                               en_dep_threshold=en_dep_thresh, tower_threshold=tower_thresh,
                                               cluster_threshold=cluster_thresh)
            island_estimator.predict_one(calo_dict[event_id])
            island_estimator.calc_calibrated_energies(lambda x: x * 85)
            estimators[event_id] = island_estimator
            true_energies_dict[event_id] = sorted(energies_dict[event_id], reverse=True)

        def event_statistics():
            rec_n_array_rounded = np.full((n_events,), 0)
            rec_n_array_unrounded = np.full((n_events,), 0, dtype=np.float64)
            true_n_array = np.full((n_events,), 0)
            for ind, event_id in enumerate(dataset.keys()):
                rec_n_array_rounded[ind] = len(estimators[event_id].energies_list)
                rec_n_array_unrounded[ind] = sum(estimators[event_id].energy_ratio_dict.values())
                true_n_array[ind] = len(dataset.energies_dict[event_id])

            # def get_rms(a):


            hist_rounded_cols = st.columns(2)
            fig = px.histogram(true_n_array - rec_n_array_rounded, title='Rounded, Non Normalized')
            fig.update_layout(get_plotly_layout(xaxis_title='error'))
            hist_rounded_cols[0].plotly_chart(fig)
            hist_rounded_cols[0].header(np.mean(true_n_array - rec_n_array_rounded))
            # hist_rounded_cols[0].header(np.sqrt(true_n_array - rec_n_array_rounded)**2)

            fig = px.histogram((true_n_array - rec_n_array_rounded) / true_n_array, title='Rounded, Normalized')
            fig.update_layout(get_plotly_layout(xaxis_title='error normalized'))
            hist_rounded_cols[1].plotly_chart(fig)
            hist_rounded_cols[1].header(np.mean((true_n_array - rec_n_array_rounded) / true_n_array))

            hist_unrounded_cols = st.columns(2)
            fig = px.histogram(true_n_array - rec_n_array_unrounded, title='Unrounded, Non Normalized')
            fig.update_layout(get_plotly_layout(xaxis_title='error'))
            hist_unrounded_cols[0].plotly_chart(fig)
            hist_unrounded_cols[0].header(np.mean(true_n_array - rec_n_array_unrounded))

            fig = px.histogram((true_n_array - rec_n_array_unrounded) / true_n_array, title='Unrounded, Normalized')
            fig.update_layout(get_plotly_layout(xaxis_title='error normalized'))
            hist_unrounded_cols[1].plotly_chart(fig)
            hist_unrounded_cols[1].header(np.mean((true_n_array - rec_n_array_unrounded) / true_n_array))

            error_array = true_n_array - rec_n_array_rounded
            true_n_unique = np.unique(true_n_array)
            mean_error_per_true_n, std_error_per_true_n, accuracy_per_true_n = list(), list(), list()
            count_per_true_n = list()
            for cur_val in true_n_unique:
                mask = true_n_array == cur_val
                cur_error_array = error_array[mask]
                mean_error_per_true_n.append(np.mean(cur_error_array))
                std_error_per_true_n.append(np.std(cur_error_array))
                accuracy_per_true_n.append(np.sum(cur_error_array == 0) / float(len(cur_error_array)))
                count_per_true_n.append(len(cur_error_array))

            hist_df = pd.DataFrame({
                'True N': true_n_unique,
                'Mean Error': sigfiground(mean_error_per_true_n, 2),
                'Std': sigfiground(std_error_per_true_n, 2),
                'Accuracy': sigfiground(accuracy_per_true_n, 2),
                'Count': sigfiground(count_per_true_n, 2)
            })

            islands_cols = st.columns(2)
            fig = px.scatter(hist_df, x='True N', y='Mean Error', error_y='Std',
                             title=f'Mean Multiplicity Error')
            fig.update_xaxes(title_text='True N (Multiplicity)')
            fig.update_yaxes(title_text='mean(True - Rec)')
            fig.update_layout(get_plotly_layout())
            my_plotly_chart(islands_cols[0], fig)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=hist_df['True N'], y=hist_df['Accuracy'], name='Efficiency',
                                     mode='lines+markers'), secondary_y=False)
            fig.add_trace(go.Scatter(x=hist_df['True N'], y=hist_df['Count'], name='Count',
                                     mode='lines+markers'), secondary_y=True, )
            fig.update_xaxes(title_text="True N (Multiplicity)")
            fig.update_yaxes(title_text="<b>Efficiency</b>", secondary_y=False)
            fig.update_yaxes(title_text="<b>Count</b>", secondary_y=True)
            fig.update_layout(get_plotly_layout())
            islands_cols[1].header(f'Correct events: {(error_array == 0).sum()}')
            my_plotly_chart(islands_cols[1], fig)

        sth.wrap_streamlit_function(event_statistics, 'Event Statistics', True, times_agg)

        def energy_statistics():
            true_n_dict = {event_id: len(energies_event) for event_id, energies_event in
                           dataset.energies_dict.items()}

            filtered_true_energies = list()
            filtered_cluster_energies = list()
            filtered_true_n = list()
            incorrect_events_list = list()
            for event_id, estimator in estimators.items():
                cur_energies = estimators[event_id].energies_list
                if true_n_dict[event_id] == len(cur_energies):
                    for ind, cur_energy in enumerate(cur_energies):
                        filtered_true_energies.append(true_energies_dict[event_id][ind])
                        filtered_cluster_energies.append(cur_energy)
                        filtered_true_n.append(true_n_dict[event_id])
                else:
                    incorrect_events_list.append(event_id)
            filtered_true_energies = np.array(filtered_true_energies)
            filtered_cluster_energies = np.array(filtered_cluster_energies)

            filtered_true_n = np.array(filtered_true_n)
            filtered_true_n_unique = np.unique(filtered_true_n)

            fig = px.scatter(x=filtered_true_energies, y=filtered_cluster_energies,
                             title='Reconstructed VS. True energies')
            fig.update_layout(get_plotly_layout(xaxis_title='True (MeV)',
                                                yaxis_title='Reconstructed (MeV)'))
            st.plotly_chart(fig)

            mean_error_per_true_n, std_error_per_true_n, accuracy_per_true_n = list(), list(), list()
            count_per_true_n = list()
            for cur_n in filtered_true_n_unique:
                cur_mask = np.array(filtered_true_n) == cur_n
                cur_calibrated_energies = filtered_cluster_energies[cur_mask]
                cur_true_energies = filtered_true_energies[cur_mask]
                cur_error_array = cur_calibrated_energies - cur_true_energies
                mean_error_per_true_n.append(np.mean(cur_error_array))
                std_error_per_true_n.append(np.std(cur_error_array))
                count_per_true_n.append(len(cur_error_array))

            hist_df2 = pd.DataFrame({
                'True N': filtered_true_n_unique,
                'Mean Error': sigfiground(mean_error_per_true_n, 2),
                'Std': sigfiground(std_error_per_true_n, 2),
                'Count': sigfiground(count_per_true_n, 2)
            })
            islands_cols2 = st.columns(2)
            fig = px.scatter(hist_df2, x='True N', y='Mean Error', error_y='Std',
                             title=f'Mean Energy reconstruction error',
                             labels={'Mean Error': 'mean(True - Rec)'})
            fig.update_layout(get_plotly_layout(xaxis_title='True N (Multiplicity)'))
            my_plotly_chart(islands_cols2[0], fig)

            fig = px.bar(hist_df2, x='True N', y='Count', text_auto=True)
            fig.update_layout(get_plotly_layout(title_text="Energy Count", xaxis_title='True N (Multiplicity)',
                                                yaxis_title='Count'))
            my_plotly_chart(islands_cols2[1], fig)

            incorrect_with_info = [
                (event_id, len(dataset.energies_dict[event_id]), len(estimators[event_id].energies_list))
                for event_id in incorrect_events_list[:15]]
            st.header(f'Incorrect events\n{incorrect_with_info}')
        sth.wrap_streamlit_function(energy_statistics, 'Energy Statistics', True, times_agg=times_agg)

        # bins = 60
        # rec_energies, rec_energies_weights = zip(*[(energy, ratio)
        #                                            for estimator in estimators.values()
        #                                            for energy, ratio in estimator.energy_ratio_dict.items()])
        # true_energies = np.concatenate(list(energies_dict.values()))
        # rec_hist, bin_edges = np.histogram(rec_energies, weights=rec_energies_weights, bins=bins)
        # true_hist, _ = np.histogram(true_energies, bins=bin_edges)
        # overlap = [min(t, r) for t, r in zip(true_hist, rec_hist)]
        # true_relative = [(t - r if t > r else 0) for t, r in zip(true_hist, rec_hist)]
        # rec_relative = [(r - t if r > t else 0) for t, r in zip(true_hist, rec_hist)]
        # fig = go.Figure()
        # fig.add_trace(go.Bar(x=bin_edges, y=overlap, name='Overlap', marker=dict(color='#888888')))
        # fig.add_trace(go.Bar(x=bin_edges, y=true_relative, name='True', marker=dict(color='#1F77b4')))
        # fig.add_trace(go.Bar(x=bin_edges, y=rec_relative, name='Reconstructed', marker=dict(color='#FF7F0E')))
        # fig.update_layout(get_plotly_layout(title='Energy Histogram', barmode='stack',
        #                                     xaxis_title='Energy (Mev)', yaxis_title='Count'))
        # st.plotly_chart(fig)

        energy_hist_comp_cols = st.columns(2)

        bins = 60
        rec_energies, rec_energies_weights = zip(*[(energy, ratio)
                                                   for estimator in estimators.values()
                                                   for energy, ratio in estimator.energy_ratio_dict.items()])
        true_energies = np.concatenate(list(energies_dict.values()))
        rec_hist_unrounded, bin_edges = np.histogram(rec_energies, weights=rec_energies_weights, bins=bins)
        true_hist, _ = np.histogram(true_energies, bins=bin_edges)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bin_edges, y=true_hist, name='True', opacity=1,
                             marker=dict(color='#1F77b4')))
        fig.add_trace(go.Bar(x=bin_edges, y=rec_hist_unrounded, name='Reconstructed', opacity=1,
                             marker=dict(color='#FF7F0E')))
        fig.update_layout(get_plotly_layout(title='Energy Histogram - Unrounded', barmode='group',
                                            xaxis_title='Energy (Mev)', yaxis_title='Count'))
        energy_hist_comp_cols[0].plotly_chart(fig)

        rec_energies, rec_energies_weights = zip(*[(energy, round(ratio))
                                                   for estimator in estimators.values()
                                                   for energy, ratio in estimator.energy_ratio_dict.items()])
        rec_hist_rounded, bin_edges = np.histogram(rec_energies, weights=rec_energies_weights, bins=bins)

        rec_multiplicity_per_event = np.sum(rec_hist_rounded) / n_events
        energy_hist_comp_cols[1].header(
            f'Reconstructed multiplicity per event: {sigfiground(rec_multiplicity_per_event)}')
        true_multiplicity_per_event = np.sum(true_hist) / n_events
        energy_hist_comp_cols[1].header(
            f'True multiplicity per event: {sigfiground(true_multiplicity_per_event)}')

        more_hist_cols = st.columns(2)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=bin_edges, y=rec_hist_unrounded - rec_hist_rounded, opacity=1,
                             marker=dict(color='#FF7F0E')))
        fig.update_layout(get_plotly_layout(title='Reconstructed Energy Histogram - Unrounded - Rounded',
                                            barmode='group',
                                            xaxis_title='Energy (Mev)', yaxis_title='Count'))
        more_hist_cols[0].plotly_chart(fig)

        fig = px.scatter(x=(bin_edges[1:] + bin_edges[:-1]) / 2, y=rec_hist_unrounded/true_hist)
        fig.add_shape(type="line", x0=bin_edges[0], y0=1, x1=bin_edges[-1], y1=1,
                      line=dict(color="Red", width=3), name='y=1')
        fig.update_layout(
            get_plotly_layout(title='Energy Ratio - Rec/True (Unrounded)',
                              xaxis_title='Energy (Mev)', yaxis_title='Ratio'))
        more_hist_cols[1].plotly_chart(fig)

    sth.wrap_streamlit_function(show_islands, 'Islands Predictor', value=True, times_agg=times_agg)
