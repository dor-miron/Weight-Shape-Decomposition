from functools import partial
import streamlit as st
from plotly.subplots import make_subplots
from fit_to_gaussians_1d.fit_to_gaussians_1d import get_n_1d_gaussians_model
from quantum_clustering.ws_decomposition import count_clusters_by_z_line, energy_to_x
import numpy as np
import altair as alt
import pandas as pd
from plotly import graph_objects as go
from plotly import express as px
from streamlit_helper import LINE_SEPARATION, get_plotly_layout
from pipelines.estimators import IslandEstimator
from utils.other_utils import sigfiground
import streamlit_helper as sth

CLUSTER_THRESHOLD_VALUE = 15.0

DEFAULT_KERNEL_VALUE = 7
RAW_THRESHOLD_DEFAULT = 0.000
default_sigma_list = [0.5, 1.0, 1.0]

def event2label(event_id):
    return f'{event_id.split("_")[-1]}'

def event_exploration(dataset, times_agg):
    from quantum_clustering.ws_decomposition import weight_shape_decomp, convert_to_array_and_expand
    from quantum_clustering.gradient_descent import cluster_by_gradient_descent

    calo_dict, energies_dict = dataset.calo_dict, dataset.energies_dict

    # INIT WIDGETS
    with times_agg("init widgets"):
        event_id_array = np.array(list(dataset.keys()))
        energy_list = list(dataset.energies_dict.values())
        sorting_permutation = np.argsort(energy_list)
        ordered_event_id_list = event_id_array[sorting_permutation]
        n_list = [len(energy) for energy in energy_list]
        unique_n_list = np.unique(n_list)
        event_cols = st.sidebar.columns(2)
        chosen_n = event_cols[0].selectbox('Select N', options=unique_n_list, index=3)
        chosen_n_event_list = event_id_array[n_list == chosen_n]
        event_id = event_cols[1].selectbox('Select event', options=sorted(chosen_n_event_list),
                                           format_func=event2label)
        calo_event, energy_event = calo_dict[event_id], energies_dict[event_id]
        true_x_values = energy_to_x(energy_event)

    convert_to_array_and_expand = st.cache(convert_to_array_and_expand)

    def show_plot_2d():
        raw_data = convert_to_array_and_expand(calo_event, t=1)
        raw_2d = raw_data.sum(axis=1)
        fig = px.imshow(raw_2d.T, origin='lower', title='Raw data')
        for x_val in true_x_values:
            fig.add_vline(x=x_val, line_width=2, line_color='white', line_dash='dash')
        fig.update_layout(get_plotly_layout(1, 1))
        st.plotly_chart(fig)

    is_show_plot_2d = st.sidebar.checkbox('Show 2D plot', value=True)
    if is_show_plot_2d:
        show_plot_2d()

    def show_qc():
        st.header('Quantum clustering')
        columns1 = st.sidebar.columns(2)
        expansion_factor = columns1[0].radio('Expansion Factor', [1, 2, 3], index=0)

        axis_name_list = ['X', 'Y', 'Z']
        sigma = [st.sidebar.slider(f'Sigma-{axis_name_list[i]}', min_value=0.1, max_value=5.0, step=0.1,
                                   value=default_sigma_list[i])
                 for i in range(3)]
        kernel_size = st.sidebar.slider('Kernel Size', min_value=3, max_value=21, step=2,
                                        value=DEFAULT_KERNEL_VALUE)
        raw_threshold = st.sidebar.number_input('Raw Threshold', min_value=0.000, max_value=0.006,
                                                value=RAW_THRESHOLD_DEFAULT, step=0.0001, format="%f")

        raw_data = convert_to_array_and_expand(calo_event, t=expansion_factor)

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
    sth.wrap_streamlit_function(show_qc, 'Quantum clustering', value=False, times_agg=times_agg)

    def island_clustering():
        cluster_threshold_value = sth.checkbox_with_number_input(
            st.sidebar, 'Min Rec Energy', default=0, value=CLUSTER_THRESHOLD_VALUE, key='clust_thresh')

        estimator = IslandEstimator(cluster_threshold=cluster_threshold_value)
        estimator.predict_one(calo_event)
        plotly_df = pd.DataFrame(
            {'Energy Deposition': estimator.raw_1d_, 'Cluster Number': estimator.ind2cluster_})

        st_cols = st.columns(2)
        fig = px.bar(plotly_df, x=plotly_df.index, y='Energy Deposition', color='Cluster Number')
        fig.update_layout(get_plotly_layout(1, 1))
        st_cols[0].plotly_chart(fig)

        sorted_energies = sigfiground(estimator.sum_per_cluster, 3)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=sorted_energies, text=sorted_energies, mode="markers+text",
                                 textposition="top center"))
        fig.update_layout(get_plotly_layout())
        st_cols[1].plotly_chart(fig)

        estimator.calc_calibrated_energies(lambda x: x * 85)
        st.header(sigfiground(estimator.energies_list, ndigits=6))
        st.header(f'Predicted: {len(estimator.energies_list)}')

        fig = estimator.get_fig_cluster_to_position_compare()
        fig.update_layout(get_plotly_layout())
        sth.my_plotly_chart(st, fig)

    sth.wrap_streamlit_function(island_clustering, 'Island Clustering', value=True, times_agg=times_agg)

    def gauss_fit_1d():
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
    sth.wrap_streamlit_function(gauss_fit_1d, '1D Gauss fit', value=False, times_agg=times_agg)

    def gradient_clustering():
        cluster_by_gradient_descent = st.cache(cluster_by_gradient_descent)
        clustered_array = cluster_by_gradient_descent(-V.sum(axis=1))

        hm = partial(go.Heatmap, coloraxis='coloraxis1')
        fig = make_subplots(1, 1)
        fig.add_trace(hm(z=clustered_array.T), 1, 1)
        fig.update_layout(get_plotly_layout())
        st.plotly_chart(fig)
    sth.wrap_streamlit_function(gradient_clustering, 'Gradient Clustering', value=False, times_agg=times_agg)

    def z_line():
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
    sth.wrap_streamlit_function(z_line, 'Z Line', value=False, times_agg=times_agg)
