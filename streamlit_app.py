import streamlit as st

import EcalDataIO
from Parzen import plot_all, weight_shape_decomp, get_data
from pathlib import Path

from matplotlib import pyplot as plt
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

file = 5
data_dir = Path("data\\")

sigma = st.sidebar.slider('Sigma', min_value=0.1, max_value=2.0, step=0.1, value=1.0)

kernel_size = st.sidebar.slider('Kernel size', min_value=3, max_value=21, step=2, value=13)

columns = st.sidebar.columns(2)

array_names = ['Psi', 'V', 'W', 'S', 'Psi x Shape', 'PxS - P']
default_plot = [True, False, False, True, True, True]
to_plot_bool = [columns[0].checkbox(array_names[i], value=default_plot[i]) for i in range(len(array_names))]

elihu_chosen_event_ids = ['708', '813', '261', '103']
event_id = columns[1].radio('Event ID', elihu_chosen_event_ids, index=1)

en_dep = EcalDataIO.ecalmatio(data_dir / f"signal.al.elaser.IP0{file}.edeplist.mat")
energies = EcalDataIO.energymatio(data_dir / f"signal.al.elaser.IP0{file}.energy.mat")
data_tuple = (en_dep, energies)

# Cache
# get_data = st.cache(get_data)
# weight_shape_decomp = st.cache(weight_shape_decomp)
# plot_all = st.cache(plot_all)

raw_data = get_data(data_tuple, event_id, t=1)
P, V, W, S = weight_shape_decomp(raw_data, kernel_size, sigma)
PxS = P*S
P_PxS_dec = PxS - P

fig = plot_all(P, V, W, S, PxS, P_PxS_dec, to_plot_bool=to_plot_bool)

st.pyplot(fig)



