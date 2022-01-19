from os import path

import streamlit as st

import EcalDataIO
from Parzen import plot_all, weight_shape_decomp, get_data, energy_to_x
from pathlib import Path

from matplotlib import pyplot as plt
plt.rcParams['figure.facecolor'] = 'black'
TEXT_COLOR = 'white'
plt.rcParams['text.color'] = TEXT_COLOR
plt.rcParams['axes.labelcolor'] = TEXT_COLOR
plt.rcParams['xtick.color'] = TEXT_COLOR
plt.rcParams['ytick.color'] = TEXT_COLOR
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100


file = 5
data_dir = path.join(path.curdir, 'data')
en_dep = EcalDataIO.ecalmatio(path.join(data_dir, f"signal.al.elaser.IP0{file}.edeplist.mat"))
energies = EcalDataIO.energymatio(path.join(data_dir, f"signal.al.elaser.IP0{file}.energy.mat"))
data_tuple = (en_dep, energies)

sigma = st.sidebar.slider('Sigma', min_value=0.1, max_value=2.0, step=0.1, value=1.0)

kernel_size = st.sidebar.slider('Kernel size', min_value=3, max_value=21, step=2, value=13)

columns = st.sidebar.columns(2)

array_names = ['Psi', 'V', 'W', 'S', 'Psi x Shape', 'PxS - P']
default_plot = [True, False, False, True, True, True]
to_plot_bool = [columns[0].checkbox(array_names[i], value=default_plot[i]) for i in range(len(array_names))]

event_id_list = ['708', '813', '261', '103', '819', '343', '815', '719', '618', '420', '291']
ordered_event_id_list = sorted(event_id_list, key=lambda x: len(energies[x]))
def event2label(event_id):
    return f'{event_id} - {len(energies[event_id])}'
event_id = columns[1].radio('Event ID', ordered_event_id_list, index=1, format_func=event2label)

# Cache
# get_data = st.cache(get_data)
# weight_shape_decomp = st.cache(weight_shape_decomp)
# plot_all = st.cache(plot_all)

raw_data, e_list = get_data(data_tuple, event_id, t=1)
P, V, W, S = weight_shape_decomp(raw_data, kernel_size, sigma)
PxS = P*S
P_PxS_dec = PxS - P

st.subheader(f'N={len(e_list)}')
x_values = energy_to_x(e_list)
fig = plot_all(P, V, W, S, PxS, P_PxS_dec, to_plot_bool=to_plot_bool, xlines=x_values)

st.pyplot(fig)



