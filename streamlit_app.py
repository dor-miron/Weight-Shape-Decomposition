from os import path

import streamlit as st

import EcalDataIO
from Parzen import plot_all, weight_shape_decomp, get_data, energy_to_x
from pathlib import Path
import numpy as np

from matplotlib import pyplot as plt
plt.rcParams['figure.facecolor'] = 'black'
TEXT_COLOR = 'white'
plt.rcParams['text.color'] = TEXT_COLOR
plt.rcParams['axes.labelcolor'] = TEXT_COLOR
plt.rcParams['xtick.color'] = TEXT_COLOR
plt.rcParams['ytick.color'] = TEXT_COLOR
# plt.rcParams['figure.figsize'] = [12, 14]
# plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = "22"


file = 5
data_dir = path.join(path.curdir, 'data')
en_dep = EcalDataIO.ecalmatio(path.join(data_dir, f"signal.al.elaser.IP0{file}.edeplist.mat"))
energies = EcalDataIO.energymatio(path.join(data_dir, f"signal.al.elaser.IP0{file}.energy.mat"))
data_tuple = (en_dep, energies)

columns1 = st.sidebar.columns(2)
expansion_factor = columns1[0].radio('Expansion Factor', [1, 2, 3], index=0)
plot_labels = columns1[1].checkbox('Plot Labels', value=True)

axis_name_list = ['X', 'Y', 'Z']
sigma = [st.sidebar.slider(f'Sigma-{axis_name_list[i]}', min_value=0.1, max_value=5.0, step=0.1, value=2.0)
         for i in range(3)]

kernel_size = st.sidebar.slider('Kernel Size', min_value=3, max_value=21, step=2, value=13)

columns2 = st.sidebar.columns(2)

array_names = ['Raw', 'Psi', 'V', 'W', 'Shape', 'Psi x Shape', 'Psi x Shape^2']
default_plot = [True, False, False, True, True, True, True]
to_plot_bool = [columns2[0].checkbox(array_names[i], value=default_plot[i]) for i in range(len(array_names))]

event_id_list = ['708', '813', '261', '103', '819', '343', '815', '719', '618', '420', '291']
ordered_event_id_list = sorted(event_id_list, key=lambda x: len(energies[x]))
def event2label(event_id):
    return f'{event_id} - {len(energies[event_id])}'
event_id = columns2[1].radio('Event ID', ordered_event_id_list, index=1, format_func=event2label)

# Cache
# get_data = st.cache(get_data)
# weight_shape_decomp = st.cache(weight_shape_decomp)
# plot_all = st.cache(plot_all)

raw_data, e_list = get_data(data_tuple, event_id, t=expansion_factor)
sigma_effective = expansion_factor * np.array(sigma)
kernel_size_effective = (kernel_size - 1) * 2 + 1
P, V, W, S = weight_shape_decomp(raw_data, kernel_size_effective, sigma_effective)
PxS = P*S
PxS2 = PxS * S

st.subheader(f'N={len(e_list)}')
x_values = energy_to_x(e_list)

data_arrays = [raw_data, P, V, W, S, PxS, PxS2]
assert np.all(len(array_names) == len(some_array) for some_array in [default_plot, data_arrays])
data_name_tuple = np.array(
    [(data_arrays[i], array_names[i]) for i in range(len(data_arrays))],
    dtype='object')
to_plot_list = data_name_tuple if (to_plot_bool is None) \
        else data_name_tuple[to_plot_bool, :]

xlines = x_values * expansion_factor if plot_labels else None
fig = plot_all(to_plot_list, xlines=xlines)

st.pyplot(fig)



