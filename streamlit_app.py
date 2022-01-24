from os import path
import streamlit as st
import EcalDataIO
from Parzen import plot_all, weight_shape_decomp, get_data, energy_to_x, count_clusters_by_z_line
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


# """ PYPLOT PARAMS """
plt.rcParams['figure.facecolor'] = 'black'
TEXT_COLOR = 'white'
plt.rcParams['text.color'] = TEXT_COLOR
plt.rcParams['axes.labelcolor'] = TEXT_COLOR
plt.rcParams['xtick.color'] = TEXT_COLOR
plt.rcParams['ytick.color'] = TEXT_COLOR
plt.rcParams['figure.figsize'] = [12, 14]
# plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = "22"

def event2label(event_id):
    return f'{event_id} - {len(energies[event_id])}'


RUN_ONCE = 'RUN_ONCE'
DATA_TUPLE = 'DATA_TUPLE'

# """ GET DATA """
if RUN_ONCE not in st.session_state:
    file = 5
    data_dir = path.join(path.curdir, 'data')
    en_dep = EcalDataIO.ecalmatio(path.join(data_dir, f"signal.al.elaser.IP0{file}.edeplist.mat"))
    energies = EcalDataIO.energymatio(path.join(data_dir, f"signal.al.elaser.IP0{file}.energy.mat"))
    st.session_state[DATA_TUPLE] = (en_dep, energies)
st.session_state[RUN_ONCE] = True
data_tuple = st.session_state[DATA_TUPLE]
en_dep, energies = data_tuple

# """ INIT WIDGETS """
columns1 = st.sidebar.columns(2)
expansion_factor = columns1[0].radio('Expansion Factor', [1, 2, 3], index=1)
plot_labels = columns1[1].checkbox('Plot Labels', value=True)

axis_name_list = ['X', 'Y', 'Z']
default_sigma_list = [1.0, 2.0, 2.0]
sigma = [st.sidebar.slider(f'Sigma-{axis_name_list[i]}', min_value=0.1, max_value=5.0, step=0.1,
                           value=default_sigma_list[i])
         for i in range(3)]

kernel_size = st.sidebar.slider('Kernel Size', min_value=3, max_value=21, step=2, value=9)

columns2 = st.sidebar.columns(2)

PxS_NAME = 'Psi x Shape'
array_names = ['Raw', 'Psi', 'V', 'W', 'Shape', PxS_NAME, 'Psi x Shape^2']
default_plot = [True, False, False, False, True, True, False]
to_plot_bool = [columns2[0].checkbox(array_names[i], value=default_plot[i]) for i in range(len(array_names))]

event_id_list = ['708', '813', '261', '103', '819', '343', '815', '719', '618', '420', '291']
ordered_event_id_list = sorted(event_id_list, key=lambda x: len(energies[x]))
event_id = columns2[1].radio('Event ID', ordered_event_id_list, index=1, format_func=event2label)

# """ CALCULATE ARRAYS """
# get_data = st.cache(get_data)
raw_data, e_list = get_data(data_tuple, event_id, t=expansion_factor)
sigma_effective = expansion_factor * np.array(sigma)
kernel_size_effective = (kernel_size - 1) * 2 + 1
P, V, W, S = weight_shape_decomp(raw_data, kernel_size_effective, sigma_effective)
PxS = P*S
PxS2 = PxS * S

x_values = energy_to_x(e_list)

# """ PREPARE FOR PLOT """
data_arrays = [raw_data, P, V, W, S, PxS, PxS2]
assert np.all(len(array_names) == len(some_array) for some_array in [default_plot, data_arrays])
data_name_tuple = np.array(
    [(data_arrays[i], array_names[i]) for i in range(len(data_arrays))],
    dtype='object')
to_plot_list = data_name_tuple if (to_plot_bool is None) \
        else data_name_tuple[to_plot_bool, :]

# """ PLOT """
xlines = x_values * expansion_factor if plot_labels else None
fig = plot_all(to_plot_list, xlines=xlines)
st.pyplot(fig)

show_z_line = st.sidebar.checkbox('Show Z Line', value=True)
if show_z_line:
    z_line = st.sidebar.slider('Z Line', min_value=0, max_value=30, step=1, value=10)
    peaks_x, peaks_y, chosen_line = count_clusters_by_z_line(PxS, z_value=z_line)
    fig, ax = plt.subplots(1, 1)
    ax.plot(chosen_line)
    # peaks = np.asarray(peaks[0], np.int)
    ax.plot(peaks_x, peaks_y, '*')
    ax.set_title(f'PxS, Z={z_line}, Pred/True={len(peaks_x)}/{len(e_list)}')
    st.pyplot(fig)



