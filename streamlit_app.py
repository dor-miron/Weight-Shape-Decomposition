import streamlit as st
from dataset_exploration import dataset_exploration
from event_exploration import event_exploration
from afori_utils.debug_utils import TimesAgg
from streamlit_helper import LINE_SEPARATION
import streamlit_helper as sth
from utils.data_utils import Dataset

DS5 = 'FL=5'
DS8 = 'FL=8'
DS_SINGLE_10GEV = 'Single - 10 Gev'

DEFAULT_MAX_N_VALUE = 110
DEFAULT_MIN_DIST = None

MENU_OPTIONS = ['Nothing', 'Dataset exploration', 'Event exploration']
DEFAULT_MENU_INDEX = 1

st.set_page_config(
    page_title='QC Dashboard',
    page_icon=":shark:",
    layout='wide',
    initial_sidebar_state="expanded"
)

top_container = st.container()
top_container.text("Loading...")

times_agg = TimesAgg()

# """ GET DATA """
dataset_key = st.sidebar.radio('Data Set', [DS5, DS8, DS_SINGLE_10GEV], index=0)
key2path = {
    DS5: ("signal.al.elaser.IP05.edeplist.mat", "signal.al.elaser.IP05.energy.mat"),
    DS8: ("signal.al.elaser.IP08.edeplist.mat", "signal.al.elaser.IP08.energy.mat"),
    DS_SINGLE_10GEV: ("alw_10k_10GeV_edep.mat", "alw_10k_10GeV_trueXY.mat"),
}

st.text(f'Saved session keys: {list(st.session_state.keys())}')

with times_agg('Get data'), st.spinner('Get data'):
    cache_key = f"data_{dataset_key}"
    if cache_key in st.session_state.keys():
        dataset = st.session_state[cache_key]
    else:
        dataset = Dataset(r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\data')
        dataset.add_dataset(*key2path[dataset_key], dataset_name=dataset_key)
        st.session_state[cache_key] = dataset
    n_events_unfiltered = dataset.__len__()

with times_agg('Filter by number of particles'):
    max_n = sth.checkbox_with_number_input(st.sidebar, 'Max N', step=1, min_value=0, value=DEFAULT_MAX_N_VALUE)
    dataset = dataset.filter_by_number_of_particles(max_n=max_n, inplace=False)

with times_agg('Filter by minimal distance'):
    min_dist = sth.checkbox_with_number_input(st.sidebar, 'Min Distance (mm)', step=0.01, min_value=0.0,
                                              default=0.0, value=DEFAULT_MIN_DIST, key='MINDISTFILT')
    dataset = dataset.filter_by_distance(min_dist, inplace=False)

n_events_filtered = dataset.__len__()
st.header(f'After filter - {n_events_filtered}/{n_events_unfiltered} events')

st.sidebar.text(LINE_SEPARATION)

choice = st.sidebar.radio('Choice', MENU_OPTIONS, index=DEFAULT_MENU_INDEX)
st.sidebar.text(LINE_SEPARATION)
if choice == 'Dataset exploration':
    dataset_exploration(dataset, times_agg)
elif choice == 'Event exploration':
    event_exploration(dataset, times_agg)

st.text(times_agg.__str__())

top_container.text("Finished Loading")
