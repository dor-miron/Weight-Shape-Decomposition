from os import path
import streamlit as st
import EcalDataIO
from Parzen import plot_all, weight_shape_decomp, get_data, energy_to_x
import numpy as np
from matplotlib import pyplot as plt

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



