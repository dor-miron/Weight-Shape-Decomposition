import numpy as np
from streamlit import bootstrap
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from utils.other_utils import get_random_string

LINE_SEPARATION = "_______________________"
ISLANDS_EXPLANATION = "The island predictor works as follows:\n"
" - Take a 3D image\n"
" - Sum over Z and Y axes\n"
" - Over the 1D array:\n"
"   - Take maximal value and assign it to new cluster, then remove from array\n"
"   - Take new maximal value. \n"
"       - If it neighbors and existing cluster, then assign to existing cluster\n"
"       - Else, assign to new cluster\n"
"   - Remove assigned value from array and repeat last stage\n"


def get_plotly_layout(rows=1, cols=1, **kwargs):
    plotly_layout = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgb(220,220,220)',
                  size=28),
        height=500 * rows, width=1100 * cols,
        # coloraxis=dict(colorscale='viridis'),
    )
    plotly_layout.update(kwargs)
    return plotly_layout

def my_plotly_chart(where, fig):
    where.plotly_chart(fig)
    where.button("Show in browser", on_click=fig.show, key=get_random_string(8))


if __name__ == '__main__':
    real_script = 'streamlit_app.py'
    bootstrap.run(real_script, f'run.py {real_script}', [], {})
