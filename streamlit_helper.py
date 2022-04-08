import numpy as np
from streamlit import bootstrap
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from utils.other_utils import get_random_string

LINE_SEPARATION = "_______________________"

def get_plotly_layout(rows=1, cols=1):
    plotly_layout = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgb(220,220,220)',
                  size=28),
        height=500 * rows, width=1100 * cols,
        coloraxis=dict(colorscale='viridis'),
    )
    return plotly_layout

def my_plotly_chart(where, fig):
    where.plotly_chart(fig)
    where.button("Show in browser", on_click=fig.show, key=get_random_string(8))


if __name__ == '__main__':
    real_script = 'streamlit_app.py'
    bootstrap.run(real_script, f'run.py {real_script}', [], {})