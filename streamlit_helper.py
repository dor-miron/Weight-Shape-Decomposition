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
        plot_bgcolor='rgba(40,40,40,255)',
        font=dict(color='rgb(220,220,220)',
                  size=28),
        height=500 * rows, width=1100 * cols,
        # coloraxis=dict(colorscale='viridis'),
        xaxis=dict(tickcolor='white', ticks='inside')
    )
    plotly_layout.update(kwargs)
    return plotly_layout

def my_plotly_chart(where, fig):
    where.plotly_chart(fig)
    where.button("Show in browser", on_click=fig.show, key=get_random_string(8))

def checkbox_with_number_input(where, label, default=float('inf'), **kwargs):
    if 'value' in kwargs and 'check_value' not in kwargs:
        kwargs['check_value'] = False if kwargs['value'] is None else kwargs['value']
        if not kwargs['check_value']:
            kwargs['value'] = st.state.NoValue()

    cols = where.columns([1, 6])
    check_value = kwargs.pop('check_value', False)
    disabled = not cols[0].checkbox('', check_value)
    kwargs['disabled'] = disabled
    max = cols[1].number_input(label + (' (Disabled)' if disabled else ''),
                               **kwargs)
    return default if disabled else max


if __name__ == '__main__':
    real_script = 'streamlit_app.py'
    bootstrap.run(real_script, f'run.py {real_script}', [], {})
