import random, string

import pandas as pd
from lmfit.models import LinearModel, Model
import numpy as np
import sigfig
import plotly.express as px
import plotly.graph_objects as go


def calibrate_energies(true, measured, slope=None, intercept=None, return_plot=False):
    def func(x, slope, intercept):
        return x * slope + intercept

    vary_slope = slope is None
    slope_val = 0 if vary_slope else slope
    vary_intercept = intercept is None
    intercept_val = 0 if vary_intercept else intercept

    model = Model(func)
    model.set_param_hint('slope', value=slope_val, vary=vary_slope)
    model.set_param_hint('intercept', value=intercept_val, vary=vary_intercept)
    result = model.fit(true, x=measured)

    predicted = result.best_fit
    sorter = np.argsort(true)
    if return_plot:
        fig = go.Figure()
        fig.add_scatter(x=true[sorter], y=true[sorter], name='True')
        fig.add_scatter(x=true[sorter], y=predicted[sorter], mode='markers', name='Reconstructed')
        fig.update_traces(hovertemplate='%{x}')
        fig.update_layout(dict(title=f'Calibration - {len(true)} energies considered',
                               xaxis_title='True', yaxis_title='Reconstructed',
                               hovermode='x unified'))
        return result, fig
    else:
        return result


def order_of_magnitude(val):
    return int(np.floor(np.log10(val)))

def log_histogram(x, **kwargs):
    start_oom, stop_oom = order_of_magnitude(min(x)), order_of_magnitude(max(x))
    bins = np.logspace(start=start_oom, stop=stop_oom + 1, num=(stop_oom - start_oom + 1) * 10)
    hist = np.histogram(x, bins=bins)
    height_list = hist[0]
    bins_middle = (bins[1:] + bins[:-1]) / 2.0
    bins_width = (bins[1:] -bins[:-1]) / 2.0
    fig = px.bar(x=bins_middle, y=height_list, log_x=True, **kwargs)

    point_list = list()
    point_list.append((bins[0], 0))
    for i, cur_y in enumerate(height_list):
        point_list.append((bins[i], cur_y))
        point_list.append((bins[i + 1], cur_y))
    point_list.append((bins[-1], 0))
    x, y = zip(*point_list)
    fig.add_scatter(x=x, y=y, mode='lines', hoverinfo='skip')

    fig.update_xaxes(ticks='outside', tickmode='array', tickvals=sigfiground(bins))
    # fig.update_layout(dict(xaxis=dict(gridcolor="#FFFFFF")))

    return fig

def get_sum_per_island(array, threshold=0):
    first_ind = None
    sum_list = list()
    ind_tuple_list = list()
    cumsum = np.cumsum(array)
    for ind, val in enumerate(array):
        if val > threshold and first_ind is None:
            first_ind = ind
        elif val <= threshold and first_ind is not None:
            sum_list.append(cumsum[ind] - cumsum[first_ind - 1])
            ind_tuple_list.append((first_ind, ind))
            first_ind = None

    ind2cluster = np.full_like(array, 0)
    for ind, tup in enumerate(ind_tuple_list):
        ind2cluster[tup[0]:tup[1] + 1] = ind + 1

    return sigfiground(sum_list), ind2cluster


def sigfiground(x, ndigits=3):
    if type(x) is dict:
        return {key: sigfig.round(value, 2) for key, value in x.items()}
    elif hasattr(x, '__iter__'):
        rounded_array = np.full_like(x, -1)
        for ind, val in enumerate(x):
            try:
                rounded_array[ind] = sigfig.round(val, ndigits)
            except:
                print()
        return rounded_array
    else:
        return sigfig.round(x, ndigits)


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


if __name__ == '__main__':
    vals = np.abs(np.random.normal(scale=1, size=10000)) + 1
    log_histogram(vals).show()