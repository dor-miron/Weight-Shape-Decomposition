import random, string

import pandas as pd
from lmfit.models import LinearModel, Model
import numpy as np
import sigfig
import plotly.express as px

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
        fig = px.line(x=true[sorter], y=[true[sorter], predicted[sorter]],
                      title=f'Calibration - {len(true)} energies considered')
        fig.update_layout(dict(xaxis_title='True', yaxis_title='Predicted'))
        return result, fig
    else:
        return result

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