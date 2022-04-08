import random, string

import numpy as np
import sigfig

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


def sigfiground(array, ndigits=3):
    rounded_array = np.full_like(array, -1)
    for ind, val in enumerate(array):
        rounded_array[ind] = sigfig.round(val, ndigits)

    return rounded_array

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str