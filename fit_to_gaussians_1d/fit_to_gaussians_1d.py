from functools import partial
from os import path
from typing import List

import plotly.graph_objects as go
import lmfit.model
import numpy as np
from lmfit import Parameters
from plotly.subplots import make_subplots
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from utils import EcalDataIO
from quantum_clustering.ws_decomposition import convert_to_array_and_expand, weight_shape_decomp
import matplotlib.pyplot as plt
from lmfit.models import Gaussian2dModel, GaussianModel
from plotly import express as px


def star_search_1d(data, x, n, model=GaussianModel(), max_center_deviation=5, plot_debug=False) -> List[
    lmfit.model.ModelResult]:
    mcd = max_center_deviation

    result_list = list()
    fitted_image_list = list()
    cur_image_list = list()
    max_index_list = list()
    cur_image = data
    for i in range(n):
        print(f'Fitting gaussian #{i + 1}')
        ix, iy = np.where(cur_image == cur_image.max())
        ix, iy = ix[0], iy[0]
        max_val = cur_image[ix, iy]
        params = Parameters()
        params.add('amplitude', value=max_val, min=0, max=max_val)
        params.add('center', value=ix, min=ix - mcd, max=ix + mcd)
        params.add('sigma', value=2, min=0, max=10)

        # Fit model
        # weights = 1 / (np.abs(x - ix) + 1)
        weights = np.full_like(x, 1)
        out = model.fit(cur_image, params=params, x=x, method='basinhopping',
                        weights=weights)
        result_list.append(out)

        # Decrement from image
        fitted_image = model.eval(out.params, x=x, y=y)
        cur_image = cur_image - fitted_image

        # Append result
        max_index_list.append((ix, iy))
        fitted_image_list.append(fitted_image)
        cur_image_list.append(cur_image)

    return result_list

def get_n_1d_gaussians_model(data, x, n, guess=True):
    data, x = data.ravel(), x.ravel()
    model_list = list()
    params = Parameters()

    for i in range(0, n):
        cur_model = GaussianModel(prefix=f'g{i}_')
        model_list.append(cur_model)
        if guess:
            params.update(cur_model.guess(data, x=x))
        else:
            params.update(cur_model.make_params())

    sum_model = model_list[0]
    for model in model_list[1:]:
        sum_model += model

    return sum_model, params

def split_sum_to_multiple_fits(model, params):
    pass

def main():
    file = 5
    data_dir = path.join(path.curdir, '../data')
    calo = EcalDataIO.ecalmatio(path.join(data_dir, f"signal.al.elaser.IP0{file}.edeplist.mat"))
    energies = EcalDataIO.energymatio(path.join(data_dir, f"signal.al.elaser.IP0{file}.energy.mat"))

    event_id, n = '813', 1
    calo_event, energy_event = calo[event_id], energies[event_id]
    expanded_array, e_list = convert_to_array_and_expand(calo_event, energy_event, t=1)

    expansion_factor, sigma, kernel_size = 1, (1, 2, 2), 7
    sigma_effective = expansion_factor * np.array(sigma)
    kernel_size_effective = (kernel_size - 1) * 2 + 1
    P, V, S, W = weight_shape_decomp(expanded_array, kernel_size_effective, sigma_effective)
    PxS = P * S

    data_to_fit = PxS.sum(axis=2).sum(axis=1)
    X, Y = np.meshgrid(np.arange(data_to_fit.shape[0]), np.arange(data_to_fit.shape[1]), indexing='ij')

    gauss_model, params = get_n_1d_gaussians_model(data_to_fit, X, Y, n)
    out = gauss_model.fit(data_to_fit, params=params, x=X, y=Y, method='basinhopping')
    print(out.fit_report())

    """ PLOT """

    fitted_image = gauss_model.eval(out.params, x=X, y=Y)
    fig, ax_list = plt.subplots(3, 1)

    img = ax_list[0].imshow(data_to_fit, origin='lower')
    # fig.colorbar(img, ax=ax_list[0:2])
    fig.colorbar(img, ax=ax_list[0])
    ax_list[0].set_title('Real')
    img = ax_list[1].imshow(fitted_image, origin='lower')
    fig.colorbar(img, ax=ax_list[1])
    ax_list[1].set_title('Fit')
    img = ax_list[2].imshow(data_to_fit - fitted_image, origin='lower')
    fig.colorbar(img, ax=ax_list[2])
    ax_list[2].set_title('Residuals')
    fig.show()

if __name__ == '__main__':
    main()