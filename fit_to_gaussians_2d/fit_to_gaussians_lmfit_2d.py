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
from lmfit.models import Gaussian2dModel

def iter_callback(params, iter, resid, *args, **kwargs):
    if iter % 5 == 0:
        print(iter)

def detect_peaks_with_maximum_filter(image, threshold=0.01):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to
    #successfully subtract it form local_max, otherwise a line will
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks,
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

def star_search_2d(data, x, y, n, model=Gaussian2dModel(), max_center_deviation=5, plot_debug=False) -> List[
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
        params.add('centerx', value=ix, min=ix - mcd, max=ix + mcd)
        params.add('centery', value=iy, min=iy - mcd, max=iy + mcd)
        params.add('sigmax', value=2, min=0, max=10)
        params.add('sigmay', value=10, min=0, max=20)
        params.add('sigma_ratio', expr='sigmax/sigmay', max=0.5, min=0.1)

        out = model.fit(cur_image, params=params, x=x, y=y, method='basinhopping')
        result_list.append(out)

        fitted_image = model.eval(out.params, x=x, y=y)
        cur_image = cur_image - fitted_image

        max_index_list.append((ix, iy))
        fitted_image_list.append(fitted_image)
        cur_image_list.append(cur_image)

    if plot_debug:
        # t = 1
        # fig, ax_list = plt.subplots(n + t, 2)
        # ax_list[0][0].imshow(data.T, origin='lower')
        # ax_list[0][0].set_title('Original Data')
        # ax_list[0][1].imshow(np.stack(fitted_image_list, axis=0).sum(axis=0).T, origin='lower')
        # ax_list[0][1].set_title('Sum of fits')
        # for i in range(0, n):
        #     ax_list[i+t, 0].imshow(cur_image_list[i].T, origin='lower')
        #     ax_list[i+t, 1].imshow(fitted_image_list[i].T, origin='lower')
        # fig.show()

        layout = dict(
            # yaxis=dict(scaleanchor="x", scaleratio=1),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgb(220,220,220)'),
            height=300 * (n+1), width=750,
            title_text="Side By Side Subplots",
            coloraxis=dict(colorscale='viridis')
        )
        hover_template = 'X: %{x}<br>' + 'Z: %{y}<br>'
        hm = partial(go.Heatmap, coloraxis='coloraxis1', hover_template=hover_template)

        fig = make_subplots(n + 1, 2)
        fig.add_trace(hm(z=data.T), 1, 1)
        fig.add_trace(hm(z=np.stack(fitted_image_list, axis=0).sum(axis=0).T), 1, 2)
        for i in range(0, n):
            cur_params = result_list[i].best_values
            sx, sy = cur_params['sigmax'], cur_params['sigmay']
            fig.add_trace(hm(z=cur_image_list[i].T), i + 2, 1)
            fig.add_trace(hm(z=fitted_image_list[i].T), i + 2, 2,
                          hover_template=hover_template + f'<br>sx={sx}, sy={sy}')

        # fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
        fig.update_layout(layout)
        fig.show()

    return result_list


def get_n_2d_gaussians_model(data, x, y, n, guess=True):
    data, x, y = data.ravel(), x.ravel(), y.ravel()
    model_list = list()
    params = Parameters()

    for i in range(0, n):
        cur_model = Gaussian2dModel(prefix=f'g{i}_')
        model_list.append(cur_model)
        if guess:
            params.update(cur_model.guess(data, x=x, y=y))

    final_model = model_list[0]
    for model in model_list[1:]:
        final_model += model
    return final_model, params


def fit_real():
    file = 5
    data_dir = path.join(path.curdir, '../data')
    calo = EcalDataIO.ecalmatio(path.join(data_dir, f"signal.al.elaser.IP0{file}.edeplist.mat"))
    energies = EcalDataIO.energymatio(path.join(data_dir, f"signal.al.elaser.IP0{file}.energy.mat"))

    event_id, n = '708', 2
    calo_event, energy_event = calo[event_id], energies[event_id]
    expanded_array, e_list = convert_to_array_and_expand(calo_event, energy_event, t=1)

    expansion_factor, sigma, kernel_size = 1, (1, 2, 2), 7
    sigma_effective = expansion_factor * np.array(sigma)
    kernel_size_effective = (kernel_size - 1) * 2 + 1
    P, V, S, W = weight_shape_decomp(expanded_array, kernel_size_effective, sigma_effective)
    PxS = P * S

    data_to_fit = PxS.sum(axis=1)
    X, Y = np.meshgrid(np.arange(data_to_fit.shape[0]), np.arange(data_to_fit.shape[1]), indexing='ij')

    gauss_model_1 = Gaussian2dModel()
    result_list = star_search_2d(data_to_fit, X, Y, n, model=gauss_model_1, plot_debug=True)

    gauss_model, params = get_n_2d_gaussians_model(data_to_fit, X, Y, n)
    out = gauss_model.fit(data_to_fit, params=params, x=X, y=Y, method='basinhopping')
    print(out.fit_report())

    """ PLOT """

    fitted_image = gauss_model.eval(out.params, x=X, y=Y)
    fig,  ax_list = plt.subplots(3, 1)

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

def fit_fake():
    import numpy as np
    from fit_to_gaussians_2d import one_2d_gaussian
    import matplotlib.pyplot as plt

    N = 100
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    cov = np.array([[1, 0],
                    [0, 1]])
    mu = [0, 0]
    a = 1

    data = np.stack([X, Y], axis=-1)

    # gauss_2d = one_2d_gaussian(data, a, mu, cov)
    gauss_2d = one_2d_gaussian(data, a, mu[0], mu[1], cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1])
    gauss_model, params = get_n_2d_gaussians_model(gauss_2d, x=X, y=Y, n=2)

    # for param in gauss_model.param_names:

    out = gauss_model.fit(gauss_2d, params=params, x=X, y=Y)
    print(out.fit_report())
    plt.imshow(gauss_model.eval(out.params, x=X, y=Y))
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    fit_real()
