from os import path
from typing import Callable
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import EcalDataIO
import os
import streamlit as st
import plotly.offline as pyo

# pyo.init_notebook_mode()

# project_path = Path(__file__).parent
project_path = os.getcwd()
res_path = project_path  # Path for saving 2d image results


def psy(x, y, z, data, d_space, sigma):
    """
    Calculate Psy((x,y,z)) as defined by the article.
    Parameters:
        x,y,z: the X vector space - lists or single point.
        data: The data points (myu in the article).
        d_space: a list of the data space points.
        sigma: The sigma parameter.
    """
    diff = np.linalg.norm((x, y, z) - d_space, axis=1, ord=2)
    gauss = np.exp(-diff / 2 * (sigma ** 2)) * data.ravel()
    return sum(gauss)


def prob(x, y, z, i: tuple, data, sigma, d_space, Psy: Callable):
    """
    Calculate P_i((x,y,z)) as defined by the article.
    Parameters:
        x,y,z: the X vector space - lists or single point.
        i: The relevant point in space to calculate P_i.
        data: The data points (myu in the article).
        d_space: a list of the data space points.
        sigma: The sigma parameter.
        Psy: The above Psy function.
    """

    X_i = (x, y, z)
    if data[i[0], i[1], i[2]] == 0:
        return 0
    diff = np.linalg.norm(np.asarray(X_i) - np.asarray(i), axis=0, ord=2)
    gauss = np.exp(-diff / 2 * (sigma ** 2)) * data[i[0], i[1], i[2]]
    return gauss / Psy(x, y, z, data, d_space=d_space, sigma=sigma)


def entropy(x, y, z, data, d_space, sigma, Prob: Callable, Psy: Callable):
    """
    Calculate H((x,y,z)) as defined by the article.
    Parameters:
        x,y,z: the X vector space - lists or single point.
        data: The data points (myu in the article).
        d_space: a list of the data space points.
        sigma: The sigma parameter.
        Prob: the above prob function.
        Psy: The above Psy function.
    """

    res = []
    for x_i, y_i, z_i in d_space:
        i = (int(x_i), int(y_i), int(z_i))
        P_i = Prob(x, y, z, i, data, sigma=sigma, d_space=d_space, Psy=Psy)
        if P_i == 0:
            continue
        else:
            E = -P_i * np.log(P_i)
            res.append(E)

    return sum(res)


def potential(x, y, z, data, d_space, sigma, Prob: Callable, Psy: Callable):
    """
    Calculate V((x,y,z)) as defined by the article.
    Parameters:
        x,y,z: the X vector space - lists or single point.
        data: The data points (myu in the article).
        d_space: a list of the data space points.
        sigma: The sigma parameter.
        Prob: the above prob function.
        Psy: The above Psy function.
    """

    res = []
    for x_i, y_i, z_i in d_space:
        i = (int(x_i), int(y_i), int(z_i))
        P_i = Prob(x, y, z, i, data, d_space=d_space, sigma=sigma, Psy=Psy)
        if P_i == 0:
            continue
        else:
            diff = np.linalg.norm(np.asarray((x, y, z)) - np.asarray(i), axis=0, ord=2)
            E = (diff * P_i) / 2 * (sigma ** 2)
            res.append(E)

    return sum(res)


###########################################################
# Currently unused.
def np_bivariate_normal_pdf(domain, mean, variance):
    X = np.arange(-domain + mean, domain + mean, variance)
    Y = np.arange(-domain + mean, domain + mean, variance)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = ((1. / np.sqrt(2 * np.pi)) * np.exp(-.5 * R ** 2))
    return X + mean, Y + mean, Z


###########################################################


#####################################################################
# Currently unused.
def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    kernel = np.outer(kernel, gauss)
    return kernel / np.sum(kernel)


#####################################################################


def get_K(dim=7, sigma=1, disp=False):
    """
    Generates the K kernel as defined in the article (Gaussian kernel)
    Parameters:
        dim: Kernel will be dimXdimXdim in dimensions.
        sigma: The sigma parameter.
        disp: Wether to display a 3D image of the kernel.
    """
    KK = int(dim / 2)

    kernel = np.zeros((dim, dim, dim))
    k_space = np.stack([y.ravel() for y in np.mgrid[:dim, :dim, :dim]] + [kernel.ravel()], axis=1)[:, 0:3]

    for x, y, z in k_space:
        diff = -np.linalg.norm(np.asarray((x, y, z) - np.asarray((KK, KK, KK))), axis=0, ord=2)
        a = np.exp(diff / (2 * (sigma ** 2)))
        kernel[int(x), int(y), int(z)] = a

    if disp:
        fig = go.Figure(data=go.Volume(
            x=k_space[:, 0], y=k_space[:, 1], z=k_space[:, 2],
            value=kernel.ravel(),
            isomin=np.min(kernel),
            isomax=np.max(kernel),
            opacity=0.1,
            surface_count=25,
        ))
        fig.show()

    return k_space, kernel


def get_L(K, k_space, disp=False):
    """
    Generates the L kernel as defined in the article.
    Parameters:
        K: The gaussian kernel from above.
        K_space: Spanning of the kernel space as points.
        disp: Wether to display a 3D image of the kernel.
    """
    L_kernel = -K * np.log(K)

    if disp:
        fig = go.Figure(data=go.Volume(
            x=k_space[:, 0], y=k_space[:, 1], z=k_space[:, 2],
            value=L_kernel.ravel(),
            isomin=np.min(L_kernel),
            isomax=np.max(L_kernel),
            opacity=0.1,
            surface_count=25,
        ))
        fig.show()

    return L_kernel


def conv3d(data, kernel):
    """Preform a 3D convolution on data using the given kernel, using the functional interface format of pytorch."""
    # Shaping
    data = (data.unsqueeze(0)).unsqueeze(0)
    kernel = torch.Tensor(kernel)
    kernel = (kernel.unsqueeze(0)).unsqueeze(0)

    # Conv and Numpy transform back
    output = F.conv3d(data, kernel, padding='same')
    output = ((output.squeeze(0)).squeeze(0)).numpy()

    return output


def figures_to_html(figs, filename="N=5 .html", N=0, K=0, sig=0):
    """Generate an HTML page containing all the figures given in figs(list)."""
    dashboard = open(filename, 'w')
    dashboard.write(f"<html><head></head><body><h1>N={N}, Kernel_size={K}, sigma={sig}</h1>" + "\n")
    for fig in figs:
        inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
        dashboard.write(inner_html)
    dashboard.write("</body></html>" + "\n")


def weight_shape_decomp(d, K_DIM, sigma):
    k_space, K = get_K(dim=K_DIM, sigma=sigma, disp=False)
    L = get_L(K, k_space, disp=False)

    psi = conv3d(torch.Tensor(d), K)
    C_2 = conv3d(torch.Tensor(d), L)
    V = C_2 / psi
    S = np.exp(-V)
    W = psi / S

    return psi, V, W, S


def plot_all(psi, V, W, S, PxS, P_PxS_dec, to_plot_bool=(True, True, True, True, True, True)):
    if sum(to_plot_bool) == 0:
        return

    to_plot_full_list = np.array([
        (psi, 'Psi'),
        (V, 'V'),
        (W, 'Weight'),
        (S, 'Shape'),
        (PxS, 'Psi x Shape'),
        (P_PxS_dec, 'PxS - P')
    ], dtype='object')

    x = 5 if True else 2
    to_plot_list = to_plot_full_list if (to_plot_bool is None) \
        else to_plot_full_list[to_plot_bool, :]

    fig, ax_list = plt.subplots(len(to_plot_list), 1, sharey='row')
    if not hasattr(ax_list, '__iter__'):
        ax_list = [ax_list]

    for ind, ax in enumerate(ax_list):
        data, title = to_plot_list[ind]
        img = ax.imshow(data.sum(axis=1).T, origin='lower', aspect='auto', interpolation='antialiased')
        ax.set_title(title)
        plt.colorbar(img, ax=ax)

    fig.tight_layout()
    return fig


def get_data(data_tuple, event_id, t=1):
    """ t is the expansion ratio """
    en_dep, energies = data_tuple

    tmp = en_dep[event_id]
    en = energies[event_id]

    x_dim, y_dim, z_dim = t * 110, t * 11, t * 21
    d_tens = torch.zeros((x_dim, y_dim, z_dim))

    for z, x, y in tmp:
        d_tens[t * x, t * y, t * z] = tmp[(z, x, y)]

    return d_tens + 1e-9


def main():

    file = 5
    K = 13  # Kernel Size
    cut = 0.9  # Cut off percentage for S

    data_dir = path.join(path.curdir, 'data')
    en_dep = EcalDataIO.ecalmatio(path.join(data_dir, f"signal.al.elaser.IP0{file}.edeplist.mat"))
    energies = EcalDataIO.energymatio(path.join(data_dir, f"signal.al.elaser.IP0{file}.energy.mat"))
    data_tuple = (en_dep, energies)

    elihu_chosen_event_ids = ['708', '813', '261', '103']
    event_id = '813'
    sigma = 1.1

    raw_data = get_data(data_tuple, event_id, t=1)
    P, V, W, S = weight_shape_decomp(raw_data, K, sigma)
    fig = plot_all(P, V, W, S, P*S, P - P*S)
    fig.show()

    # figures_to_html([fig_1, fig_2], N=N, K=K, sig=sigma)

if __name__ == '__main__':
    main()