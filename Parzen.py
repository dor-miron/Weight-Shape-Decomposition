from pathlib import Path
from typing import Callable
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import EcalDataIO
import os
import plotly.offline as pyo
pyo.init_notebook_mode()

# project_path = Path(__file__).parent
project_path = os.getcwd()
res_path = project_path # Path for saving 2d image results


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


def parzen(d, d_space, K_DIM, sigma, N, cut, s_count, disp=False, cmap='blackbody'):
    """
    Preform the Weight-Shape decomposition as defined in the article using the convolution methods.
    Parameters:
        d: The relevant data sample to be convolved.
        d_space: The data space spanned as a list of points.
        K_DIM: Kernel dimensions.
        sigma: The sigma param.
        N: The total number of showers in the sample.
        cut: The percentage of values to filter above from the final visualization.
        s_count: Display parameter for plotly. The larger it is the heavier the display.
        disp: Display the 3D image and 2D image of the sample.
        cmap: The colormap to display.
    """
    k_space, K = get_K(dim=K_DIM, sigma=sigma, disp=False)
    L = get_L(K, k_space, disp=False)

    Psy = conv3d(torch.Tensor(d), K)
    C_2 = conv3d(torch.Tensor(d), L)
    V = C_2 / Psy
    # V = np.nan_to_num(V)
    S = np.exp(-V)
    # W = Psy / S
    # plt.hist(S.ravel(), bins=100)
    # plt.show()

    X, Y, Z = d_space[:, 0], d_space[:, 1], d_space[:, 2]
    # Apply threshold cut off by percentage
    cut_off = np.percentile(S, cut)

    # Generate S values histogram
    # plt.hist(S.flatten(), bins=100)
    # plt.show()
    # plt.clf()

    fig = go.Figure(data=go.Volume(
        x=X, y=Y, z=Z,
        value=S.ravel(),
        isomin=cut_off,
        isomax=np.max(S),
        opacity=0.1,
        # opacityscale=[[-0.5, 1], [-0.2, 0], [0.2, 0], [0.5, 1]],
        colorscale=cmap,
        surface_count=s_count,
    ))
    fig.update_layout(title=f'S: \n N={N}, K_dim = {K_DIM}, sigma={sigma},'
                            f' cut_precent={cut}, cutoff_value={cut_off:.2f}', scene_aspectmode='data')
    if disp:
        fig.show()
        
        # Generate 2d image of XZ plane while summing over Y axis
        p_space = np.stack([y.ravel() for y in np.mgrid[:220, :42]])
        X, Z = p_space[0, :], p_space[1, :]
        data = S[:, 0, :]

        for i in range(0, 22):
            data += S[:, i, :]

        plt.scatter(X, Z, c=data.ravel(), label='Data', cmap=cmap, marker=',')
        plt.title(f'S {N} sigma_{sigma}')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        plt.savefig(res_path + f'\\{N}_Shape_sig_{sigma}_file_{file}.png')
        plt.show()
        plt.clf()

    return fig


def get_data(data_dir, file, idx, s_count, disp=True, cmap='blackbody'):
    """
    Load a data sample.
    Parameters:
        data_dir: Path conatining the data files.
        file: Weather the file is 5 or 3.
        idx: The sample index.
        s_count: Display parameter for plotly. The larger it is the heavier the display.
        disp: Display the 3D image and 2D image of the sample.
        cmap: The colormap to display.
    """
    en_dep = EcalDataIO.ecalmatio(data_dir / f"signal.al.elaser.IP0{file}.edeplist.mat")
    energies = EcalDataIO.energymatio(data_dir / f"signal.al.elaser.IP0{file}.energy.mat")

    # d_tens = torch.zeros((110, 11, 21))  # Formatted as [x_idx, y_idx, z_idx]
    d_tens = torch.zeros((220, 22, 42))  # Formatted as [x_idx, y_idx, z_idx]

    key = list(en_dep.keys())[idx]
    tmp = en_dep[key]
    en = energies[key]
    N = len(en)

    for z, x, y in tmp:
        d_tens[2*x, 2*y, 2*z] = tmp[(z, x, y)]


    # To prevent division by zero - add a small value to the sample.
    d = d_tens.numpy() + 0.0000000001

    # d_space = np.stack([y.ravel() for y in np.mgrid[:110, :11, :21]] + [d.ravel()], axis=1)[:, 0:3]
    d_space = np.stack([y.ravel() for y in np.mgrid[:220, :22, :42]] + [d.ravel()], axis=1)[:, 0:3]

    X, Y, Z = d_space[:, 0], d_space[:, 1], d_space[:, 2]

    fig = go.Figure(data=go.Volume(
        x=X, y=Y, z=Z,
        value=d.ravel(),
        isomin=np.min(d),
        isomax=np.max(d),
        opacity=0.1,
        # opacityscale=[[-0.5, 1], [-0.2, 0], [0.2, 0], [0.5, 1]],
        colorscale=cmap,
        surface_count=s_count,
    ))
    fig.update_layout(title=f'Sample: N = {N}', scene_aspectmode='data')
    if disp:
        fig.show()

        # Generate 2d image of XZ plane while summing over Y axis
        p_space = np.stack([y.ravel() for y in np.mgrid[:220, :42]])
        X, Z = p_space[0, :], p_space[1, :]
        data = d[:, 0, :]
        
        for i in range(0, 22):
            data += d[:, i, :]

        plt.scatter(X, Z, c=data.ravel(), label='Data', cmap=cmap, marker=',')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title(f'sample {N}')
        plt.colorbar()
        plt.savefig(res_path + f'\\{N}_Sample_file_{file}.png')
        plt.show()
        plt.clf()

    return d, d_space, N, fig


if __name__ == '__main__':
    SCOUNT = 10
    file = 5
    C_m = 'gray_r'
    data_dir = Path("data\\")
    disp = True
    K = 14  # Kernel Size
    cut = 0.9   # Cut off percentage for S
    for idx in [276, 6, 718, 84]: # 05 file
        d, d_space, N, fig_1 = get_data(data_dir, file, idx, SCOUNT, disp=disp, cmap=C_m)
        for sigma in [1.1]:
            # sigma = 1.5   # Set a constant sigma
            fig_2 = parzen(d, d_space, K, sigma, N, cut, SCOUNT, disp=disp, cmap=C_m)

    # figures_to_html([fig_1, fig_2], N=N, K=K, sig=sigma)
