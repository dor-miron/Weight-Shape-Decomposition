from itertools import product

from scipy.optimize import curve_fit
from scipy import linalg
import numpy as np
from numpy import pi

def one_2d_gaussian(x, a, mu0, mu1, cov00, cov01, cov10, cov11):
    """

    Args:
        x: shape (..., 2)
        a: shape (1)
        mu: shape (2)
        cov: covariance matrix - shape (2, 2)

    Returns:

    """
    mu = np.array([mu0, mu1])
    cov = np.array([[cov00, cov01],
                   [cov10, cov11]])
    trans = x - mu
    try:
        cov_inv = linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = linalg.inv(cov+np.array([[0.01,0], [0,0]]))

    normal = np.sqrt( (2*pi) ** 2 * linalg.det(cov) )
    val = trans[..., np.newaxis, :] @ cov_inv @ trans[..., np.newaxis]
    result = np.squeeze(a * np.exp(-val) / normal)
    print('here')
    return result

def n_2d_gaussians_sum(x, y, a_list, mu_list, cov_list):
    gauss_list = [one_2d_gaussian(x, y, a, mu, cov)
                  for a, mu, cov in product(a_list, mu_list, cov_list)]
    return np.sum(gauss_list)

# def n_2d_gaussians_sum_for_scipy(x, y, a_list, mu_list, cov_list):
#     gauss_list = [one_2d_gaussian(x, y, a, mu, cov)
#                   for a, mu, cov in product(a_list, mu_list, cov_list)]
#     return np.sum(gauss_list)


def fit_to_gaussians(x, n):
    shape = np.array(x).shape
    X, Y = np.meshgrid(shape[0], shape[1], indexing='ij')
    data = np.stack([X, Y], axis=-1)

